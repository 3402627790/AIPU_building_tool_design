//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Tom Gao
 * @Date: 2022-05-11 16:50:32
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-05-11 17:52:03
 */

#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
namespace aipubt
{
// insert cast after eltwise for Z1 eltwise with LeakyRelu/PRelu/Clip
void adapt_eltwise_out_type(GraphPtr g)
{
    auto criteria = [](NodePtr node) -> bool {
        bool ret = true;
        auto target = get_target();
        ret = ret && (node->type() == OpType_Eltwise) &&
              (node->inputs().size() == 2) && (node->outputs().size() == 1) &&
              (Tensor::dtype_bits(node->inputs()[0]->type()) == 16) &&
              (Tensor::dtype_bits(node->inputs()[1]->type()) == 16) &&
              (Tensor::dtype_bits(node->outputs()[0]->type()) == 8);
        if (ret && (target->isainfo().type == ISA_ZY) && node->params().count("with_activation"))
        {
            std::string activation_type = node->params()["with_activation"];
            if ((activation_type == Constants::act_leakyrelu) || (activation_type == Constants::act_prelu) || (activation_type == Constants::act_clip))
            {
                return true;
            }
        }
        return false;
    };

    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto out_tensor = n->outputs()[0];
        TensorPtr elt_out = std::make_shared<Tensor>(n->name() + "_before_cast", n->inputs()[0]->shape(), n->inputs()[0]->type());
        n->remove_output(0);
        n->add_output(elt_out);
        auto cast_out = ops::cast(elt_out, n->outputs()[0]->type());
        auto cast = cast_out->node();
        cast->remove_output(0);
        cast->add_output(out_tensor);
        g->add_node(cast);
        ADD_LOCATION_BY_NODE(cast, n);
    }
}
}; // namespace aipubt

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
 * @Date: 2021-12-30 17:42:43
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2021-12-30 17:42:43
 */

#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include <algorithm>
namespace aipubt
{
void adapt_eltwise_in_type(GraphPtr g)
{
    auto criteria = [](NodePtr node) -> bool {
        bool ret = true;
        auto target = get_target();
        ret = ret && (node->type() == OpType_Eltwise) &&
              (node->inputs().size() == 2) && (node->outputs().size() == 1) &&
              (node->inputs()[0]->type() == Tensor::UINT8) &&
              (node->inputs()[1]->type() == Tensor::UINT16) &&
              (node->outputs()[0]->type() == Tensor::UINT16) &&
              (target->isainfo().type == ISA_ZY);
        return ret;
    };

    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto node_name = g->get_valid_node_name(n->name() + "_input_0_cast");
        NodePtr cast = Node::create(node_name, OpType_Cast);
        auto tensor_name = g->get_valid_tensor_name(cast->name());
        TensorPtr cast_out = std::make_shared<Tensor>(tensor_name, n->inputs()[0]->shape(), Tensor::UINT16);
        auto input1 = n->inputs()[1];
        cast->add_input(n->inputs()[0]);
        cast->add_output(cast_out);

        n->clear_inputs();
        n->add_input(cast_out);
        n->add_input(input1);
        g->add_node(cast);
        ADD_LOCATION_BY_NODE(cast, n);
    }
}
}; // namespace aipubt

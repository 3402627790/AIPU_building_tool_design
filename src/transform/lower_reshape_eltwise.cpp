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
 * @Author: Neo Wang
 * @Date: 2020-05-26 17:25:56
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-04-28 14:21:07
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>
#include <vector>

namespace aipubt
{

static bool criteria(const std::vector<NodePtr> &nodes)
{
    auto reshape = nodes[0];
    auto eltwise = nodes[1];
    if (reshape->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    auto eltwise_out_shape = eltwise->outputs()[0]->shape();
    auto reshape_in_shape = reshape->inputs()[0]->shape();
    if ((eltwise_out_shape.dim() == 3 || eltwise_out_shape.dim() == 2 ||
         (eltwise_out_shape.dim() == 4 && reshape_in_shape[0] > eltwise_out_shape[0])) &&
        reshape_in_shape.dim() == 4)
    {
        return true;
    }

    return false;
}
bool lower_reshape_eltwise(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Reshape}, {"1", OpType_Eltwise}},
        {{"0", "1"}},
    };
    bool is_changed = false;
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (!criteria(m))
            {
                continue;
            }
            /**
             *             2D/3D---\                2D/3D reshape1 4D-----\
             *                      |--eltwise ==>                         |--eltwise---4D reshape0 2D/3D
             * 4D reshape0 2D/3D---/                               4D-----/
             */
            auto reshape0 = m[0];
            auto eltwise = m[1];

            auto elt_out = eltwise->outputs()[0];
            auto re0_out = reshape0->outputs()[0];
            auto re0_inp = reshape0->inputs()[0];
            auto elt_inp = eltwise->inputs()[0];
            if (elt_inp == re0_out)
            {
                elt_inp = eltwise->inputs()[1];
            }

            auto re1_out_tensor = std::make_shared<Tensor>(g->get_valid_tensor_name(elt_inp->name() + "/elt"), re0_inp->shape(), elt_inp->type());
            re1_out_tensor->quantization() = elt_inp->quantization();
            auto reshape1 = ops::reshape(elt_inp, re0_inp->shape())->node();
            reshape1->clear_outputs();
            reshape1->add_output(re1_out_tensor);
            g->add_node(reshape1);
            ADD_LOCATION_BY_NODE(reshape1, m[0], m[1]);

            eltwise->add_input(re0_inp, eltwise->remove_input(re0_out));
            eltwise->add_input(re1_out_tensor, eltwise->remove_input(elt_inp));
            re0_out->quantization() = elt_out->quantization();
            re0_out->shape() = re0_inp->shape();
            eltwise->clear_outputs();
            eltwise->add_output(re0_out);

            reshape0->clear_outputs();
            reshape0->clear_inputs();
            reshape0->add_output(elt_out);
            reshape0->add_input(re0_out);

            is_changed |= true;
            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}; // namespace aipubt

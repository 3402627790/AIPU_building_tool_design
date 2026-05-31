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
    auto add = nodes[1];
    if (reshape->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    auto add_in0_shape = add->inputs()[0]->shape();
    auto add_in1_shape = add->inputs()[1]->shape();
    auto add_out_shape = add->outputs()[0]->shape();
    auto reshape_in_shape = reshape->inputs()[0]->shape();
    auto reshape_out_shape = reshape->outputs()[0]->shape();
    if (add_out_shape != reshape_out_shape)
    {
        return false;
    }
    if (add_in0_shape == add_out_shape)
    {
        // broad_cast add_in1
        // support [C]
        // support [1], not [1,1,1,1]
        if ((add_in1_shape.size() != 1 && add_in1_shape.size() != reshape_in_shape[-1]) || add_in1_shape.dim() != 1)
        {
            return false;
        }
    }
    else if (add_in1_shape == add_out_shape)
    {
        // broad_cast add_in0
        // support [C]
        // support [1], not [1,1,1,1]
        if ((add_in0_shape.size() != 1 && add_in0_shape.size() != reshape_in_shape[-1]) || add_in0_shape.dim() != 1)
        {
            return false;
        }
    }

    return true;
}
bool lower_reshape_add(GraphPtr g)
{
    std::set<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Reshape}, {"1", OpType_Add}},
         {{"0", "1"}}},
        {{{"0", OpType_Reshape}, {"1", OpType_Sub}},
         {{"0", "1"}}},
        {{{"0", OpType_Reshape}, {"1", OpType_Mul}},
         {{"0", "1"}}},
    };
    bool is_changed = false;
    for (auto &pattern : patterns)
    {
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
                 * [C]/[1]---\                 [C]/[1]-----\
                 *           |--add/sub/mul ==>            |--add/sub/mul---reshape
                 * reshape---/                        -----/
                 */
                auto reshape_lower = m[0];
                auto eltwise = m[1];

                auto elt_out = eltwise->outputs()[0];
                auto reshape_out = reshape_lower->outputs()[0];
                auto reshape_inp = reshape_lower->inputs()[0];

                eltwise->add_input(reshape_lower->inputs()[0], eltwise->remove_input(reshape_lower->outputs()[0]));
                reshape_lower->clear_inputs();
                reshape_lower->clear_outputs();

                eltwise->clear_outputs();
                eltwise->add_output(reshape_out);
                reshape_out->shape() = reshape_inp->shape();
                reshape_out->quantization() = elt_out->quantization();
                reshape_out->quantization().set_aipu_dtype(elt_out->type());
                reshape_out->set_type(elt_out->type());
                reshape_lower->add_input(reshape_out);
                reshape_lower->add_output(elt_out);

                is_changed |= true;
                is_fused = true;
            }
            if (!is_fused)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}
}; // namespace aipubt

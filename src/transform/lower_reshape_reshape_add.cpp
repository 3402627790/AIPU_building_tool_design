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
    auto n0 = nodes[0];
    auto n1 = nodes[1];
    if (n0->outputs()[0]->is_io_tensor() || n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (n0->is_dynamic() || n1->is_dynamic())
    {
        return false;
    }
    if (n0->inputs()[0]->shape().size() == 1 || n1->inputs()[0]->shape().size() == 1)
    {
        return true;
    }
    return false;
}
bool lower_reshape_reshape_add(GraphPtr g)
{
    std::set<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Reshape}, {"1", OpType_Reshape}, {"2", OpType_Add}},
         {{"0", "2"}, {"1", "2"}}},
        {{{"0", OpType_Reshape}, {"1", OpType_Reshape}, {"2", OpType_Sub}},
         {{"0", "2"}, {"1", "2"}}},
        {{{"0", OpType_Reshape}, {"1", OpType_Reshape}, {"2", OpType_Mul}},
         {{"0", "2"}, {"1", "2"}}},
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
                 * reshape---\                 reshape-----\
                 *           |--add/sub/mul ==>            |--add/sub/mul---reshape
                 * reshape---/                        -----/ 
                 */
                auto reshape0 = m[0];
                auto reshape1 = m[1];
                auto eltwise = m[2];
                NodePtr reshape_lower = reshape0;
                NodePtr reshape_hold = reshape1;
                if (reshape0->inputs()[0]->shape().size() == 1)
                {
                    reshape_lower = reshape1;
                    reshape_hold = reshape0;
                }

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
                reshape_lower->add_input(reshape_out);
                reshape_lower->add_output(elt_out);
                
                if (reshape_hold->outputs()[0]->shape().dim() != 1 && 
                    reshape_hold->outputs()[0]->shape().dim() != reshape_inp->shape().dim())
                {
                    std::vector<int> l(reshape_inp->shape().dim(), 1);
                    reshape_hold->outputs()[0]->shape().reshape(l);
                    reshape_hold->params()["shape"] = l;
                }

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

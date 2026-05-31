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

    std::vector<int> perm0 = n0->params()["perm"];
    std::vector<int> perm1 = n1->params()["perm"];
    if (perm0 != perm1)
    {
        return false;
    }
    return true;
}
bool lower_trans_trans_eltwise(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Transpose}, {"1", OpType_Transpose}, {"2", OpType_Eltwise}},
        {{"0", "2"}, {"1", "2"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
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
             * trans---\              -----\
             *          |--eltwise ==>      |--eltwise---trans
             * trans---/              -----/ 
             */
            auto trans0 = m[0];
            auto trans1 = m[1];
            auto eltwise = m[2];

            auto elt_out = eltwise->outputs()[0];
            auto trans_out = trans0->outputs()[0];
            auto trans_inp = trans0->inputs()[0];

            eltwise->add_input(trans0->inputs()[0], eltwise->remove_input(trans0->outputs()[0]));
            eltwise->add_input(trans1->inputs()[0], eltwise->remove_input(trans1->outputs()[0]));
            trans0->clear_inputs();
            trans1->clear_inputs();

            trans0->clear_outputs();
            trans1->clear_outputs();

            eltwise->clear_outputs();
            eltwise->add_output(trans_out);
            trans_out->shape() = (trans_inp->shape().layout() == DataLayout_Flat) ? trans1->inputs()[0]->shape() : trans_inp->shape();

            trans_out->quantization() = elt_out->quantization();
            trans0->add_input(trans_out);
            trans0->add_output(elt_out);

            g->remove_node(trans1);
            is_changed = true;
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

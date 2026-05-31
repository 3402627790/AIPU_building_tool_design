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
 * @Last Modified time: 2022-03-24 17:09:16
 */
#include "core/core.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

bool fuse_transpose(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Transpose}, {"1", OpType_Transpose}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    std::set<NodePtr> removed;
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            std::vector<int> perm0 = m[0]->params()["perm"];
            std::vector<int> perm1 = m[1]->params()["perm"];
            std::vector<int> perm;
            if (removed.count(m[0]) || removed.count(m[1]))
            {
                continue;
            }
            // TODO  add reshape to support not same dim case
            if (perm0.size() != perm1.size())
            {
                continue;
            }
            for (uint32_t i = 0; i < perm0.size(); ++i)
            {
                perm.push_back(perm0[perm1[i]]);
            }
            m[1]->params()["perm"] = perm;
            auto out = m[0]->outputs()[0];
            m[1]->clear_inputs();
            m[1]->add_input(m[0]->inputs()[0]);
            if (out->consumers().size() == 0 && !out->is_io_tensor())
            {
                m[0]->clear_inputs();
                m[0]->clear_outputs();
                g->remove_node(m[0]);
                removed.insert(m[0]);
            }
            is_changed = true;
            is_merged = true;
        }
        if (!is_merged)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}; // namespace aipubt

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
 * @Author: Xiaolong Zhang
 * @Date: 2023-06-2 14:15:49
 * @Last Modified by: Xialong Zhang
 * @Last Modified time: 2023-06-2 14:24:36
 */

#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{

static inline bool criteria(NodePtr n1, NodePtr n2)
{
    return n1->params()["method"] == "EQUAL" && n2->params()["method"] == "NOT";
}

bool fuse_equal_not(GraphPtr g)
{
    static aipubt::GraphMatcherPattern pattern = {
        {{"0", aipubt::OpType_Logical}, {"1", aipubt::OpType_Logical}},
        {{"0", "1"}},
    };
    aipubt::GraphMatcher matcher(pattern);
    matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);

    bool is_changed = false;
    for (auto &&m : matched)
    {
        if (!criteria(m[0], m[1]))
        {
            continue;
        }

        auto not_equal_t = ops::not_equal(m[0]->inputs()[0], m[0]->inputs()[1]);
        auto not_equal_n = not_equal_t->node();
        not_equal_n->clear_outputs();
        not_equal_n->add_output(m[1]->outputs());

        ADD_LOCATION_BY_NODE(not_equal_n, m[0], m[1]);
        g->remove_node(m[0]);
        g->remove_node(m[1]);
        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt
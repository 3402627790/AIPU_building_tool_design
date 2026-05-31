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
 * @Last Modified time: 2022-03-24 17:00:27
 */
#include "core/core.h"
#include "passes.h"

namespace aipubt
{
static inline bool creteria(NodePtr n1, NodePtr n2)
{
    if (n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    return true;
}

bool fuse_reshape(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Reshape}, {"1", OpType_Reshape}},
        {{"0", "1"}},
    };
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    std::set<NodePtr> removed;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (!creteria(m[0], m[1]))
            {
                continue;
            }
            if (removed.count(m[0]) || removed.count(m[1]))
            {
                continue;
            }
            m[1]->clear_inputs();
            m[1]->add_input(m[0]->inputs());
            if (m[1]->is_dynamic())
            {
                if (!m[1]->params().count("ds_shape"))
                {
                    std::vector<Expr> s = m[1]->outputs()[0]->shape().symbol();
                    m[1]->params()["ds_shape"] = s;
                }
                if (m[1]->params().count("ds_output_shape"))
                {
                    m[1]->params().erase("ds_output_shape");
                }
            }

            MODIFY_NODE(m[1]);
            auto out = m[0]->outputs()[0];
            if (out->consumers().size() == 0 && !out->is_io_tensor())
            {
                m[0]->clear_inputs();
                m[0]->clear_outputs();
                g->remove_node(m[0]);
                removed.insert(m[0]);
                ADD_LOCATION_BY_NODE(m[1], m[0]);
            }
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
} // namespace aipubt

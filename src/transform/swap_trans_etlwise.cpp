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
 * @Last Modified time: 2022-04-28 11:12:27
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static bool criteria(const std::vector<NodePtr> &nodes)
{
    auto n0 = nodes[0];
    auto n1 = nodes[2];
    if (n0->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (nodes[1]->inputs().size() != 2)
    {
        return false;
    }
    std::vector<int> perm0 = n0->params()["perm"];
    std::vector<int> perm1 = n1->params()["perm"];
    if (perm0.size() != perm1.size())
    {
        return false;
    }
    // rotate only and if only the two sides trans can be eliminated
    for (uint32_t i = 0; i < perm0.size(); ++i)
    {
        if ((int)i != perm0[perm1[i]])
        {
            return false;
        }
    }
    return true;
}

bool swap_trans_eltwise(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Transpose}, {"1", OpType_Eltwise}, {"2", OpType_Transpose}},
         {{"0", "1"}, {"1", "2"}}},
        {{{"0", OpType_Transpose}, {"1", OpType_Concat}, {"2", OpType_Transpose}},
         {{"0", "1"}, {"1", "2"}}},
    };
    bool is_changed = false;

    for (auto pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
        auto matched = matcher.match(g);
        std::set<NodePtr> viewed;
        for (auto &m : matched)
        {
            // eltwise node only shows one times
            if (!criteria(m) || viewed.count(m[1]))
            {
                continue;
            }
            viewed.insert(m[1]);
            /**
             * 0----------etlwise-----------3       0-----trans1----5-----eltwise-----trans2----3
             *             /   \                ==>                       /  \
             *            /     \                        /---trans0--2   /    \
             * 1-trans0--/2      \---trans1--4       1--/---------------/      \----------------4
             */
            auto trans0 = m[0];
            auto trans1 = m[2];
            auto elt = m[1];
            auto t1 = trans0->inputs()[0];
            auto t4 = trans1->outputs()[0];
            auto t3 = elt->outputs()[0];
            auto t0 = elt->inputs()[0];
            auto t2 = trans0->outputs()[0];
            if (t0 == t2)
            {
                t0 = elt->inputs()[1];
            }

            elt->clear_outputs();
            trans1->clear_outputs();
            trans1->clear_inputs();
            std::vector<int> perm0 = trans0->params()["perm"];

            elt->add_output(t4);
            auto t5 = std::make_shared<Tensor>(g->get_valid_tensor_name(elt->name()), t0->shape(), t0->type());
            std::vector<int> perm1 = trans1->params()["perm"];
            for (uint32_t i = 0; i < perm1.size(); ++i)
            {
                t5->shape()[i] = t0->shape()[perm1[i]];
            }
            trans1->add_input(t0);
            trans1->add_output(t5);
            MODIFY_NODE(trans1);

            auto reserve_trans0 = trans0->children().size() > 1;
            if (!reserve_trans0)
            {
                trans0->clear_outputs();
                trans0->clear_inputs();
                g->remove_node(trans0);
            }

            auto trans2 = ops::transpose(t4, trans0->params()["perm"])->node();
            trans2->remove_output(0);
            trans2->add_output(t3);
            ADD_LOCATION_BY_NODE(trans2, m[0], m[1], m[2]);

            elt->add_input(t1, elt->remove_input(t2));
            elt->add_input(t5, elt->remove_input(t0));
            MODIFY_NODE(elt);

            if (elt->type() == OpType_Concat)
            {
                int axis = elt->params()["axis"];
                elt->params()["axis"] = perm0[axis];
            }

            if (t3->consumers().size() == 0 && !t3->is_io_tensor())
            {
                trans2->clear_inputs();
                trans2->clear_outputs();
                g->remove_node(trans2);
            }
            is_changed = true;
        }
    }
    return is_changed;
}
}; // namespace aipubt

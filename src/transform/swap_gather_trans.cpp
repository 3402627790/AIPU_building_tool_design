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
 * @Last Modified time: 2023-03-27 11:25:36
 */
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
    if (n0->inputs().size() != 2)
    {
        return false;
    }
    if (!((n0->inputs()[0]->shape().dim() != 1) ^ (n0->inputs()[1]->shape().dim() != 1)))
    {
        return false;
    }
    int axis = n0->params()["axis"];
    if (axis < 0)
    {
        return false;
    }
    auto g_1d_idx = n0->inputs()[0]->shape().dim() == 1 ? 0 : 1;
    auto g_nd_idx = n0->inputs()[0]->shape().dim() == 1 ? 1 : 0;
    if (n0->inputs()[g_nd_idx]->shape().dim() != n0->outputs()[0]->shape().dim())
    {
        return false;
    }
    if (n0->inputs()[g_1d_idx]->shape()[0] != n0->inputs()[g_nd_idx]->shape()[axis])
    {
        return false;
    }
    if (n0->is_dynamic())
    {
        if (!n0->inputs()[g_nd_idx]->shape().symbol()[axis]->is_constant())
        {
            return false;
        }
        if (n0->inputs()[g_1d_idx]->shape()[0] !=
            (int)n0->inputs()[g_nd_idx]->shape().symbol()[axis]->value())
        {
            return false;
        }
    }

    return true;
}
bool swap_gather_trans(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Gather}, {"1", OpType_Transpose}},
        {{"0", "1"}}};
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
             *  (1D)--\                           (1D)-------------\
             *         |--gather---transpose ==>                   |--gather
             *  (ND)--/                           (ND)--transpose--/
             */
            auto gather = m[0];
            auto transpose = m[1];

            std::vector<int> perm = transpose->params()["perm"];
            int axis = gather->params()["axis"];
            auto _argindex = std::find(perm.begin(), perm.end(), axis);
            if (_argindex != perm.end())
            {
                axis = (int)(_argindex - perm.begin());
            }
            else
            {
                continue;
            }

            // auto g_1d_idx = gather->inputs()[0]->shape().dim() == 1 ? 0 : 1;
            auto g_nd_idx = gather->inputs()[0]->shape().dim() == 1 ? 1 : 0;

            auto g_nd_in = gather->inputs()[g_nd_idx]; // no change
            auto g_out = gather->outputs()[0];
            auto t_out = transpose->outputs()[0]; // no change

            gather->clear_outputs();
            gather->add_output(t_out);

            transpose->clear_inputs();
            transpose->clear_outputs();
            transpose->add_input(g_nd_in);
            transpose->add_output(g_out);
            gather->add_input(g_out, gather->remove_input(g_nd_in));
            g_out->shape() = t_out->shape();
            g_out->set_type(g_nd_in->type());
            g_out->quantization() = g_nd_in->quantization();

            gather->params()["axis"] = axis;

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

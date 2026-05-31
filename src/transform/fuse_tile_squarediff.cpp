//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Matt Chen
 * @Date: 2022-03-30 13:38:32
 * @Last Modified by: Matt Chen
 * @Last Modified time: 2022-07-05 10:40:45
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace aipubt
{

static inline bool criteria(NodePtr tile, NodePtr sq, int &c_dim)
{
    auto inp = tile->inputs()[0]->shape();
    int count_n1 = 0;
    for (int i = 0; i < inp.dim(); ++i)
    {
        if (inp[i] != 1)
        {
            count_n1++;
            c_dim = i;
        }
    }
    if (count_n1 > 1)
    {
        return false;
    }

    // h/w broadcast
    std::vector<int> repeats = tile->params()["repeats"];
    if (repeats[repeats.size() - 1] != 1)
    {
        return false;
    }

    // axis check, only support [1,1,c],[1,1,1,c]
    if (tile->inputs()[0]->shape().dim() != tile->outputs()[0]->shape().dim())
    {
        return false;
    }
    else
    {
        if (tile->inputs()[0]->shape().dim() == 3)
        {
            if (tile->inputs()[0]->shape()[1] != 1)
            {
                return false;
            }
        }
        if (tile->inputs()[0]->shape().dim() == 4)
        {
            if (tile->inputs()[0]->shape()[1] != 1 or tile->inputs()[0]->shape()[2] != 1)
            {
                return false;
            }
        }
    }

    bool both_tile = std::all_of(sq->inputs().begin(), sq->inputs().end(), [](const TensorPtr t)
                                 { return t->node()->type() == OpType_Tile; });
    if (both_tile)
    {
        return false;
    }
    if (sq->is_dynamic() &&
        tile->outputs()[0]->shape().symbol() != sq->outputs()[0]->shape().symbol())
    {
        return false;
    }
    return true;
}

bool fuse_tile_squareddiff(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Tile}, {"1", OpType_SquaredDifference}},
        {{"0", "1"}}};
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    auto matched = matcher.match(g);
    std::set<NodePtr> removed;
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            int c_dim = -1;
            if (!criteria(m[0], m[1], c_dim))
            {
                continue;
            }
            if (removed.count(m[0]) || removed.count(m[1]))
            {
                continue;
            }

            auto inp00 = m[0]->inputs()[0];
            auto oup00 = m[0]->outputs()[0];

            auto inp10 = m[1]->inputs()[0];
            auto inp11 = m[1]->inputs()[1];
            auto oup10 = m[1]->outputs()[0];

            m[1]->clear_inputs();
            m[1]->add_input(inp10);
            m[1]->add_input(inp00);

            if (oup00->consumers().size() == 0 && !oup00->is_io_tensor())
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
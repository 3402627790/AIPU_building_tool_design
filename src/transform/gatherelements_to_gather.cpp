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
 * @Author: Wuyang Zhao
 * @Date: 2025-07-31 17:47:45
 */

#include "builder/lowering/lowering.h"
#include <core/config.h>
#include "core/core.h"
#include "core/ops/ops.h"
#include <numeric>

namespace aipubt
{
static inline bool criteria(NodePtr tile, NodePtr n1)
{
    auto input = n1->inputs()[0];
    auto indices = n1->inputs()[1];
    auto output = n1->outputs()[0];
    int32_t axis = n1->params()["axis"];
    auto tile_inp = tile->inputs()[0];
    std::vector<int> repeats = tile->params()["repeats"];
    axis = axis >= 0 ? axis : input->shape().dim() + axis;

    // axis < input_dim - 1: Gather performance better than GatherElements
    if (axis == input->shape().dim() - 1)
    {
        return false;
    }
 
    if (indices != tile->outputs()[0] || tile_inp->shape().dim() == 1 || tile->outputs()[0]->consumers().size() != 1)
    {
        return false;
    }
    if (indices->shape().dim() != input->shape().dim() || 
        indices->shape() != output->shape())
    {
        return false;
    }

    for (int32_t i = 0; i < indices->shape().dim(); i++)
    {
        if (indices->shape()[i] != input->shape()[i] && axis != i)
        {
            return false;
        }
    }
    for (int32_t i = 0; i < tile_inp->shape().dim(); i++)
    {
        // repeat.size() == tile_inp->shape().dim()
        if (i <= axis && repeats[i] != 1)
        {
            return false;
        }
        if (i > axis && tile_inp->shape()[i] != 1)
        {
            return false;
        }
    }

    return true;
}

bool gatherelements_to_gather(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Tile}, {"1", OpType_GatherElements}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_convert = false;
        for (auto &&m : matched)
        {
            if (!criteria(m[0], m[1]))
            {
                continue;
            }
            /*
             [1x300x1] |                                 [1x300x1] |
                Tile(indices)   Input                           Reshape  Input
            [1x300x256] \         / [1x8400x256]           [1x300] \       / [1x8400x256]
                         \       /                  ==>             \     / 
                      GatherElements                                 Gather
                             | [1x300x256]                             | [1x300x256]     
            */
            
            auto inp = m[0]->inputs()[0];
            int32_t axis = m[1]->params()["axis"];
            axis = axis >= 0 ? axis : m[1]->inputs()[0]->shape().dim() + axis;
            std::vector<int> shape(axis + 1);
            for (int32_t i = 0; i < axis + 1; i++)
            {
                shape[i] = inp->shape()[i];
            }
            auto new_shape = TensorShape(shape);
            // rank_input + rank_indices - 1 -batch_dims_value == rank_o
            int32_t batch_dims = m[1]->inputs()[0]->shape().dim() + new_shape.dim() - m[1]->outputs()[0]->shape().dim() - 1; 

            auto reshape_node = ops::reshape(inp, new_shape)->node();
            if (enable_ds)
            {
                std::vector<Expr> shape(axis + 1);
                for (int32_t i = 0; i < axis + 1; i++)
                {
                    shape[i] = inp->shape().symbol()[i];
                }
                reshape_node->params()["ds_shape"] = shape;
            }
            reshape_node->infer_shape(true);
            g->add_node(reshape_node);
            ADD_LOCATION_BY_NODE(reshape_node, m[0]);
            
            auto outp = m[1]->outputs()[0];
            auto gather_node = ops::gather(m[1]->inputs()[0], reshape_node->outputs()[0], axis, batch_dims)->node();
            gather_node->remove_output(0);
            gather_node->add_output(outp);
            gather_node->infer_shape(true);
            g->add_node(gather_node);
            ADD_LOCATION_BY_NODE(gather_node, m[1]);
            m[0]->clear_inputs();
            m[0]->clear_outputs();
            g->remove_node(m[0]);
            m[1]->clear_inputs();
            m[1]->clear_outputs();
            g->remove_node(m[1]);

            is_convert = true;
            is_changed = true;
        }
        if (!is_convert)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}
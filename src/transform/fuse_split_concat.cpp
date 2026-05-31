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
 * =====================================================================================
 *
 *       Filename:  fuse_split_concat.cpp
 *
 *    Description:  fuse split+concat to equivalent transpose/reshape
 *    1. Input [N, H, C] -> Split [1, H, C] * N -> Concat [1, H, N*C]
 *    => Input [N, H, C] -> Reshape [1, N, H, C] -> Transpose [1, H, N, C] -> Reshape [1, H, N*C] (using Transpose2DPlugin)
 *    (or => Input [N, H, C] -> Transpose [H, N, C] -> Reshape [1, H, N*C] (using TransposePlugin))
 *    2. Input [1, H, N*C] -> Split [1, H, C] * N -> Concat [N, H, C]
 *    => Input [1, H, N*C] -> Reshape [1, H, N, C] -> Transpose [1, N, H, C] -> Reshape [N, H, C] (using Transpose2DPlugin)
 *    (or => Input [1, H, N*C] -> Reshape [H, N, C] -> Transpose [N, H, C] (using TransposePlugin))
 *    When H=1, we can further remove the split+concat node
 *    1. Input [N, 1, C] -> Split [1, 1, C] * N -> Concat [1, 1, N*C]
 *    => Input [N, 1, C] -> Reshape [1, 1, N*C]
 *    2. Input [1, 1, N*C] -> Split [1, 1, C] * N -> Concat [N, 1, C]
 *    => Input [1, 1, N*C] -> Reshape [N, 1, C]
 *
 *        Version:  1.0
 *        Created:  11/16/22 10:42:25
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>

namespace aipubt
{
static bool check_batch_to_channel(NodePtr split_node, NodePtr concat_node)
{
    if (split_node->inputs().size() != 1)
    {
        return false;
    }

    auto in_shape = split_node->inputs()[0]->shape();
    if (in_shape.dim() != 3)
    {
        return false;
    }

    int out_num = split_node->outputs().size();
    if (out_num <= 1)
    {
        return false;
    }
    if (in_shape[0] != out_num)
    {
        return false;
    }

    for (auto output : split_node->outputs())
    {
        auto consumers = output->consumers();
        if (std::find(consumers.begin(), consumers.end(), concat_node) == consumers.end())
        {
            return false;
        }

        auto out_shape = output->shape();
        if (out_shape.dim() != 3)
        {
            return false;
        }

        if (out_shape[0] != 1)
        {
            return false;
        }

        if (out_shape[1] != in_shape[1])
        {
            return false;
        }

        if (in_shape[2] != out_shape[2])
        {
            return false;
        }
    }

    if (concat_node->inputs().size() != split_node->outputs().size())
    {
        return false;
    }

    if (concat_node->outputs().size() != 1)
    {
        return false;
    }

    auto out_shape = concat_node->outputs()[0]->shape();
    if (out_shape.dim() != 3)
    {
        return false;
    }

    if (out_shape[0] != 1)
 *    2. Input [1, 1, N*C] -> Split [1, 1, C] * N -> Concat [N, 1, C]
 *    => Input [1, 1, N*C] -> Reshape [N, 1, C]
 *
 *        Version:  1.0
 *        Created:  11/16/22 10:42:25
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>

namespace aipubt
{
static bool check_batch_to_channel(NodePtr split_node, NodePtr concat_node)
{
    if (split_node->inputs().size() != 1)
    {
        return false;
    }

    auto in_shape = split_node->inputs()[0]->shape();
    if (in_shape.dim() != 3)
    {
        return false;
    }

    int out_num = split_node->outputs().size();
    if (out_num <= 1)
    {
        return false;
    }
    if (in_shape[0] != out_num)
    {
        return false;
    }

    for (auto output : split_node->outputs())
    {
        auto consumers = output->consumers();
        if (std::find(consumers.begin(), consumers.end(), concat_node) == consumers.end())
        {
            return false;
        }

        auto out_shape = output->shape();
        if (out_shape.dim() != 3)
        {
            return false;
        }

        if (out_shape[0] != 1)
        {
            return false;
        }

        if (out_shape[1] != in_shape[1])
        {
            return false;
        }

        if (in_shape[2] != out_shape[2])
        {
            return false;
        }
    }

    if (concat_node->inputs().size() != split_node->outputs().size())
    {
        return false;
    }

    if (concat_node->outputs().size() != 1)
    {
        return false;
    }

    auto out_shape = concat_node->outputs()[0]->shape();
    if (out_shape.dim() != 3)
    {
        return false;
    }

    if (out_shape[0] != 1)
    auto out_shape = concat_node->outputs()[0]->shape();
    if (out_shape.dim() != 3)
    {
        return false;
    }

    int axis = concat_node->params()["axis"];
    if (axis != 0)
    {
        return false;
    }

    int in_num = concat_node->inputs().size();
    if (in_num <= 1)
    {
        return false;
    }
    if (out_shape[0] != in_num)
    {
        return false;
    }

    for (auto input : concat_node->inputs())
    {
        auto in_shape = input->shape();
        if (in_shape.dim() != 3)
        {
            return false;
        }

        if (in_shape[0] != 1)
        {
            return false;
        }

        for (auto i = 1; i < 2; i++)
        {
            if (in_shape[i] != out_shape[i])
            {
                return false;
            }
        }
    }

    return true;
}

bool fuse_split_concat(GraphPtr graph)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Split}, {"1", OpType_Concat}},
        {{"0", "1"}},
    };
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);

    auto matched = matcher.match(graph);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            auto split_node = m[0];
            auto concat_node = m[1];

            auto is_batch_to_channel = check_batch_to_channel(split_node, concat_node);
            auto is_channel_to_batch = check_channel_to_batch(split_node, concat_node);
            if (!is_batch_to_channel && !is_channel_to_batch)
            {
                continue;
            }

            bool is_identical = (split_node->inputs()[0]->shape()[1] == 1);

            DEBUG() << "Fuse split: " << split_node->name()
                    << " concat: " << concat_node->name()
                    << " is_batch_to_channel: " << is_batch_to_channel
                    << " is_channel_to_batch: " << is_channel_to_batch
                    << " is_identical: " << is_identical;

            if (is_identical)
            {
                auto concat_output = concat_node->outputs()[0];
                auto reshape_node = ops::reshape(split_node->inputs()[0], concat_output->shape())->node();

                reshape_node->remove_output(0);
                reshape_node->add_output(concat_output);
                ADD_LOCATION_BY_NODE(reshape_node, split_node);
            }
            else
            {
                auto split_input = split_node->inputs()[0];
                auto split_input_shape = split_input->shape();
                TensorShape in_shape;
                if (is_batch_to_channel)
                {
                    in_shape = TensorShape({1, split_input_shape[0], split_input_shape[1], split_input_shape[2]}, DataLayout_NHWC);
                }
                else if (is_channel_to_batch)
                {
                    auto split_output_shape = split_node->outputs()[0]->shape();
                    int split_num = split_node->outputs().size();
                    in_shape = TensorShape({1, split_input_shape[1], split_num, split_output_shape[2]}, DataLayout_NHWC);
                }
                auto in_reshape_node = ops::reshape(split_input, in_shape)->node();
                ADD_LOCATION_BY_NODE(in_reshape_node, split_node);

                std::vector<uint32_t> perm = {0, 2, 1, 3};
                auto transpose_node = ops::transpose(in_reshape_node->outputs()[0], perm)->node();
                ADD_LOCATION_BY_NODE(transpose_node, split_node);

                auto concat_output = concat_node->outputs()[0];
                auto reshape_node = ops::reshape(transpose_node->outputs()[0], concat_output->shape())->node();

                reshape_node->remove_output(0);
                reshape_node->add_output(concat_output);
                ADD_LOCATION_BY_NODE(reshape_node, split_node);
            }

            graph->remove_node(split_node);
            graph->remove_node(concat_node);

            is_changed = true;
            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(graph);
    }

    return is_changed;
}
} // namespace aipubt

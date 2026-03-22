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
 *       Filename:  fuse_transpose_reshape_matmul.cpp
 *
 *    Description:  eliminate transpose and reshape nodes if it's the second input of matmul node.
 * The graph pattern matmul(Q, transpose(K)) is used by the scaled dot-product attention of transformer
 * network. It's natively supported by AIFF since Z3.
 *
 *        Version:  1.0
 *        Created:  10/28/22 15:00:33
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "builder/target.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "builder/lowering/lowering.h"

namespace aipubt
{
static bool check_transpose_node(NodePtr node)
{
    std::vector<int> perm = node->params()["perm"];
    return (node->outputs()[0]->shape().dim() == 3)
        && (perm[0] == 0)
        && (perm[1] == 2)
        && (perm[2] == 1);
}

static bool check_reshape_node(NodePtr node)
{
    auto in_shape = node->inputs()[0]->shape();
    if (in_shape.dim() != 3)
    {
        return false;
    }

    auto out_shape = node->outputs()[0]->shape();
    if (out_shape.dim() != 4)
    {
        return false;
    }

    int idx = -1;
    for (auto i = 0; i < 2; i++)
    {
        if (out_shape[i] == in_shape[0])
        {
            idx = i;
            break;
        }
    }
    if (idx < 0)
    {
        return false;
    }

    return (out_shape[1 - idx] == 1)
        && (out_shape[2] == in_shape[1]) && (out_shape[3] == in_shape[2]);
}

bool fuse_transpose_reshape_matmul(GraphPtr graph)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Transpose}, {"1", OpType_Reshape}, {"2", OpType_MatMul}},
        {{"0", "1"}, {"1", "2"}},
    };
    static GraphMatcher matcher(pattern);
    // TODO: mode?
    // matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);

    auto matched = matcher.match(graph);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            auto transpose_node = m[0];
            auto reshape_node = m[1];
            auto matmul_node = m[2];

            if (!check_transpose_node(transpose_node))
            {
                continue;
            }

            if (!check_reshape_node(reshape_node))
            {
                continue;
            }

            int matmul_input_idx = -1;
            auto reshape_output = reshape_node->outputs()[0];
            for (auto i = 0; i < 2; i++)
            {
                if (matmul_node->inputs()[i] == reshape_output)
                {
                    matmul_input_idx = i;
                    break;
                }
            }
            if (matmul_input_idx < 0)
            {
                continue;
            }

            DEBUG() << "Fuse transpose: " << transpose_node->name()
                    << " reshape: " << reshape_node->name()
                    << " matmul: " << matmul_node->name();

            matmul_node->remove_input(matmul_input_idx);
            auto transpose_input = transpose_node->inputs()[0];
            matmul_node->add_input(transpose_input, matmul_input_idx);
            if (matmul_input_idx == 0)
            {
                matmul_node->params()["trans_a"] = !matmul_node->params()["trans_a"];
            }
            else
            {
                matmul_node->params()["trans_b"] = !matmul_node->params()["trans_b"];
            }

            auto transpose_output = transpose_node->outputs()[0];
            if (reshape_output->consumers().size() == 0)
            {
                graph->remove_node(reshape_node);
            }
            if (transpose_output->consumers().size() == 0)
            {
                graph->remove_node(transpose_node);
            }

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


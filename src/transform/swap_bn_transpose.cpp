//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2023 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Ryan Lai
 * @Date: 2023-11-30  13:15:32
 * @Last Modified by: Ryan Lai
 * @Last Modified time: 2023-11-30 15:34:00
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"

/***
 *
 * A special pass for codeformer
 *
 * It REVERSE some transpose of the "lower_transpose" pass
 * because in codeformer:
 * bn with axis 1 have bad performance
 *
 * TODO: remove me and refine lower_transpose
 */

namespace aipubt
{

static int get_out_axis(const NodePtr &n, int axis)
{
    int out_axis = axis;
    if (n->type() == OpType_Transpose)
    {
        std::vector<int> perm = n->params()["perm"];
        for (size_t i = 0; i < perm.size(); i++)
        {
            if (perm[i] == axis)
            {
                out_axis = i;
                break;
            }
        }
    }
    return out_axis;
}

/**
 * @brief swap_axis_with_child_transpose
 * @param node
 * @param transpose
 * @return
 */
bool swap_axis_with_child_transpose(NodePtr node, NodePtr transpose)
{
    if (1 != node->children().size())
        return false;
    if (node->children()[0] != transpose)
        return false;
    if (!node->params().count("axis"))
        return false;

    const TensorPtr sg_in = node->inputs()[0];
    const TensorPtr sg_out = transpose->outputs()[0];

    TensorPtr tensor_between_node_transpose = node->outputs()[0];
    tensor_between_node_transpose->shape() = sg_out->shape();

    // change axis for bn
    int ori_axis = node->params()["axis"];
    // axis == 3 have the best performance, NOT switch
    if (ori_axis == 3)
        return false;
    node->params()["axis"] = get_out_axis(transpose, ori_axis);

    // swap!
    transpose->clear_inputs();
    transpose->clear_outputs();
    node->clear_inputs();
    node->clear_outputs();
    transpose->add_input(sg_in);
    transpose->add_output(tensor_between_node_transpose);
    node->add_input(tensor_between_node_transpose);
    node->add_output(sg_out);

    return true;
}

/**
 * @brief swap_bn_transpose
 * @param g
 * @return
 */
bool swap_bn_transpose(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_BatchNorm}, {"1", OpType_Transpose}},
        {{"0", "1"}},
    };

    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    bool is_changed = false;
    auto matched = matcher.match(g);
    for (auto &&nodes : matched)
    {
        NodePtr node = nodes[0];
        NodePtr trans = nodes[1];
        swap_axis_with_child_transpose(node, trans);
        is_changed = true;
    }
    return is_changed;
}

} // namespace aipubt

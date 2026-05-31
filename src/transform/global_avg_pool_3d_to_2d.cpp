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
 *       Filename:  global_avg_pool_3d_to_2d.cpp
 *
 *    Description:  Convert GlobalAvgPool 3D node to multiple GlobalAvgPool 2D nodes
 *                  Given a GlobalAvgPool 3D node: [N, D, H, W, C] --> [N, 1, 1, 1, C]
 *                  we transform it to:
 *                  1. Reshape: [N, D, H, W, C] --> [N*D, H, W, C]
 *                  2. GlobalAvgPool 2D on H*W: [N*D, H, W, C] --> [N*D, 1, 1, C]
 *                  3. Reshape: [N*D, 1, 1, C] --> [N, 1, D, C]
 *                  4. GlobalAvgPool 2D on D: [N, 1, D, C] --> [N, 1, 1, C]
 *                  5. Reshape: [N, 1, 1, C] --> [N, 1, 1, 1, C]
 *       Refer to:  https://jira.armchina.com/browse/CP-26397
 *
 *        Version:  1.0
 *        Created:  11/04/25 15:44:54
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <array>
#include <string>
#include <utility>

namespace aipubt
{

static bool criteria(NodePtr node)
{
    if (node->type() != OpType_Pooling3D)
    {
        return false;
    }

    std::string method = node->params()["method"];
    if (method != "AVG")
    {
        return false;
    }

    auto input = node->inputs()[0];
    auto shape = input->shape();
    if (shape.dim() != 5)
    {
        return false;
    }

    std::array<std::pair<int, std::string>, 3> kernel_param_names{
        std::make_pair(1, "kernel_z"),
        std::make_pair(2, "kernel_y"),
        std::make_pair(3, "kernel_x"),
    };
    for (const auto &kv : kernel_param_names)
    {
        int shape_idx = kv.first;
        std::string param_name = kv.second;
        if (!node->params().count(param_name))
        {
            return false;
        }
        int kernel = node->params()[param_name];
        if (kernel != shape[shape_idx])
        {
            return false;
        }
    }

    std::array<std::string, 6> pad_param_names = {
        "pad_x_begin",
        "pad_x_end",
        "pad_y_begin",
        "pad_y_end",
        "pad_z_begin",
        "pad_z_end",
    };
    for (const auto &pad_param_name : pad_param_names)
    {
        if (!node->params().count(pad_param_name))
        {
            return false;
        }
        else
        {
            int pad = node->params()[pad_param_name];
            if (pad != 0)
            {
                return false;
            }
        }
    }

    for (const auto &tensor_list : {node->inputs(), node->outputs()})
    {
        for (const auto &tensor : tensor_list)
        {
            if (!Tensor::is_float(tensor->type()))
            {
                DEBUG() << "global_avg_pool_3d_to_2d doesn't support int IR for precision reasons.";
                return false;
            }
        }
    }

    if (node->is_dynamic())
    {
        auto symbol = input->shape().symbol();
        for (const auto &kv : kernel_param_names)
        {
            int shape_idx = kv.first;
            if (!symbol[shape_idx]->is_constant())
            {
                return false;
            }
        }
    }

    return true;
}

bool global_avg_pool_3d_to_2d(GraphPtr graph)
{
    bool is_changed = false;
    auto nodes = graph->nodes();
    for (auto node : nodes)
    {
        if (!criteria(node))
        {
            continue;
        }

        auto input_tensor = node->inputs()[0];
        auto input_shape = input_tensor->shape();
        auto input_shape_symbol = input_tensor->shape().symbol();

        std::vector<tongue::Expr> new_input_exprs;
        TensorShape new_input_shape({input_shape[0] * input_shape[1], input_shape[2], input_shape[3], input_shape[4]}, DataLayout_NHWC);
        if (node->is_dynamic())
        {
            new_input_exprs = {
                input_shape_symbol[0] * input_shape_symbol[1],
                input_shape_symbol[2],
                input_shape_symbol[3],
                input_shape_symbol[4],
            };
        }
        auto input_reshape_tensor = ops::reshape(input_tensor, new_input_shape, new_input_exprs);
        auto input_reshape_node = input_reshape_tensor->node();
        graph->add_node(input_reshape_node);
        ADD_LOCATION_BY_NODE(input_reshape_node, node);

        std::vector<int> kernel = {new_input_shape.h(), new_input_shape.w()};
        std::vector<int> strides = {1, 1};
        std::vector<int> dilations = {1, 1};
        auto hw_pool_tensor = ops::avg_pool(input_reshape_tensor, kernel, strides, dilations);
        auto hw_pool_node = hw_pool_tensor->node();
        graph->add_node(hw_pool_node);
        ADD_LOCATION_BY_NODE(hw_pool_node, node);

        TensorShape d_pool_input_shape({input_shape[0], 1, input_shape[1], input_shape[4]}, DataLayout_NHWC);
        std::vector<tongue::Expr> d_pool_input_exprs;
        if (node->is_dynamic())
        {
            d_pool_input_exprs = {
                input_shape_symbol[0],
                1,
                input_shape_symbol[1],
                input_shape_symbol[4],
            };
        }
        auto pool_reshape_tensor = ops::reshape(hw_pool_tensor, d_pool_input_shape, d_pool_input_exprs);
        auto pool_reshape_node = pool_reshape_tensor->node();
        graph->add_node(pool_reshape_node);
        ADD_LOCATION_BY_NODE(pool_reshape_node, node);

        kernel = {d_pool_input_shape.h(), d_pool_input_shape.w()};
        auto d_pool_tensor = ops::avg_pool(pool_reshape_tensor, kernel, strides, dilations);
        auto d_pool_node = d_pool_tensor->node();
        graph->add_node(d_pool_node);
        ADD_LOCATION_BY_NODE(d_pool_node, node);

        TensorShape new_output_shape({input_shape[0], 1, 1, 1, input_shape[4]}, DataLayout_NDHWC);
        auto output_reshape_tensor = ops::reshape(d_pool_tensor, new_output_shape);
        auto output_reshape_node = output_reshape_tensor->node();
        graph->add_node(output_reshape_node);
        ADD_LOCATION_BY_NODE(output_reshape_node, node);
        output_reshape_node->replace_output_by(output_reshape_tensor, node->outputs()[0]);

        graph->remove_node(node);

        is_changed = true;
    }
    return is_changed;
}
} // namespace aipubt

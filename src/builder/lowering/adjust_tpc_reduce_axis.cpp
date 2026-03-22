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
 *       Filename:  adjust_tpc_reduce_axis.cpp
 *
 *    Description:  Adjust reduce shape and axis if it's using TPC plugin
 *
 *    Assume that the reduce node is already processed so that always only one of the following conditions meets:
 *    * shape=1D, axis=[0]
 *    * shape=2D, axis=[0]/[1]
 *    * shape=3D, axis=[1]
 *    * shape=4D (uses AIFF instead)
 *
 *        Version:  1.0
 *        Created:  10/15/24 16:34:26
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
#include "core/config.h"
#include <algorithm>
#include <memory>
#include <vector>

namespace aipubt
{
static int get_actual_axis(int dim, int axis)
{
    if (axis < 0)
    {
        return axis + dim;
    }
    else
    {
        return axis;
    }
};

static bool criterion(NodePtr node)
{
    if (node->type() != OpType_Reduce)
    {
        return false;
    }

    auto dim = node->inputs()[0]->shape().dim();
    if (dim > 3)
    {
        return false;
    }

    std::vector<int> axis_vec = node->params()["axis"];
    if (axis_vec.size() != 1)
    {
        return false;
    }
    int axis = get_actual_axis(dim, axis_vec[0]);
    if (axis != dim - 1)
    {
        return true;
    }

    return false;
}

void adjust_tpc_reduce_axis(GraphPtr graph)
{
    auto nodes = graph->nodes();
    for (auto node : nodes)
    {
        if (!criterion(node))
        {
            continue;
        }

        auto input_tensor = node->inputs()[0];
        auto input_shape = input_tensor->shape();
        std::vector<int> old_axis_vec = node->params()["axis"];
        auto old_dim = input_shape.dim();
        std::for_each(old_axis_vec.begin(), old_axis_vec.end(), [old_dim](int &axis) {
            if (axis < 0)
            {
                axis += old_dim;
            }
        });

        std::vector<uint32_t> perm;
        std::vector<int> new_input_shape_vec(2, 1);
        std::vector<Expr> new_input_sym_shape_vec(2, 1);
        for (auto i = 0; i < input_shape.dim(); i++)
        {
            auto it = std::find(old_axis_vec.begin(), old_axis_vec.end(), i);
            if (it == old_axis_vec.end())
            {
                perm.push_back(i);
                new_input_shape_vec[0] *= input_shape[i];
                new_input_sym_shape_vec[0] *= input_shape.symbol()[i];
            }
        }
        for (auto axis : old_axis_vec)
        {
            perm.push_back(axis);
            new_input_shape_vec[1] *= input_shape[axis];
            new_input_sym_shape_vec[1] *= input_shape.symbol()[axis];
        }
        std::vector<int> new_output_shape_vec = new_input_shape_vec;
        new_output_shape_vec.back() = 1;

        auto transpose_tensor = ops::transpose(input_tensor, perm);
        auto transpose_node = transpose_tensor->node();
        ADD_LOCATION_BY_NODE(transpose_node, node);
        graph->add_node(transpose_node);
        transpose_node->infer_shape(true);

        TensorShape new_input_shape(new_input_shape_vec, DataLayout_Flat);
        auto input_reshape_node = ops::reshape(transpose_tensor, new_input_shape)->node();
        graph->add_node(input_reshape_node);
        ADD_LOCATION_BY_NODE(input_reshape_node, node);
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            input_reshape_node->params()["ds_shape"] = new_input_sym_shape_vec;
        }
        input_reshape_node->infer_shape(true);
        node->add_input(input_reshape_node->outputs()[0], node->remove_input(input_tensor));

        std::vector<int> new_axis_vec = {1};
        node->params()["axis"] = new_axis_vec;

        auto output_tensor = node->outputs()[0];
        auto output_shape = output_tensor->shape();
        TensorShape new_output_shape(new_output_shape_vec, DataLayout_Flat);
        auto new_output_tensor = std::make_shared<Tensor>(output_tensor->name() + "/reshape", new_output_shape, output_tensor->type());
        new_output_tensor->quantization() = output_tensor->quantization();
        node->remove_output(0);
        node->add_output(new_output_tensor);
        auto output_reshape_node = ops::reshape(new_output_tensor, output_shape)->node();
        output_reshape_node->remove_output(0);
        output_reshape_node->add_output(output_tensor);
        graph->add_node(output_reshape_node);
        ADD_LOCATION_BY_NODE(output_reshape_node, node);
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            std::vector<Expr> ds_shape = output_shape.symbol();
            output_reshape_node->params()["ds_shape"] = ds_shape;
        }
        output_reshape_node->infer_shape(true);

        node->infer_shape(true);
    }
}

}; // namespace aipubt

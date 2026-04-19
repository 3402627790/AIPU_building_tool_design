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
 * @Author: ALex Zheng
 * @Date: 2021-07-13 10:55:59
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-07-10 17:18:41
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    static bool batch_dims_to_positive(int32_t &batch_dims, int32_t rank)
    {
        int32_t batch_dims_tmp = batch_dims < 0 ? batch_dims + rank : batch_dims;
        if ((batch_dims_tmp < 0) || (batch_dims_tmp > rank))
        {
            return false;
        }
        batch_dims = batch_dims_tmp;
        return true;
    }
    TensorPtr gather(TensorPtr input,
                     TensorPtr indices,
                     int32_t axis,
                     int32_t batch_dims)
    {
        AXIS_CHECK(axis, input);
        if (!batch_dims_to_positive(batch_dims, indices->shape().dim()))
        {
            ERROR() << "batch_dims should be in [-" << indices->shape().dim()
                    << "," << indices->shape().dim() << ") of input, but now: " << axis;
            return nullptr;
        }
        if (batch_dims > axis)
        {
            ERROR() << "Batch_dims must be less than or equal to axis";
            return nullptr;
        }

        if ((batch_dims > input->shape().dim()) || batch_dims > indices->shape().dim())
        {
            ERROR() << "batch_dims  must be less than or equal to the rank (inputs), but now: " << batch_dims;
            return nullptr;
        }

        GraphPtr g = input->node()->graph();
        std::vector<int32_t> new_shape_dims;
        new_shape_dims.insert(new_shape_dims.begin(), input->shape().begin(), input->shape().begin() + axis);
        new_shape_dims.insert(new_shape_dims.begin() + axis, indices->shape().begin() + batch_dims, indices->shape().end());
        new_shape_dims.insert(new_shape_dims.begin() + axis + indices->shape().dim() - batch_dims,
                              input->shape().begin() + axis + 1, input->shape().end());

        TensorShape new_shape(new_shape_dims, input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("gather"), new_shape);
        add_tensor_info(input, output);

        auto node = Node::create(g->get_valid_node_name("gather"), OpType_Gather);
        node->add_input(input);
        node->add_input(indices);
        node->add_output(output);
        auto &params = node->params();
        params["axis"] = axis;
        params["batch_dims"] = batch_dims;

        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    TensorPtr gather_nd(TensorPtr input,
                        TensorPtr indices,
                        int32_t batch_dims)
    {
        GraphPtr g = input->node()->graph();
        std::vector<int32_t> new_shape_dims;
        int32_t indices_dim = indices->shape().dim();
        new_shape_dims.insert(new_shape_dims.begin(), indices->shape().begin(), indices->shape().begin() + indices_dim - 1);
        if (indices->shape()[indices_dim - 1] + batch_dims < input->shape().dim())
        {
            new_shape_dims.insert(new_shape_dims.begin() + indices_dim - 1, input->shape().begin() + indices->shape()[indices_dim - 1] + batch_dims, input->shape().end());
        }
        TensorShape new_shape(new_shape_dims);
        new_shape.set_layout(input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("gather_nd"), new_shape);
        add_tensor_info(input, output);
        auto node = Node::create(g->get_valid_node_name("gather_nd"), OpType_GatherND);
        node->add_input(input);
        node->add_input(indices);
        node->add_output(output);
        auto &params = node->params();
        params["batch_dims"] = batch_dims;

        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    TensorPtr embedding_lookup(TensorPtr params,
                               TensorPtr indices)
    {

        return gather(params, indices);
    }

    TensorPtr gather_elements(TensorPtr input, TensorPtr indices, int32_t axis)
    {
        AXIS_CHECK(axis, input);
        if (input->shape().dim() <= 1 || indices->shape().dim() <= 1)
        {
            ERROR() << "Both input and indices rank should be > 1, but now: " << input->shape().dim() << ", " << indices->shape().dim();
            return {};
        }
        GraphPtr g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("gather_elements"), indices->shape());
        add_tensor_info(input, output);
        auto node = Node::create(g->get_valid_node_name("gather_elements"), OpType_GatherElements);
        node->add_input(input);
        node->add_input(indices);
        node->add_output(output);
        node->params()["axis"] = axis;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

}; // namespace ops
}; // namespace aipubt
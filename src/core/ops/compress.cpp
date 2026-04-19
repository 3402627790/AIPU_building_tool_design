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
 * @Date: 2021-04-14 11:04:33
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:29:18
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    template <typename T>
    int32_t get_output_axis_dim(TensorPtr condition, int32_t axis)
    {
        int32_t output_dim_x = 0;
        auto data = condition->data<T>();
        if (data == nullptr)
            return 0;
        auto index_size = condition->size();
        for (size_t i = 0; i < index_size; i++)
        {
            if (data[i] == 1)
                output_dim_x++;
        }
        return output_dim_x;
    }

    TensorPtr compress(TensorPtr input,
                       TensorPtr condition,
                       int32_t axis)
    {
        // convert neg to pos,
        int32_t dim = input->shape().dim();
        axis = axis < 0 ? axis + dim : axis;
        if (axis < 0 || axis >= dim)
        {
            ERROR() << "Invalid axis, should be in rank [" << -dim << ", " << dim << ")!";
            return nullptr;
        }
        TensorPtr condition_data = nullptr;
        if (condition->node() != nullptr && condition->node()->type() == OpType_Constant &&
            condition->node()->constants().count("weights"))
        {
            condition_data = condition->node()->constants()["weights"];
        }
        if (condition_data == nullptr)
        {
            condition_data = condition;
        }
        if (condition_data->data() == nullptr)
        {
            ERROR() << "The condition tensor only support constants!";
            return nullptr;
        }

        int32_t output_dim_x = 0;
        switch (condition_data->type())
        {
        case Tensor::BOOL:
            output_dim_x = get_output_axis_dim<bool>(condition_data, axis);
            break;
        case Tensor::UINT8:
            output_dim_x = get_output_axis_dim<uint8_t>(condition_data, axis);
            break;
        case Tensor::INT8:
            output_dim_x = get_output_axis_dim<int8_t>(condition_data, axis);
            break;
        case Tensor::UINT16:
            output_dim_x = get_output_axis_dim<uint16_t>(condition_data, axis);
            break;
        case Tensor::INT16:
            output_dim_x = get_output_axis_dim<int16_t>(condition_data, axis);
            break;
        case Tensor::UINT32:
            output_dim_x = get_output_axis_dim<uint32_t>(condition_data, axis);
            break;
        case Tensor::INT32:
            output_dim_x = get_output_axis_dim<int32_t>(condition_data, axis);
            break;
        default:
            ERROR() << "Unsupport 'condition' data type, only for: BOOL,UINT8,INT8,UINT16,INT16,UINT32,INT32.\n";
            return nullptr;
        }

        if (output_dim_x == 0)
        {
            ERROR() << "Unsupport 'condition' data values, only for: 0,1.\n"
                    << "It must have a value, '1'.";
            return nullptr;
        }

        DEBUG() << "output_dim_x: " << output_dim_x;
        auto g = input->node()->graph();
        TensorShape output_shape(input->shape());
        output_shape.set_layout(input->shape().layout());
        output_shape[axis] = output_dim_x;
        DEBUG() << "output_shape: " << output_shape;
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("compress"), output_shape);
        add_tensor_info(input, output);

        auto node = Node::create(g->get_valid_node_name("compress"), OpType_Compress);
        node->add_input(input);
        node->add_input(condition);
        node->add_output(output);
        node->params()["axis"] = axis;
        g->add_node(node);
        quantize(node);
        return output;
    }

}; // namespace ops
}; // namespace aipubt
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
 * @Date: 2021-07-09 16:41:04
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-15 11:11:38
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr fully_connected(TensorPtr input,
                              TensorPtr filter,
                              TensorPtr bias,
                              const Activation &with_activation,
                              const Quantization &quant,
                              const TensorShape &output_shape)
    {
        GraphPtr g = input->node()->graph();

        if (filter->shape().dim() != 2)
        {
            ERROR() << "fully_connected only support 2-D filter";
            return nullptr;
        }

        TensorPtr new_input = input;

        // reshape to 2d
        bool is_reshape = false;
        if (input->shape().dim() != 2 || input->shape()[-1] != filter->shape()[-1])
        {
            uint32_t input_size = filter->shape()[1];
            if (input->shape().size() % input_size != 0)
            {
                ERROR() << "fully_connected input cannot reshape to 2D";
                return nullptr;
            }
            uint32_t batch_size = input->shape().size() / input_size;
            TensorShape new_shape(input->shape());
            new_shape.reshape(batch_size, input_size);
            new_input = reshape(input, new_shape);
            is_reshape = true;
        }

        TensorShape new_shape({new_input->shape()[0], filter->shape()[0]}, new_input->shape().layout());
        if (output_shape.dim() != 0)
        {
            if ((new_shape[0] != output_shape[0]) || (new_shape[1] != output_shape[1]))
            {
                WARN() << "The output shape is not correct, use calculated values!";
            }
        }

        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("fully_connected"), new_shape);
        add_tensor_info(new_input, output, quant);

        auto node = Node::create(g->get_valid_node_name("fully_connected"), OpType_FullyConnected);
        node->add_input(new_input);
        node->add_output(output);
        node->constants()["weights"] = filter;
        node->constants()["biases"] = bias;
        node->params()["num_output"] = output->shape()[-1];
        g->add_node(node);
        append_activation(node, with_activation);
        node->infer_shape(true);
        quantize(node);

        // reshape back
        if (is_reshape && output_shape.dim() != 0)
        {
            output = reshape(output, output_shape);
        }
        return output;
    }
}; // namespace ops
}; // namespace aipubt
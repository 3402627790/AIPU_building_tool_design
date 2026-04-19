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
 * @Last Modified time: 2022-08-13 11:34:59
 */

#include <core/compute_tensor.h>
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr col2im(TensorPtr input,
                     TensorPtr image_shape,
                     TensorPtr block_shape,
                     const std::vector<int32_t> &strides,
                     Padding padding,
                     const std::vector<int32_t> &dilations,
                     const Quantization &quant)
    {
        // check block image to be constant
        if ((image_shape->data<uint8_t *>() == nullptr) || (block_shape->data<uint8_t *>() == nullptr))
        {
            ERROR() << "Only image_shape and block_shape are supported as constant tensors.";
            return nullptr;
        }

        if ((input->shape().dim() != 2) && (input->shape().dim() != 3))
        {
            ERROR() << "Only supports input as [N, C * n-ary- product(block_shape), L], [C * n-ary- product(block_shape), L].";
            return nullptr;
        }

        if ((image_shape->shape().size() < 2) && (block_shape->shape().size() < 2))
        {
            ERROR() << "The image_shape and block_shape are a 1-dimensional tensor with size of at least 2.";
            return nullptr;
        }
        if (!Tensor::is_int(image_shape->type()))
        {
            ERROR() << "The image shape only support fixed point dtype.";
            return nullptr;
        }
        if (!Tensor::is_int(block_shape->type()))
        {
            ERROR() << "The block shape only support fixed point dtype.";
            return nullptr;
        }

        TensorPtr new_input = input;
        if (input->shape().dim() == 2)
        {
            TensorShape new_shape(input->shape());
            new_shape.reshape(1, input->shape()[0], input->shape()[1]);
            new_input = reshape(input, new_shape);
        }

        std::vector<int32_t> pads = {0, 0, 0, 0};
        for (size_t i = 0; (i < padding.size()) && (i < pads.size()); i++)
        {
            pads[i] = padding[i];
        }

        std::vector<int32_t> valid_dilation = {1, 1};
        for (size_t i = 0; (i < dilations.size()) && (i < valid_dilation.size()); ++i)
        {
            valid_dilation[i] = dilations[i];
        }
        std::vector<int32_t> valid_strides = {1, 1};
        for (size_t i = 0; (i < strides.size()) && (i < valid_strides.size()); ++i)
        {
            valid_strides[i] = strides[i];
        }

        int32_t batch_size = new_input->shape()[0];
        int32_t input_plane = new_input->shape()[1];
        int32_t kernel_batch = 1;
        int32_t kernel_height = 1;
        int32_t kernel_width = 1;
        if (block_shape->shape().size() == 2)
        {
            std::vector<int32_t> block = ComputeTensor(block_shape).astype(Tensor::Dtype::INT32).flatten();
            kernel_height = block[0];
            kernel_width = block[1];
        }
        else if (block_shape->shape().size() == 3)
        {
            std::vector<int32_t> block = ComputeTensor(block_shape).astype(Tensor::Dtype::INT32).flatten();
            kernel_batch = block[0];
            kernel_height = block[1];
            kernel_width = block[2];
        }

        int32_t output_plane = input_plane / (kernel_batch * kernel_height * kernel_width);
        if (output_plane == 0)
        {
            ERROR() << "Block shape error.";
            return nullptr;
        }
        std::vector<int32_t> output_dims = {batch_size, output_plane};
        std::vector<int32_t> image = ComputeTensor(image_shape).astype(Tensor::Dtype::INT32).flatten();
        for (size_t i = 0; i < image.size(); ++i)
        {
            output_dims.push_back(image[i]);
        }

        GraphPtr g = new_input->node()->graph();
        TensorShape output_shape(output_dims, new_input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("col2im"), output_shape);
        add_tensor_info(new_input, output, quant);

        auto node = Node::create(g->get_valid_node_name("col2im"), OpType_Col2Im);
        node->add_input(input);
        node->add_input(image_shape);
        node->add_input(block_shape);
        node->add_output(output);
        auto &params = node->params();
        params["kernel_y"] = kernel_height;
        params["kernel_x"] = kernel_width;
        params["pad_top"] = pads[0];
        params["pad_bottom"] = pads[1];
        params["pad_left"] = pads[2];
        params["pad_right"] = pads[3];
        params["dilation_y"] = valid_dilation[0];
        params["dilation_x"] = valid_dilation[1];
        params["stride_y"] = valid_strides[0];
        params["stride_x"] = valid_strides[1];
        g->add_node(node);
        quantize(node);

        return output;
    }

}; //namespace ops
}; // namespace aipubt

//
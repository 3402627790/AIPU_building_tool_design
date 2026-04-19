
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
 * @Date: 2021-07-09 15:03:00
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:33:09
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr depthwise_conv2d(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               const std::vector<int32_t> &pads,
                               const Activation &with_activation,
                               const std::vector<int32_t> &dilations,
                               const Quantization &quant,
                               int32_t quantization_dim,
                               const TensorShape &output_shape)
    {
        // When input[c] is 1, we replace dwconv with conv, because HWA will not handle this situation.
        if (input->shape().c() == 1)
        {
            return conv2d(input, filter, bias, strides, pads,
                          with_activation, dilations, quant, output_shape);
        }

        GraphPtr g = input->node()->graph();
        std::vector<int32_t> valid_strides = {1, 1};
        if (extract_valid_stride(strides, valid_strides) < 0)
        {
            ERROR() << "Unsupport strides size";
            return nullptr;
        }
        std::vector<int32_t> padding = {0, 0, 0, 0};
        for (size_t i = 0; (i < padding.size()) && (i < pads.size()); i++)
        {
            padding[i] = pads[i];
        }

        std::vector<int32_t> dilation_new = {1, 1};
        for (size_t i = 0; (i < dilations.size()) && (i < dilation_new.size()); i++)
        {
            dilation_new[i] = dilations[i];
        }
        // cal output shape
        int32_t output_n = input->shape().n();
        int32_t output_h = compute_out_size(input->shape().h(), filter->shape().h(), valid_strides[0],
                                            dilation_new[0], padding[0], padding[1]);
        int32_t output_w = compute_out_size(input->shape().w(), filter->shape().w(), valid_strides[1],
                                            dilation_new[1], padding[2], padding[3]);
        int32_t output_c = filter->shape().n();
        if (output_shape.dim() != 0)
        {
            if ((output_shape.n() != output_n) || (output_shape.h() != output_h) || (output_shape.w() != output_w) || (output_shape.c() != output_c))
                WARN() << "The output shape is not correct, use calculated values!";
        }
        if (filter->shape()[0] != 1)
        {
            int32_t scale_size = static_cast<int32_t>(filter->quantization().scales().size());

            if (scale_size != filter->shape()[0])
            {
                float file_scale = filter->quantization().scale();
                for (int32_t i = scale_size; i < filter->shape()[0]; i++)
                {
                    filter->quantization().push_scale(file_scale);
                }

                float bias_scale = bias->quantization().scale();
                for (int32_t i = scale_size; i < filter->shape()[0]; i++)
                {
                    bias->quantization().push_scale(bias_scale);
                }
            }
        }

        TensorShape new_shape(output_n, output_h, output_w, output_c);
        new_shape.set_layout(input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("depthwise_conv2d"), new_shape);
        add_tensor_info(input, output, quant);
        output->quantization().set_quantization_dim(quantization_dim);
        auto node = Node::create(g->get_valid_node_name("depthwise_conv2d"), OpType_DepthwiseConv);
        node->add_input(input);
        node->add_output(output);
        auto &params = node->params();
        params["kernel_y"] = filter->shape().h();
        params["kernel_x"] = filter->shape().w();
        params["stride_y"] = valid_strides[0];
        params["stride_x"] = valid_strides[1];
        params["pad_top"] = padding[0];
        params["pad_bottom"] = padding[1];
        params["pad_left"] = padding[2];
        params["pad_right"] = padding[3];
        params["dilation_y"] = dilation_new[0];
        params["dilation_x"] = dilation_new[1];
        params["group"] = input->shape().c();
        params["num_output"] = output->shape().c();
        params["multiplier"] = output->shape().c() / input->shape().c();

        node->constants()["weights"] = filter;
        node->constants()["biases"] = bias;
        g->add_node(node);
        append_activation(node, with_activation);
        quantize(node);
        return output;
    }

    TensorPtr depthwise_conv2d(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               Padding padding,
                               const Activation &with_activation,
                               const std::vector<int32_t> &dilations,
                               const Quantization &quant,
                               int32_t quantization_dim,
                               const TensorShape &output_shape)
    {
        int32_t kernel_y = filter->shape().h(), kernel_x = filter->shape().w();
        std::vector<int32_t> valid_strides = {1, 1};
        if (extract_valid_stride(strides, valid_strides) < 0)
        {
            ERROR() << "Unsupport strides size";
            return nullptr;
        }
        if(padding.size() == 0)
        {
            padding.values.resize(4, 0);
            if (padding == "SAME")
            {
                cal_explicit_padding(input->shape().h(), valid_strides[0], kernel_y, padding, padding[0], padding[1]);
                cal_explicit_padding(input->shape().w(), valid_strides[1], kernel_x, padding, padding[2], padding[3]);
            }
        }

        return depthwise_conv2d(input, filter, bias, strides, padding.values, with_activation,
                                dilations, quant, quantization_dim, output_shape);
    }
}; //namespace ops
}; // namespace aipubt
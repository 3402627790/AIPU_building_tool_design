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
 * @Author: Aaron Gao
 * @Date: 2022-08-13 11:33:51
 * @Last Modified by:   Aaron Gao
 * @Last Modified time: 2022-08-13 11:33:51
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr _conv2d(TensorPtr input,
                      TensorPtr filter,
                      TensorPtr bias,
                      const std::vector<int32_t> &strides,
                      const std::vector<int32_t> &pads,
                      const Activation &with_activation,
                      const std::vector<int32_t> &dilations,
                      const Quantization &quant,
                      const TensorShape &output_shape,
                      const bool with_winograd = false,
                      const std::vector<int32_t> Fmnrs = {})
    {
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

        int32_t output_n = input->shape().n();
        int32_t output_h = compute_out_size(input->shape().h(), filter->shape().h(), valid_strides[0],
                                            dilation_new[0], padding[0], padding[1]);
        int32_t output_w = compute_out_size(input->shape().w(), filter->shape().w(), valid_strides[1],
                                            dilation_new[1], padding[2], padding[3]);
        int32_t output_c = filter->shape().n();
        if (output_shape.dim() != 0)
        {
            if ((output_shape.n() != output_n) || (output_shape.h() != output_h) || (output_shape.w() != output_w) || (output_shape.c() != output_c))
            {
                WARN() << "The output shape is not correct, use calculated values!";
            }
        }
        int32_t num_groups = input->shape().c() / filter->shape().c();
        if (input->shape().c() % filter->shape().c() != 0)
        {
            ERROR() << "Must be satisfied: input.channel = filter.channel * num_groups";
            return nullptr;
        }

        TensorShape new_shape(output_n, output_h, output_w, output_c);
        new_shape.set_layout(input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("conv2d"), new_shape);
        add_tensor_info(input, output, quant);

        auto node = Node::create(g->get_valid_node_name("conv2d"), OpType_Convolution);
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
        params["group"] = num_groups;
        params["num_output"] = output_c;

        // The with_winograd and Fmnrs is used for Winograd algorithm mode.
        if (with_winograd)
        {
            // Currently,lib only support 2D.
            if ((Fmnrs.size() == 2) &&
                (Fmnrs[0] == 2) &&
                (Fmnrs[1] == 3))
            {
                params["with_winograd"] = NodeParamValue(with_winograd);
                params["Fmnrs"] = NodeParamValue(Fmnrs);
            }
            else
            {
                WARN() << "The paramter \"Fmnrs\" is unqualified when with_winograd=True";
            }
        }

        node->constants()["weights"] = filter;
        node->constants()["biases"] = bias;
        g->add_node(node);
        node->infer_shape(true);
        append_activation(node, with_activation);

        quantize(node);
        return output;
    }

    TensorPtr conv2d(TensorPtr input,
                     TensorPtr filter,
                     TensorPtr bias,
                     const std::vector<int32_t> &strides,
                     Padding padding,
                     const Activation &with_activation,
                     const std::vector<int32_t> &dilations,
                     const Quantization &quant,
                     const TensorShape &output_shape)
    {
        int32_t kernel_y = filter->shape().h(), kernel_x = filter->shape().w();
        std::vector<int32_t> valid_strides = {1, 1};
        if (extract_valid_stride(strides, valid_strides) < 0)
        {
            ERROR() << "Unsupport strides size";
            return nullptr;
        }
        if (padding.size() == 0)
        {
            padding.values.resize(4, 0);
            if (padding == "SAME")
            {
                cal_explicit_padding(input->shape().h(), strides[0], kernel_y, padding, padding[0], padding[1]);
                cal_explicit_padding(input->shape().w(), strides[1], kernel_x, padding, padding[2], padding[3]);
            }
        }
        return _conv2d(input, filter, bias, strides, padding, with_activation, dilations, quant, output_shape);
    }

    TensorPtr conv2d_winograd(TensorPtr input,
                              TensorPtr filter,
                              TensorPtr bias,
                              const std::vector<int32_t> &strides,
                              Padding padding,
                              const Activation &with_activation,
                              const std::vector<int32_t> &dilations,
                              const Quantization &quant,
                              const TensorShape &output_shape,
                              const bool with_winograd,
                              const std::vector<int32_t> Fmnrs)
    {
        int32_t kernel_y = filter->shape().h(), kernel_x = filter->shape().w();
        std::vector<int32_t> valid_strides = {1, 1};
        if (extract_valid_stride(strides, valid_strides) < 0)
        {
            ERROR() << "Unsupport strides size";
            return nullptr;
        }
        if (padding.size() == 0)
        {
            padding.values.resize(4, 0);
            if (padding == "SAME")
            {
                cal_explicit_padding(input->shape().h(), strides[0], kernel_y, padding, padding[0], padding[1]);
                cal_explicit_padding(input->shape().w(), strides[1], kernel_x, padding, padding[2], padding[3]);
            }
        }
        return _conv2d(input, filter, bias, strides, padding, with_activation, dilations, quant, output_shape, with_winograd, Fmnrs);
    }

    TensorPtr _conv1d(TensorPtr input,
                      TensorPtr filter,
                      TensorPtr bias,
                      const std::vector<int32_t> &strides,
                      const std::vector<int32_t> &pads,
                      const Activation &with_activation,
                      const std::vector<int32_t> &dilations,
                      const Quantization &quant,
                      const TensorShape &output_shape)
    {
        // 1. input reshape to 4d
        if (input->shape().dim() != 3)
        {
            ERROR() << "input shape size must be 3,in fact is " << input->shape().size();
            return nullptr;
        }

        std::vector<int32_t> padding = {0, 0};
        for (size_t i = 0; (i < padding.size()) && (i < pads.size()); i++)
        {
            padding[i] = pads[i];
        }

        TensorShape new_input_shape(input->shape().n(), 1, input->shape().h(), input->shape().w());
        TensorPtr input_4d = reshape(input, new_input_shape);

        // 2. filter reshape to 4d
        TensorShape new_filter_shape(filter->shape().n(), 1, filter->shape().h(), filter->shape().w());
        new_filter_shape.set_layout(filter->shape().layout());
        filter->shape() = new_filter_shape;

        std::vector<int32_t> valid_strides = {1, 1};
        if (strides.size() > 0)
        {
            valid_strides[1] = strides[0];
        }
        std::vector<int32_t> dilation_new = {1, 1};
        if (dilations.size() > 0)
        {
            dilation_new[1] = dilations[0];
        }

        // 3. conv2d
        TensorShape new_output_shape(1, 1, 1, 1);
        if (output_shape.dim() != 0)
        {
            if (output_shape.dim() == 3)
            {
                new_output_shape.n() = output_shape.n();
                new_output_shape.w() = output_shape.h();
                new_output_shape.c() = output_shape.w();
            }
            else
            {
                WARN() << "conv1d shape must be 3";
            }
        }
        new_output_shape.set_layout(input->shape().layout());

        TensorPtr output = _conv2d(input_4d, filter, bias, valid_strides, {0, 0, pads[0], pads[1]},
                                   with_activation, dilation_new, quant, new_output_shape);
        // 4. output reshape to 3d
        TensorShape ret_shape(output->shape());
        ret_shape.reshape(output->shape().n(), output->shape().w(), output->shape().c());
        output = reshape(output, ret_shape);
        return output;
    }

    TensorPtr conv1d(TensorPtr input,
                     TensorPtr filter,
                     TensorPtr bias,
                     const std::vector<int32_t> &strides,
                     Padding padding,
                     const Activation &with_activation,
                     const std::vector<int32_t> &dilations,
                     const Quantization &quant,
                     const TensorShape &output_shape)
    {
        // 1. input reshape to 4d
        if (input->shape().dim() != 3)
        {
            ERROR() << "input shape size must be 3,in fact is " << input->shape().size();
            return nullptr;
        }

        int32_t kernel_x = filter->shape().h();
        std::vector<int32_t> valid_strides = {1};
        if (extract_valid_stride(strides, valid_strides) < 0)
        {
            ERROR() << "Unsupport strides size";
            return nullptr;
        }

        if (padding.size() == 0)
        {
            padding.values.resize(2, 0);
            if (padding == "SAME")
                cal_explicit_padding(input->shape().h(), valid_strides[0], kernel_x, padding, padding[0], padding[1]);
        }

        return _conv1d(input, filter, bias, strides, padding, with_activation, dilations, quant, output_shape);
    }

    TensorPtr _conv3d(TensorPtr input,
                      TensorPtr filter,
                      TensorPtr bias,
                      const std::vector<int32_t> &strides,
                      const std::vector<int32_t> &pads,
                      const Activation &with_activation,
                      const std::vector<int32_t> &dilations,
                      const Quantization &quant,
                      const TensorShape &output_shape)
    {
        aipu_assert(input->shape().dim() == 5);
        aipu_assert(filter->shape().dim() == 5);
        aipu_assert(bias->shape().dim() == 1);
        aipu_assert(strides.size() == 3);
        aipu_assert(pads.size() == 6);
        aipu_assert(dilations.size() == 1 || dilations.size() == 3);

        GraphPtr g = input->node()->graph();
        std::vector<int32_t> new_dilations(dilations);
        if (new_dilations.size() == 1)
        {
            new_dilations.push_back(new_dilations[0]);
            new_dilations.push_back(new_dilations[0]);
        }

        int32_t output_n = input->shape().n();
        int32_t output_d = compute_out_size(input->shape()[1], filter->shape()[3], strides[0],
                                            new_dilations[0], pads[0], pads[1]);
        int32_t output_h = compute_out_size(input->shape()[2], filter->shape()[1], strides[1],
                                            new_dilations[1], pads[2], pads[3]);
        int32_t output_w = compute_out_size(input->shape()[3], filter->shape()[2], strides[2],
                                            new_dilations[2], pads[4], pads[5]);
        int32_t output_c = filter->shape().n();

        TensorShape new_shape({output_n, output_d, output_h, output_w, output_c}, output_shape.layout());
        if (output_shape.dim() != 0)
        {
            if (output_shape != new_shape)
            {
                WARN() << "The output shape is not correct, use calculated values! provided: "
                       << output_shape << ", calculated: " << new_shape;
            }
        }
        new_shape.set_layout(input->shape().layout());

        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("conv3d"), new_shape);
        add_tensor_info(input, output, quant);

        auto node = Node::create(g->get_valid_node_name("conv3d"), OpType_Convolution3D);
        node->add_input(input);
        node->add_output(output);

        auto &params = node->params();
        params["kernel_z"] = filter->shape()[3];
        params["kernel_y"] = filter->shape()[1];
        params["kernel_x"] = filter->shape()[2];
        params["stride_z"] = strides[0];
        params["stride_y"] = strides[1];
        params["stride_x"] = strides[2];
        params["pad_z_begin"] = pads[0];
        params["pad_z_end"] = pads[1];
        params["pad_y_begin"] = pads[2];
        params["pad_y_end"] = pads[3];
        params["pad_x_begin"] = pads[4];
        params["pad_x_end"] = pads[5];
        params["dilation_z"] = new_dilations[0];
        params["dilation_y"] = new_dilations[1];
        params["dilation_x"] = new_dilations[2];
        params["group"] = 1;
        params["num_output"] = output_c;

        append_activation(node, with_activation);

        node->constants()["weights"] = filter;
        node->constants()["biases"] = bias;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    TensorPtr conv3d(TensorPtr input,
                     TensorPtr filter,
                     TensorPtr bias,
                     const std::vector<int32_t> &strides,
                     Padding padding,
                     const Activation &with_activation,
                     const std::vector<int32_t> &dilations,
                     const Quantization &quant,
                     const TensorShape &output_shape)
    {
        aipu_assert(input->shape().dim() == 5);
        aipu_assert(filter->shape().dim() == 5);
        aipu_assert(strides.size() == 1 || strides.size() == 3);

        auto new_strides(strides);
        if (new_strides.size() == 1)
        {
            new_strides.push_back(new_strides[0]);
            new_strides.push_back(new_strides[0]);
        }
        if (padding.size() == 0)
        {
            padding.values.resize(6, 0);
        }
        if (padding == "SAME")
        {
            cal_explicit_padding(input->shape()[1], new_strides[0], filter->shape()[3], padding, padding[0], padding[1]);
            cal_explicit_padding(input->shape()[2], new_strides[1], filter->shape()[1], padding, padding[2], padding[3]);
            cal_explicit_padding(input->shape()[3], new_strides[2], filter->shape()[2], padding, padding[4], padding[5]);
        }
        return _conv3d(input,
                       filter,
                       bias,
                       new_strides,
                       padding,
                       with_activation,
                       dilations,
                       quant,
                       output_shape);
    }

}; // namespace ops
}; // namespace aipubt
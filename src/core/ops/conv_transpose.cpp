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
 * @Date: 2022-08-13 11:34:25
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-10 21:49:46
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    static inline int32_t compute_out_size_transpose_conv(int32_t image_size,
                                                          int32_t filter_size,
                                                          int32_t stride,
                                                          int32_t dilation,
                                                          int32_t padding_head,
                                                          int32_t padding_tail)
    {
        return (image_size - 1) * stride +
               (filter_size - 1) * dilation + 1 - padding_head - padding_tail;
    }

    TensorPtr _conv2d_transpose(TensorPtr input,
                                TensorPtr filter,
                                TensorPtr bias,
                                const std::vector<int32_t> &strides,
                                const std::vector<int32_t> &pads,
                                int32_t group,
                                const Activation &with_activation,
                                const std::vector<int32_t> &dilations,
                                const Quantization &quant,
                                const TensorShape &output_shape)
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
        std::vector<std::pair<uint32_t, uint32_t>> pad_pads = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
        bool need_pad = !std::all_of(padding.begin(), padding.end(), [](int32_t pad)
                                     { return pad >= 0; });
        if (padding[0] < 0)
        {
            pad_pads[1].first = -padding[0];
            padding[0] = 0;
        }
        if (padding[1] < 0)
        {
            pad_pads[1].second = -padding[1];
            padding[1] = 0;
        }
        if (padding[2] < 0)
        {
            pad_pads[2].first = -padding[2];
            padding[2] = 0;
        }
        if (padding[3] < 0)
        {
            pad_pads[2].second = -padding[3];
            padding[3] = 0;
        }

        std::vector<int32_t> dilation_new = {1, 1};
        for (size_t i = 0; (i < dilations.size()) && (i < dilation_new.size()); i++)
        {
            dilation_new[i] = dilations[i];
        }

        int32_t output_n = input->shape().n();
        int32_t output_h = compute_out_size_transpose_conv(
            input->shape().h(), filter->shape().h(), strides[0],
            dilation_new[0], padding[0], padding[1]);
        int32_t output_w = compute_out_size_transpose_conv(
            input->shape().w(), filter->shape().w(), strides[1],
            dilation_new[1], padding[2], padding[3]);
        int32_t output_c = filter->shape().n() * group;
        if (output_shape.dim() != 0)
        {
            if ((output_shape.n() != output_n) ||
                (output_shape.h() != output_h) ||
                (output_shape.w() != output_w) ||
                (output_shape.c() != output_c))
                WARN() << "The output shape is not correct, use calculated values!";
        }

        TensorShape new_shape(output_n, output_h, output_w, output_c);
        new_shape.set_layout(input->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("conv2d_transpose"), new_shape);
        add_tensor_info(input, output, quant);

        auto node = Node::create(g->get_valid_node_name("conv2d_transpose"), OpType_ConvTranspose);
        node->add_input(input);
        node->add_output(output);
        int32_t max_stride_h = input->shape().h() + padding[0] + padding[1];
        int32_t max_stride_w = input->shape().w() + padding[2] + padding[3];
        if (valid_strides[0] > max_stride_h)
        {
            valid_strides[0] = max_stride_h;
        }
        if (valid_strides[1] > max_stride_w)
        {
            valid_strides[1] = max_stride_w;
        }

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
        params["group"] = group;
        params["num_output"] = output_c;

        node->constants()["weights"] = filter;
        node->constants()["biases"] = bias;
        g->add_node(node);
        append_activation(node, with_activation);
        quantize(node);
        if (need_pad)
        {
            output = pad(output, pad_pads);
        }

        return output;
    }

    TensorPtr conv2d_transpose(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               const std::vector<int32_t> &pads,
                               int32_t group,
                               const Activation &with_activation,
                               const std::vector<int32_t> &dilations,
                               const Quantization &quant,
                               const TensorShape &output_shape)
    {
        return _conv2d_transpose(
            input, filter, bias, strides, pads, group,
            with_activation, dilations, quant, output_shape);
    }

    TensorPtr conv2d_transpose(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               Padding padding,
                               int32_t group,
                               const Activation &with_activation,
                               const std::vector<int32_t> &dilations,
                               const Quantization &quant,
                               const TensorShape &output_shape)
    {
        std::vector<int32_t> valid_strides = {1, 1};
        if (extract_valid_stride(strides, valid_strides) < 0)
        {
            ERROR() << "Unsupport strides size";
            return nullptr;
        }

        int32_t kernel_y = filter->shape().h(), kernel_x = filter->shape().w();
        if (padding.size() == 0)
        {
            padding.values.resize(4, 0);
            if (padding == "SAME")
            {
                cal_explicit_padding(
                    input->shape().h(), valid_strides[0], kernel_y,
                    padding, padding[0], padding[1]);
                cal_explicit_padding(
                    input->shape().w(), valid_strides[1], kernel_x,
                    padding, padding[2], padding[3]);
            }
        }
        return _conv2d_transpose(
            input, filter, bias, strides, padding, group,
            with_activation, dilations, quant, output_shape);
    }

    TensorPtr _conv1d_transpose(TensorPtr input,
                                TensorPtr filter,
                                TensorPtr bias,
                                const std::vector<int32_t> &strides,
                                const std::vector<int32_t> &pads,
                                int32_t group,
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

        TensorShape new_input_shape(input->shape());
        new_input_shape.reshape(input->shape().n(), 1, input->shape().h(), input->shape().w());
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

        // 3.
        TensorShape new_output_shape(1, 1, 1, 1);
        if (output_shape.dim() == 0)
        {
            new_output_shape = output_shape;
        }
        else if (output_shape.dim() == 3)
        {
            new_output_shape.n() = output_shape.n();
            new_output_shape.w() = output_shape.h();
            new_output_shape.c() = output_shape.w();
        }
        else
        {
            WARN() << "conv1d_transpose shape must be 3";
        }

        new_output_shape.set_layout(input->shape().layout());
        TensorPtr output = _conv2d_transpose(
            input_4d, filter, bias, valid_strides, {0, 0, padding[0], padding[1]},
            group, with_activation, dilation_new, quant, new_output_shape);
        // output reshape to 3d
        TensorShape ret_shape(output->shape());
        ret_shape.reshape(output->shape().n(), output->shape().w(), output->shape().c());
        output = reshape(output, ret_shape);
        return output;
    }

    TensorPtr conv1d_transpose(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               Padding padding,
                               int32_t group,
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
            {
                cal_explicit_padding(
                    input->shape().h(), valid_strides[0], kernel_x,
                    padding, padding[0], padding[1]);
            }
        }
        return _conv1d_transpose(
            input, filter, bias, strides, padding, group,
            with_activation, dilations, quant, output_shape);
    }

    static bool cal_pads(const std::vector<int32_t> &origin_pads,
                         std::vector<int32_t> &pads_for_conv,
                         std::vector<std::pair<uint32_t, uint32_t>> &pads_for_pad)
    {
        for (size_t i = 0; (i < pads_for_conv.size()) && (i < origin_pads.size()); i++)
        {
            pads_for_conv[i] = origin_pads[i];
        }
        bool need_pad = false;
        for (uint32_t i = 0; i < pads_for_conv.size(); ++i)
        {
            if (pads_for_conv[i] < 0)
            {
                pads_for_pad[i / 2 + 1].first = i % 2 == 0 ? 0 - pads_for_conv[i] : pads_for_pad[i / 2 + 1].first;
                pads_for_pad[i / 2 + 1].second = i % 2 == 1 ? 0 - pads_for_conv[i] : pads_for_pad[i / 2 + 1].second;
                pads_for_conv[i] = 0;
                need_pad = true;
            }
        }

        return need_pad;
    }

    TensorPtr _conv3d_transpose(TensorPtr input,
                                TensorPtr filter,
                                TensorPtr bias,
                                const std::vector<int32_t> &strides,
                                const std::vector<int32_t> &pads,
                                int32_t group,
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

        std::vector<int32_t> pads_for_conv = {0, 0, 0, 0, 0, 0};
        std::vector<std::pair<uint32_t, uint32_t>> pads_for_pad = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
        bool need_pad = cal_pads(pads, pads_for_conv, pads_for_pad);


        std::vector<int32_t> new_dilations(dilations);
        if (new_dilations.size() == 1)
        {
            new_dilations.push_back(new_dilations[0]);
            new_dilations.push_back(new_dilations[0]);
        }

        int32_t output_n = input->shape().n();
        int32_t output_d = compute_out_size_transpose_conv(
            input->shape()[1], filter->shape()[3], strides[0],
            new_dilations[0], pads_for_conv[0], pads_for_conv[1]);
        int32_t output_h = compute_out_size_transpose_conv(
            input->shape()[2], filter->shape()[1], strides[1],
            new_dilations[1], pads_for_conv[2], pads_for_conv[3]);
        int32_t output_w = compute_out_size_transpose_conv(
            input->shape()[3], filter->shape()[2], strides[2],
            new_dilations[2], pads_for_conv[4], pads_for_conv[5]);
        int32_t output_c = filter->shape().n() * group;
        TensorShape new_shape({output_n, output_d, output_h, output_w, output_c}, output_shape.layout());
        if (output_shape.dim() != 0)
        {
            if (output_shape != new_shape)
            {
                WARN() << "The output shape is not correct, use calculated values, provided: "
                       << output_shape << ", calculated: " << new_shape;
            }
        }
        new_shape.set_layout(input->shape().layout());

        GraphPtr g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("conv3d_transpose"), new_shape);
        add_tensor_info(input, output, quant);

        auto node = Node::create(g->get_valid_node_name("conv3d_transpose"), OpType_ConvTranspose3D);
        node->add_input(input);
        node->add_output(output);
        // int32_t max_stride_d = input->shape()[1] + pads_for_conv[0] + pads_for_conv[1];
        // int32_t max_stride_h = input->shape()[2] + pads_for_conv[2] + pads_for_conv[3];
        // int32_t max_stride_w = input->shape()[3] + pads_for_conv[4] + pads_for_conv[5];
        // int32_t new_stride_d = strides[0] > max_stride_d ? max_stride_d : strides[0];
        // int32_t new_stride_h = strides[1] > max_stride_h ? max_stride_h : strides[1];
        // int32_t new_stride_w = strides[2] > max_stride_w ? max_stride_w : strides[2];
        auto &params = node->params();
        params["kernel_z"] = filter->shape()[3];
        params["kernel_y"] = filter->shape()[1];
        params["kernel_x"] = filter->shape()[2];
        params["stride_z"] = strides[0];
        params["stride_y"] = strides[1];
        params["stride_x"] = strides[2];
        params["pad_z_begin"] = pads_for_conv[0];
        params["pad_z_end"] = pads_for_conv[1];
        params["pad_y_begin"] = pads_for_conv[2];
        params["pad_y_end"] = pads_for_conv[3];
        params["pad_x_begin"] = pads_for_conv[4];
        params["pad_x_end"] = pads_for_conv[5];
        params["dilation_z"] = new_dilations[0];
        params["dilation_y"] = new_dilations[1];
        params["dilation_x"] = new_dilations[2];
        params["group"] = group;
        params["num_output"] = output_c;

        node->constants()["weights"] = filter;
        node->constants()["biases"] = bias;
        g->add_node(node);

        if (need_pad)
        {
            output = pad(output, pads_for_pad);
        }
        append_activation(node, with_activation);
        quantize(node);
        return output;
    }

    TensorPtr conv3d_transpose(TensorPtr input,
                               TensorPtr filter,
                               TensorPtr bias,
                               const std::vector<int32_t> &strides,
                               Padding padding,
                               int32_t group,
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
            if (padding == "SAME")
            {
                cal_explicit_padding(
                    input->shape()[1], new_strides[0], filter->shape()[3],
                    padding, padding[0], padding[1]);
                cal_explicit_padding(
                    input->shape()[2], new_strides[1], filter->shape()[1],
                    padding, padding[2], padding[3]);
                cal_explicit_padding(
                    input->shape()[3], new_strides[2], filter->shape()[2],
                    padding, padding[4], padding[5]);
            }
        }
        return _conv3d_transpose(input,
                                 filter,
                                 bias,
                                 new_strides,
                                 padding,
                                 group,
                                 with_activation,
                                 dilations,
                                 quant,
                                 output_shape);
    }

}; // namespace ops
}; // namespace aipubt
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
 * @Author: Samuel Han
 * @Date: 2023-08-16 17:12:36
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-08-21 15:18:33
 */

#include "quantization_transform/passes/passes.h"
#include "quantization_transform/utils/math_utils.h"

namespace qtlibn
{
static inline bool criteria(aipubt::NodePtr n)
{
    if (n->type() == aipubt::OpType_Resize && n->params()["method"] == "BILINEAR")
    {
        auto in_shape = n->inputs()[0]->shape();
        auto ot_shape = n->outputs()[0]->shape();

        if ((in_shape == aipubt::TensorShape(1, 1024, 1536, 3) &&
             ot_shape == aipubt::TensorShape(1, 384, 576, 3)) ||
            ((in_shape == aipubt::TensorShape(1, 384, 576, 1)) &&
             (ot_shape == aipubt::TensorShape(1, 1024, 1536, 1))))
        {
            int32_t i_h = in_shape[1];
            int32_t i_w = in_shape[2];
            int32_t i_c = in_shape[3];

            int32_t o_h = ot_shape[1];
            int32_t o_w = ot_shape[2];

            float ratio_x = float(o_w) / i_w;
            if (n->params().count("ratio_x"))
            {
                ratio_x = n->params()["ratio_x"];
            }
            float ratio_y = float(o_h) / i_h;
            if (n->params().count("ratio_y"))
            {
                ratio_y = n->params()["ratio_y"];
            }

            if (ratio_x <= 1 && ratio_y <= 1 && i_c > 8)
            {
                DEBUG() << "convert_resize_to_convolution only supports in_channel(=" << i_c << ") <= 8 when shrinking.";
                return false;
            }

            if (ratio_x > 1 && ratio_y > 1 && i_c > 1)
            {
                DEBUG() << "convert_resize_to_convolution only supports in_channel(=" << i_c << ") <= 1 when enlarging.";
                return false;
            }

            auto h_gcd = math_utils::gcd(i_h, o_h);
            auto w_gcd = math_utils::gcd(i_w, o_w);

            int32_t i_w_ratio = i_w / w_gcd;
            int32_t i_h_ratio = i_h / h_gcd;
            int32_t o_w_ratio = o_w / w_gcd;
            int32_t o_h_ratio = o_h / h_gcd;

            if (i_w_ratio != i_h_ratio || o_w_ratio != o_h_ratio)
            {
                DEBUG() << "convert_resize_to_convolution only supports height(=in_h/h_gcd or out_h/h_gcd) "
                        << "and width(=in_w/w_gcd or out_w/w_gcd) should be equal in input and output tensor, and h_gcd means the "
                        << "greatest common divisor between input height and output height. now in_h/h_gcd= " << i_h_ratio
                        << ", in_w/w_gcd= " << i_w_ratio << ", out_h/h_gcd= " << o_h_ratio << ", out_w/w_gcd = " << o_w_ratio;
                return false;
            }
            return true;
        }
    }
    return false;
}

aipubt::ComputeTensor generate_bias(const int32_t i_c, const int32_t o_w, const int32_t o_h)
{
    auto out_channels = i_c * o_w * o_h;
    std::vector<int32_t> bias_shape = {out_channels};
    aipubt::ComputeTensor conv_bias = aipubt::ComputeTensor::zeros(bias_shape, aipubt::Tensor::FP32);
    return conv_bias;
}
aipubt::ComputeTensor generate_weight(const int32_t i_w, const int32_t i_h, const int32_t i_c, const int32_t o_w, const int32_t o_h)
{
    aipubt::ComputeTensor src_h_ct = aipubt::ComputeTensor(nullptr);
    aipubt::ComputeTensor src_w_ct = aipubt::ComputeTensor(nullptr);

    auto get_location = [](const int32_t start, const int32_t end, const int32_t step, const int32_t factor) -> aipubt::ComputeTensor {
        aipubt::ComputeTensor ct = aipubt::ComputeTensor::arange(start, end, step) * factor;
        return ct;
    };

    int32_t filter_height = i_w > o_w || i_h > o_h ? i_h : i_h + 1;
    int32_t filter_width = i_w > o_w || i_h > o_h ? i_w : i_w + 1;

    aipubt::ComputeTensor src_loc_h = get_location(0, filter_height, 1, o_h);
    aipubt::ComputeTensor src_loc_w = get_location(0, filter_width, 1, o_w);
    aipubt::ComputeTensor dst_loc_h = get_location(0, o_h, 1, i_h);
    aipubt::ComputeTensor dst_loc_w = get_location(0, o_w, 1, i_w);
    std::vector<aipubt::ComputeTensor> src_loc = {src_loc_w, src_loc_h};
    std::vector<aipubt::ComputeTensor> dst_loc = {dst_loc_w, dst_loc_h};
    auto x_y_src_loc = aipubt::ComputeTensor::meshgrid(src_loc);
    auto x_y_dst_loc = aipubt::ComputeTensor::meshgrid(dst_loc);
    aipubt::ComputeTensor x_src_loc = x_y_src_loc[0];
    aipubt::ComputeTensor y_src_loc = x_y_src_loc[1];
    aipubt::ComputeTensor x_dst_loc = x_y_dst_loc[0];
    aipubt::ComputeTensor y_dst_loc = x_y_dst_loc[1];

    auto in_channels = i_c;
    auto out_channels = o_h * o_w * i_c;

    std::vector<int32_t> conv_shape = {out_channels, filter_height, filter_width, in_channels};
    aipubt::ComputeTensor conv_weight = aipubt::ComputeTensor::zeros(conv_shape, aipubt::Tensor::FP32);

    int32_t y_up = 0;
    int32_t y_down = 0;
    int32_t x_left = 0;
    int32_t x_right = 0;
    int32_t sub_out_channels = o_h * o_w;
    std::vector<int32_t> one_conv_shape = {sub_out_channels, filter_height, filter_width, 1};
    aipubt::ComputeTensor one_conv_weight = aipubt::ComputeTensor::zeros(one_conv_shape, aipubt::Tensor::FP32);

    for (int32_t c = 0; c < sub_out_channels; c++)
    {
        auto c_y = c / o_h;

        auto c_x = c - c_y * o_h;
        int64_t y_dst = x_y_dst_loc[1][c_y][c_x];
        int64_t x_dst = x_y_dst_loc[0][c_y][c_x];

        for (int32_t i = 0; i < i_w; i++)
        {
            int64_t yi0 = y_src_loc[i][0];
            int64_t yi10 = y_src_loc[i + 1][0];
            if (y_dst >= yi0 && y_dst < yi10)
            {
                y_up = i;
                y_down = i + 1;
            }
        }
        for (int32_t j = 0; j < i_h; j++)
        {
            int64_t xi0 = x_src_loc[0][j];
            if (x_dst >= xi0)
            {
                if (x_dst < (int64_t)x_src_loc[0][j + 1])
                {
                    x_left = j;
                    x_right = j + 1;
                }
            }
        }

        int64_t y_dst_v = y_dst_loc[c_y][0];
        int64_t x_dst_v = x_dst_loc[0][c_x];
        float y_up_v = y_src_loc[y_up][0];
        float y_down_v = y_src_loc[y_down][0];
        float x_left_v = x_src_loc[0][x_left];
        float x_right_v = x_src_loc[0][x_right];
        float y_sc = 1.0 - (y_dst_v - y_up_v) * 1.0 / (y_down_v - y_up_v);
        float x_sc = 1.0 - (x_dst_v - x_left_v) * 1.0 / (x_right_v - x_left_v);

        one_conv_weight[{{c, c + 1}, {y_up, y_up + 1}, {x_left, x_left + 1}, {0, 1}}] = y_sc * x_sc;
        // one_conv_weight[c][y_up][x_left][0] = y_sc * x_sc;
        one_conv_weight[c][y_up][x_right][0] = y_sc * (1 - x_sc);
        one_conv_weight[c][y_down][x_left][0] = (1 - y_sc) * x_sc;
        one_conv_weight[c][y_down][x_right][0] = (1 - y_sc) * (1 - x_sc);
        // INFO() << (one_conv_weight.flatten())[{16*c, 16*c+16}];
    }

    for (auto s = 0; s < sub_out_channels; s++)
    {
        for (auto ss = 0; ss < i_c; ss++)
        {
            int32_t ofs = s + ss * sub_out_channels;
            conv_weight[{{ofs, ofs + 1}, {}, {}, {ss, ss + 1}}] = one_conv_weight[{{s, s + 1}, {}, {}, {0, 1}}];
            // INFO() << conv_weight.flatten();
        }
    }

    return conv_weight;
}

void convert_resize_to_convolution(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        if (!criteria(node))
        {
            continue;
        }

        int32_t i_h = node->inputs()[0]->shape()[1];
        int32_t i_w = node->inputs()[0]->shape()[2];
        int32_t i_c = node->inputs()[0]->shape()[3];

        int32_t o_h = node->outputs()[0]->shape()[1];
        int32_t o_w = node->outputs()[0]->shape()[2];

        int32_t h_gcd = math_utils::gcd(i_h, o_h);
        int32_t w_gcd = math_utils::gcd(i_w, o_w);

        int32_t i_w_ratio = i_w / w_gcd;
        int32_t i_h_ratio = i_h / h_gcd;
        int32_t o_w_ratio = o_w / w_gcd;
        // int32_t o_h_ratio = o_h / h_gcd;

        if (i_w_ratio == 1 || i_h_ratio == 1)
        {
            i_w_ratio *= 2;
            i_h_ratio *= 2;
            o_w_ratio *= 2;
            // o_h_ratio *= 2;
        }

        auto i_scale = i_w_ratio;
        auto o_scale = o_w_ratio;

        aipubt::ComputeTensor conv_weight = generate_weight(i_scale, i_scale, i_c, o_scale, o_scale);
        aipubt::ComputeTensor conv_bias = generate_bias(i_c, o_scale, o_scale);

        aipubt::TensorPtr conv_weight_t = conv_weight;
        aipubt::TensorPtr conv_bias_t = conv_bias;
        conv_weight_t->quantization().set_data_unchanged(false);
        // conv_weight_t->quantization().set_primitive_dtype(aipubt::Tensor::FP32);
        conv_weight_t->quantization().set_bits(8);
        conv_weight_t->quantization().set_quantized_range(aipubt::Quantization::Restricted_Range);
        conv_weight_t->quantization().set_granularity(aipubt::Quantization::Granularity::Per_Tensor);
        conv_bias_t->quantization().set_data_unchanged(false);
        // conv_bias_t->quantization().set_primitive_dtype(aipubt::Tensor::FP32);
        conv_bias_t->quantization().set_bits(32);
        conv_bias_t->quantization().set_granularity(aipubt::Quantization::Granularity::Per_Tensor);
        auto pad_output_quantization = node->inputs()[0]->quantization();
        auto conv_output_quantization = node->outputs()[0]->quantization();

        const std::string base_name = node->name();
        node->set_attr("conv_from_resize_opt", true);

        auto resize_input = node->inputs()[0];
        auto resize_output = node->outputs()[0];
        resize_input->quantization().set_unquantifiable(true);

        auto stride_x = i_scale;
        auto stride_y = i_scale;
        aipubt::TensorPtr conv_in = resize_input;

        std::vector<aipubt::TensorPtr> added_tensor = {resize_input};
        if (o_h >= i_h || o_w >= i_w)
        {
            std::vector<std::pair<uint32_t, uint32_t>> pads = {{0, 0}, {0, 1}, {0, 1}, {0, 0}};
            float pad_constant_value = 0.0;
            auto pad_out = aipubt::ops::pad(resize_input, pads, aipubt::ops::PadMode::Reflect, pad_constant_value);
            pad_out->quantization() = pad_output_quantization;
            pad_out->quantization().set_unquantifiable(true);
            pad_out->node()->copy_attrs(node);
            conv_in = pad_out;

            added_tensor.push_back(pad_out);
        }
        auto padding = aipubt::ops::Padding(std::vector<int32_t>{0, 0, 0, 0});
        auto conv_out = aipubt::ops::conv2d(conv_in,
                                            conv_weight_t,
                                            conv_bias_t,
                                            std::vector<int32_t>{stride_y, stride_x},
                                            padding,
                                            aipubt::ops::Activation("NONE"),
                                            std::vector<int32_t>{1, 1});
        conv_out->quantization() = conv_output_quantization;
        conv_out->quantization().set_unquantifiable(true);
        conv_out->node()->copy_attrs(node);
        added_tensor.push_back(conv_out);

        if (i_c > 1)
        {
            auto split_outs = aipubt::ops::split(conv_out, i_c, -1);
            for (auto &so : split_outs)
            {
                so->quantization() = conv_output_quantization;
                so->quantization().set_unquantifiable(true);
                so->node()->copy_attrs(node);
                added_tensor.push_back(so);
            }

            auto concat_out = aipubt::ops::concat(split_outs, 0);
            concat_out->node()->copy_attrs(node);
            concat_out->quantization() = conv_output_quantization;
            concat_out->quantization().set_unquantifiable(true);
            added_tensor.push_back(concat_out);

            auto d2s_out = aipubt::ops::depth_to_space(concat_out, o_scale);
            d2s_out->node()->copy_attrs(node);
            d2s_out->quantization() = conv_output_quantization;
            d2s_out->quantization().set_unquantifiable(true);
            added_tensor.push_back(d2s_out);

            auto trans_out = aipubt::ops::transpose(d2s_out, std::vector<uint32_t>{3, 1, 2, 0});
            trans_out->node()->copy_attrs(node);
            auto trans_node = trans_out->node();
            trans_node->clear_outputs();
            trans_node->add_output(resize_output);
        }
        else
        {
            auto d2s_out = aipubt::ops::depth_to_space(conv_out, o_scale);
            d2s_out->node()->copy_attrs(node);
            auto d2s_node = d2s_out->node();
            d2s_node->clear_outputs();
            d2s_node->add_output(resize_output);
        }

        for (auto &add_t : added_tensor)
        {
            add_t->quantization().set_unquantifiable(false);
        }
        node->clear_inputs();
        node->clear_outputs();
        g->remove_node(node);
    }
}
} // namespace qtlibn

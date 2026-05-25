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
 * @Date: 2022-04-13 19:19:11
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-01-31 15:00:35
 */

#include "quantization_transform/ops/roialign.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{

void RoiAlignOp::quantize(aipubt::NodePtr node)
{
    auto _get_enlarge_scale = [](float raw_value,
                                 int32_t enlarge_bits,
                                 int32_t limitation_value,
                                 float &enlarge_value,
                                 int8_t &act_enlarge_bits) -> void {
        if (raw_value > limitation_value)
        {
            ERROR() << "now raw_value = " << raw_value << ", has already greater than limitation value = " << limitation_value
                    << ", and we will clip the raw_value to limitation_value directly";
            enlarge_value = limitation_value;
            act_enlarge_bits = 0;
            return;
        }

        auto v = pow(2, enlarge_bits);
        while (raw_value * v > limitation_value && enlarge_bits > 0)
        {
            enlarge_bits--;
            v = std::pow(2, enlarge_bits);
        }
        act_enlarge_bits = enlarge_bits;
        enlarge_value = raw_value * std::pow(2, enlarge_bits);
    };

    auto inp0_t = node->inputs()[0];
    auto inp1_t = node->inputs()[1];
    auto out_t = node->outputs()[0];

    // because the input zp maybe different with output zp in android roialign, so we calculate the output quantization
    // NoQuantization::quantize_transparent(out_t, inp0_t);
    Op::quantize(node);

    int32_t spatial_shift = 10;
    if (node->attrs().count("spatial_shift"))
    {
        spatial_shift = node->get_attr<aipubt::NodeParamValue>("spatial_shift");
    }

    if (!node->attrs().count(Constants::params_quantized) || !node->get_attr(Constants::params_quantized))
    {
        aipubt::NodeParamValue spatial_values = param_utils::get_param(node, "spatial_scale_value");
        float spatial_x_scale;
        int8_t spatial_x_shift;
        float spatial_y_scale;
        int8_t spatial_y_shift;
        int32_t limitation_value = 32767;
        _get_enlarge_scale((float)spatial_values[1], spatial_shift, limitation_value, spatial_x_scale, spatial_x_shift);
        _get_enlarge_scale((float)spatial_values[0], spatial_shift, limitation_value, spatial_y_scale, spatial_y_shift);
        if (spatial_x_shift != spatial_y_shift)
        {
            int8_t diff = spatial_x_shift - spatial_y_shift;
            if (diff > 0)
            {
                spatial_shift = spatial_y_shift;
                spatial_x_scale = round(spatial_x_scale / std::pow(2, diff));
            }
            else
            {
                spatial_shift = spatial_x_shift;
                spatial_y_scale = round(spatial_y_scale / std::pow(2, diff));
            }
        }
        node->params()["spatial_scale_value"] = std::vector<int32_t>{(int32_t)spatial_y_scale, (int32_t)spatial_x_scale};
        node->params()["spatial_scale_type"] = std::vector<aipubt::Tensor::Dtype>{aipubt::Tensor::UINT16, aipubt::Tensor::UINT16};
        node->params()["spatial_shift_value"] = spatial_shift;
        node->params()["spatial_shift_type"] = aipubt::Tensor::INT8;
    }

    auto inp1_aipu_scale = inp1_t->quantization().scale();
    auto inp1_bits = inp1_t->quantization().bits();
    float inp1_do_scale;
    int8_t inp1_do_shift;
    quant_utils::get_scale_approximation_params(1. / inp1_aipu_scale, inp1_do_scale, inp1_do_shift, inp1_bits);

    float out_do_scale;
    int8_t out_do_shift;
    auto inp0_aipu_scale = inp0_t->quantization().scale();
    auto out_aipu_scale = out_t->quantization().scale();
    auto out_bits = out_t->quantization().bits();
    auto sample = node->params()["sample"];
    int32_t sample_h = sample.arr[0];
    int32_t sample_w = sample.arr[1];
    float out_ts = out_aipu_scale / inp0_aipu_scale;
    if (sample_h > 0 && sample_w > 0)
    {
        out_ts /= (sample_h * sample_w);
    }
    quant_utils::get_scale_approximation_params(out_ts, out_do_scale, out_do_shift, out_bits);
    // this is for lib limitation: do_shift+spatial_shift should >16
    if ((out_do_shift + spatial_shift) <= 16)
    {
        int8_t diff = 17 - out_do_shift - spatial_shift;
        if (out_do_shift + diff > 31)
        {
            WARN() << "if meeting lib limiation, the output doshift will out of range 31.";
        }
        else if (out_do_scale * std::pow(2, diff) > 32767)
        {
            WARN() << "if meeting lib limiation, the output doscale will out of range 32767.";
        }
        else
        {
            out_do_shift += diff;
            out_do_scale = out_do_scale * pow(2, diff);
        }
    }

    auto unsigned_out_dtype = dtype_utils::bits_to_dtype(out_bits, false);
    auto unsigned_inp1_dtype = dtype_utils::bits_to_dtype(inp1_bits, false);

    node->params()["scale_value"] = std::vector<uint32_t>{static_cast<uint32_t>(out_do_scale), static_cast<uint32_t>(inp1_do_scale)};
    node->params()["shift_value"] = std::vector<int8_t>{out_do_shift, inp1_do_shift};
    node->params()["scale_type"] = std::vector<aipubt::Tensor::Dtype>{unsigned_out_dtype, unsigned_inp1_dtype};
    node->params()["shift_type"] = std::vector<aipubt::Tensor::Dtype>{aipubt::Tensor::INT8, aipubt::Tensor::INT8};

    if (sample_h > 0 && sample_w > 0)
    {
        int32_t quantization_bits = 12;
        if (node->attrs().count("quantization_bits"))
        {
            quantization_bits = node->get_attr<aipubt::NodeParamValue>("quantization_bits");
        }
        auto pooled_shape = node->params()["pooled_shape"];
        int32_t out_h = pooled_shape.arr[0];
        int32_t out_w = pooled_shape.arr[1];
        float out_hw_do_scale;
        int8_t out_hw_do_shift;
        auto q_dtype = dtype_utils::bits_to_dtype(quantization_bits, false);
        quant_utils::get_scale_approximation_params(1. / out_h, out_hw_do_scale, out_hw_do_shift, quantization_bits);
        node->params()["bin_scale_value"].push_back(static_cast<uint32_t>(out_hw_do_scale));
        node->params()["bin_scale_type"].push_back(q_dtype);
        node->params()["bin_shift_value"].push_back(out_hw_do_shift);
        node->params()["bin_shift_type"].push_back(aipubt::Tensor::INT8);
        quant_utils::get_scale_approximation_params(1. / out_w, out_hw_do_scale, out_hw_do_shift, quantization_bits);
        node->params()["bin_scale_value"].push_back(static_cast<uint32_t>(out_hw_do_scale));
        node->params()["bin_scale_type"].push_back(q_dtype);
        node->params()["bin_shift_value"].push_back(out_hw_do_shift);
        node->params()["bin_shift_type"].push_back(aipubt::Tensor::INT8);

        float sample_do_scale;
        int8_t sample_do_shift;

        float sample_hw_ts = 1.0 / (2 * sample_h);
        quant_utils::get_scale_approximation_params(sample_hw_ts, sample_do_scale, sample_do_shift, quantization_bits);
        node->params()["grid_scale_value"].push_back(static_cast<uint32_t>(sample_do_scale));
        node->params()["grid_scale_type"].push_back(q_dtype);
        node->params()["grid_shift_value"].push_back(sample_do_shift);
        node->params()["grid_shift_type"].push_back(aipubt::Tensor::INT8);
        sample_hw_ts = 1.0 / (2 * sample_w);
        quant_utils::get_scale_approximation_params(sample_hw_ts, sample_do_scale, sample_do_shift, quantization_bits);
        node->params()["grid_scale_value"].push_back(static_cast<uint32_t>(sample_do_scale));
        node->params()["grid_scale_type"].push_back(q_dtype);
        node->params()["grid_shift_value"].push_back(sample_do_shift);
        node->params()["grid_shift_type"].push_back(aipubt::Tensor::INT8);
    }
}

REGISTER_QUANTIZATION_OP(RoiAlign);

} // namespace qtlibn

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
 * @Date: 2022-04-15 13:52:34
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-23 19:19:00
 */

#include "quantization_transform/ops/logsoftmax.h"

namespace qtlibn
{

template <typename dtype>
static aipubt::TensorPtr generate_log_lut(uint32_t lut_len, std::map<std::string, float> param)
{
    dtype *lut = new dtype[lut_len];
    float output_qmin = (float)param["output_qmin"];
    float output_qmax = (float)param["output_qmax"];
    float aipu_output_scale = (float)param["aipu_output_scale"];
    float aipu_output_zp = (float)param["aipu_output_zp"];
    float step = float(1.0 / float(lut_len - 1));

    float value = 0;
    uint32_t log_lut_idx = 1;
    for (float idx = step; idx < 1.0 && log_lut_idx <= lut_len; idx += step, log_lut_idx++)
    {
        value = log(idx);
        value = quant_utils::linear_quantize_clip(value, aipu_output_scale, aipu_output_zp, output_qmin, output_qmax);
        lut[log_lut_idx] = (dtype)value;
    }
    lut[0] = lut[1];
    aipubt::TensorPtr lut_tensor = std::make_shared<aipubt::Tensor>("log_lut", (void *)lut, lut_len * sizeof(dtype));
    lut_tensor->set_type(dtype_utils::bytes_to_signed_qdtype(sizeof(dtype)));
    aipubt::TensorShape lut_shape(lut_len);
    lut_tensor->shape() = lut_shape;
    return lut_tensor;
}

static std::map<aipubt::Tensor::Dtype, std::function<aipubt::TensorPtr(uint32_t, std::map<std::string, float>)>> GENERATE_LOG_LUT{
    {aipubt::Tensor::INT8, generate_log_lut<int8_t>},
    {aipubt::Tensor::INT16, generate_log_lut<int16_t>},
    {aipubt::Tensor::INT32, generate_log_lut<int32_t>},
};

void LogSoftmaxOp::quantize(aipubt::NodePtr node)
{

    int32_t in_bits = node->inputs()[0]->quantization().bits();
    uint32_t lut_len = pow(2, in_bits);
    if (node->attrs().count("lut_items_in_bits"))
    {
        int32_t lut_bits = node->get_attr<aipubt::NodeParamValue>(Constants::lut_items_in_bits);
        lut_len = std::min(uint32_t(pow(2, lut_bits)), lut_len);
    }
    if (lut_len > 65535)
    {
        WARN() << "input tensor use more than 16 bits quantization, which will cause softmax's lut tensor very long";
    }

    // set output_type
    int32_t output_bits = node->outputs()[0]->quantization().bits();
    auto output_dtype = dtype_utils::bits_to_dtype(output_bits, true);
    dtype_utils::set_aipu_dtype(node->outputs()[0], output_dtype);
    std::string target = "Z2";
    if (node->attrs().count("min_compatible_zhouyi_target"))
    {
        target = (std::string)node->get_attr<aipubt::NodeParamValue>("min_compatible_zhouyi_target");
    }

    if (quant_utils::optimized_target_level(target) <= 0)
    {
        if (node->placeholders().size() < 1)
        {
            ERROR() << "please check the placeholders size, expect placeholders size is 1, "
                    << "but currently placeholders size is " << node->placeholders().size();
        }
        SoftmaxOp::generate_softmax_lut(node, node->inputs()[0]);
        aipubt::TensorPtr exp_lut = node->constants()["lut"];
        node->constants().erase("lut");

        // log lut
        auto placeholder_scheme = aipubt::Quantization::Symmetric;
        auto placeholder_dtype = dtype_utils::bits_to_dtype(output_bits, false);
        auto placeholder_granularity = aipubt::Quantization::Per_Tensor;
        auto placeholder_range = node->outputs()[0]->quantization().quantized_range();
        auto p = node->placeholders()[0];
        p->quantization().set_bits(output_bits);
        p->quantization().set_scheme(placeholder_scheme);
        p->quantization().set_granularity(placeholder_granularity);
        p->quantization().set_quantized_range(placeholder_range);
        dtype_utils::set_aipu_dtype(p, placeholder_dtype);

        Op::quantize(node);

        float do_scale;
        int8_t do_shift;
        float local_rescale = node->placeholders()[0]->quantization().scale();
        quant_utils::get_scale_approximation_params(local_rescale, do_scale, do_shift, output_bits);

        node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
        node->params()["scale_type"] = dtype_utils::bits_to_dtype(output_bits, false);
        node->params()["shift_value"] = do_shift;
        node->params()["shift_type"] = aipubt::Tensor::INT8;

        float aipu_output_scale = node->outputs()[0]->quantization().scale();
        float aipu_output_zp = node->outputs()[0]->quantization().offset();
        aipubt::Tensor::Dtype aipu_output_dtype = node->outputs()[0]->quantization().aipu_dtype();
        float output_qmin = dtype_utils::dtype_to_range_min(aipu_output_dtype);
        float output_qmax = dtype_utils::dtype_to_range_max(aipu_output_dtype);
        std::map<std::string, float> log_lut_param = {
            {"aipu_output_scale", aipu_output_scale},
            {"aipu_output_zp", aipu_output_zp},
            {"output_qmin", output_qmin},
            {"output_qmax", output_qmax}};
        aipubt::TensorPtr log_lut_tensor = GENERATE_LOG_LUT[aipu_output_dtype](lut_len, log_lut_param);
        node->constants()["lut_log"] = log_lut_tensor;
        node->constants()["lut_exp"] = exp_lut;
    }
    else
    {
        auto out = node->outputs()[0];
        std::vector<float_t> min_value = {-16.};
        std::vector<float_t> max_value = {0.};
        out->quantization().set_mins(min_value);
        out->quantization().set_maxs(max_value);
        dtype_utils::set_aipu_dtype(node->outputs()[0], true);

        Op::quantize(node);

        float aipu_output_scale = node->outputs()[0]->quantization().scale();
        float aipu_input_scale = node->inputs()[0]->quantization().scale();
        auto qmax_val = dtype_utils::bits_to_range_max(in_bits, false);

        // generate lut
        float step = float(qmax_val) / (lut_len - 1);
        aipubt::Tensor::Dtype lut_dtype = aipubt::Tensor::Dtype::FP32;
        float *lut = new float[lut_len];
        auto lut_idx = lut_len - 1;
        for (float val = 0; val <= qmax_val && lut_idx >= 0; val += step, lut_idx--)
        {
            lut[lut_idx] = exp(-val / aipu_input_scale);
        }
        aipubt::TensorPtr lut_exp_tensor = std::make_shared<aipubt::Tensor>("log_exp", (void *)lut, lut_len * sizeof(float));
        lut_exp_tensor->set_type(lut_dtype);
        aipubt::TensorShape lut_shape(lut_len);
        lut_exp_tensor->shape() = lut_shape;
        node->constants()["lut_exp"] = lut_exp_tensor;
        std::vector<float> scs = {float(aipu_output_scale / aipu_input_scale), float(aipu_output_scale)};
        node->params()["quantize_scale"] = scs;

        node->params()["scale_value"] = static_cast<uint32_t>(1);
        node->params()["scale_type"] = dtype_utils::bits_to_dtype(output_bits, false);
        node->params()["shift_value"] = 0;
        node->params()["shift_type"] = aipubt::Tensor::INT8;
    }
}

REGISTER_QUANTIZATION_OP(LogSoftmax);

} // namespace qtlibn

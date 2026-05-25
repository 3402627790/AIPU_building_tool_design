//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-03-23 19:22:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-02-02 17:47:54
 */

#include "quantization_transform/ops/reduce.h"
#include "core/compute_tensor.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{
void ReduceOp::forward(aipubt::NodePtr node)
{
    // Currently use float forward to generate placeholder for L2 method
    if (node->params()["method"] != "L2")
    {
        return;
    }

    std::vector<int32_t> axes = node->params()["axis"];

    auto inp = aipubt::ComputeTensor::create(node->inputs()[0]);
    auto pow2 = inp * inp;
    auto sum = pow2.sum(axes);
    auto out = sum.sqrt();

    node->placeholders().push_back(sum);

    node->outputs()[0]->share_data_from(*out);
    node->outputs()[0]->set_type(aipubt::Tensor::FP32);
}

template <typename dtype>
static aipubt::TensorPtr generate_sum_lut(uint32_t lut_len, std::map<std::string, float> param)
{
    dtype *lut = new dtype[lut_len];
    float output_qmin = (float)param["output_qmin"];
    float output_qmax = (float)param["output_qmax"];
    float aipu_output_scale = (float)param["aipu_output_scale"];
    float aipu_output_zp = (float)param["aipu_output_zp"];
    float step = float(1.0 / float(lut_len - 1));
    bool sign = std::is_signed<dtype>::value;

    float value = 0;
    uint32_t sum_lut_idx = 1;
    for (float idx = step; idx < 1.0 && sum_lut_idx <= lut_len; idx += step, sum_lut_idx++)
    {
        value = sqrt(idx);
        value = quant_utils::linear_quantize_clip(value, aipu_output_scale,
                                                  aipu_output_zp, output_qmin, output_qmax);
        lut[sum_lut_idx] = (dtype)value;
    }
    lut[0] = lut[1];
    aipubt::TensorPtr lut_tensor = std::make_shared<aipubt::Tensor>("sum_lut",
                                                                    (void *)lut, lut_len * sizeof(dtype));
    if (sign)
    {
        lut_tensor->set_type(dtype_utils::bytes_to_signed_qdtype(sizeof(dtype)));
    }
    else
    {
        lut_tensor->set_type(dtype_utils::bytes_to_unsigned_qdtype(sizeof(dtype)));
    }
    aipubt::TensorShape lut_shape(lut_len);
    lut_tensor->shape() = lut_shape;
    return lut_tensor;
}

static std::map<aipubt::Tensor::Dtype, std::function<aipubt::TensorPtr(uint32_t, std::map<std::string, float>)>> GENERATE_SUM_LUT{
    {aipubt::Tensor::INT8, generate_sum_lut<int8_t>},
    {aipubt::Tensor::INT16, generate_sum_lut<int16_t>},
    {aipubt::Tensor::INT32, generate_sum_lut<int32_t>},
    {aipubt::Tensor::UINT8, generate_sum_lut<uint8_t>},
    {aipubt::Tensor::UINT16, generate_sum_lut<uint16_t>},
    {aipubt::Tensor::UINT32, generate_sum_lut<uint32_t>},
};

uint32_t convert_less_mbit(int32_t &data, int32_t threshold_min, int32_t threshold_max)
{
    uint32_t left_shift = 0;
    while ((data > threshold_max) || (data < threshold_min))
    {
        data >>= 1;
        left_shift += 1;
    }
    return left_shift;
}

void ReduceOp::tflite_quantize(aipubt::NodePtr node)
{

    std::string method = node->params()["method"];
    int32_t obits = node->outputs()[0]->quantization().bits();
    int32_t multiplier_bits = obits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }
    if (method == "MEAN")
    {
        auto out_scales = node->outputs()[0]->quantization().scales();
        auto out_offsets = node->outputs()[0]->quantization().offsets();
        auto inp_scales = node->inputs()[0]->quantization().scales();
        auto inp_offsets = node->inputs()[0]->quantization().offsets();

        float do_scale;
        int8_t do_shift;
        int inp_shape_prod = node->inputs()[0]->shape().size();
        int out_shape_prod = node->outputs()[0]->shape().size();
        float div_dim = (float)inp_shape_prod / (float)out_shape_prod;
        // int32_t acc_bits = ceil(log2(div_dim));
        // acc_bits += node->inputs()[0]->quantization().bits();
        DEBUG() << "[tflite quantize]reduce mean: now multiplier_bits = " << multiplier_bits;
        float temp = out_scales / inp_scales;
        float local_scale = temp / div_dim;
        quant_utils::get_scale_approximation_params(local_scale, do_scale, do_shift, multiplier_bits);

        temp = temp * inp_offsets[0] * 256;
        temp = temp > 0 ? temp + 0.5f : temp - 0.5f;
        int32_t bias = static_cast<int32_t>(temp) - out_offsets[0] * 256;

        node->params()["scale_value"] = (int)do_scale;
        node->params()["shift_value"] = (int)do_shift;
        node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
        node->params()["shift_type"] = aipubt::Tensor::INT8;

        node->params()["bias"] = bias;
        node->params()["tflite_quantization"] = true;
    }
}

void ReduceOp::quantize(aipubt::NodePtr node)
{
    std::string method = node->params()["method"];
    std::vector<int32_t> axes = node->params()["axis"];
    auto input_shape = node->inputs()[0]->shape();

    bool qinvariant = node->inputs()[0]->quantization().qinvariant();
    bool sign = aipubt::Tensor::is_signed(node->inputs()[0]->type());

    if (method == "ANY" || method == "ALL")
    {
        node->outputs()[0]->quantization().set_bits(8);
        node->outputs()[0]->quantization().set_qinvariant(true);
        node->outputs()[0]->quantization().set_scale(1.0);
        node->outputs()[0]->quantization().set_offset(0);
        dtype_utils::set_aipu_dtype(node->outputs()[0], false);
    }
    else if (method == "MIN" || method == "MAX")
    {
        NoQuantization::quantize_transparent(node->outputs()[0], node->inputs()[0]);
    }
    else
    {
        static float do_scale;
        static int8_t do_shift;
        int32_t obits = node->outputs()[0]->quantization().bits();
        if (node->placeholders().size() > 0)
        {
            //if placeholders().size > 1, this code need to refine
            auto p = node->placeholders()[0];
            auto placeholder_scheme = aipubt::Quantization::Symmetric;
            auto placeholder_granularity = aipubt::Quantization::Per_Tensor;
            auto placeholder_range = node->outputs()[0]->quantization().quantized_range();

            p->quantization().set_bits(obits);
            p->quantization().set_scheme(placeholder_scheme);
            p->quantization().set_granularity(placeholder_granularity);
            p->quantization().set_quantized_range(placeholder_range);
            dtype_utils::set_aipu_dtype(p, false);
        }

        if (method == "L1" || method == "L2" || method == "VARIANCE" || method == "UNBIASED_VARIANCE")
        {
            // L1 is the sum of absolute value of tensor, L2 is the
            // sum of square value of tensor. So both outputs are unsinged
            sign = false;
        }

        dtype_utils::set_aipu_dtype(node->outputs()[0], sign);
        Op::quantize(node);

        int32_t multiplier_bits = obits;
        if (node->attrs().count("multiplier_bits"))
        {
            multiplier_bits = node->get_attr("multiplier_bits");
        }

        if (qinvariant)
        {
            node->outputs()[0]->quantization().set_scale(1);
            node->outputs()[0]->quantization().set_offset(0);
            float out_min = node->outputs()[0]->quantization().mins()[0];
            float out_max = node->outputs()[0]->quantization().maxs()[0];
            int tmp_qbit = dtype_utils::range_to_bits(out_min, out_max, sign);
            node->outputs()[0]->quantization().set_bits(tmp_qbit);
            node->outputs()[0]->quantization().set_qinvariant(qinvariant);
            do_scale = 1.0;
            do_shift = 0;
            node->params()["scale_value"] = (int)do_scale;
            node->params()["shift_value"] = (int)do_shift;
            node->params()["scale_type"] = dtype_utils::bits_to_dtype(obits, false);
            node->params()["shift_type"] = aipubt::Tensor::INT8;
        }
        else
        {
            float in_scale = node->inputs()[0]->quantization().scale();
            float out_scale = node->outputs()[0]->quantization().scale();
            float local_scale;

            //SUM, PROD operation need to quantize
            if (method == "SUM")
            {
                local_scale = out_scale / in_scale;
                quant_utils::get_scale_approximation_params(local_scale, do_scale, do_shift, multiplier_bits);
                node->params()["scale_value"] = (int)do_scale;
                node->params()["shift_value"] = (int)do_shift;
                node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
                node->params()["shift_type"] = aipubt::Tensor::INT8;
            }
            else if (method == "PROD")
            {
                float re_input_scale = 1 / in_scale;
                local_scale = out_scale;
                int32_t q_bits = std::max(16, multiplier_bits);
                float input_do_scale;
                int8_t input_do_shift;
                quant_utils::get_scale_approximation_params(re_input_scale, input_do_scale, input_do_shift, q_bits);
                quant_utils::get_scale_approximation_params(local_scale, do_scale, do_shift, q_bits);
                int32_t in_int_do_scale = static_cast<int32_t>(input_do_scale);
                int32_t int_do_scale = static_cast<int32_t>(do_scale);
                int32_t threshold_min = dtype_utils::bits_to_range_min(15, false);
                uint32_t threshold_max = dtype_utils::bits_to_range_max(15, false);

                int32_t step = 1;
                for (auto &i : axes)
                {
                    step *= input_shape[i];
                }

                int32_t tmp_data = 1;
                int32_t left_shifts = 0;
                for (int i = 0; i < step; i++)
                {
                    tmp_data *= in_int_do_scale;
                    uint32_t left_shift = convert_less_mbit(tmp_data, threshold_min, threshold_max);
                    left_shifts += left_shift;
                }
                int32_t tmp_data_2 = tmp_data * int_do_scale;
                uint32_t left_shift_2 = convert_less_mbit(tmp_data_2, threshold_min, threshold_max);
                left_shifts += left_shift_2;
                int_do_scale = tmp_data_2;
                do_shift = input_do_shift * step + do_shift - left_shifts;
                node->params()["scale_value"] = int_do_scale;
                node->params()["shift_value"] = do_shift;
                node->params()["scale_type"] = dtype_utils::bits_to_dtype(q_bits, false);
                node->params()["shift_type"] = aipubt::Tensor::UINT32;
            }
            else if (method == "VARIANCE" || method == "UNBIASED_VARIANCE")
            {
                int16_t num = 1;
                aipubt::TensorShape inp_shape = node->inputs()[0]->shape();
                std::vector<aipubt::NodeParamValue> axis = node->params()["axis"];
                for (auto a : axis)
                {
                    num *= inp_shape[a];
                }
                if (method == "UNBIASED_VARIANCE" && num > 1)
                {
                    num = num - 1;
                }
                float total_scale = out_scale / (in_scale * in_scale) / (float)num;
                float input_do_scale;
                int8_t input_do_shift;
                if (!node->attrs().count("multiplier_bits"))
                {
                    //use default multiple bits 16
                    multiplier_bits = 16;
                }
                quant_utils::get_scale_approximation_params(total_scale, input_do_scale, input_do_shift, multiplier_bits);
                node->params()["scale_value"] = static_cast<int32_t>(input_do_scale);
                node->params()["shift_value"] = input_do_shift;
                node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
                node->params()["shift_type"] = aipubt::Tensor::INT8;
            }
            else if (method == "MEAN")
            {
                //MEAN
                if (node->attrs().count("tflite_quantization") && (bool)node->get_attr("tflite_quantization"))
                {
                    tflite_quantize(node);
                }
                else
                {
                    int inp_shape_prod = node->inputs()[0]->shape().size();
                    int out_shape_prod = node->outputs()[0]->shape().size();
                    float div_dim = (float)inp_shape_prod / (float)out_shape_prod;
                    // int32_t acc_bits = ceil(log2(div_dim));
                    // acc_bits += node->inputs()[0]->quantization().bits();
                    // multiplier_bits = std::min(multiplier_bits, 32 - acc_bits);
                    DEBUG() << "reduce mean: now multiplier_bits = " << multiplier_bits;
                    float local_scale = (out_scale / in_scale) / div_dim;
                    quant_utils::get_scale_approximation_params(local_scale, do_scale, do_shift, multiplier_bits);

                    node->params()["scale_value"] = (int)do_scale;
                    node->params()["shift_value"] = (int)do_shift;
                    node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
                    node->params()["shift_type"] = aipubt::Tensor::INT8;
                }
            }
            else if (method == "L1")
            {
                local_scale = out_scale / in_scale;
                quant_utils::get_scale_approximation_params(local_scale, do_scale, do_shift, multiplier_bits);

                node->params()["scale_value"] = (int)do_scale;
                node->params()["shift_value"] = (int)do_shift;
                node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
                node->params()["shift_type"] = aipubt::Tensor::INT8;
            }
            else if (method == "L2")
            {
                int8_t bits_i = node->inputs()[0]->quantization().bits();
                uint32_t lut_len = 1 << bits_i;
                if (node->attrs().count(Constants::lut_items_in_bits))
                {
                    int32_t bits = node->get_attr<aipubt::NodeParamValue>(Constants::lut_items_in_bits);
                    lut_len = 1 << bits;
                }

                float local_rescale = node->placeholders()[0]->quantization().scale();
                quant_utils::get_scale_approximation_params(local_rescale / (in_scale * in_scale), do_scale, do_shift, multiplier_bits);

                node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
                node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
                node->params()["shift_value"] = do_shift;
                node->params()["shift_type"] = aipubt::Tensor::INT8;

                float aipu_output_scale = node->outputs()[0]->quantization().scale();
                float aipu_output_zp = node->outputs()[0]->quantization().offset();
                aipubt::Tensor::Dtype aipu_output_dtype = node->outputs()[0]->quantization().aipu_dtype();
                float output_qmin = dtype_utils::dtype_to_range_min(aipu_output_dtype);
                float output_qmax = dtype_utils::dtype_to_range_max(aipu_output_dtype);

                std::map<std::string, float> sum_lut_param = {
                    {"aipu_output_scale", aipu_output_scale},
                    {"aipu_output_zp", aipu_output_zp},
                    {"output_qmin", output_qmin},
                    {"output_qmax", output_qmax}};

                aipubt::TensorPtr sum_lut_tensor = GENERATE_SUM_LUT[aipu_output_dtype](lut_len, sum_lut_param);
                node->constants()["sqrt_lut"] = sum_lut_tensor;
            }
        }
    }
}

REGISTER_QUANTIZATION_OP(Reduce);
} // namespace qtlibn

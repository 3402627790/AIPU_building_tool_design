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
 * @Date: 2021-11-30 17:13:09
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-11-07 11:40:50
 */

#include "quantization_transform/generate_lut.h"
#include "utils/string_utils.h"

namespace qtlibn
{

aipubt::TensorPtr generate_lut(aipubt::NodePtr node,
                               LUTFunc cal_lut_func,
                               std::map<std::string, float> cal_param,
                               int32_t input_idx,
                               int32_t output_idx)
{
    auto aipu_input_scale = node->inputs()[input_idx]->quantization().scale();
    auto aipu_input_zp = node->inputs()[input_idx]->quantization().offset();
    auto aipu_output_scale = node->outputs()[output_idx]->quantization().scale();
    auto aipu_output_zp = node->outputs()[output_idx]->quantization().offset();
    auto aipu_input_dtype = node->inputs()[input_idx]->quantization().aipu_dtype();
    auto aipu_output_dtype = node->outputs()[output_idx]->quantization().aipu_dtype();

    int bits = aipubt::Tensor::dtype_bits(aipu_input_dtype);
    // int32_t bits = node->inputs()[input_idx]->quantization().bits();
    float input_qmin = dtype_utils::dtype_to_range_min(aipu_input_dtype);
    float input_qmax = dtype_utils::dtype_to_range_max(aipu_input_dtype);
    float output_qmin = dtype_utils::dtype_to_range_min(aipu_output_dtype);
    float output_qmax = dtype_utils::dtype_to_range_max(aipu_output_dtype);
    size_t new_dtype_size = aipubt::Tensor::dtype_size(aipu_output_dtype);

    aipubt::Tensor::Dtype lut_dtype = output_qmin >= 0 ? dtype_utils::bytes_to_unsigned_qdtype(new_dtype_size) : dtype_utils::bytes_to_signed_qdtype(new_dtype_size);

    uint32_t lut_len = pow(2, bits);
    if (node->attrs().count(Constants::lut_items_in_bits))
    {
        int32_t lut_in_bits = node->get_attr<aipubt::NodeParamValue>(Constants::lut_items_in_bits);
        lut_in_bits = std::min(bits, lut_in_bits);
        lut_len = pow(2, lut_in_bits);
    }
    bool use_float_input = cal_param.count("USE_FLOAT_INPUT") && cal_param["USE_FLOAT_INPUT"] == 1.0;
    float input_min = input_qmin;
    float input_max = input_qmax;
    if (use_float_input)
    {
        input_min = cal_param["float_input_min"];
        input_max = cal_param["float_input_max"];
    }

    aipubt::ComputeTensor ct = aipubt::ComputeTensor::linspace(input_min, input_max, lut_len);
    if (!use_float_input)
    {
        ct = (ct + aipu_input_zp) / aipu_input_scale;
    }
    ct = cal_lut_func(ct, cal_param);
    ct = (ct * aipu_output_scale - aipu_output_zp).round().clamp(output_qmin, output_qmax).astype(lut_dtype);

    aipubt::TensorPtr lut_tensor = ct;
    lut_tensor->set_name("lut_tensor");
    lut_tensor->set_type(lut_dtype);
    aipubt::TensorShape lut_shape(lut_len);
    lut_tensor->shape() = lut_shape;

    int32_t lut_bits = log2(lut_len);
    if (lut_len - pow(2, lut_bits) != 0)
    {
        WARN() << "[QTLib]lut size is not the pow2, now lut_len = " << lut_len;
    }
    lut_tensor->quantization().set_bits(new_dtype_size * 8);

    return lut_tensor;
}

aipubt::TensorPtr generate_lut(aipubt::NodePtr node, LUTFunc cal_lut_func, std::map<std::string, float> cal_param)
{
    return generate_lut(node, cal_lut_func, cal_param, 0, 0);
}

void generate_lut_fp(aipubt::NodePtr node, LUTFunc cal_lut_func, std::map<std::string, float> cal_param)
{
    auto inpt = node->inputs()[0];
    int32_t lut_items_in_bits = cal_param["aiff_lut_items_in_bits"];
    auto bk_index_scales = inpt->quantization().scales();
    auto bk_index_offsets = inpt->quantization().offsets();
    auto bk_mins = inpt->quantization().mins();
    auto bk_maxs = inpt->quantization().maxs();
    auto index_scales = bk_index_scales;
    auto index_offsets = bk_index_offsets;

    bool out_signed = cal_param.count("out_signed") ? cal_param["out_signed"] : true;

    if (cal_param.count("clip_min") && cal_param.count("clip_max"))
    {
        float inp_min = cal_param["clip_min"];
        float inp_max = cal_param["clip_max"];
        inpt->quantization().set_mins({inp_min});
        inpt->quantization().set_maxs({inp_max});
        // INFO() << "clip_mi = " << inp_min << ", clip_max = " << inp_max;
    }
    inpt->quantization().set_bits(lut_items_in_bits);
    inpt->quantization().set_scheme(aipubt::Quantization::Asymmetric);
    inpt->quantization().set_aipu_dtype(dtype_utils::bits_to_dtype(lut_items_in_bits, out_signed));

    quant_utils::calculate_scale_zp(inpt, true);
    index_scales = inpt->quantization().scales();
    index_offsets = inpt->quantization().offsets();
    // INFO() << "index_scale = " << index_scales[0] << ", index_offsets = " << index_offsets[0];

    inpt->quantization().set_scales(bk_index_scales);
    inpt->quantization().set_offsets(bk_index_offsets);
    inpt->quantization().set_mins(bk_mins);
    inpt->quantization().set_maxs(bk_maxs);

    if (index_scales.size() == 0 || index_offsets.size() == 0)
    {
        ERROR() << "when generate luts for float op = " << node << ", its's input scale/offset is None";
        return;
    }

    aipubt::ComputeTensor lut = aipubt::ComputeTensor::arange(0, std::pow(2.0, lut_items_in_bits));
    lut = (lut + index_offsets[0]) / index_scales[0];

    float value_offset = cal_param.count("value_offset") ? cal_param["value_offset"] : 0.0;

    lut = cal_lut_func(lut, cal_param) + value_offset;
    lut = dtype_utils::to_fp24(lut.astype(aipubt::Tensor::FP32));
    lut->set_type(aipubt::Tensor::FP32);
    node->constants()["lut"] = lut;
    node->params()["lut_mode"] = "NORMAL";
    node->params()["is_perf_mode"] = true;
    node->params()["index_scale_value"] = (float)index_scales[0];
    node->params()["index_scale_type"] = aipubt::Tensor::FP32;
    node->params()["index_offset_value"] = (float)index_offsets[0];
    node->params()["index_offset_type"] = aipubt::Tensor::FP32;
    node->params()["value_offset_value"] = value_offset;
    node->params()["value_offset_type"] = aipubt::Tensor::FP32;
}

struct TargetRelativeQInfo
{
    uint8_t level;
    uint8_t aiff_lut_items_in_bits;
    uint8_t aiff_bias_effective_bits;
};

static std::map<std::string, TargetRelativeQInfo> TARGET_QINFO = {
    {"Z1", {0, 8, 16}},
    {"Z2", {0, 8, 16}},
    {"Z3", {0, 8, 16}},
    {"X1", {0, 8, 16}},
    {"X2", {1, 8, 32}},
    {"X3", {2, 9, 32}},
    {"X3P", {2, 9, 32}},
    {"X3S", {2, 9, 32}},
};

bool ApproxLut::is_perf_mode(const std::string &min_target)
{
    bool _is_perf_mode = false;
    if (TARGET_QINFO.count(min_target))
    {
        auto level = (TARGET_QINFO[min_target]).level;
        if (level >= 2)
        {
            _is_perf_mode = true;
        }
    }
    return _is_perf_mode;
}

int32_t ApproxLut::get_aiff_lut_items_in_bits(aipubt::NodePtr node)
{
    auto min_target = get_min_target(node);
    int32_t aiff_lut_items_in_bits = 9;
    if (TARGET_QINFO.count(min_target))
    {
        aiff_lut_items_in_bits = TARGET_QINFO[min_target].aiff_lut_items_in_bits;
    }
    return aiff_lut_items_in_bits;
}

std::string ApproxLut::get_min_target(aipubt::NodePtr node)
{
    auto target = aipubt::get_target();
    auto full_target = target->name();
    auto starget = utils::split_string(full_target, '_');
    std::string min_target = "X2";
    if (starget.size() > 0)
    {
        min_target = starget[0];
    }
    if (node->attrs().count("min_compatible_zhouyi_target"))
    {
        min_target = std::string(node->get_attr<aipubt::NodeParamValue>("min_compatible_zhouyi_target"));
    }
    DEBUG() << "get min target in qatlib = " << min_target;
    return min_target;
}

bool ApproxLut::is_dynamic_lut(aipubt::NodePtr node, const int32_t &idx)
{
    bool use_dynamic_lut = false;

    if (node->attrs().count("approx_params"))
    {
        auto approx_params = node->get_attr<aipubt::NodeParamValue>("approx_params");
        if (approx_params.arr.size() > 1)
        {
            use_dynamic_lut = approx_params[1] == 0 ? false : true;
            if (use_dynamic_lut)
            {
                auto inpt = node->inputs()[idx];
                use_dynamic_lut = inpt->quantization().empty() ? false : true;
            }
            // INFO() << "use dynamci lut = " << use_dynamic_lut;
        }
    }
    return use_dynamic_lut;
}

void ApproxLut::get_min_max(aipubt::TensorPtr t, float &base_min, float &base_max)
{
    auto s_mins = t->quantization().mins();
    auto s_maxs = t->quantization().maxs();
    if (s_mins.size() && s_maxs.size())
    {
        base_min = s_mins[0];
        base_max = s_maxs[0];
        return;
    }
    else
    {
        auto scales = t->quantization().scales();
        auto offsets = t->quantization().offsets();
        if (scales.size() == 0 || offsets.size() == 0)
        {
            return;
        }

        std::vector<float> mins;
        std::vector<float> maxs;
        std::function<float(float, float, int32_t)> formula = quant_utils::dequantize_formula_aipu;
        quant_utils::dequantize_from_primitive_quant_info(
            t->type(),
            scales,
            offsets,
            mins,
            maxs,
            formula);
        base_min = mins[0];
        base_max = maxs[0];
    }
}

bool is_aiff_perf_mode(aipubt::NodePtr node)
{
    // android already setted this params.
    if (node->params().count("is_perf_mode"))
    {
        return node->params()["is_perf_mode"];
    }

    auto approx_lut = ApproxLut();
    std::string min_target = approx_lut.get_min_target(node);
    bool use_aiff_perf_mode = approx_lut.is_perf_mode(min_target);

    // bt flow, opt maybe set approx_params
    if (node->attrs().count("approx_params"))
    {
        auto approx_params = node->get_attr<aipubt::NodeParamValue>("approx_params");
        int32_t method = 0;
        if (approx_params.arr.size() > 0)
        {
            method = approx_params[0];
        }
        else
        {
            method = approx_params;
        }
        // INFO() << "method = " << method;
        use_aiff_perf_mode = method == 1;
    }
    return use_aiff_perf_mode;
}

bool is_dynamic_lut(aipubt::NodePtr node, const int32_t &idx)
{
    auto approx_lut = ApproxLut();
    return approx_lut.is_dynamic_lut(node, idx);
}

void get_min_max(aipubt::TensorPtr t, float &base_min, float &base_max)
{
    auto approx_lut = ApproxLut();
    approx_lut.get_min_max(t, base_min, base_max);
}
int32_t get_aiff_lut_items_in_bits(aipubt::NodePtr node)
{
    auto approx_lut = ApproxLut();
    return approx_lut.get_aiff_lut_items_in_bits(node);
}

}; // namespace qtlibn

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
 * @Date: 2022-03-23 17:07:39
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-10-10 22:29:45
 */

#include "quantization_transform/ops/activation.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_sigmoid_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    return 1.0f / (1.0f + (0.0f - value).exp());
}

static aipubt::ComputeTensor cal_tanh_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    // float in_clamp_min = 0.0, in_clamp_max = 10.0;
    // float out_clamp_min = 0.0, out_clamp_max = 1.0;
    // if (value < 0)
    // {
    //     in_clamp_min = -10.0;
    //     in_clamp_max = 10.0;
    //     out_clamp_min = -1.0;
    //     out_clamp_max = 1.0;
    // }
    // value = quant_utils::clip<float, float>(value, in_clamp_min, in_clamp_max);
    // value = (exp(value) - exp(-value)) / (exp(value) + exp(-value));
    // value = quant_utils::clip<float, float>(value, out_clamp_min, out_clamp_max);
    float pos_in_clip_min = 0.0, pos_in_clip_max = 10.0;
    float pos_out_clip_min = 0.0, pos_out_clip_max = 1.0;
    float neg_in_clip_min = -10.0, neg_in_clip_max = 10.0;
    float neg_out_clip_min = -1.0, neg_out_clip_max = 1.0;

    auto neg_mask = value < 0;
    auto neg_value_clip = (value * neg_mask).clamp(neg_in_clip_min, neg_in_clip_max);
    auto pos_mask = 1 - neg_mask;
    auto pos_value_clip = (value * pos_mask).clamp(pos_in_clip_min, pos_in_clip_max);
    auto cur_value = neg_value_clip + pos_value_clip;

    auto exp_value = cur_value.exp();
    auto exp_neg_value = (0.f - cur_value).exp();
    cur_value = (exp_value - exp_neg_value) / (exp_value + exp_neg_value);

    neg_value_clip = (cur_value * neg_mask).clamp(neg_out_clip_min, neg_out_clip_max);
    pos_value_clip = (cur_value * pos_mask).clamp(pos_out_clip_min, pos_out_clip_max);

    return neg_value_clip + pos_value_clip;
}

static aipubt::ComputeTensor cal_softplus_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    return (value.exp() + 1.0f).log();
}

static aipubt::ComputeTensor cal_celu_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    float alpha = cal_param["alpha"];
    auto min_v = ((value / alpha).exp() - 1.f) * alpha;
    return value.maximum(0.0f) + min_v.minimum(0.0f);
    return value;
}

static aipubt::ComputeTensor cal_shrink_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    auto bias = cal_param["bais"];
    auto lambd = cal_param["lambd"];

    auto lt_mask = value < -lambd;
    auto masked = lt_mask * bias + value;
    auto gt_mask = value > lambd;
    masked = gt_mask * (-bias) + masked;
    return masked;
}

static aipubt::ComputeTensor cal_selu_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    auto alpha = cal_param["alpha"];
    auto gamma = cal_param["gamma"];
    auto max_value = value.maximum(0.f);
    auto min_value = ((value.exp() - 1.f) * alpha).minimum(0.f);
    return (max_value + min_value) * gamma;
}

static aipubt::ComputeTensor cal_softsign_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    return value / (value.abs() + 1.f);
}

static aipubt::ComputeTensor cal_thresholdedrelu_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    auto alpha = cal_param["alpha"];
    auto replace_value = cal_param["replace_value"];
    auto mask = value > alpha;
    auto mask_value = value * mask;
    auto replace_mask = 1.f - mask;
    auto replace_value_ct = value.zeros_like() + replace_value;
    replace_value_ct = replace_mask * replace_value_ct;
    return mask_value + replace_value_ct;
}

static aipubt::ComputeTensor cal_mish_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    auto softplus = (1.0f + value.exp());
    value = value * softplus.tanh();
    return value;
}

static aipubt::ComputeTensor cal_swish_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    float alpha = cal_param["alpha"];
    return value / (1.f + (alpha * -1.f * value).exp());
}

static aipubt::ComputeTensor cal_expmul_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    auto mul = cal_param["constant_value"];
    return (value + log(mul)).exp();
}

void ActivationOp::quantize(aipubt::NodePtr node)
{

    std::string method = node->params()["method"];
    // method in [SILU, RELU, CLIP, HARDSWISH, RELU6, SIGMOID, PRELU, ELU, TANH]

    static std::map<std::string, std::pair<LUTFunc, bool>> _METHOD_TO_FUNC = {
        // {method(string), {activation_function_pointer, output_is_signed}}
        {"SIGMOID", {cal_sigmoid_func, false}},
        {"SOFTPLUS", {cal_softplus_func, false}},
        {"TANH", {cal_tanh_func, true}},
        {"CELU", {cal_celu_func, true}},
        {"SHRINK", {cal_shrink_func, true}},
        {"SELU", {cal_selu_func, true}},
        {"SOFTSIGN", {cal_softsign_func, true}},
        {"THRESHOLDEDRELU", {cal_thresholdedrelu_func, true}},
        {"MISH", {cal_mish_func, true}},
        {"SWISH", {cal_swish_func, true}},
        {"UNKNOWN", {cal_expmul_func, false}},
    };

    if (_METHOD_TO_FUNC.count(method))
    {
        //(method in [TANH, SIGMOID, SOFTPLUS, CELU, SWISH, SHRINK, SELU, THRESHOLDERRELU])
        std::map<std::string, float> params = {};
        if (method == "CELU" || method == "SWISH")
        {
            float alpha = param_utils::get_param(node, "alpha");
            params.insert({"alpha", alpha});
        }
        else if (method == "SHRINK")
        {
            float bias = param_utils::get_param(node, "bias");
            float lambd = param_utils::get_param(node, "lambd");
            params.insert({"bias", bias});
            params.insert({"lambd", lambd});
        }
        else if (method == "SELU")
        {
            float alpha = param_utils::get_param(node, "alpha");
            float gamma = param_utils::get_param(node, "gamma");
            params.insert({"alpha", alpha});
            params.insert({"gamma", gamma});
        }
        else if (method == "THRESHOLDEDRELU")
        {
            float alpha = param_utils::get_param(node, "alpha");
            float replace_value = 0;
            params.insert({"alpha", alpha});
            params.insert({"replace_value", replace_value});

            bool flag = alpha < 0.0 ? true : false;
            _METHOD_TO_FUNC[method].second = flag;
        }
        else if (method == "UNKNOWN")
        {
            float mul = param_utils::get_param(node, "constant_value");
            params.insert({"constant_value", mul});
        }

        // infer output sign
        auto out_sign = _METHOD_TO_FUNC[method].second;
        dtype_utils::set_aipu_dtype(node->outputs()[0], out_sign);
        Op::quantize(node);

        node->constants()["lut"] = generate_lut(node, _METHOD_TO_FUNC[method].first, params);
    }
    else
    {
        //(method in [RELU, PRELU, LEAKYRELU, CLIP, RELU6, ELU, SILU, HardSwish, GELU])
        transform(method.begin(), method.end(), method.begin(), ::toupper);
        auto activation_optype = aipubt::string_upper_to_optype(method);
        Op *activation_op = OpRegistry::get_quant_op(activation_optype);
        activation_op->set_adaptor(this->get_adaptor());
        activation_op->quantize(node);
    }
}

void ActivationOp::approximate(aipubt::NodePtr node)
{
    std::string method = node->params()["method"];

    transform(method.begin(), method.end(), method.begin(), ::toupper);
    if (method == "SIGMOID" || method == "SWISH")
    {
        method = "SILU";
    }
    auto activation_optype = aipubt::string_upper_to_optype(method);
    Op *activation_op = OpRegistry::get_quant_op(activation_optype);
    activation_op->approximate(node);
}

REGISTER_QUANTIZATION_OP(Activation);
}; // namespace qtlibn

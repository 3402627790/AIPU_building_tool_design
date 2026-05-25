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
 * @Date: 2022-04-03 19:28:30
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-01 13:54:22
 */

#include "quantization_transform/ops/gelu.h"

namespace qtlibn
{
// static float cal_gelu_tanh_apprx(float x, std::map<std::string, float> cal_param = {})
// {
//     return 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
// }
// static float cal_gelu_none_func(float x, std::map<std::string, float> cal_param = {})
// {
//     return x * 0.5 * (1.0 + erf(x / sqrt(2.0)));
// }
static aipubt::ComputeTensor cal_gelu_tanh_apprx(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    float temp = sqrt(2.f / M_PI);
    return 0.5 * value * (1 + ((value + 0.044715 * value.pow(3)) * temp).tanh());
}
static aipubt::ComputeTensor cal_gelu_none_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    float temp = (1.0f / sqrt(2.0f));
    return value * 0.5 * (1.0 + (value * aipubt::ComputeTensor::create(temp)).erf());
}
void GELUOp::quantize(aipubt::NodePtr node)
{
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);
    std::string apprx = (std::string)node->params()["approximate"];
    Op::quantize(node);
    if (apprx == "NONE")
    {
        node->constants()["lut"] = generate_lut(node, cal_gelu_none_func);
    }
    else if (apprx == "TANH")
    {
        node->constants()["lut"] = generate_lut(node, cal_gelu_tanh_apprx);
    }
    else
    {
        ERROR() << "GELU op has unsupported approximate = " << apprx << ", and now only support [none, tanh]";
    }
}

void GELUOp::approximate(aipubt::NodePtr node)
{
    Op::approximate(node);
    auto phi_tanh_approx = [](aipubt::ComputeTensor lut, std::map<std::string, float> param = {}) -> aipubt::ComputeTensor {
        aipubt::ComputeTensor x = aipubt::ComputeTensor(2. / M_PI).sqrt();
        auto t = lut + 0.044715 * lut.pow(3.0);
        return 0.5 * (1 + (x * t).tanh());
    };
    float clip_max = 6.0;
    float clip_min = 0.0;

    // get is_perf_mode and whether use_dynamic_lut
    bool use_dynamic_lut = is_dynamic_lut(node);
    bool is_perf_mode = is_aiff_perf_mode(node);

    if (is_perf_mode)
    {
        if (use_dynamic_lut)
        {
            get_min_max(node->inputs()[0], clip_min, clip_max);
            clip_max = std::max(std::fabs(clip_min), std::fabs(clip_max));
            clip_min = 0.0;
        }

        std::map<std::string, float> cal_param = {
            {"clip_min", clip_min},
            {"clip_max", clip_max},
            {"value_offset", -0.5},
            {"out_signed", 0.0},
            {"is_perf_mode", is_perf_mode},
            {"aiff_lut_items_in_bits", static_cast<float>(get_aiff_lut_items_in_bits(node))},
        };
        generate_lut_fp(node, phi_tanh_approx, cal_param);
        node->params()["lut_mode"] = "MIRROR";
    }
    else
    {
        node->params()["is_perf_mode"] = false;
    }
}

REGISTER_QUANTIZATION_OP(GELU);
}; // namespace qtlibn

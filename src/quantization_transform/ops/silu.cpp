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
 * @Author: Ruodi Lu
 * @Date: 2020-10-13 17:58:17
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-07 10:06:56
 */

#include "quantization_transform/ops/silu.h"

namespace qtlibn
{
static aipubt::ComputeTensor cal_silu_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    return value / ((0. - value).exp() + 1.0);
}

void SiluOp::quantize(aipubt::NodePtr node)
{
    // infer output sign
    auto out_sign = aipubt::Tensor::is_signed(node->inputs()[0]->quantization().aipu_dtype());
    dtype_utils::set_aipu_dtype(node->outputs()[0], out_sign);

    Op::quantize(node);
    node->constants()["lut"] = generate_lut(node, cal_silu_func);
}

void SiluOp::approximate(aipubt::NodePtr node)
{
    Op::approximate(node);
    auto sigmoid_func = [](aipubt::ComputeTensor lut, std::map<std::string, float> param = {}) -> aipubt::ComputeTensor {
        return 1. / (1. + (0 - lut).exp());
    };
    // float clip_max = 6.0;
    float clip_max = 15.0;
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
        generate_lut_fp(node, sigmoid_func, cal_param);
        node->params()["lut_mode"] = "MIRROR";
    }
    else
    {
        node->params()["is_perf_mode"] = false;
    }
}

REGISTER_QUANTIZATION_OP(Silu);

}; // namespace qtlibn

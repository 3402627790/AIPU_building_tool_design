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
 * @Date: 2024-08-29 22:29:34
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-08-30 15:53:59
 */

#include "quantization_transform/ops/celu.h"
#include "quantization_transform/generate_lut.h"

namespace qtlibn
{

void CeluOp::approximate(aipubt::NodePtr node)
{

    Op::approximate(node);
    auto celu_func = [](aipubt::ComputeTensor lut, std::map<std::string, float> param = {}) -> aipubt::ComputeTensor {
        float alpha = param["alpha"];
        return lut.celu(alpha);
    };
    // float clip_max = 4.0;
    // float clip_min = std::log(0.00001) * alpha - 2;

    // get is_perf_mode and whether use_dynamic_lut
    // bool use_dynamic_lut = is_dynamic_lut(node);
    bool is_perf_mode = is_aiff_perf_mode(node);

    if (is_perf_mode)
    {
        float alpha = node->params()["alpha"];
        std::map<std::string, float> cal_param = {
            {"alpha", alpha},
            {"clip_min", std::log(1e-5) * alpha - 2},
            {"clip_max", 4.0},
            // {"value_offset", -0.5},
            {"out_signed", 0.0},
            {"is_perf_mode", is_perf_mode},
            {"aiff_lut_items_in_bits", static_cast<float>(get_aiff_lut_items_in_bits(node))},
        };
        generate_lut_fp(node, celu_func, cal_param);
        node->params()["lut_mode"] = "MIRROR";
    }
    else
    {
        node->params()["is_perf_mode"] = false;
    }
}

REGISTER_CUSTOMIZE_QUANTIZATION_OP(Celu);

}; // namespace qtlibn
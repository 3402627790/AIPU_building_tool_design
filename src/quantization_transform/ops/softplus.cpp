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
 * @Date: 2024-10-10 22:25:14
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-10-10 22:26:20
 */

#include "quantization_transform/ops/softplus.h"

namespace qtlibn
{

void SOFTPLUSOp::approximate(aipubt::NodePtr node)
{
    Op::approximate(node);
    auto softplus_func = [](aipubt::ComputeTensor lut, std::map<std::string, float> param = {}) -> aipubt::ComputeTensor {
        return lut.softplus();
    };
    float clip_max = 20.0;
    float clip_min = -6.0;

    // get is_perf_mode and whether use_dynamic_lut
    bool use_dynamic_lut = is_dynamic_lut(node);
    bool is_perf_mode = is_aiff_perf_mode(node);

    if (is_perf_mode)
    {
        if (use_dynamic_lut)
        {
            get_min_max(node->inputs()[0], clip_min, clip_max);
            clip_min = clip_min < -6.0 ? clip_min : -6.0;
        }

        std::map<std::string, float> cal_param = {
            {"clip_min", clip_min},
            {"clip_max", clip_max},
            // {"value_offset", -0.5},
            {"out_signed", 0.0},
            {"is_perf_mode", is_perf_mode},
            {"aiff_lut_items_in_bits", static_cast<float>(get_aiff_lut_items_in_bits(node))},
        };
        generate_lut_fp(node, softplus_func, cal_param);
    }
    else
    {
        node->params()["is_perf_mode"] = false;
    }
}

REGISTER_CUSTOMIZE_QUANTIZATION_OP(SOFTPLUS);

} // namespace qtlibn

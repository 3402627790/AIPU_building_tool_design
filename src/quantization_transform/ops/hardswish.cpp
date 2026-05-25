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
 * @Date: 2022-03-26 23:10:35
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-07 10:02:40
 */

#include "quantization_transform/ops/hardswish.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_hardswish_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    uint8_t hard_swish_opt_tag = cal_param["hard_swish_opt_tag"];
    if (hard_swish_opt_tag != 0)
    {
        value = value - 3;
    }

    // if (value <= -3)
    // {
    //     value = 0;
    // }
    // else if (value > -3 && value < 3)
    // {
    //     value = value * (value + 3) / 6;
    // }

    auto mask = (value > -3);
    value = value * mask;

    mask = (value > -3) * (value < 3);

    auto masked_value = value * mask;
    masked_value = masked_value * (masked_value + 3) / 6;
    auto unmasked_value = (value * (1 - mask));
    auto new_value = masked_value + unmasked_value;

    if (hard_swish_opt_tag == 2)
    {
        new_value = new_value + 0.375f;
    }
    return new_value;
}

void HardSwishOp::quantize(aipubt::NodePtr node)
{
    // infer output sign
    dtype_utils::set_aipu_dtype(node->outputs()[0], true);

    Op::quantize(node);
    uint8_t hard_swish_opt_tag = 0;
    if (node->attrs().count("hard_swish_opt_tag"))
    {
        hard_swish_opt_tag = (uint8_t)node->get_attr("hard_swish_opt_tag");
    }
    std::map<std::string, float> cal_param = {{"hard_swish_opt_tag", hard_swish_opt_tag}};
    node->constants()["lut"] = generate_lut(node, cal_hardswish_func, cal_param);
}

REGISTER_QUANTIZATION_OP(HardSwish);

}; // namespace qtlibn
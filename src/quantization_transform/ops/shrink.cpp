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
 * @Date: 2022-05-19 23:45:42
 * @Last Modifsied by: Samuel Han
 * @Last Modified time: 2022-05-19 23:58:06
 */

#include "quantization_transform/ops/shrink.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_shrink_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    auto bias = cal_param["bais"];
    auto lambd = cal_param["lambd"];

    auto lt_mask = value < -lambd;
    auto masked = lt_mask * bias + value;
    auto gt_mask = value > lambd;
    masked = gt_mask * (-bias) + masked;
    return masked;

    // if (value < -lambd)
    // {
    //     value += bias;
    // }
    // else if (value > lambd)
    // {
    //     value -= bias;
    // }

    // return value;
}

void SHRINKOp::quantize(aipubt::NodePtr node)
{
    dtype_utils::set_aipu_dtype(node->outputs()[0], true);

    Op::quantize(node);
    std::map<std::string, float> cal_param = {
        {"bias", (float)node->params()["bias"]},
        {"lambd", (float)node->params()["lambd"]},
    };
    node->constants()["lut"] = generate_lut(node, cal_shrink_func, cal_param);
}

REGISTER_QUANTIZATION_OP(SHRINK);
} // namespace qtlibn
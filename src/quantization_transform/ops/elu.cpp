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
 * @Date: 2022-03-26 22:57:16
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-07 10:02:28
 */

#include "quantization_transform/ops/elu.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_elu_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    float alpha = cal_param["alpha"];
    auto mask = value > 0;
    auto ltq_mask = 1.0 - mask;
    auto positive_value = value * mask;
    auto new_value = (value.exp() - 1.f) * alpha;
    auto negative_value = new_value * ltq_mask;
    return positive_value + negative_value;

    // if(value > 0)
    //     return value;
    // else
    //     return alpha * (exp(value) - 1);
}

void EluOp::quantize(aipubt::NodePtr node)
{
    // infer output sign
    dtype_utils::set_aipu_dtype(node->outputs()[0], true);

    Op::quantize(node);
    float alpha = node->params()["alpha"];
    std::map<std::string, float> cal_param = {{"alpha", alpha}};
    node->constants()["lut"] = generate_lut(node, cal_elu_func, cal_param);
}

REGISTER_QUANTIZATION_OP(Elu);

}; // namespace qtlibn
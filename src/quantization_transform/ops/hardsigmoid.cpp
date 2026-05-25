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

#include "quantization_transform/ops/hardsigmoid.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_hardsigmoid_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    float alpha = cal_param["alpha"];
    float beta = cal_param["beta"];
    float hard_sigmoid_tag = cal_param["hard_sigmoid_tag"];
    float clip_max = cal_param["clip_max"];
    float clip_min = cal_param["clip_min"];
    if (hard_sigmoid_tag != 0.0)
    {
        value = value - 1.f / 2 * alpha;
    }
    // value = std::max(clip_min, std::min(alpha * value + beta, clip_max));
    return (value * alpha + beta).minimum(clip_max).maximum(clip_min);
}

void HardSigmoidOp::quantize(aipubt::NodePtr node)
{
    float clip_max = 1.0;
    float clip_min = 0.0;
    if (node->params().count("clip_max"))
    {
        clip_max = node->params()["clip_max"];
    }
    if (node->params().count("clip_min"))
    {
        clip_min = node->params()["clip_min"];
    }
    // infer output sign
    auto out_sign = clip_min < 0.0;
    dtype_utils::set_aipu_dtype(node->outputs()[0], out_sign);

    Op::quantize(node);
    uint8_t hard_sigmoid_opt_tag = 0;
    if (node->attrs().count("hard_sigmoid_opt_tag"))
    {
        hard_sigmoid_opt_tag = (uint8_t)node->get_attr("hard_sigmoid_opt_tag");
    }
    float alpha = param_utils::get_param(node, "alpha");
    float beta = param_utils::get_param(node, "beta");

    std::map<std::string, float> cal_param = {
        {"hard_sigmoid_opt_tag", hard_sigmoid_opt_tag},
        {"alpha", alpha},
        {"beta", beta},
        {"clip_max", clip_max},
        {"clip_min", clip_min}};
    node->constants()["lut"] = generate_lut(node, cal_hardsigmoid_func, cal_param);
}

REGISTER_QUANTIZATION_OP(HardSigmoid);

}; // namespace qtlibn
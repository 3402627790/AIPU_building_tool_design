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
 * @Date: 2022-05-10 20:41:31
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-10 21:08:40
 */

#include "quantization_transform/ops/bnll.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_bnll_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    // float ret;
    // if (value > 0)
    // {
    //     ret = value + log(1 + exp(-value));
    // }
    // else
    // {
    //     ret = log(1 + exp(value));
    // }

    // return ret;

    auto mask = value > 0;
    auto pos_value = value + (1.0 + (0 - value).exp()).log();
    pos_value = mask * pos_value;

    auto neg_value = (1.0 + value.exp()).log();
    neg_value = neg_value * (1.0 - mask);
    return pos_value + neg_value;
}

void BNLLOp::quantize(aipubt::NodePtr node)
{
    // infer output sign
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);

    Op::quantize(node);
    node->constants()["lut"] = generate_lut(node, cal_bnll_func);
}

REGISTER_QUANTIZATION_OP(BNLL);

} // namespace qtlibn

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
 * @Date: 2022-05-20 14:02:35
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-20 14:04:51
 */

#include "quantization_transform/ops/round.h"

namespace qtlibn
{
static aipubt::ComputeTensor cal_round_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    return value.round();
}

void RoundOp::quantize(aipubt::NodePtr node)
{
    // infer output sign
    auto out_sign = aipubt::Tensor::is_signed(node->inputs()[0]->quantization().aipu_dtype());
    dtype_utils::set_aipu_dtype(node->outputs()[0], out_sign);

    Op::quantize(node);
    node->constants()["lut"] = generate_lut(node, cal_round_func);
}

REGISTER_QUANTIZATION_OP(Round);

}; // namespace qtlibn

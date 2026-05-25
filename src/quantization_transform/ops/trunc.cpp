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
 * @Author: Kejia Qiang
 * @Date: 2022-11-16 19:28:20
 * @Last Modified by: Kejia Qiang
 * @Last Modified time: 2022-11-16 14:52:57
 */

#include "quantization_transform/ops/trunc.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_trunc_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    return value.astype(aipubt::Tensor::INT32).astype(aipubt::Tensor::FP32);
}

void TruncOp::quantize(aipubt::NodePtr node)
{
    auto out_sign = aipubt::Tensor::is_signed(node->inputs()[0]->type());
    dtype_utils::set_aipu_dtype(node->outputs()[0], out_sign);

    Op::quantize(node);
    node->constants()["lut"] = generate_lut(node, cal_trunc_func);
}

REGISTER_QUANTIZATION_OP(Trunc);

}; // namespace qtlibn

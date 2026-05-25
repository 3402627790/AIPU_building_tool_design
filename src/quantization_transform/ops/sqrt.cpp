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
 * @Date: 2020-09-08 09:58:17
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-07 10:05:47
 */

#include "quantization_transform/ops/sqrt.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_sqrt_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    return value.maximum(0.0).sqrt();
}

void SqrtOp::quantize(aipubt::NodePtr node)
{
    // infer output dtype
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);

    Op::quantize(node);
    node->constants()["lut"] = generate_lut(node, cal_sqrt_func);
}

REGISTER_QUANTIZATION_OP(Sqrt);

}; // namespace qtlibn

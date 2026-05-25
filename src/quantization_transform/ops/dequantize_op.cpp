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
 * @Date: 2022-11-22 11:37:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2026-04-08 17:11:08
 */

#include "quantization_transform/ops/dequantize_op.h"

namespace qtlibn
{
void DeQuantizeOp::quantize(aipubt::NodePtr node)
{
    node->params()["quantize_scale"] = node->inputs()[0]->quantization().scales();
    node->params()["quantize_zp"] = node->inputs()[0]->quantization().offsets();
}

void DeQuantizeOp::approximate(aipubt::NodePtr node)
{
    Op::approximate(node);
    node->params()["quantize_scale"] = node->inputs()[0]->quantization().scales();
    node->params()["quantize_zp"] = node->inputs()[0]->quantization().offsets();
}
REGISTER_QUANTIZATION_OP(DeQuantize);

}; // namespace qtlibn
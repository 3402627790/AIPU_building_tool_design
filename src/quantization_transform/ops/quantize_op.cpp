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
 * @Date: 2022-11-22 11:38:33
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-11 14:16:26
 */

#include "quantization_transform/ops/quantize_op.h"

namespace qtlibn
{
void QuantizeOp::quantize(aipubt::NodePtr node)
{
    dtype_utils::set_aipu_dtype(node->outputs()[0], true);
    Op::quantize(node);
    node->params()["quantize_scale"] = node->outputs()[0]->quantization().scales();
    node->params()["quantize_zp"] = node->outputs()[0]->quantization().offsets();
}

REGISTER_QUANTIZATION_OP(Quantize);

}; // namespace qtlibn
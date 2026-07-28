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
 * @Date: 2026-04-15 13:44:18
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2026-04-15 13:49:28
 */

#include "quantization_transform/ops/scaleddotproductattention.h"

namespace qtlibn
{

void ScaledDotProductAttentionOp::quantize(aipubt::NodePtr node)
{
    WARN() << "[QTLIB]" << node << " does not support offline-quantize.";
}

REGISTER_QUANTIZATION_OP(ScaledDotProductAttention);

}; // namespace qtlibn


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
 * @Date: 2023-10-17 10:35:11
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-17 10:41:10
 */

#include "quantization_transform/ops/collapse_repeated.h"

namespace qtlibn
{
void CollapseRepeatedOp::quantize(aipubt::NodePtr node)
{
    int32_t in_size = node->inputs().size();
    for (int32_t i = 0; i < in_size; i++)
    {
        node->outputs()[i]->quantization() = node->inputs()[i]->quantization();
    }
}

REGISTER_QUANTIZATION_OP(CollapseRepeated);

}; // namespace qtlibn

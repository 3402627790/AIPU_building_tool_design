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
 * @Date: 2025-03-10 13:38:23
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-03-10 13:45:49
 */

#include "quantization_transform/ops/cacheupdate.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{
void CacheUpdateOp::quantize(aipubt::NodePtr node)
{
    auto inp = node->inputs()[2];
    for (auto &out : node->outputs())
    {
        out->quantization() = inp->quantization();
    }
}

REGISTER_QUANTIZATION_OP(CacheUpdate);

}; // namespace qtlibn

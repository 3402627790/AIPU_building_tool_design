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
 * @Date: 2022-03-23 19:22:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-18 19:08:57
 */

#include "quantization_transform/ops/mvn.h"
#include "quantization_transform/ops/groupnorm.h"

namespace qtlibn
{

void MVNOp::quantize(aipubt::NodePtr node)
{
    auto groupnorm = GroupNormOp();
    groupnorm.quantize(node);

    std::map<std::string, std::string> name_map = {
        {"norm_scale_value", "scale_value"},
        {"norm_scale_type", "scale_type"},
        {"norm_shift_value", "shift_value"},
        {"norm_shift_type", "shift_type"},
    };
    for (auto kv : name_map)
    {
        node->params()[kv.second] = std::move(node->params()[kv.first]);
        node->params().erase(kv.first);
    }
}

REGISTER_QUANTIZATION_OP(MVN);
}; // namespace qtlibn

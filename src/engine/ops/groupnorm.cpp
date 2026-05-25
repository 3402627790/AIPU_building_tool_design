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
 * @Date: 2023-03-23 15:29:14
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-03-23 17:04:13
 */

#include "engine/engine.h"
#include "engine/ops/norm.h"

namespace aipubt
{
namespace engine
{
    EXEOP_REGISTER(OpType_GroupNorm, norm_forward, DTCriteria::FP32_ONLY, criteria_float);
} // namespace engine

} // namespace aipubt
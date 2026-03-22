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
 * @Author: Neo Wang
 * @Date: 2023-03-09 12:16:08
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2025-10-17 17:43:52
 */
#ifndef CORE_FP16_H
#define CORE_FP16_H
#include <core/fp16/bfp16.h>
#include <core/fp16/fp4e2m1.h>
#include <core/fp16/fp8e4m3.h>
#include <core/fp16/fp8e5m2.h>
#include <core/fp16/half.h>

namespace aipubt
{
using half_float::half;

} // namespace aipubt
#endif //!CORE_FP16_H
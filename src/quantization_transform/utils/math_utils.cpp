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
 * @Date: 2022-03-14 10:24:39
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-17 17:07:01
 */

#include "quantization_transform/utils/math_utils.h"

namespace math_utils
{
int32_t gcd(int32_t x, int32_t y)
{
    int32_t z = y;
    while (x % y != 0)
    {
        z = x % y;
        x = y;
        y = z;
    }
    return z;
}

}; // namespace math_utils

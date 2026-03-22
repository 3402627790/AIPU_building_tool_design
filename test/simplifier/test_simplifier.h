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
 * @Date: 2022-03-30 15:53:09
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-30 16:37:39
 */

#ifndef __TEST_SIMPLIFIER_H__
#define __TEST_SIMPLIFIER_H__

#include "transform/passes.h"
#include <core/ops/ops.h>

void test_eliminate_cast(void);
void test_fuse_tile_eltwise();
#endif
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
 * @Date: 2020-05-17 20:10:08
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-03-17 16:16:50
 */

#ifndef __TEST_AQT_H__
#define __TEST_AQT_H__

#include <ostream>

#include "core/compute_tensor.h"
#include "core/node.h"
#include "core/tensor.h"
#include "gcompiler_api.h"
#include "quantization/qtransform.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

using namespace aipubt;

// void test_qtlibn_relu(void);
// void test_qtlibn_depthwise(void);
// void test_qtlibn_abs(void);
// void test_qtlibn_gruv3(void);
// void test_qtlibn_basiclstm(void);
void test_qtlib(void);
// void test_q(void);
// void test_tanh();
#endif
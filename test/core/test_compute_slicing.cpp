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
 * @Date: 2020-04-02 10:30:48 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-04-02 11:37:56
 */
#include <algorithm>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <builder/builder.h>
#include <builder/mem_allocator.h>
#include <builder/scheduler.h>
#include <core/compute_tensor.h>
#include <gcompiler_api.h>
#include <utils/string_utils.h>
using namespace aipubt;

static void rand_data(ComputeTensor t)
{
    int *ptr = t->data<int>();
    for (uint32_t i = 0; i < t->mem_size() / 4; ++i)
    {
        *ptr = rand();
        ptr++;
    }
}
void test_ct_slicing()
{

    auto t = ComputeTensor::zeros({2, 3, 1, 1, 2, 4});
    rand_data(t);
    INFO() << t;
    t[{ComputeTensor::Ellipse, {2}}] = 314;
    INFO() << t[{ComputeTensor::Ellipse, {2}}];
    INFO() << t[0][0][0][0][0][2];
    INFO() << "t[0] = " << t[0];
    float v = t[0][0][0][0][0][2];
    INFO() << "t[0][0][0][0][0][2]:" << v;

    t[{{0, 1}, {1, 3}, {0, -1}, {0, -1}, {0, 2}, {0, 3}}] = -100;
    INFO() << t;

    auto b = ComputeTensor::zeros({2, 3, 4});
    auto c = ComputeTensor::zeros({2, 3, 4}) + 10;
    c[0][1][2] = 11;
    b[{{1}, {2}, {0, 4}}] = c[{{0}, {1}, {0, 4}}];
    INFO() << b;
    auto d = ComputeTensor::zeros({2, 3, 4});
    d[{{1, 2}, {2, 3}, {0, 4}}] = c[{{0, 1}, {1, 2}, {0, 4}}];
    INFO() << d;
    assert(d == b);
    d = ComputeTensor::zeros({2, 3, 4});
    d[{{1, 2}, {2, 3}}] = c[{{0, 1}, {1, 2}, {0, 4}}];
    INFO() << d;
    assert(d == b);
    d = ComputeTensor::zeros({2, 3, 4});
    d[{{1, 2}, {2, 3}}] = c[{{0, 1}, {1, 2}}];
    INFO() << d;
    assert(d == b);
    d = ComputeTensor::zeros({2, 3, 4});
    d[{{1}, {2}}] = c[{{0}, {1}, {0, 4}}];
    INFO() << d;
    assert(d == b);
    d = ComputeTensor::zeros({2, 3, 4});
    d[{{1}, {2}}] = c[{{0}, {1}}];
    INFO() << d;
    assert(d == b);
    d = ComputeTensor::zeros({2, 3, 4});
    d[1][2] = c[{{0}, {1}}];
    INFO() << d;
    assert(d == b);
    d = ComputeTensor::zeros({2, 3, 4});
    d[1][2][{0, 4}] = c[{{0}, {1}}];
    INFO() << d;
    assert(d == b);
}
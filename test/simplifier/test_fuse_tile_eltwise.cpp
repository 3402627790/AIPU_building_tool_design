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
 * @Date: 2022-03-30 15:55:17
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-30 16:39:29
 */

#include "test_simplifier.h"
using namespace aipubt;

void test_fuse_tile_eltwise(void)
{
    GraphPtr g = Graph::create("test_fuse_tile_eltwise");
    auto inp0 = ops::input({1, 1, 1, 60}, Tensor::FP32, {}, g);
    auto inp1 = ops::input({2, 4, 1, 60}, Tensor::FP32, {}, g);
    auto inp2 = ops::input({2, 4, 1, 60}, Tensor::FP32, {}, g);
    auto a = ops::tile(inp0, {2, 4, 1, 1});
    auto b = ops::add(a, inp1);
    auto c = ops::add(a, inp2);
    ops::add(b, c);
    g->to_png("before.png");
    fuse_tile_eltwise(g);
    g->to_png("after.png");
}
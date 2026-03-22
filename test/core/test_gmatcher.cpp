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
#include <gcompiler_api.h>
#include <utils/string_utils.h>
using namespace aipubt;
void test_gmater0()
{
    GraphMatcher gm(
        {
            {"n0", OpType_Input},
            {"n1", OpType_Input},
            {"n2", OpType_Eltwise},
        },
        {
            {"n0", "n2"},
            {"n1", "n2"},
        });
    GraphPtr g = Graph::parse("test.def");
    if (g)
        auto m = gm.match(g);
}
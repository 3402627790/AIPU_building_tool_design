//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file
// may only be used by a person authorised under and to the extent permitted
// by a subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2018 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//
//       Release Information:
//
//--------------------------------------------------------------------------------

#include <builder/builder.h>
#include <gcompiler_api.h>

#include <core/graph.h>
#include <core/ops/ops.h>
#include <gcompiler_api.h>

#include "ops_test_base.h"

using namespace aipubt;
namespace test_ops
{

class OpTestDiv : public OpTestBase
{
public:
    OpTestDiv() {};
    virtual ~OpTestDiv() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_div");
        auto inp1 = ops::input({8, 1, 3, 1}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto inp2 = ops::input({8, 1, 3, 1}, Tensor::Dtype::UINT8, {1.0, 100}, g);
        auto a = ops::div(inp1, inp2, Quantization(1.0, 150));
        if (!a)
        {
            return false;
        }

        g->to_png("div.png");
        g->serialize("graph_div.def");

        return true;
    }
};

class OpTestDivBroadcast : public OpTestBase
{
public:
    OpTestDivBroadcast() {};
    virtual ~OpTestDivBroadcast() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_div_broadcast");
        auto inp1 = ops::input({8, 1, 3, 4}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto inp2 = ops::input({4, 3, 1}, Tensor::Dtype::UINT8, {1.0, 100}, g);
        auto a = ops::div(inp1, inp2, Quantization(1.0, 150));
        if (!a)
        {
            return false;
        }

        g->to_png("div_broadcast.png");
        g->serialize("graph_div_broadcast.def");

        return true;
    }
};

REGISTER_TEST_OP(Div, "div", true);
REGISTER_TEST_OP(DivBroadcast, "div", true);

} // namespace test_ops
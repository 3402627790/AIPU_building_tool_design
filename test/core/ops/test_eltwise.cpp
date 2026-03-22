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
#include <builder/builder.h>
#include <gcompiler_api.h>

#include <core/graph.h>
#include <core/ops/ops.h>
#include <gcompiler_api.h>

#include "ops_test_base.h"

using namespace aipubt;
namespace test_ops
{
// eltwise: add/mul/max/min/sub

class OpTestAdd : public OpTestBase
{
public:
    OpTestAdd() {};
    virtual ~OpTestAdd() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_add");
        auto inp1 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto inp2 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::add(inp1, inp2, ops::Activation("NONE"), {1.0, 128});
        if (!a)
        {
            return false;
        }

        g->to_png("add.png");
        g->serialize("graph_add.def");

        return true;
    }
};

class OpTestAddMultiBroadcast : public OpTestBase
{
public:
    OpTestAddMultiBroadcast() {};
    virtual ~OpTestAddMultiBroadcast() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_add_multibroadcast");
        auto inp1 = ops::input({4, 1, 2}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto inp2 = ops::input({5, 4, 3, 1}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::add(inp1, inp2, ops::Activation("NONE"), {1.0, 128});
        if (!a)
        {
            return false;
        }

        g->to_png("add_multibroadcast.png");
        g->serialize("graph_add_multibroadcast.def");

        return true;
    }
};

class OpTestMul : public OpTestBase
{
public:
    OpTestMul() {};
    virtual ~OpTestMul() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_mul");
        auto inp1 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto inp2 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::mul(inp1, inp2, ops::Activation("NONE"), {1.0, 128});
        if (!a)
        {
            return false;
        }

        g->to_png("mul.png");
        g->serialize("graph_mul.def");

        return true;
    }
};

class OpTestMax : public OpTestBase
{
public:
    OpTestMax() {};
    virtual ~OpTestMax() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_max");
        auto inp1 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto inp2 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::max(inp1, inp2, ops::Activation("NONE"), {1.0, 128});
        if (!a)
        {
            return false;
        }

        g->to_png("max.png");
        g->serialize("graph_max.def");

        return true;
    }
};

class OpTestMin : public OpTestBase
{
public:
    OpTestMin() {};
    virtual ~OpTestMin() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_min");
        auto inp1 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto inp2 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::min(inp1, inp2, ops::Activation("NONE"), {1.0, 128});
        if (!a)
        {
            return false;
        }

        g->to_png("min.png");
        g->serialize("graph_min.def");

        return true;
    }
};

class OpTestSub : public OpTestBase
{
public:
    OpTestSub() {};
    virtual ~OpTestSub() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_sub");
        auto inp1 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto inp2 = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::sub(inp1, inp2, ops::Activation("RELU"), {1.0, 0});
        if (!a)
        {
            return false;
        }

        g->to_png("sub.png");
        g->serialize("graph_sub.def");

        return true;
    }
};

REGISTER_TEST_OP(Add, "eltwise", true);
REGISTER_TEST_OP(AddMultiBroadcast, "eltwise", true);
REGISTER_TEST_OP(Mul, "eltwise", true);
REGISTER_TEST_OP(Max, "eltwise", true);
REGISTER_TEST_OP(Min, "eltwise", true);
REGISTER_TEST_OP(Sub, "eltwise", true);

} // namespace test_ops

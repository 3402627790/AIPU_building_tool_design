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

class OpTestMatmul4d : public OpTestBase
{
public:
    OpTestMatmul4d() {};
    virtual ~OpTestMatmul4d() {};
    bool run_test() override
    {
        INFO() << "Test matmul start.";
        GraphPtr g = Graph::create("test_matmul", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp1 = ops::input({1, 1, 2, 4}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr inp2 = ops::input({1, 1, 4, 2}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr c = ops::matmul(inp1, inp2);
        if (!c)
        {
            return false;
        }

        g->to_png("matmul_4d.png");
        g->serialize("graph_matmul_4d.def", "matmul_4d.bin");

        INFO() << "Test matmul finish.";
        return true;
    }
};

class OpTestMatmul3d : public OpTestBase
{
public:
    OpTestMatmul3d() {};
    virtual ~OpTestMatmul3d() {};
    bool run_test() override
    {
        INFO() << "Test matmul start.";
        GraphPtr g = Graph::create("test_matmul", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp1 = ops::input({1, 2, 4}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr inp2 = ops::input({1, 4, 2}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr c = ops::matmul(inp1, inp2);
        if (!c)
        {
            return false;
        }

        g->to_png("matmul_3d.png");
        g->serialize("graph_matmul_3d.def", "matmul_3d.bin");

        INFO() << "Test matmul finish.";
        return true;
    }
};

class OpTestMatmul2d : public OpTestBase
{
public:
    OpTestMatmul2d() {};
    virtual ~OpTestMatmul2d() {};
    bool run_test() override
    {
        INFO() << "Test matmul start.";
        GraphPtr g = Graph::create("test_matmul", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp1 = ops::input({2, 4}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr inp2 = ops::input({4, 2}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr c = ops::matmul(inp1, inp2);
        if (!c)
        {
            return false;
        }

        g->to_png("matmul_2d.png");
        g->serialize("graph_matmul_2d.def", "matmul_2d.bin");

        INFO() << "Test matmul finish.";
        return true;
    }
};

class OpTestMatmul2dAdjTrueAdjTrue : public OpTestBase
{
public:
    OpTestMatmul2dAdjTrueAdjTrue() {};
    virtual ~OpTestMatmul2dAdjTrueAdjTrue() {};
    bool run_test() override
    {
        INFO() << "Test matmul start.";
        GraphPtr g = Graph::create("test_matmul", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp1 = ops::input({2, 4}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr inp2 = ops::input({4, 2}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr c = ops::matmul(inp1, inp2, true, true);
        if (!c)
        {
            return false;
        }

        g->to_png("matmul_2d.png");
        g->serialize("graph_matmul_2d_true_true.def", "matmul_2d_true_true.bin");

        INFO() << "Test matmul finish.";
        return true;
    }
};

class OpTestMatmul2dAdjFalseAdjTrue : public OpTestBase
{
public:
    OpTestMatmul2dAdjFalseAdjTrue() {};
    virtual ~OpTestMatmul2dAdjFalseAdjTrue() {};
    bool run_test() override
    {
        INFO() << "Test matmul start.";
        GraphPtr g = Graph::create("test_matmul", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp1 = ops::input({2, 4}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr inp2 = ops::input({2, 4}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr c = ops::matmul(inp1, inp2, false, true);
        if (!c)
        {
            return false;
        }

        g->to_png("matmul_2d_false_true.png");
        g->serialize("graph_matmul_2d_false_true.def", "matmul_2d_false_true.bin");

        INFO() << "Test matmul finish.";
        return true;
    }
};

REGISTER_TEST_OP(Matmul4d, "matmul", true);
REGISTER_TEST_OP(Matmul3d, "matmul", true);
REGISTER_TEST_OP(Matmul2d, "matmul", true);
REGISTER_TEST_OP(Matmul2dAdjTrueAdjTrue, "matmul", true);
REGISTER_TEST_OP(Matmul2dAdjFalseAdjTrue, "matmul", true);
} // namespace test_ops
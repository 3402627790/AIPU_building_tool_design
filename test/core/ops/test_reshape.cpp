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

class OpTestReshape : public OpTestBase
{
public:
    OpTestReshape() {};
    virtual ~OpTestReshape() {};
    bool run_test() override
    {
        INFO() << "reshape test start.";
        GraphPtr g = Graph::create("test_shape", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 2, 10, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr inp1 = ops::input({1, 10, 2, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr t1 = ops::reshape(inp, TensorShape(1, 10, 2, 1));
        TensorPtr t2 = ops::pad(t1, {{0, 0}, {1, 2}});
        TensorPtr c = ops::concat({inp1, t2});
        if (!c)
        {
            return false;
        }
        g->to_png("reshape.png");
        g->serialize("graph_reshape.def");

        return true;
    }
};

class OpTestExpandDims0 : public OpTestBase
{
public:
    OpTestExpandDims0() {};
    virtual ~OpTestExpandDims0() {};
    bool run_test() override
    {
        INFO() << "expand_dims_0 test start.";
        GraphPtr g = Graph::create("test_expand_dims", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({10, 10, 3}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr t1 = ops::expand_dims(inp, 0);
        if (!t1)
        {
            return false;
        }

        g->to_png("expand_dims_0.png");
        g->serialize("graph_expand_dims_0.def");
        INFO() << "expand_dims test finish.";

        return true;
    }
};
class OpTestExpandDims1 : public OpTestBase
{
public:
    OpTestExpandDims1() {};
    virtual ~OpTestExpandDims1() {};
    bool run_test() override
    {
        INFO() << "expand_dims_1 test start.";
        GraphPtr g = Graph::create("test_expand_dims", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({10, 10, 3}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr t1 = ops::expand_dims(inp, 1);
        if (!t1)
        {
            return false;
        }

        g->to_png("expand_dims_1.png");
        g->serialize("graph_expand_dims_1.def");
        INFO() << "expand_dims_1 test finish.";

        return true;
    }
};

class OpTestExpandDimsNeg3 : public OpTestBase
{
public:
    OpTestExpandDimsNeg3() {};
    virtual ~OpTestExpandDimsNeg3() {};
    bool run_test() override
    {
        INFO() << "expand_dims_Neg3 test start.";
        GraphPtr g = Graph::create("test_expand_dims_Neg3", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({10, 10, 3}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr t1 = ops::expand_dims(inp, 0);
        if (!t1)
        {
            return false;
        }

        g->to_png("expand_dims_Neg3.png");
        g->serialize("graph_expand_dims_Neg3.def");
        INFO() << "expand_dims_Neg3 test finish.";

        return true;
    }
};
class OpTestSqueezeAxis : public OpTestBase
{
public:
    OpTestSqueezeAxis() {};
    virtual ~OpTestSqueezeAxis() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_squeeze_axis", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 3, 4, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr t1 = ops::squeeze(inp, {0, 3});
        if (!t1)
        {
            return false;
        }

        g->to_png("test_squeeze_axis.png");
        g->serialize("graph_squeeze_axis.def");

        return true;
    }
};

class OpTestSqueeze : public OpTestBase
{
public:
    OpTestSqueeze() {};
    virtual ~OpTestSqueeze() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_squeeze", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 3, 4, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr t1 = ops::squeeze(inp);
        if (!t1)
        {
            return false;
        }

        g->to_png("test_squeeze.png");
        g->serialize("graph_squeeze.def");

        return true;
    }
};

class OpTestSqueezeErr1 : public OpTestBase
{
public:
    OpTestSqueezeErr1() {};
    virtual ~OpTestSqueezeErr1() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_squeeze_err1", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 3, 4, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr t1 = ops::squeeze(inp, {2});
        if (!t1)
        {
            return false;
        }

        g->to_png("test_squeeze_err1.png");
        g->serialize("graph_squeeze_err1.def");

        return true;
    }
};

class OpTestSqueezeErr2 : public OpTestBase
{
public:
    OpTestSqueezeErr2() {};
    virtual ~OpTestSqueezeErr2() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_squeeze_err2", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 3, 4, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        TensorPtr t1 = ops::squeeze(inp, {0, 3, 5});
        if (!t1)
        {
            return false;
        }

        g->to_png("test_squeeze_err2.png");
        g->serialize("graph_squeeze_err2.def");

        return true;
    }
};

REGISTER_TEST_OP(SqueezeAxis, "reshape", true);
REGISTER_TEST_OP(Squeeze, "reshape", true);
REGISTER_TEST_OP(SqueezeErr1, "reshape", false);
REGISTER_TEST_OP(SqueezeErr2, "reshape", false);

REGISTER_TEST_OP(Reshape, "reshape", true);
REGISTER_TEST_OP(ExpandDims0, "reshape", true);
REGISTER_TEST_OP(ExpandDims1, "reshape", true);
REGISTER_TEST_OP(ExpandDimsNeg3, "reshape", true);
} // namespace test_ops
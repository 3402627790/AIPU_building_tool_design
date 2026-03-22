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
// activation: relu/relu1/relu6/sigmoid/tanh/softplus/leaky_relu
class OpTestRelu : public OpTestBase
{
public:
    OpTestRelu(){};
    virtual ~OpTestRelu(){};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_relu");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::relu(inp, Quantization(1.0, 0));
        if (!a)
        {

            return false;
        }

        g->to_png("relu.png");
        g->serialize("graph_relu.def");
        return true;
    }
};

class OpTestRelu1 : public OpTestBase
{
public:
    OpTestRelu1(){};
    virtual ~OpTestRelu1(){};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_relu1");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::relu1(inp, Quantization(1.0, 0));
        if (!a)
        {

            return false;
        }

        g->to_png("relu1.png");
        g->serialize("graph_relu1.def");
        return true;
    }
};

class OpTestRelu6 : public OpTestBase
{
public:
    OpTestRelu6(){};
    virtual ~OpTestRelu6(){};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_relu6");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::relu6(inp, Quantization(1.0, 0));
        if (!a)
        {

            return false;
        }

        g->to_png("relu6.png");
        g->serialize("graph_relu6.def");
        return true;
    }
};

class OpTestShrink : public OpTestBase
{
public:
    OpTestShrink(){};
    virtual ~OpTestShrink(){};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_shrink");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::shrink(inp, 0.8, 1.0);
        if (!a)
        {

            return false;
        }

        g->to_png("shrink.png");
        g->serialize("graph_shrink.def");
        return true;
    }
};

class OpTestSigmoid : public OpTestBase
{
public:
    OpTestSigmoid(){};
    virtual ~OpTestSigmoid(){};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_sigmoid");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::sigmoid(inp, {0.00390625, 0});
        if (!a)
        {

            return false;
        }

        g->to_png("sigmoid.png");
        g->serialize("graph_sigmoid.def");
        return true;
    }
};

class OpTestTanh : public OpTestBase
{
public:
    OpTestTanh(){};
    virtual ~OpTestTanh(){};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_tanh");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::tanh(inp, {0.0078125f, 128});
        if (!a)
        {

            return false;
        }

        g->to_png("tanh.png");
        g->serialize("graph_tanh.def");
        return true;
    }
};

class OpTestLeakyRelu : public OpTestBase
{
public:
    OpTestLeakyRelu(){};
    virtual ~OpTestLeakyRelu(){};
    bool run_test() override
    {
        INFO() << "construct leakyrelu graph";
        GraphPtr g = Graph::create("test_leakyrelu");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::leaky_relu(inp, 0.1f, {0.1, 128});
        if (!a)
        {

            return false;
        }

        g->to_png("leakyrelu.png");
        g->serialize("graph_leakyrelu.def");
        return true;
    }
};

class OpTestThresholdedRelu : public OpTestBase
{
public:
    OpTestThresholdedRelu(){};
    virtual ~OpTestThresholdedRelu(){};
    bool run_test() override
    {
        INFO() << "construct thresholded_relu graph";
        GraphPtr g = Graph::create("test_thresholded_relu");
        auto inp = ops::input({2, 10, 15, 4}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::thresholded_relu(inp, 2.1, {0.1, 128});
        if (!a)
        {

            return false;
        }

        g->to_png("thresholded_relu.png");
        g->serialize("graph_thresholded_relu.def");
        return true;
    }
};

class OpTestSoftPlus : public OpTestBase
{
public:
    OpTestSoftPlus(){};
    virtual ~OpTestSoftPlus(){};
    bool run_test() override
    {
        INFO() << "construct softplus graph";
        GraphPtr g = Graph::create("test_softplus");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::softplus(inp, {0.1, 128});
        if (!a)
        {
            return false;
        }

        g->to_png("softplus.png");
        g->serialize("graph_softplus.def");
        return true;
    }
};

class OpTestElu : public OpTestBase
{
public:
    OpTestElu(){};
    virtual ~OpTestElu(){};
    bool run_test() override
    {
        INFO() << "construct elu graph";
        GraphPtr g = Graph::create("test_elu");
        auto inp = ops::input({2, 10}, Tensor::Dtype::INT8, {1.0, 0}, g);
        auto a = ops::elu(inp);
        if (!a)
        {

            return false;
        }

        g->to_png("elu.png");
        g->serialize("graph_elu.def");
        return true;
    }
};

class OpTestClip : public OpTestBase
{
public:
    OpTestClip(){};
    virtual ~OpTestClip(){};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_clip");
        auto inp = ops::input({2, 10}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::clip(inp, -128.f, 127.f, Quantization(1.0, 0));
        if (!a)
        {

            return false;
        }
        std::string run_mode = "fsym";
        auto qt = qtlibn::QuantizationTransform(g, run_mode, 8);
        qt.quantize();

        g->to_png("clip.png");
        g->serialize("graph_clip.def");
        return true;
    }
};

REGISTER_TEST_OP(Relu, "activation", true);
REGISTER_TEST_OP(Relu1, "activation", true);
REGISTER_TEST_OP(Relu6, "activation", true);
REGISTER_TEST_OP(Shrink, "activation", true);
REGISTER_TEST_OP(Sigmoid, "activation", true);
REGISTER_TEST_OP(Tanh, "activation", true);
REGISTER_TEST_OP(LeakyRelu, "activation", true);
REGISTER_TEST_OP(ThresholdedRelu, "activation", true);
REGISTER_TEST_OP(SoftPlus, "activation", true);
REGISTER_TEST_OP(Elu, "activation", true);
REGISTER_TEST_OP(Clip, "activation", true);
} // namespace test_ops

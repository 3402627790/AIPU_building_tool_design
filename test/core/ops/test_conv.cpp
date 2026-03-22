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

class OpTestConv2d : public OpTestBase
{
public:
    OpTestConv2d() {};
    virtual ~OpTestConv2d() {};
    bool run_test() override
    {
        INFO() << "Conv2d test start.";
        GraphPtr g = Graph::create("test_conv2d", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 3, 3, 1}, Tensor::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2, 2, 2, 2};
        uint8_t *filter_data = new uint8_t[4];
        memcpy(filter_data, filter_data_1, 4);
        auto filter = ops::make_weight(g, filter_data, {1, 2, 2, 1}, Tensor::UINT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4};
        int32_t *bais_data = new int32_t[1];
        memcpy(bais_data, bais_data_1, 4);
        auto bias = ops::make_bias(g, bais_data, 4, Tensor::INT32, Quantization(0.25, 0));

        int32_t n_data_1[] = {4};
        int32_t *n_data = new int32_t[1];
        memcpy(n_data, n_data_1, 4);
        auto negative_slop = ops::make_tensor(g, n_data, 1, Tensor::INT32, Quantization(0.25, 0));

        TensorPtr c = ops::conv2d(inp, filter, bias, {1, 1}, {0, 0, 0, 0}, ops::Activation("CLIP", -10.f, 20.f), {1, 1}, Quantization(0.5, 1));
        // TensorPtr c = ops::conv2d(inp, filter, bias, {1, 1}, {0, 0, 0, 0}, ops::Activation("LEAKYRELU"), 0.02, {1, 1}, Quantization(0.5, 1));
        // TensorPtr c = ops::conv2d(inp, filter, bias, {1, 1}, {0, 0, 0, 0}, ops::Activation("PRELU"), negative_slop, {1, 1}, Quantization(0.5, 1));
        if (!c)
        {
            return false;
        }
        qtlibn::QuantizationTransform asym_sym(g, "Sym", 8);
        asym_sym.quantize();

        g->to_png("conv2d.png");
        g->serialize("graph_conv2d.def");

        INFO() << "Conv2d test run finish.";
        return true;
    }
};

class OpTestConv1d : public OpTestBase
{
public:
    OpTestConv1d() {};
    virtual ~OpTestConv1d() {};
    bool run_test() override
    {
        INFO() << "Conv1d test start.";
        GraphPtr g = Graph::create("test_conv1d", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 9, 3}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8,
                                   2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8,
                                   2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8};
        uint8_t *filter_data = new uint8_t[12];
        memcpy(filter_data, filter_data_1, 12);
        auto filter = ops::make_weight(g, filter_data, {3, 4, 3}, Tensor::UINT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4, 4, 4};
        int32_t *bais_data = new int32_t[3];
        memcpy(bais_data, bais_data_1, 3 * 4);
        auto bias = ops::make_bias(g, bais_data, 3 * 4, Tensor::INT32, Quantization(0.25, 0));

        int32_t n_data_1[] = {4, 5, 6};
        int32_t *n_data = new int32_t[3];
        memcpy(n_data, n_data_1, 3 * 4);
        auto negative_slop = ops::make_tensor(g, n_data, 3, Tensor::INT32, Quantization(0.25, 0));

        // TensorPtr c = ops::conv1d(inp, filter, bias, {1}, {0, 0}, ops::Activation("NONE"), {1});
        // TensorPtr c = ops::conv1d(inp, filter, bias, {1}, {0, 0}, ops::Activation("CLIP"), -10, 20, {1}, Quantization(0.5, 1));
        // TensorPtr c = ops::conv1d(inp, filter, bias, {1}, {0, 0,}, ops::Activation("LEAKYRELU"), 20, {1}, Quantization(0.5, 1));
        TensorPtr c = ops::conv1d(inp, filter, bias, {2}, {
                                                              0,
                                                              0,
                                                          },
                                  ops::Activation("PRELU", negative_slop), {1}, Quantization(0.5, 1));

        if (!c)
        {
            return false;
        }
        qtlibn::QuantizationTransform asym_sym(g, "Sym", 8);
        asym_sym.quantize();

        g->to_png("conv1d.png");
        g->serialize("graph_conv1d.def", "conv1d_weight.bin");

        INFO() << "Conv1d test run finish.";
        return true;
    }
};

class OpTestConv2dPadding : public OpTestBase
{
public:
    OpTestConv2dPadding() {};
    virtual ~OpTestConv2dPadding() {};
    bool run_test() override
    {
        INFO() << "Conv2d padding test start.";
        GraphPtr g = Graph::create("test_conv2d_padding", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 3, 3, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2, 2, 2, 2};
        uint8_t *filter_data = new uint8_t[4];
        memcpy(filter_data, filter_data_1, 4);
        auto filter = ops::make_weight(g, filter_data, {1, 2, 2, 1}, Tensor::UINT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4};
        int32_t *bais_data = new int32_t[1];
        memcpy(bais_data, bais_data_1, 4);
        auto bias = ops::make_bias(g, bais_data, 4, Tensor::INT32, Quantization(0.25, 0));
        TensorPtr c = ops::conv2d(inp, filter, bias, {1, 1}, "SAME", ops::Activation("RELU"), {1, 1});
        if (!c)
        {
            return false;
        }

        g->to_png("conv2d_padding.png");
        g->serialize("graph_conv2d_padding.def");

        INFO() << "Conv2d padding test run finish.";
        return true;
    }
};

class OpTestConv1dPadding : public OpTestBase
{
public:
    OpTestConv1dPadding() {};
    virtual ~OpTestConv1dPadding() {};
    bool run_test() override
    {
        INFO() << "Conv1d test start.";
        GraphPtr g = Graph::create("test_conv1d", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 9, 3}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8};
        uint8_t *filter_data = new uint8_t[12];
        memcpy(filter_data, filter_data_1, 12);
        auto filter = ops::make_weight(g, filter_data, {1, 4, 3}, Tensor::UINT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4};
        int32_t *bais_data = new int32_t[1];
        memcpy(bais_data, bais_data_1, 4);
        auto bias = ops::make_bias(g, bais_data, 4, Tensor::INT32, Quantization(0.25, 0));
        TensorPtr c = ops::conv1d(inp, filter, bias, {1}, "SAME", ops::Activation("RELU"));
        if (!c)
        {
            return false;
        }

        g->to_png("conv1d_padding.png");
        g->serialize("graph_conv1d_padding.def");

        INFO() << "Conv1d padding test run finish.";
        return true;
    }
};

REGISTER_TEST_OP(Conv2d, "conv", true);
REGISTER_TEST_OP(Conv1d, "conv", true);

// REGISTER_TEST_OP(Conv2dPadding, "conv", true);
// REGISTER_TEST_OP(Conv1dPadding, "conv", true);
} // namespace test_ops
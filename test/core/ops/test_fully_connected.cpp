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
 * @Author: ALex Zheng
 * @Date: 2021-07-09 17:01:45
 * @Last Modified by: ALex Zheng
 * @Last Modified time: 2021-07-09 17:31:09
 */
#include <builder/builder.h>
#include <gcompiler_api.h>

#include <core/graph.h>
#include <core/ops/ops.h>
#include <gcompiler_api.h>

#include "ops_test_base.h"

using namespace aipubt;
namespace test_ops
{

class OpTestFC : public OpTestBase
{
public:
    OpTestFC() {};
    virtual ~OpTestFC() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_fully_connected", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({3, 1}, Tensor::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2};
        uint8_t *filter_data = new uint8_t[1];
        memcpy(filter_data, filter_data_1, 1);
        auto filter = ops::make_weight(g, filter_data, {1, 1}, Tensor::INT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4};
        int32_t *bais_data = new int32_t[1];
        memcpy(bais_data, bais_data_1, 4);
        auto bias = ops::make_bias(g, bais_data, 4, Tensor::INT32, Quantization(0.5, 0));
        TensorPtr c = ops::fully_connected(inp, filter, bias, ops::Activation("NONE"), Quantization(1, 0), {3, 1});
        if (!c)
        {
            return false;
        }

        g->to_png("test_fully_connected.png");
        g->serialize();

        return true;
    }
};

class OpTestFC2 : public OpTestBase
{
public:
    OpTestFC2() {};
    virtual ~OpTestFC2() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_fully_connected", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({3, 1}, Tensor::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2};
        uint8_t *filter_data = new uint8_t[1];
        memcpy(filter_data, filter_data_1, 1);
        auto filter = ops::make_weight(g, filter_data, {1, 1}, Tensor::INT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4};
        int32_t *bais_data = new int32_t[1];
        memcpy(bais_data, bais_data_1, 4);
        auto bias = ops::make_bias(g, bais_data, 4, Tensor::INT32, Quantization(0.5, 0));
        TensorPtr c = ops::fully_connected(inp, filter, bias, ops::Activation("NONE"), Quantization(1, 0));
        if (!c)
        {
            return false;
        }

        g->to_png("test_fully_connected.png");
        g->serialize("graph_fc2.def");

        return true;
    }
};

class OpTestFCWarn : public OpTestBase
{
public:
    OpTestFCWarn() {};
    virtual ~OpTestFCWarn() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_fully_connected", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({3, 1}, Tensor::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2};
        uint8_t *filter_data = new uint8_t[1];
        memcpy(filter_data, filter_data_1, 1);
        auto filter = ops::make_weight(g, filter_data, {1, 1}, Tensor::UINT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4};
        int32_t *bais_data = new int32_t[1];
        memcpy(bais_data, bais_data_1, 4);
        auto bias = ops::make_bias(g, bais_data, 4, Tensor::INT32, Quantization(0.5, 0));
        TensorPtr c = ops::fully_connected(inp, filter, bias, ops::Activation("NONE"), Quantization(1, 0), {3, 3});
        if (!c)
        {
            return false;
        }

        g->to_png("test_fully_connected.png");
        g->serialize("graph_fc_warn.def");

        return true;
    }
};

class OpTestFCShape : public OpTestBase
{
public:
    OpTestFCShape() {};
    virtual ~OpTestFCShape() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_fully_connected", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 10}, Tensor::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147};
        uint8_t *filter_data = new uint8_t[30];
        memcpy(filter_data, filter_data_1, 30);
        auto filter = ops::make_weight(g, filter_data, {3, 10}, Tensor::UINT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4, 8, 12};
        int32_t *bais_data = new int32_t[3];
        memcpy(bais_data, bais_data_1, 12);
        auto bias = ops::make_bias(g, bais_data, 12, Tensor::INT32, Quantization(0.5, 0));
        TensorPtr c = ops::fully_connected(inp, filter, bias, ops::Activation("NONE"), Quantization(1, 0), {2, 3});
        if (!c)
        {
            return false;
        }

        g->to_png("test_fully_connected.png");
        g->serialize("graph_fc_shape.def");

        return true;
    }
};

class OpTestFCShapeWarn : public OpTestBase
{
public:
    OpTestFCShapeWarn() {};
    virtual ~OpTestFCShapeWarn() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_fully_connected", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 4, 1, 5}, Tensor::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147};
        uint8_t *filter_data = new uint8_t[30];
        memcpy(filter_data, filter_data_1, 30);
        auto filter = ops::make_weight(g, filter_data, {3, 10}, Tensor::UINT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {4, 8, 12};
        int32_t *bais_data = new int32_t[3];
        memcpy(bais_data, bais_data_1, 12);
        auto bias = ops::make_bias(g, bais_data, 12, Tensor::INT32, Quantization(0.5, 0));
        TensorPtr c = ops::fully_connected(inp, filter, bias, ops::Activation("NONE"), Quantization(1, 0), {3, 2});
        if (!c)
        {
            return false;
        }

        g->to_png("test_fully_connected.png");
        g->serialize("graph_fc_shape_err.def");

        return true;
    }
};

REGISTER_TEST_OP(FC, "fully_connected", true);
REGISTER_TEST_OP(FC2, "fully_connected", true);
REGISTER_TEST_OP(FCWarn, "fully_connected", true);
REGISTER_TEST_OP(FCShape, "fully_connected", true);
REGISTER_TEST_OP(FCShapeWarn, "fully_connected", true);

} // namespace test_ops
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
 * @Date: 2021-07-09 15:03:00
 * @Last Modified by: ALex Zheng
 * @Last Modified time: 2021-07-26 16:02:41
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

class OpTestDepthwiseConv2D : public OpTestBase
{
public:
    OpTestDepthwiseConv2D() {};
    virtual ~OpTestDepthwiseConv2D() {};
    bool run_test() override
    {
        INFO() << "DepthwiseConv2D run start.";
        GraphPtr g = Graph::create("test_depthwise_conv2d", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 2, 2, 2}, Tensor::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {1, 2, 1, 0, 1, 1, 1, 0};
        uint8_t *filter_data = new uint8_t[8];
        memcpy(filter_data, filter_data_1, 8);

        auto filter = ops::make_weight(g, filter_data, {1, 2, 2, 2}, Tensor::INT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {0, 0};
        int32_t *bais_data = new int32_t[2];
        memcpy(bais_data, bais_data_1, 8);
        auto bias = ops::make_bias(g, bais_data, 8, Tensor::INT32, Quantization(0.25, 0));
        TensorPtr c = ops::depthwise_conv2d(inp, filter, bias, {1, 1}, {0, 0, 0, 0},
                                            ops::Activation("RELU"), {1, 1});
        if (!c)
        {
            return false;
        }

        g->to_png("test_depthwise_conv2d.png");
        g->serialize("graph_depthwise_conv2d.def");

        INFO() << "DepthwiseConv2D run finish.";
        return true;
    }
};

class OpTestDepthwiseConv2DPadding : public OpTestBase
{
public:
    OpTestDepthwiseConv2DPadding() {};
    virtual ~OpTestDepthwiseConv2DPadding() {};
    bool run_test() override
    {
        INFO() << "DepthwiseConv2D Padding test start.";
        GraphPtr g = Graph::create("test_depthwise_conv2d_padding", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 2, 2, 2}, Tensor::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {1, 2, 1, 0, 1, 1, 1, 0};
        uint8_t *filter_data = new uint8_t[8];
        memcpy(filter_data, filter_data_1, 8);

        auto filter = ops::make_weight(g, filter_data, {1, 2, 2, 2}, Tensor::INT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {0, 0};
        int32_t *bais_data = new int32_t[2];
        memcpy(bais_data, bais_data_1, 8);
        auto bias = ops::make_bias(g, bais_data, 8, Tensor::INT32, Quantization(0.25, 0));
        TensorPtr c = ops::depthwise_conv2d(inp, filter, bias, {1, 1}, "SAME",
                                            ops::Activation("RELU"), {1, 1});
        if (!c)
        {
            return false;
        }

        g->to_png("test_depthwise_conv2d_padding.png");
        g->serialize("graph_depthwise_conv2d_padding.def");

        INFO() << "DepthwiseConv2D Padding run finish.";
        return true;
    }
};

REGISTER_TEST_OP(DepthwiseConv2D, "depthwise_conv2d", true);
REGISTER_TEST_OP(DepthwiseConv2DPadding, "depthwise_conv2d", true);

} // namespace test_ops
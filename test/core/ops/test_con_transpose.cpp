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
#include "ops_test_base.h"
#include <builder/builder.h>
#include <gcompiler_api.h>

#include <core/graph.h>
#include <core/ops/ops.h>
#include <gcompiler_api.h>

using namespace aipubt;
namespace test_ops
{
class OpTestConv2dTranspose : public OpTestBase
{
public:
    OpTestConv2dTranspose() {};
    virtual ~OpTestConv2dTranspose() {};
    bool run_test() override
    {
        INFO() << "test Conv2dTranspose run start.";
        GraphPtr g = Graph::create("test_conv2d_transpose", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 2, 2, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2, 6, 10, 14, 18, 22, 26, 30, 34, 4, 8, 12, 16, 20, 24, 28, 32, 36};
        uint8_t *filter_data = new uint8_t[2 * 3 * 3 * 1];
        memcpy(filter_data, filter_data_1, 2 * 3 * 3 * 1);
        auto filter = ops::make_weight(g, filter_data, {2, 3, 3, 1}, Tensor::INT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {-6, -8};
        int32_t *bais_data = new int32_t[2];
        memcpy(bais_data, bais_data_1, 4 * 2);
        auto bias = ops::make_bias(g, bais_data, 4 * 2, Tensor::INT32, Quantization(0.25, 0));

        TensorPtr c = ops::conv2d_transpose(inp, filter, bias, {1, 1}, {0, 1, 2, 0}, 1, ops::Activation("LeakyRelu", 0.02f), {1, 1}, Quantization(0.5, 1));
        if (!c)
        {
            return false;
        }
        qtlibn::QuantizationTransform asym_sym(g, "Sym", 8);
        asym_sym.quantize();

        g->to_png("test_conv2d_transpose.png");
        g->serialize("graph_conv2d_transpose.def", "conv2d_transpose_weight.bin");

        return true;
    }
};

class OpTestConv1dTranspose : public OpTestBase
{
public:
    OpTestConv1dTranspose() {};
    virtual ~OpTestConv1dTranspose() {};
    bool run_test() override
    {

        INFO() << " test Conv1dTranspose run start.";
        GraphPtr g = Graph::create("test_conv1d_transpose", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 2, 1}, Tensor::Dtype::UINT8, {1.0, 0}, g);
        uint8_t filter_data_1[] = {2, 6, 10, 14, 18, 22};
        uint8_t *filter_data = new uint8_t[2 * 3 * 1];
        memcpy(filter_data, filter_data_1, 2 * 3 * 1);
        auto filter = ops::make_weight(g, filter_data, {2, 3, 1}, Tensor::INT8, Quantization(0.5, 0));

        int32_t bais_data_1[] = {-6, -8};
        int32_t *bais_data = new int32_t[2];
        memcpy(bais_data, bais_data_1, 4 * 2);
        auto bias = ops::make_bias(g, bais_data, 4 * 2, Tensor::INT32, Quantization(0.25, 0));

        TensorPtr c = ops::conv1d_transpose(inp, filter, bias, {1}, {
                                                                        -1,
                                                                        0,
                                                                    },
                                            1, ops::Activation("Clip", -10.f, 20.f), {1}, Quantization(0.5, 1));
        if (!c)
        {
            return false;
        }
        qtlibn::QuantizationTransform asym_sym(g, "Sym", 8);
        asym_sym.quantize();

        g->to_png("test_conv1d_transpose.png");
        g->serialize("graph_conv1d_transpose.def", "conv1d_transpose_weight.bin");

        return true;
    }
};

REGISTER_TEST_OP(Conv2dTranspose, "conv_transpose", true);
REGISTER_TEST_OP(Conv1dTranspose, "conv_transpose", true);

} // namespace test_ops
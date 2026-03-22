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

class OpTestBiasAdd : public OpTestBase
{
public:
    OpTestBiasAdd() {};
    virtual ~OpTestBiasAdd() {};
    bool run_test() override
    {
        INFO() << "Test bias_add start.";
        GraphPtr g = Graph::create("test_bias_add", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 2, 2, 4}, Tensor::INT16, {1.0, 128}, g);
        int16_t bias_data_1[] = {100, 100, 100, 100};
        int16_t *bias_data = new int16_t[4];
        memcpy(bias_data, bias_data_1, 2 * 4);
        auto bias = ops::make_bias(g, bias_data, 8, Tensor::INT16, Quantization(1.0, 0));
        TensorPtr c = ops::bias_add(inp, bias);
        if (!c)
        {
            return false;
        }

        g->to_png("bias_add.png");
        g->serialize("graph_bias_add_16bit.def", "bias_add2_16bit_weight.bin");

        INFO() << "Test bias_add finish.";
        return true;
    }
};

class OpTestBiasAdd2 : public OpTestBase
{
public:
    OpTestBiasAdd2() {};
    virtual ~OpTestBiasAdd2() {};
    bool run_test() override
    {
        INFO() << "Test bias_add start.";
        GraphPtr g = Graph::create("test_bias_add", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({1, 2, 2, 16}, Tensor::INT8, {1.0, 0}, g);
        int8_t bias_data_1[] = {0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64,
                                0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64, 0x64};
        int8_t *bias_data = new int8_t[16];
        memcpy(bias_data, bias_data_1, 16);
        auto bias = ops::make_bias(g, bias_data, 16, Tensor::INT8, Quantization(1.0, 0));
        TensorPtr c = ops::bias_add(inp, bias);
        if (!c)
        {
            return false;
        }

        g->to_png("bias_add2.png");
        g->serialize("graph_bias_add_8bit.def", "bias_add2_8bit_weight.bin");

        INFO() << "Test bias_add finish.";
        return true;
    }
};

REGISTER_TEST_OP(BiasAdd, "bias_add", true);
REGISTER_TEST_OP(BiasAdd2, "bias_add", true);
} // namespace test_ops
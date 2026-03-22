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

#include <numeric>

#include "ops_test_base.h"

using namespace aipubt;
namespace test_ops
{

class OpTestConstant : public OpTestBase
{
public:
    OpTestConstant(){};
    virtual ~OpTestConstant(){};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_constant_add");

        constexpr int32_t data_size = 100;
        std::vector<uint8_t> data(data_size);
        std::iota(data.begin(), data.end(), -50);
        uint8_t *const_data = new uint8_t[data_size];
        memcpy(const_data, data.data(), data_size);
        auto c = ops::constant(g, const_data, TensorShape(data_size), Tensor::UINT8, Quantization(0.1, 128));
        auto i = ops::input({1, 3, 3, 100}, Tensor::Dtype::UINT8, {1.0, 128});
        auto a = ops::add(i, c, ops::Activation("NONE"), {2.0, 128});
        if (!a)
        {
            return false;
        }

        g->to_png("constant_add.png");
        g->serialize("graph_constant_add.def");

        return true;
    }
};

REGISTER_TEST_OP(Constant, "constant", true);

} // namespace test_ops
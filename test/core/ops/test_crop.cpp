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

class OpTestCrop : public OpTestBase
{
public:
    OpTestCrop() {};
    virtual ~OpTestCrop() {};
    bool run_test() override
    {
        INFO() << "Test crop start.";
        GraphPtr g = Graph::create("test_crop", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp1 = ops::input({1, 3, 2, 4}, Tensor::INT8, {1.0, 0}, g);
        TensorPtr c = ops::crop(inp1, {{0, 0}, {1, 2}, {0, 1}, {-1, -2}});
        if (!c)
        {
            return false;
        }

        g->to_png("crop.png");
        g->serialize("graph_crop.def", "crop.bin");

        INFO() << "Test crop finish.";
        return true;
    }
};
REGISTER_TEST_OP(Crop, "crop", true);
} // namespace test_ops
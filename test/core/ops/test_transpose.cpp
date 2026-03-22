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
/*
 * @Author: ALex Zheng
 * @Date: 2021-07-07 17:40:08
 * @Last Modified by: ALex Zheng
 * @Last Modified time: 2021-07-07 17:58:43
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

class OpTestTranspose : public OpTestBase
{
public:
    OpTestTranspose() {};
    virtual ~OpTestTranspose() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_transpose", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 3, 4, 5}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        TensorPtr t1 = ops::transpose(inp, {3, 0, 1, 2});
        if (!t1)
        {
            return false;
        }
        g->to_png("test_transpose.png");
        g->serialize("graph_transpose.def");
        return true;
    }
};

class OpTestTransposeErr : public OpTestBase
{
public:
    OpTestTransposeErr() {};
    virtual ~OpTestTransposeErr() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_transpose_err", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 3, 4, 5}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        TensorPtr t1 = ops::transpose(inp, {3, 0, 1});
        if (!t1)
        {
            return false;
        }
        g->to_png("test_transpose_err.png");
        g->serialize("graph_transpose_err.def");
        return true;
    }
};

REGISTER_TEST_OP(Transpose, "transpose", true);
REGISTER_TEST_OP(TransposeErr, "transpose", false);

} // namespace test_ops
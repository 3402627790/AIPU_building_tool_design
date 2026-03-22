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
 * @Date: 2021-07-13 10:55:59
 * @Last Modified by: ALex Zheng
 * @Last Modified time: 2021-07-14 08:50:26
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
class OpTestGather1 : public OpTestBase
{
public:
    OpTestGather1() {};
    virtual ~OpTestGather1() {};
    bool run_test() override
    {
        INFO() << " test gather run start.";
        GraphPtr g = Graph::create("test_gather", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 2, 4}, Tensor::UINT8, {1.0, 0}, g);

        TensorPtr inp1 = ops::input({3, 2}, Tensor::INT32, {1.0, 0}, g);
        // int32_t indices_data[] = {2, 0, 0, 1, 1, 2};
        int32_t axis = 1;
        TensorPtr c = ops::gather(inp, inp1, axis);
        if (!c)
        {
            return false;
        }

        g->to_png("test_gather.png");
        g->serialize("graph_gather.def", "gather_weight.bin");

        INFO() << " test gather finish.";
        return true;
    }
};

class OpTestEmbedding : public OpTestBase
{
public:
    OpTestEmbedding() {};
    virtual ~OpTestEmbedding() {};
    bool run_test() override
    {
        INFO() << " test embedding run start.";
        GraphPtr g = Graph::create("test_embedding", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 2, 4}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr inp1 = ops::input({3}, Tensor::INT32, {1.0, 0}, g);
        TensorPtr c = ops::embedding_lookup(inp, inp1);
        if (!c)
        {
            return false;
        }

        g->to_png("test_embedding.png");
        g->serialize("graph_embedding.def", "embdding_weight.bin");

        INFO() << " test embedding finish.";
        return true;
    }
};

class OpTestGatherNd : public OpTestBase
{
public:
    OpTestGatherNd() {};
    virtual ~OpTestGatherNd() {};
    bool run_test() override
    {
        INFO() << " test gather_nd run start.";
        GraphPtr g = Graph::create("test_gather_nd", aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 2}, Tensor::UINT8, {1.0, 0}, g);
        TensorPtr inp1 = ops::input({2, 2}, Tensor::INT32, {1.0, 0}, g);
        TensorPtr c = ops::gather_nd(inp, inp1, 0);
        if (!c)
        {
            return false;
        }

        g->to_png("test_gather_nd.png");
        g->serialize("graph_gather_nd.def");

        INFO() << " test gather_nd finish.";
        return true;
    }
};

REGISTER_TEST_OP(Gather1, "gather", true);
REGISTER_TEST_OP(GatherNd, "gather", true);
REGISTER_TEST_OP(Embedding, "gather", true);
} // namespace test_ops
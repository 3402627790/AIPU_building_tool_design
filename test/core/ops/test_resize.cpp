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

#include "ops_test_base.h"

using namespace aipubt;
namespace test_ops
{

class OpTestBilinear : public OpTestBase
{
public:
    OpTestBilinear() {};
    virtual ~OpTestBilinear() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_bilinear");
        auto inp = ops::input({2, 2, 2, 2}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        std::vector<uint32_t> output_info = {3, 3};
        auto a = ops::resize(inp, output_info, {}, ops::ResizeMethod::Bilinear, ops::ResizeMode::HalfPixel);
        if (!a)
        {
            return false;
        }

        g->to_png("resize_bilinear.png");
        g->serialize("graph_bilinear.def");

        return true;
    }
};

class OpTestNearestNeighbor : public OpTestBase
{
public:
    OpTestNearestNeighbor() {};
    virtual ~OpTestNearestNeighbor() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_nearest_neighbor");
        auto inp = ops::input({2, 2, 2, 2}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        std::vector<uint32_t> output_info = {3, 3};
        auto a = ops::resize(inp, output_info, {}, ops::ResizeMethod::NearestNeighbor, ops::ResizeMode::AlignCorners);
        if (!a)
        {
            return false;
        }

        g->to_png("resize_nearest_neighbor.png");
        g->serialize("graph_nearest_neigbor.def");

        return true;
    }
};

class OpTestNearestNeighborF : public OpTestBase
{
public:
    OpTestNearestNeighborF() {};
    virtual ~OpTestNearestNeighborF() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("test_nearest_neighbor_f");
        auto inp = ops::input({2, 2, 2, 2}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        std::vector<float> output_info = {3.0, 4.0};
        auto a = ops::resize(inp, {}, output_info, ops::ResizeMethod::NearestNeighbor, ops::ResizeMode::AlignCorners);
        if (!a)
        {
            return false;
        }

        g->to_png("resize_nearest_neighbor.png");
        g->serialize("graph_nearest_neigbor.def");

        return true;
    }
};

REGISTER_TEST_OP(Bilinear, "resize", true);
REGISTER_TEST_OP(NearestNeighbor, "resize", true);
REGISTER_TEST_OP(NearestNeighborF, "resize", true);

} // namespace test_ops
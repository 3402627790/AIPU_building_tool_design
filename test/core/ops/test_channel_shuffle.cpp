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
 * @Date: 2021-07-15 14:52:30
 * @Last Modified by: ALex Zheng
 * @Last Modified time: 2021-07-15 15:32:43
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

class OpTestChannelShuffleDim4Axis3 : public OpTestBase
{
public:
    OpTestChannelShuffleDim4Axis3(){};
    virtual ~OpTestChannelShuffleDim4Axis3(){};
    bool run_test() override
    {
        std::string name = "channel_shuffle_dim4_axis3";
        INFO() << "Test" << name << "start.";
        GraphPtr g = Graph::create(name, aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 2, 3, 12});
        int32_t groups = 3;
        int32_t split = 1;
        auto c = ops::channel_shuffle(inp, groups, split);
        if (c.size() == 0)
        {
            return false;
        }

        g->to_png(name + ".png");
        g->serialize("graph_" + name + ".def", "weight_" + name + ".bin");

        INFO() << "Test " << name << "finish.";
        return true;
    }
};

class OpTestChannelShuffleDim4AxisN1 : public OpTestBase
{
public:
    OpTestChannelShuffleDim4AxisN1(){};
    virtual ~OpTestChannelShuffleDim4AxisN1(){};
    bool run_test() override
    {
        std::string name = "channel_shuffle_dim4_axis_neg1";
        INFO() << "Test" << name << "start.";
        GraphPtr g = Graph::create(name, aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 2, 3, 12});
        int32_t groups = 3;
        int32_t split = 1;
        auto c = ops::channel_shuffle(inp, groups, split);
        if (c.size() == 0)
        {
            return false;
        }

        g->to_png(name + ".png");
        g->serialize("graph_" + name + ".def", "weight_" + name + ".bin");

        INFO() << "Test " << name << "finish.";
        return true;
    }
};

class OpTestChannelShuffleDim2Axis0 : public OpTestBase
{
public:
    OpTestChannelShuffleDim2Axis0(){};
    virtual ~OpTestChannelShuffleDim2Axis0(){};
    bool run_test() override
    {
        std::string name = "channel_shuffle_dim2_axis0";
        INFO() << "Test" << name << "start.";
        GraphPtr g = Graph::create(name, aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({6, 2});
        int32_t groups = 3;
        int32_t split = 1;
        auto c = ops::channel_shuffle(inp, groups, split);
        if (c.size() == 0)
        {
            return false;
        }

        g->to_png(name + ".png");
        g->serialize("graph_" + name + ".def", "weight_" + name + ".bin");

        INFO() << "Test " << name << "finish.";
        return true;
    }
};

class OpTestChannelShuffleDim2 : public OpTestBase
{
public:
    OpTestChannelShuffleDim2(){};
    virtual ~OpTestChannelShuffleDim2(){};
    bool run_test() override
    {
        std::string name = "channel_shuffle_dim2";
        INFO() << "Test" << name << "start.";
        GraphPtr g = Graph::create(name, aipubt::DataLayout::DataLayout_NHWC);
        TensorPtr inp = ops::input({2, 6});
        int32_t groups = 2;
        int32_t split = 1;
        auto c = ops::channel_shuffle(inp, groups, split);
        if (c.size() == 0)
        {
            return false;
        }

        g->to_png(name + ".png");
        g->serialize("graph_" + name + ".def", "weight_" + name + ".bin");

        INFO() << "Test " << name << "finish.";
        return true;
    }
};

REGISTER_TEST_OP(ChannelShuffleDim4Axis3, "channel_shuffle", true);
REGISTER_TEST_OP(ChannelShuffleDim4AxisN1, "channel_shuffle", true);
REGISTER_TEST_OP(ChannelShuffleDim2Axis0, "channel_shuffle", true);
REGISTER_TEST_OP(ChannelShuffleDim2, "channel_shuffle", true);
} // namespace test_ops
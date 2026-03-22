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
 * @Author: Samuel Han
 * @Date: 2020-02-27 14:03:05
 * @Last Modified by: Yuchou Gan
 * @Last Modified time: 2020-04-15 14:40:28
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
static bool criteria(NodePtr n)
{
    if (n->type() == OpType_Mul || n->type() == OpType_Add || n->type() == OpType_Sub || n->type() == OpType_Div)
    {
        auto inp1 = n->inputs()[0];
        auto inp2 = n->inputs()[1];
        // not shape().size(), just for [1,1,1024] add [1,1024]
        return (inp1->shape() != inp2->shape());
    }
    return false;
}
static void adapt_arithmetic_op_pad_shape(NodePtr n)
{
    bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    auto inp1 = n->inputs()[0];
    auto inp2 = n->inputs()[1];
    auto shape1 = inp1->shape();
    auto shape2 = inp2->shape();
    std::vector<int> s;
    std::vector<Expr> s1;
    std::vector<Expr> s2;
    if (shape1.dim() < shape2.dim())
    {
        s = shape1;
        s1 = shape1.symbol();
        for (int i = 0; i < (shape2.dim() - shape1.dim()); ++i)
        {
            s.insert(s.begin(), 1);
            s1.insert(s1.begin(), 1);
        }
        shape1 = s;
    }
    else if (shape1.dim() > shape2.dim())
    {
        s = shape2;
        s2 = shape2.symbol();
        for (int i = 0; i < (shape1.dim() - shape2.dim()); ++i)
        {
            s.insert(s.begin(), 1);
            s2.insert(s2.begin(), 1);
        }
        shape2 = s;
    }
    if (inp1->shape() != shape1)
    {
        auto reshape_out = ops::reshape(inp1, shape1);
        auto reshape = reshape_out->node();
        if (enable_ds)
        {
            reshape->params()["ds_shape"] = s1;
        }
        reshape->infer_shape(true);
        n->add_input(reshape_out, n->remove_input(inp1));
        n->infer_shape(true);
        ADD_LOCATION_BY_NODE(reshape, n);
    }
    if (inp2->shape() != shape2)
    {
        auto reshape_out = ops::reshape(inp2, shape2);
        auto reshape = reshape_out->node();
        if (enable_ds)
        {
            reshape->params()["ds_shape"] = s2;
        }
        reshape->infer_shape(true);
        n->add_input(reshape_out, n->remove_input(inp2));
        n->infer_shape(true);
        ADD_LOCATION_BY_NODE(reshape, n);
    }
}

static void adapt_arithmetic_op_reshape_input(NodePtr n)
{
    //find input
    auto inp1 = n->inputs()[0];
    auto inp2 = n->inputs()[1];
    auto shape1 = inp1->shape();
    auto shape2 = inp2->shape();
    TensorPtr inp = inp1;
    if (shape1.dim() == shape2.dim())
    {
        int cdim = shape1.dim() - 1;
        std::vector<int> v;
        std::vector<Expr> s;
        for (int i = 0; i < shape1.dim(); ++i)
        {
            if (shape1[i] != 1)
            {
                if (shape2[i] == 1)
                {
                    inp = inp2;
                }
            }
            if (shape1[i] != 1 && shape2[i] != 1)
            {
                cdim = i;
                break;
            }
        }
        // for case like: [1,1,3,2] [1,1,3,1]
        for (int i = cdim; i < shape1.dim(); ++i)
        {
            if (shape1[i] == shape2[i])
            {
                continue;
            }
            if (shape2[i] == 1)
            {
                inp = inp2;
                break;
            }
            else
            {
                inp = inp1;
                break;
            }
        }
        // for case like: [1,1,1024] [1,1,1024]
        if (shape1 == shape2)
        {
            /*
                1. input/constant has no inputs()
                2. the graph after adapt_arithmetic_op_pad_shape shoule below:
                     input0[1,1024] --> reshape[1,1,1024] \
                                                         add[1,1,1024]---->
                     input1[1,1,1024] --------------------/
                    <=========>shoule be =========>
                     input0[1,1024] --> reshape[1,1,1024]--> reshape[1024] \
                                                                          add[1,1,1024]---->
                     input1[1,1,1024] -------------------------------------/
                    <=========>shoule NOT be =========>
                     input0[1,1024] --> reshape[1,1,1024] \
                                                         add[1,1,1024]---->
                     input1[1,1,1024] --> reshape[1024] ->/
             */
            auto inp2_parent = inp2->node();
            auto inp2_parent_tensor = inp2_parent->outputs()[0];
            if (inp2_parent->inputs().size() != 0 &&
                inp2_parent->type() == OpType_Reshape)
            {
                inp2_parent_tensor = inp2_parent->inputs()[0];
            }
            auto inp1_parent = inp1->node();
            auto inp1_parent_tensor = inp1_parent->outputs()[0];
            if (inp1_parent->inputs().size() != 0 &&
                inp1_parent->type() == OpType_Reshape)
            {
                inp1_parent_tensor = inp1_parent->inputs()[0];
            }

            if (inp2_parent_tensor->shape().dim() <
                inp1_parent_tensor->shape().dim())
            {
                inp = inp2;
            }
            else
            {
                inp = inp1;
            }
        }
        for (int i = cdim; i < inp->shape().dim(); ++i)
        {
            v.push_back(inp->shape()[i]);
            s.push_back(inp->shape().symbol()[i]);
        }
        TensorShape shape = v;
        auto reshape = ops::reshape(inp, shape);

        auto reshape_node = reshape->node();
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            reshape_node->params()["ds_shape"] = s;
        }
        reshape_node->infer_shape(true);

        n->add_input(reshape, n->remove_input(inp));
        ADD_LOCATION_BY_TENSOR(reshape, n);
    }
}

// insert reshape to reduce dim of input to make Layerlib happy
void adapt_arithmetic_op(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        TensorPtr inp = nullptr;
        TensorShape shape;
        if (!criteria(n))
        {
            continue;
        }
        adapt_arithmetic_op_pad_shape(n);
        adapt_arithmetic_op_reshape_input(n);
    }
}
} // namespace aipubt

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
 * @Author: Alex Zheng
 * @Date: 2025-02-21 10:56:07
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2025-02-21 17:56:07
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <numeric>

namespace aipubt
{
namespace
{
    bool criteria(NodePtr n)
    {
        static std::set<OpType> supported_ops = {
            OpType_Bitwise,
            OpType_Eltwise,
        };
        if (supported_ops.count(n->type()) == 0)
            return false;
        if ((n->type() == OpType_Bitwise) && (n->params()["method"] == "NOT"))
            return false;

        auto inp0 = n->inputs()[0];
        auto inp1 = n->inputs()[1];
        if (n->is_dynamic())
        {
            // dynamic support:  input0 [1,2,1], input [n,2,c]
            //  1. input1 dim size =input2 dim size
            //  2. only one symbol tensor
            //  3. boadcast dimension must be 1
            auto inp0 = n->inputs()[0];
            auto inp1 = n->inputs()[1];
            if (!inp0->shape().symbol().is_constant() && !inp0->shape().symbol().is_constant())
            {
                return false;
            }
            if (inp0->shape().dim() != inp1->shape().dim())
            {
                return false;
            }
            if (inp0->shape().symbol().is_constant())
            {
                for (auto i = 0; i < inp0->shape().dim(); ++i)
                {
                    if (!inp1->shape().symbol()[i]->is_constant() &&
                        inp0->shape()[i] != 1)
                    {
                        return false;
                    }
                }
            }
            else if (inp1->shape().symbol().is_constant())
            {
                for (auto i = 0; i < inp0->shape().dim(); ++i)
                {
                    if (!inp0->shape().symbol()[i]->is_constant() &&
                        inp1->shape()[i] != 1)
                    {
                        return false;
                    }
                }
            }
        }
        else
        {
            if (inp0->shape() == inp1->shape())
                return false;
        }
        return true;
    }

    void input_tile_repeats(NodePtr n,
                            std::vector<uint32_t> &repeats_0,
                            std::vector<uint32_t> &repeats_1,
                            std::vector<tongue::Expr> &ds_repeats_0,
                            std::vector<tongue::Expr> &ds_repeats_1)
    {
        auto inp0 = n->inputs()[0];
        auto inp1 = n->inputs()[1];
        for (auto i = 0; i < inp0->shape().dim(); ++i)
        {
            int inp0_shape = inp0->shape()[i];
            int inp1_shape = inp1->shape()[i];

            if (inp0_shape == inp1_shape)
            {
                continue;
            }
            else if (inp0_shape > inp1_shape)
            {
                if (inp0_shape % inp1_shape == 0)
                {
                    repeats_1[i] = inp0_shape / inp1_shape;
                }
                else
                {
                    ERROR() << " inp0_shape[" << i << "]: " << inp0_shape
                            << " inp1_shape[" << i << "]: " << inp1_shape
                            << " can not tile";
                    return;
                }
            }
            else
            {
                if (inp1_shape % inp0_shape == 0)
                {
                    repeats_0[i] = inp1_shape / inp0_shape;
                }
                else
                {
                    ERROR() << " inp0_shape[" << i << "]: " << inp0_shape
                            << " inp1_shape[" << i << "]: " << inp1_shape
                            << " can not tile";
                    return;
                }
            }
        }
        if (n->is_dynamic())
        {
            ds_repeats_1.assign(inp1->shape().symbol().dim(), 1);
            if (inp0->shape().symbol().is_constant())
            {
                ds_repeats_0.assign(inp0->shape().symbol().dim(), 1);
                for (auto i = 0; i < inp0->shape().dim(); ++i)
                {
                    ds_repeats_0[i] = inp1->shape().symbol()[i].floordiv(inp0->shape()[i]);
                }
            }
            else if (inp1->shape().symbol().is_constant())
            {
                ds_repeats_1.assign(inp1->shape().symbol().dim(), 1);
                for (auto i = 0; i < inp1->shape().dim(); ++i)
                {
                    ds_repeats_1[i] = inp0->shape().symbol()[i].floordiv(inp1->shape()[i]);
                }
            }
        }

        return;
    }
} // namespace

bool binary_op_broadcast(GraphPtr g)
{
    auto do_reshape = [](TensorPtr inp0, TensorPtr inp1)
    {
        auto new_shape = inp0->shape();
        std::vector<int> s = new_shape;
        for (int i = 0; i < (inp1->shape().dim() - inp0->shape().dim()); ++i)
        {
            s.insert(s.begin(), 1);
        }
        new_shape = s;
        auto reshape = ops::reshape(inp0, new_shape);
        return reshape;
    };

    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto inp0 = n->inputs()[0];
        auto inp1 = n->inputs()[1];
        if (inp0->shape().dim() < inp1->shape().dim())
        {
            auto reshape = do_reshape(inp0, inp1);
            n->add_input(reshape, n->remove_input(inp0));
            ADD_LOCATION_BY_TENSOR(reshape, n);
        }
        else if (inp0->shape().dim() > inp1->shape().dim())
        {
            auto reshape = do_reshape(inp1, inp0);
            n->add_input(reshape, n->remove_input(inp1));
            ADD_LOCATION_BY_TENSOR(reshape, n);
        }
        inp0 = n->inputs()[0];
        inp1 = n->inputs()[1];
        std::vector<uint32_t> repeats_0(inp0->shape().dim(), 1);
        std::vector<uint32_t> repeats_1(inp1->shape().dim(), 1);
        std::vector<tongue::Expr> ds_repeats_0 = {};
        std::vector<tongue::Expr> ds_repeats_1 = {};
        input_tile_repeats(n, repeats_0, repeats_1, ds_repeats_0, ds_repeats_1);

        if (std::accumulate(repeats_0.begin(), repeats_0.end(), 1, std::multiplies<uint32_t>()) != 1)
        {
            auto inp = ops::tile(inp0, repeats_0, ds_repeats_0);
            n->replace_input_by(inp0, inp);
            if (n->outputs()[0]->shape() != inp->shape())
                n->outputs()[0]->shape() = inp->shape();
            ADD_LOCATION_BY_TENSOR(inp, n);
            is_changed = true;
        }
        if (std::accumulate(repeats_1.begin(), repeats_1.end(), 1, std::multiplies<uint32_t>()) != 1)
        {
            auto inp = ops::tile(inp1, repeats_1, ds_repeats_1);
            n->replace_input_by(inp1, inp);
            if (n->outputs()[0]->shape() != inp->shape())
                n->outputs()[0]->shape() = inp->shape();
            ADD_LOCATION_BY_TENSOR(inp, n);
            is_changed = true;
        }
    }
    return is_changed;
} // namespace aipubt
}; // namespace aipubt

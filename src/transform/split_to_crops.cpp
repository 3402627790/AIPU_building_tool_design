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
 * @Author: Xiaolong Zhang
 * @Date: 2023-06-2 14:15:49
 * @Last Modified by: Xialong Zhang
 * @Last Modified time: 2023-06-2 14:24:36
 */

#include "core/config.h"
#include "core/core.h"

namespace aipubt
{
static inline bool criteria(NodePtr n)
{
    if (n->type() == OpType_Split)
    {
        std::set<Tensor::Dtype> support_dtype = {
            Tensor::INT8,
            Tensor::UINT8,
            Tensor::INT16,
            Tensor::UINT16,
            Tensor::FP16,
            Tensor::BFP16,
        };
        if (!support_dtype.count(n->inputs()[0]->type()))
        {
            return false;
        }

        // auto get_c_base = [](Tensor::Dtype dtype) -> int {
        //     int c_base = 16; // int16/uint16/fp16
        //     if (dtype == Tensor::INT8 || dtype == Tensor::UINT8)
        //     {
        //         c_base = 32;
        //     }
        //     return c_base;
        // };

        auto check_split_out = [n, support_dtype /*, get_c_base*/]() -> bool
        {
            bool ret = true;
            for (auto &o : n->outputs())
            {
                if (ret && (!support_dtype.count(o->type())))
                {
                    ret &= false;
                }
                if (ret && (o->type() != n->inputs()[0]->type()))
                {
                    ret &= false;
                }
                // if (ret && (o->shape().c_total() % get_c_base(o->type()) != 0))
                // {
                //     ret &= false;
                // }
            }
            return ret;
        };
        if (n->outputs().size() > 1 && check_split_out())
        {
            return true;
        }
    }
    return false;
}

static bool split2crops(GraphPtr g, NodePtr n)
{
    bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    std::vector<int> splits = n->params()["splits"];
    std::vector<int> begin, end;

    for (uint32_t i = 0; i < splits.size(); ++i)
    {
        if (i == 0)
        {
            begin.push_back(0);
            end.push_back(splits[i]);
        }
        else
        {
            begin.push_back(end[i - 1]);
            end.push_back(splits[i] + end[i - 1]);
        }
    }

    std::vector<tongue::Expr> ds_begin, ds_end;
    if (n->params().count("ds_splits"))
    {
        std::vector<tongue::Expr> ds_splits = n->params()["ds_splits"];
        for (uint32_t i = 0; i < ds_splits.size(); ++i)
        {
            if (i == 0)
            {
                ds_begin.push_back(0);
                ds_end.push_back(ds_splits[i]);
            }
            else
            {
                ds_begin.push_back(ds_end[i - 1]);
                ds_end.push_back(ds_splits[i] + ds_end[i - 1]);
            }
        }
    }

    int axis = n->params()["axis"];
    for (uint32_t i = 0; i < begin.size(); ++i)
    {
        auto crop_name = g->get_valid_node_name(n->name() + "_crop_");
        auto crop_node = Node::create(crop_name, OpType_Crop);

        // fill params
        auto in_shape = n->inputs()[0]->shape();
        std::vector<std::vector<int>> crops;
        std::vector<std::vector<Expr>> ds_crops;
        for (int d = 0; d < in_shape.dim(); ++d)
        {
            crops.push_back({0, in_shape[d]});
            if (enable_ds || n->is_dynamic())
            {
                ds_crops.push_back({0, in_shape.symbol()[d]});
            }
        }

        crops[axis] = {begin[i], end[i]};
        crop_node->params()["crops"] = crops;
        if (enable_ds || n->is_dynamic())
        {
            if (ds_begin.size() > i)
            {
                ds_crops[axis] = {ds_begin[i], ds_end[i]};
            }
            else
            {
                ds_crops[axis] = {begin[i], end[i]};
            }
            crop_node->params()["ds_crops"] = ds_crops;
        }

        // add to graph
        crop_node->add_input(n->inputs()[0]);
        crop_node->add_output(n->outputs()[i]);
        crop_node->infer_shape(true);
        g->add_node(crop_node);
        ADD_LOCATION_BY_NODE(crop_node, n);
    }
    g->remove_node(n);
    return true;
}

bool split_to_crops(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }

        is_changed |= split2crops(g, n);
    }
    return is_changed;
}
}; // namespace aipubt
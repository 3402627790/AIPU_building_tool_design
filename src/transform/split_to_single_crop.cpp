//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-05-26 17:25:56
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-04-28 14:21:07
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>
#include <vector>

namespace aipubt
{
static bool criteria(const std::vector<NodePtr> &nodes)
{
    auto reshape = nodes[0];
    auto add = nodes[1];
    if (reshape->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    auto add_in0_shape = add->inputs()[0]->shape();
    auto add_in1_shape = add->inputs()[1]->shape();
    auto add_out_shape = add->outputs()[0]->shape();
    auto reshape_in_shape = reshape->inputs()[0]->shape();
    auto reshape_out_shape = reshape->outputs()[0]->shape();
    if (add_out_shape != reshape_out_shape)
    {
        return false;
    }
    if (add_in0_shape == add_out_shape)
    {
        // broad_cast add_in1
        // support [C]
        // support [1], not [1,1,1,1]
    return false;
}

static bool split2crop(NodePtr n, uint32_t valid_out_port)
{
    auto g = n->graph();
    std::vector<int> split_params;
    for (auto s : n->params()["splits"].arr)
    {
        split_params.push_back(s);
    }

    int begin = std::accumulate(split_params.begin(), split_params.begin() + valid_out_port, 0);
    int end = begin + split_params[valid_out_port];
    // fill params
    auto inp_shape = n->inputs()[0]->shape();
    int axis = n->params()["axis"];
    aipu_assert(axis <= (int)inp_shape.size());
    std::vector<std::pair<int32_t, int32_t>> crop_params;
    for (auto s : inp_shape)
    {
        crop_params.push_back({0, s});
    }
    crop_params[axis] = {begin, end};

    // dynamic crops
    std::vector<std::pair<tongue::Expr, tongue::Expr>> ds_crops = {};
    if (n->is_dynamic() && n->params().count("ds_splits"))
    {
        for (auto s : inp_shape.symbol())
        {
            ds_crops.push_back({0, s});
        }
        std::vector<tongue::Expr> ds_splits = n->params()["ds_splits"];
        tongue::Expr expr_begin = 0;
        for (uint32_t i = 0; i < valid_out_port; ++i)
        {
            expr_begin += ds_splits[i];
        }
        tongue::Expr expr_end = expr_begin + ds_splits[valid_out_port];
        ds_crops[axis] = {expr_begin, expr_end};
    }
    auto crop_out = ops::crop(n->inputs()[0], crop_params, ds_crops);

    // add to graph
    auto crop_node = crop_out->node();
    n->outputs()[valid_out_port]->shape() = crop_out->shape();
    crop_node->remove_output(0);
    crop_node->add_output(n->outputs()[valid_out_port]);
    ADD_LOCATION_BY_NODE(crop_node, n);
    g->remove_node(n, true);
    return true;
}

bool split_to_single_crop(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        uint32_t valid_out_port = -1;
        if (!criteria(n, valid_out_port))
        {
            continue;
        }

        is_changed |= split2crop(n, valid_out_port);
    }
    return is_changed;
}
}; // namespace aipubt
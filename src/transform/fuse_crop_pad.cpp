//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Tom Gao
 * @Date: 2022-03-22 10:25:43
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-10 21:52:49
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr c, NodePtr p)
{
    if (c->loc() != nullptr && c->loc()->is_original()) // the crop must be not original
    {
        return false;
    }
    if (p->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    std::vector<std::vector<int>> pads = p->params()["pads"];
    std::vector<std::vector<int>> crops = c->params()["crops"];
    if (pads.size() != crops.size() || c->inputs()[0]->shape() != p->outputs()[0]->shape())
    {
        return false;
    }
    bool ret = true;
    for (size_t i = 0; i < pads.size(); ++i) // we already check the total shape
    {
        ret &= (pads[i][0] == crops[i][0]);
    }

    // Symbol shape and static 'pads' check can cover scenarios with dynamic shapes.
    if (c->inputs()[0]->shape().symbol() != p->outputs()[0]->shape().symbol())
    {
        return false;
    }

    return ret;
}

bool fuse_crop_pad(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Crop}, {"1", OpType_Pad}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!creteria(m[0], m[1]))
            {
                continue;
            }
            auto p = m[1];
            auto c = m[0];
            auto inp = c->inputs()[0];
            auto out = p->outputs()[0];
            for (auto child : p->children())
            {
                child->add_input(inp, child->remove_input(out));
                MODIFY_NODE(child);
            }
            g->remove_node(p);
            if (c->children().size() == 0 && !c->outputs()[0]->is_io_tensor())
            {
                g->remove_node(c);
            }
            is_changed = true;
            is_merged = true;
        }
        if (!is_merged)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}

static inline bool creteria_to_crop(NodePtr c, NodePtr p)
{
    if (c->loc() != nullptr && c->loc()->is_original()) // the crop must be not original
    {
        return false;
    }
    if (p->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    std::vector<std::vector<int>> pads = p->params()["pads"];
    std::vector<std::vector<int>> crops = c->params()["crops"];
    if (pads.size() != crops.size() || !(c->inputs()[0]->shape() > p->outputs()[0]->shape()))
    {
        return false;
    }
    bool ret = true;
    for (size_t i = 0; i < pads.size(); ++i)
    {
        ret &= (pads[i][0] <= crops[i][0]) && (pads[i][1] <= c->inputs()[0]->shape()[i] - crops[i][1]);
    }
    // The expression cannot perform size comparison, so it is excluded.
    auto ds_params_check = [](NodePtr n, std::string name)
    {
        if (n->params().count(name))
        {
            std::vector<std::vector<tongue::Expr>> ds_params = n->params()[name];
            for (auto exprs : ds_params)
            {
                if (std::any_of(exprs.begin(), exprs.end(), [](tongue::Expr e)
                                { return !e->is_constant(); }))
                {
                    return false;
                }
            }
        }
        return true;
    };
    if (!ds_params_check(p, "ds_pads") || !ds_params_check(c, "ds_crops"))
    {
        return false;
    }
    return ret;
}

bool fuse_crop_pad_to_crop(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Crop}, {"1", OpType_Pad}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!creteria_to_crop(m[0], m[1]))
            {
                continue;
            }
            auto p = m[1];
            auto c = m[0];
            std::vector<std::vector<int>> pads = p->params()["pads"];
            std::vector<std::vector<int>> crops = c->params()["crops"];
            std::vector<std::pair<int, int>> new_crops;
            for (size_t i = 0; i < pads.size(); ++i)
            {
                new_crops.push_back({crops[i][0] - pads[i][0],
                                     crops[i][1] + pads[i][1]});
            }
            auto inp = c->inputs()[0];
            auto out = p->outputs()[0];
            auto crop_out = ops::crop(inp, new_crops);
            auto crop_node = crop_out->node();
            ADD_LOCATION_BY_NODE(crop_node, c);
            crop_node->remove_output(0);
            crop_node->add_output(out);
            g->remove_node(p);
            if (c->children().size() == 0 && !c->outputs()[0]->is_io_tensor())
            {
                g->remove_node(c);
            }
            is_changed = true;
            is_merged = true;
        }
        if (!is_merged)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}; // namespace aipubt

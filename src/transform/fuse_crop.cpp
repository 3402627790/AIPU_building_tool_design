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
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-03-25 08:57:25
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr n1, NodePtr n2)
{
    std::vector<std::vector<int>> crops_0 = n1->params()["crops"];
    std::vector<std::vector<int>> crops_1 = n2->params()["crops"];
    if (crops_0.size() != crops_1.size() || n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    return true;
}

static bool is_symbol_ds_crops(NodePtr n)
{
    if (n->params().count("ds_crops") == 0)
        return false;
    std::vector<std::vector<tongue::Expr>> ds_crops = n->params()["ds_crops"];
    for (auto ds_crop : ds_crops)
    {
        for (auto expr : ds_crop)
        {
            if (!expr->is_constant())
            {
                return true;
            }
        }
    }
    return false;
}

bool fuse_crop(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Crop}, {"1", OpType_Crop}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
    std::set<NodePtr> removed;
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!creteria(m[0], m[1]) || removed.count(m[1]) || removed.count(m[0]))
            {
                continue;
            }
            std::vector<std::vector<int>> crops_0 = m[0]->params()["crops"];
            std::vector<std::vector<int>> crops_1 = m[1]->params()["crops"];
            std::vector<std::pair<int32_t, int32_t>> crops;
            for (uint32_t i = 0; i < crops_0.size(); ++i)
            {
                crops.push_back({crops_0[i][0] + crops_1[i][0],
                                 crops_0[i][0] + crops_1[i][1]});
            }

            std::vector<std::pair<tongue::Expr, tongue::Expr>> ds_crops;
            if (m[0]->is_dynamic() || m[1]->is_dynamic())
            {
                std::vector<std::vector<tongue::Expr>> ds_crops_0;
                std::vector<std::vector<tongue::Expr>> ds_crops_1;
                if (is_symbol_ds_crops(m[0]))
                {
                    ds_crops_0 = m[0]->params()["ds_crops"];
                }
                if (is_symbol_ds_crops(m[1]))
                {
                    ds_crops_1 = m[1]->params()["ds_crops"];
                }
                for (uint32_t i = 0; i < crops_0.size(); ++i)
                {
                    tongue::Expr ds_crops_start;
                    tongue::Expr ds_crops_begin;
                    tongue::Expr ds_crops_end;
                    if (ds_crops_0.size() != 0)
                    {
                        ds_crops_start = ds_crops_0[i][0];
                    }
                    else
                    {
                        ds_crops_start = tongue::Expr(crops_0[i][0]);
                    }
                    if (ds_crops_1.size() != 0)
                    {
                        ds_crops_begin = ds_crops_start + ds_crops_1[i][0];
                        ds_crops_end = ds_crops_start + ds_crops_1[i][1];
                    }
                    else
                    {
                        ds_crops_begin = ds_crops_start + tongue::Expr(crops_1[i][0]);
                        ds_crops_end = ds_crops_start + tongue::Expr(crops_1[i][1]);
                    }
                    ds_crops.push_back({ds_crops_begin, ds_crops_end});
                }
            }

            auto inp = m[0]->inputs()[0];
            auto out = m[1]->remove_output(0);
            auto crop_out = ops::crop(inp, crops, ds_crops);
            auto crop_node = crop_out->node();
            crop_node->remove_output(0);
            crop_node->add_output(out);
            ADD_LOCATION_BY_NODE(crop_node, m[0], m[1]);

            g->remove_node(m[1]);
            removed.insert(m[1]);
            if (m[0]->children().size() == 0)
            {
                g->remove_node(m[0]);
                removed.insert(m[0]);
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

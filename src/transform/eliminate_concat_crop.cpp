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
 * @Author: Neo Wang 
 * @Date: 2020-05-26 17:25:56 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-24 09:12:04
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr n, NodePtrList &crops)
{
    if (n->type() == OpType_Concat || n->type() == OpType_InternalConcatenated)
    {
        //1. are all children  crop
        if (!n->outputs()[0]->is_io_tensor() && n->inputs().size() == n->outputs()[0]->consumers().size() &&
            all_of(n->children(), [](NodePtr p) {
                return p->type() == OpType_InternalCrop || p->type() == OpType_Crop;
            }))
        {
            uint32_t axis = n->params()["axis"];
            auto inp = n->outputs()[0];
            std::vector<std::pair<std::vector<int>, NodePtr>> interval;
            for (auto c : n->children())
            {
                // crops=[[0,1],[0,2],[0,7],[0,2048]]
                std::vector<std::vector<int>> crop = c->params()["crops"];
                for (uint32_t i = 0; i < crop.size(); ++i)
                {
                    if (i == axis)
                    {
                        interval.push_back({crop[i], c});
                    }
                    else if (crop[i][0] != 0 || crop[i][1] != inp->shape()[i])
                    {
                        return false;
                    }
                }
            }
            std::sort(interval.begin(), interval.end(), [](std::pair<std::vector<int>, NodePtr> &a, std::pair<std::vector<int>, NodePtr> &b) {
                return a.first[0] < b.first[0];
            });
            if (interval[0].first[0] != 0)
            {
                return false;
            }
            crops.push_back(interval[0].second);
            for (uint32_t i = 1; i < interval.size(); ++i)
            {
                if (interval[i - 1].first[1] != interval[i].first[0])
                {
                    crops.clear();
                    return false;
                }
                crops.push_back(interval[i].second);
            }
            return true;
        }
    }
    return false;
}

bool eliminate_concat_crop(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        NodePtrList crops;

        if (!creteria(n, crops))
        {
            continue;
        }
        for (uint32_t i = 0; i < n->inputs().size(); ++i)
        {
            auto r = ops::reshape(n->inputs()[i], n->inputs()[i]->shape())->node();
            ADD_LOCATION_BY_NODE(r, n, crops[i]);
            r->remove_output(0);
            r->add_output(crops[i]->outputs()[0]);
        }
        g->remove_node(n);
        for (auto c : crops)
        {
            g->remove_node(c);
        }
        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt

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
 * @Last Modified time: 2021-11-04 11:46:44
 */
#include "core/core.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr n)
{
    if (n->type() != OpType_Slice)
    {
        return false;
    }

    if (!n->params().count("strides"))
    {
        return false;
    }

    std::vector<int> strides = n->params()["strides"];
    if (std::any_of(strides.begin(), strides.end(), [](int i)
                    { return i != 1; }))
    {
        return false;
    }
    if (n->params().count("ds_strides"))
    {
        std::vector<tongue::Expr> ds_strides = n->params()["ds_strides"];
        for (auto s : ds_strides)
        {
            if (!s->is_constant())
                return false;
        }
    }
    return true;
}

bool slice_to_crop(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }
        auto crop = Node::create(n->name(), OpType_Crop);

        auto layout = n->inputs()[0]->shape().layout();
        std::vector<int> shape = n->inputs()[0]->shape();
        std::vector<int> begin = n->params()["begin"];
        std::vector<int> end = n->params()["end"];
        if (begin.size() == 4 && shape.size() > begin.size())
        {
            if (layout == DataLayout_NCHWC4 || layout == DataLayout_NCHWC8 ||
                layout == DataLayout_NCHWC16 || layout == DataLayout_NCHWC32)
            {
                shape = {shape[0], shape[2], shape[3], shape[1] * shape[4]};
            }
        }
        for (uint32_t i = 0; i < begin.size(); ++i)
        {
            if (begin[i] < 0)
            {
                begin[i] += shape[i];
            }
            if (end[i] < 0)
            {
                end[i] += shape[i];
            }
            // clamp to [0, dim_size]
            if (begin[i] < 0)
            {
                begin[i] = 0;
            }
            else if (begin[i] > shape[i])
            {
                begin[i] = shape[i];
            }
            if (end[i] < 0)
            {
                end[i] = 0;
            }
            else if (end[i] > shape[i])
            {
                end[i] = shape[i];
            }
            if (begin[i] > end[i])
            {
                int begin_idx = begin[i];
                begin[i] = end[i];
                end[i] = begin_idx;
            }
            crop->params()["crops"].push_back({begin[i], end[i]});
        }
        if (n->params().count("ds_begin"))
        {
            std::vector<tongue::Expr> ds_begin = n->params()["ds_begin"];
            std::vector<tongue::Expr> ds_end = n->params()["ds_end"];
            NodeParamValue params_ds_crops;
            for (uint32_t i = 0; i < ds_begin.size(); ++i)
            {
                params_ds_crops.push_back({ds_begin[i], ds_end[i]});
            }
            crop->params()["ds_crops"] = params_ds_crops;
        }

        crop->constants() = std::move(n->constants());
        ADD_LOCATION_BY_NODE(crop, n);
        g->replace_node_by(n, crop);
        crop->infer_shape(true);
        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt

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
 * @Author: Chenyang Zhu
 * @Date: 2022-12-01 15:59:50
 * @Last Modified by: Chenyang Zhu
 * @Last Modified time: 2022-12-01 15:59:50
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr n)
{
    if (n->type() == OpType_ChannelShuffle)
    {
        int param_splits = (int)n->params()["splits"];
        int param_group = (int)n->params()["group"];
        if (param_splits == 1 && param_group == 1)
        {
            return true;
        }
    }
    return false;
}

bool eliminate_channel_shuffle(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }

        auto shape = n->outputs()[0]->shape();
        auto reshape_out = ops::reshape(n->inputs()[0], shape);
        auto reshape_node = reshape_out->node();

        ADD_LOCATION_BY_NODE(reshape_node, n);
        g->replace_node_by(n, reshape_node);
    }
    return is_changed;
}
}; // namespace aipubt

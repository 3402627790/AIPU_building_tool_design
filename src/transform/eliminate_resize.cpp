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
 * @Author: Acheng Pan
 * @Date: 2022-10-24 16:45:40
 * @Last Modified by: Acheng Pan
 * @Last Modified time: 2022-10-24 17:06:54
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
    if (n->type() == OpType_Resize)
    {
        bool ret = n->inputs()[0]->shape() == n->outputs()[0]->shape();

        std::string mode = n->params()["mode"];
        transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
        // the interval = (i + 0.5) * scale which is not 1 when scale == 1
        ret = ret && (mode != "tf_half_pixel_for_nn");
        return ret;
    }
    return false;
}

bool eliminate_resize(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }
        auto reshape_node = ops::reshape(n->inputs()[0], n->outputs()[0]->shape())->node();
        ADD_LOCATION_BY_NODE(reshape_node, n);
        g->replace_node_by(n, reshape_node);
        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt

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
 * @Date: 2022-06-30 14:15:49
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-06-30 14:24:36
 */

#include "core/core.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool criteria(NodePtr n)
{
    if (n->type() == OpType_Split)
    {
        if (n->outputs().size() == 1 /*&& n->inputs()[0]->shape() == n->outputs()[0]->shape()*/)
        {
            return true;
        }
    }
    return false;
}

bool eliminate_split(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }

        is_changed |= g->eliminate_node(n);
    }
    return is_changed;
}
}; // namespace aipubt
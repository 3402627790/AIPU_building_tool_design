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
 * @Author: Lang Hu
 * @Date: 2024-04-03 10:10:56
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-04-03 10:10:56
 */
#include "core/core.h"
#include "passes.h"

namespace aipubt
{

static inline bool creteria(NodePtr n)
{
    if (n->type() != OpType_Pad)
    {
        return false;
    }

    if (n->is_dynamic())
    {
        return false;
    }

    if (n->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    if (n->inputs()[0]->shape() != n->outputs()[0]->shape())
    {
        return false;
    }

    return true;
}

bool eliminate_pad(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }

        is_changed |= g->eliminate_node(n);
    }
    return is_changed;
}

} // namespace aipubt

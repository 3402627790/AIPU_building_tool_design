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
 * @Last Modified time: 2021-11-04 17:14:15
 */
#include "core/core.h"
#include "passes.h"

namespace aipubt
{

static inline bool creteria(NodePtr n)
{
    if (n->type() != OpType_Reshape)
    {
        return false;
    }

    auto in_shape = n->inputs()[0]->shape();
    auto out_shape = n->outputs()[0]->shape();

    // check dynamic is same
    if (n->is_dynamic() && (in_shape.symbol() != out_shape.symbol()))
    {
        return false;
    }
    return in_shape == out_shape;
}

bool eliminate_reshape(GraphPtr g)
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

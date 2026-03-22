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
 * @Author: Kyle Huang
 * @Date: 2020-06-28 11:32:56
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2020-06-28 11:32:56
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{

static inline bool creteria(NodePtr n)
{
    if (n->type() == OpType_Pooling)
    {
        if (!n->params().count("kernel_y") || !n->params().count("kernel_x") ||
            !n->params().count("stride_y") || !n->params().count("stride_x"))
        {
            return false;
        }
        int kx = (int)n->params()["kernel_x"];
        int ky = (int)n->params()["kernel_y"];
        int sx = (int)n->params()["stride_x"];
        int sy = (int)n->params()["stride_y"];
        int pl = (int)n->params()["pad_left"];
        int pr = (int)n->params()["pad_right"];
        int pt = (int)n->params()["pad_top"];
        int pb = (int)n->params()["pad_bottom"];
        if (kx == 1 && ky == 1 && sx == 1 && sy == 1 && pl == 0 && pr == 0 && pt == 0 && pb == 0)
        {
            return true;
        }
    }
    return false;
}

void pooling_to_reshape(GraphPtr g)
{
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
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
}
}; // namespace aipubt

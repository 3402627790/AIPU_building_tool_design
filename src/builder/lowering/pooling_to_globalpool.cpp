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
 * @Date: 2023-11-28 15:09:33
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2023-11-28 15:09:33
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"

namespace aipubt
{

static bool creteria(NodePtr n)
{
    if (n->type() == OpType_Pooling)
    {
        if (!n->params().count("kernel_y") || !n->params().count("kernel_x") ||
            !n->params().count("stride_y") || !n->params().count("stride_x"))
        {
            return false;
        }

        auto out_shape = n->outputs()[0]->shape();

        if (out_shape.h() == 1 && out_shape.w() == 1)
        {
            return true;
        }
    }
    return false;
}

void pooling_to_globalpool(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }

        auto pool_node = Node::create(n->name(), OpType_GlobalPool);
        pool_node->params()["method"] = n->params()["method"];
        pool_node->params()["keep_dims"] = true;

        ADD_LOCATION_BY_NODE(pool_node, n);
        g->replace_node_by(n, pool_node);
    }
}
}; // namespace aipubt

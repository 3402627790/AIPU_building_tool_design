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
 * @Date: 2026-01-19 10:18:37
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-01-19 10:18:37
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
static inline bool criteria(NodePtr n)
{
    if (get_real_op_type(n) != OpType_Switch)
    {
        return false;
    }

    auto inputs = n->inputs();
    for (auto inp : inputs)
    {
        if (inp->is_io_tensor() || inp->consumers().size() != 1)
        {
            return false;
        }

        if (inp->node()->type() != OpType_If) // only for moe
        {
            return false;
        }
    }
    return true;
}

void remove_switch(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }

        auto real_node = get_real_node(n);
        auto internal_node = Node::create(real_node->name(), OpType_InternalSwitch);
        internal_node->attrs() = std::move(real_node->attrs());
        ADD_LOCATION_BY_NODE(internal_node, n);
        g->replace_node_by(n, internal_node);
    }
}
} // namespace aipubt

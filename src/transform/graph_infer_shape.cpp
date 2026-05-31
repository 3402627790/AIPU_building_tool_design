//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-03-30 13:55:50
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-30 16:35:20
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
bool graph_infer_shape(GraphPtr g)
{
    g->topological_sort();
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            continue;
        }
        if (n->type() == OpType_Input || n->type() == OpType_SubGraphInput ||
            n->type() == OpType_Constant || n->type() == OpType_InternalConstant)
        {
            continue;
        }
        auto shapes = n->infer_shape();
        if (shapes.size() != n->outputs().size() ||
            shapes.size() == 0)
        {
            // DEBUG() << shapes.size();
            // DEBUG() << n->outputs().size();
            WARN() << "Failed to infer shape for node: " << n->name() << "(" << n->type() << ")";
            continue;
        }
        for (uint32_t i = 0; i < n->outputs().size(); ++i)
        {
            n->outputs()[i]->shape() =shapes[i];
        }
    }
    return true;
}
} // namespace aipubt
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
 * @Author: Chao Liu
 * @Date: 2023-08-1 16:25:56
 * @Last Modified time: 2023-08-1 16:25:56
 */

#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>
#include <vector>

namespace aipubt
{
static bool lower_creteria(const NodePtr &node)
{
    if (node->type() != OpType_Gather)
    {
        return false;
    }
    if (node->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    // need more critical lower_creteria, if original graph is only axis=3 gather, than will insert more two transpose
    std::vector<int> axis = node->params()["axis"];
    if ((axis.size() != 1) || (axis[0] != 3))
    {
        return false;
    }
    return true;
}

bool lower_gather(GraphPtr g)
{
    auto nodes = g->nodes();
    bool is_changed = false;
    for (auto &&n : nodes)
    {
        if (!lower_creteria(n))
            continue;
        auto gather = n;
        if (gather->parents().size() != 2)
            continue;
        uint32_t const_idx = 65536;
        for (uint32_t i = 0; i < gather->parents().size(); ++i)
        {
            if (gather->parents()[i]->type() == OpType_Constant)
                const_idx = i;
        }
        if (const_idx >= gather->parents().size())
            continue;
        auto idx_node = gather->parents()[const_idx];
        auto input_node = gather->parents()[1 - const_idx];

        auto transpose_node0 = ops::transpose(input_node->outputs()[0], {0, 1, 3, 2})->node();
        g->add_node(transpose_node0);
        ADD_LOCATION_BY_NODE(transpose_node0, input_node);

        uint32_t axis = 2;
        auto new_gather = ops::gather(transpose_node0->outputs()[0], idx_node->outputs()[0], axis)->node();
        g->add_node(new_gather);
        ADD_LOCATION_BY_NODE(new_gather, input_node);

        auto transpose_node1 = ops::transpose(new_gather->outputs()[0], {0, 1, 3, 2})->node();
        g->add_node(transpose_node1);
        ADD_LOCATION_BY_NODE(transpose_node1, input_node);
        for (auto child : gather->children())
        {
            child->replace_input_by(gather->outputs()[0], transpose_node1->outputs()[0]);
        }
        g->remove_node(gather);
        is_changed = true;
    }
    return is_changed;
}
} // namespace aipubt

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
 * @Date: 2020-06-08 10:37:00
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-06-08 14:14:41
 */
#include "builder/lowering/lowering.h"
#include <algorithm>
#include <core/core.h>
#include <set>
#include <vector>
namespace aipubt
{

// this pass will change internal crop to crop for some memory alignment issue
void patch_internal_split(GraphPtr g)
{
    auto criterion = [](NodePtr node) -> bool {
        if (node->inputs().empty())
        {
            return false;
        }
        if (OpType_InternalSpliter == node->type())
        {
            for (auto out : node->outputs())
            {
                if (out->mem_size() % 32 != 0)
                {
                    return true;
                }
            }
        }
        return false;
    };

    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }

        NodePtr reshape_node = Node::create(n->name(), OpType_Split);

        // reshape_node->inputs() = n->inputs();
        // reshape_node->outputs() = n->outputs();
        ADD_LOCATION_BY_NODE(reshape_node, n);
        g->replace_node_by(n, reshape_node);
        reshape_node->params() = n->params();
        reshape_node->constants() = n->constants();
        for (auto out : reshape_node->outputs())
        {
            out->memory_offset()->base_offset = nullptr;
            out->memory_offset()->relative_offset = 0;
            out->memory_offset()->align = 128;
        }
    }
}
}; // namespace aipubt
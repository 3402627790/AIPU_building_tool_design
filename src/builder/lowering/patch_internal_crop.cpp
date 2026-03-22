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
void patch_internal_crop(GraphPtr g)
{
    auto criterion = [](NodePtr node) -> bool {
        if (node->inputs().empty())
        {
            return false;
        }
        if (OpType_InternalCrop == node->type())
        {
            int32_t offset = node->params().at("start_offset");
            return offset % 16 != 0;
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

        NodePtr reshape_node = Node::create(n->name(), OpType_Crop);

        reshape_node->params() = std::move(n->params());
        reshape_node->constants() = std::move(n->constants());
        reshape_node->params().erase("start_offset");

        ADD_LOCATION_BY_NODE(reshape_node, n);
        g->replace_node_by(n, reshape_node);
    }
}
}; // namespace aipubt
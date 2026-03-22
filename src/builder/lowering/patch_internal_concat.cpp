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
 * @Date: 2022-05-26 18:00:00
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 022-05-26 18:00:00
 */
#include "builder/lowering/lowering.h"
#include <algorithm>
#include <core/core.h>
#include <set>
#include <vector>
namespace aipubt
{

// this pass will change internal crop to concat
void patch_internal_concat(GraphPtr g)
{
    auto criterion = [](NodePtr node) -> bool {
        if (node->inputs().empty())
        {
            return false;
        }
        if (OpType_InternalConcatenated != node->type())
        {
            return false;
        }
        auto input_tensors = node->inputs();
        std::set<TensorPtr> s(input_tensors.begin(), input_tensors.end());
        bool hasDuplicates = s.size() != input_tensors.size();
        return hasDuplicates;
    };

    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }
        NodePtr concat = Node::create(n->name(), OpType_Concat);

        concat->params() = std::move(n->params());
        concat->constants() = std::move(n->constants());

        ADD_LOCATION_BY_NODE(concat, n);
        g->replace_node_by(n, concat);
    }
}
}; // namespace aipubt
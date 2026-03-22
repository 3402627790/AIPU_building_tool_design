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
 * @Author: Yuchou.Gan
 * @Date: 2020-02-25 17:58:05
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-21 16:24:47
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"

namespace aipubt
{
void add_address_to_dot(GraphPtr g)
{
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        // make the gbuilder add a address in dot png
        n->set_attr(Constants::dot_address, true);
        auto sgn = SubGraphNode::from_node(n);
        if (sgn && !sgn->has_plugin())
        {
            add_address_to_dot(sgn->subgraph());
        }
        if (n->children().size() == 1)
        {
            auto child = n->children()[0];
            if (child->type() == OpType_InternalConcatenated)
            {
                n->set_attr("with_IC_child", true);
            }
        }
    }
}
}; // namespace aipubt

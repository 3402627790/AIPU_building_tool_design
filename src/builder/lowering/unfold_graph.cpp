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
 * @Date: 2024-04-11 18:11:13
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-04-11 18:11:13
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"

namespace aipubt
{

void unfold_graph(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (n->attrs().count(Constants::skip_unfold) &&
            n->get_attr<bool>(Constants::skip_unfold))
        {
            continue;
        }

        if (SubGraphNode::is_subgraph_node(n) &&
            !n->attrs().count(Constants::plugin_attrs_name))
        {
            SubGraphNode::from_node(n)->unfold();
        }
    }
}

} // namespace aipubt

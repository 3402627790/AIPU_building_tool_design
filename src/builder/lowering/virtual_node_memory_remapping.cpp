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
 * @Date: 2020-12-31 10:27:40
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-03-27 11:12:53
 */
#include "builder/lowering/lowering.h"
#include "builder/mem_remapping.h"
#include "builder/plugin.h"
#include "core/core.h"
#include "core/node.h"
#include "core/plugin_loader.h"

namespace aipubt
{
void virtual_node_memory_remapping(GraphPtr g)
{
    g->topological_sort(); //remapping must ensure the running order
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        // clean memory mapping
        if (n->type() != OpType_SubGraphInput)
        {
            for (auto out : n->outputs())
            {
                out->memory_offset()->set_base_offset(nullptr);
            }
        }
    }
    for (auto &&n : m_nodes)
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            if (!n->attrs().count(Constants::plugin_attrs_name))
            {
                auto sgn = SubGraphNode::from_node(n);
                virtual_node_memory_remapping(sgn->subgraph());
                continue;
            }
            else
            {
                auto plugin_info = n->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
                if (plugin_info->plugin_type & GBPlugin_Internal)
                {
                    auto plugin = plugin_info->init_handle();
                    auto sgnode = std::dynamic_pointer_cast<SubGraphNode>(n);
                    auto ml = n->get_attr<NodePtrList>("_match_list");
                    BaseNodePtrList bml;
                    for (auto p : ml)
                    {
                        bml.push_back(p);
                    }
                    plugin->memory_remapping(sgnode, bml);
                    continue;
                }
            }
        }
        memory_remapping_op(n);
    }
}
} // namespace aipubt

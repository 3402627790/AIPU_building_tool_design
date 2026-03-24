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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-01-29 14:23:29
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-01-05 16:47:56
 */

#include "mem_remapping.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "utils/utils.h"
#include <mutex>
#include <string>

namespace aipubt
{
MemoryRemappingOpRegister::MemoryRemappingOpRegister(OpType t, MemoryRemappingFunc func)
{
    MemoryRemappingOpRegister::get_memory_remapping_map()[t] = func;
}

static OpType subgraph_op_type(NodePtr n)
{
    if (!SubGraphNode::is_subgraph_node(n))
    {
        return n->type();
    }
    SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(n);
    if (sgnode->subgraph()->nodes().size() != 1)
    {
        return n->type();
    }
    return sgnode->subgraph()->nodes()[0]->type();
}

void memory_remapping_op(NodePtr n)
{
    auto &map = MemoryRemappingOpRegister::get_memory_remapping_map();
    auto op_type = subgraph_op_type(n);
    if (map.count(op_type) && map[op_type])
    {
        map[op_type](n);
    }
}

} // namespace aipubt

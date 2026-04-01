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
 * @Date: 2021-03-27 19:58:14 
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-23 14:25:44
 */
#ifndef CORE_LOOP_NODE_H
#define CORE_LOOP_NODE_H
#include "core/graph.h"
#include "core/node.h"
#include "core/subgraph.h"
#include "core/tensor.h"

namespace aipubt
{
class LoopNode;
using LoopNodePtr = std::shared_ptr<LoopNode>;
using LoopNodeRef = std::weak_ptr<LoopNode>;
class LoopNode : public SubGraphNode
{
public:
    LoopNode(GraphPtr g, NodePtr loopnode, GraphPtr body);
    LoopNode(SubGraphPtr sg, const std::string &name);

    static LoopNodePtr create(GraphPtr g, NodePtr loopnode, GraphPtr body);
    static LoopNodePtr create(SubGraphPtr sg, const std::string &name);

    NodePtr head() const
    {
        return m_head;
    }
    NodePtr tail() const
    {
        return m_tail;
    }

protected:
    static void to_sgnode(LoopNodePtr node);

    NodePtr m_head;
    NodePtr m_tail;
};
} //namespace aipubt

#endif //!CORE_LOOP_NODE_H
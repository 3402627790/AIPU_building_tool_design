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
 * @Date: 2025-05-09 10:34:33
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-05-09 10:34:33
 */
#ifndef CORE_IF_NODE_H
#define CORE_IF_NODE_H
#include "core/graph.h"
#include "core/node.h"
#include "core/subgraph.h"
#include "core/tensor.h"

namespace aipubt
{
class IfNode;
using IfNodePtr = std::shared_ptr<IfNode>;
using IfNodeRef = std::weak_ptr<IfNode>;
class IfNode : public SubGraphNode
{
public:
    IfNode(GraphPtr g, NodePtr ifnode, GraphPtr then_sg, GraphPtr else_sg);
    IfNode(SubGraphPtr sg, const std::string &name);

    static IfNodePtr create(GraphPtr g, NodePtr ifnode, GraphPtr then_sg, GraphPtr else_sg);
    static IfNodePtr create(SubGraphPtr sg, const std::string &name);

    void get_branch_nodes(std::vector<NodePtr> &then_nodes, std::vector<NodePtr> &else_nodes) const;

    NodePtr head() const
    {
        return m_head;
    }
    NodePtr sep() const
    {
        return m_sep;
    }
    NodePtr tail() const
    {
        return m_tail;
    }

protected:
    static void to_sgnode(IfNodePtr node);

    NodePtr m_head;
    NodePtr m_sep;
    NodePtr m_tail;
};
} //namespace aipubt

#endif //!CORE_IF_NODE_H
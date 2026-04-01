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
 * @Date: 2021-03-27 19:58:22 
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-23 14:25:44
 */
#include "core/loopnode.h"
#include "core/constants.h"

namespace aipubt
{
LoopNode::LoopNode(GraphPtr g, NodePtr loopnode, GraphPtr body) : SubGraphNode(nullptr)
{
    m_name = loopnode->name();
    m_type = loopnode->type();
    m_params = std::move(loopnode->params());
    int depend_input_num = m_params["dependency_inputs_num"];
    int scan_output_num = m_params["scan_outputs_num"];

    std::vector<NodePtr> sg_nodes;
    m_head = Node::create(g->get_valid_node_name(loopnode->name() + "/head"), OpType_ForRangeHeader);
    sg_nodes.push_back(m_head);
    auto loop_inputs = loopnode->inputs();
    for (auto t : loop_inputs)
    {
        m_head->add_input(t);
        loopnode->remove_input(t);
    }
    auto call_back = std::make_shared<Tensor>(g->get_valid_tensor_name(loopnode->name() + "/call_back"),
                                              TensorShape(1), Tensor::INT32);
    auto frame_ptr = std::make_shared<Tensor>(g->get_valid_tensor_name(loopnode->name() + "/frame_ptr"),
                                              TensorShape(1), Tensor::INT32);
    auto scan_len = std::make_shared<Tensor>(g->get_valid_tensor_name(loopnode->name() + "/scan_len"),
                                             TensorShape(scan_output_num), Tensor::INT32);
    auto max_count = std::make_shared<Tensor>(*loop_inputs[0]);
    max_count->set_name(g->get_valid_tensor_name(loopnode->name() + "/max_count"));
    max_count->set_io_tensor(false);
    m_head->add_output(call_back);
    m_head->add_output(frame_ptr);
    m_head->add_output(scan_len);
    m_head->add_output(max_count);
    for (auto t : body->get_input_tensors())
    {
        t->set_name(g->get_valid_tensor_name(loopnode->name() + "/" + t->name()));
        t->set_io_tensor(false);
        m_head->add_output(t);
    }

    auto body_outputs = body->get_output_tensors();
    auto nodes = body->nodes();
    for (auto n : nodes)
    {
        if (n->type() == OpType_Input)
        {
            n->remove_output(n->outputs()[0]);
            body->remove_node(n);
            continue;
        }
        n->set_name(g->get_valid_node_name(loopnode->name() + "/" + n->name()));
        for (auto t : n->outputs())
        {
            t->set_name(g->get_valid_tensor_name(loopnode->name() + "/" + t->name()));
        }
        g->add_node(n);
        body->remove_node(n, false);
        sg_nodes.push_back(n);
    }

    m_tail = Node::create(g->get_valid_node_name(loopnode->name() + "/tail"), OpType_ForRangeEnder);
    sg_nodes.push_back(m_tail);
    m_tail->add_input(call_back);
    m_tail->add_input(frame_ptr);
    m_tail->add_input(scan_len);
    m_tail->add_input(max_count);
    m_tail->add_input(m_head->outputs()[4]); // index
    m_tail->add_input(m_head->outputs()[5]); // cond_in
    for (int i = 0; i < depend_input_num; ++i)
    {
        m_tail->add_input(m_head->outputs()[6 + i]);
    }
    for (auto t : body_outputs)
    {
        t->set_io_tensor(false);
        m_tail->add_input(t);
    }

    auto loop_outputs = loopnode->outputs();
    for (auto t : loop_outputs)
    {
        m_tail->add_output(t);
        loopnode->remove_output(t);
    }

    g->add_node(m_head);
    g->add_node(m_tail);
    g->remove_node(loopnode);
    m_sg = SubGraph::create(g, sg_nodes);
    set_attr(".subgraph_color", "antiquewhite2");
}

LoopNode::LoopNode(SubGraphPtr sg, const std::string &name) : SubGraphNode(sg, name)
{
    m_type = OpType_Loop;
    NodePtr head = nullptr, tail = nullptr;
    for (auto &&n : sg->nodes())
    {
        if (n->type() == OpType_ForRangeHeader)
        {
            head = n;
        }
        if (n->type() == OpType_ForRangeEnder)
        {
            tail = n;
        }
    }
    if (!head || !tail)
    {
        ERROR() << "Missing header/ender for loopnode: " << m_name;
        return;
    }
    m_head = head;
    m_tail = tail;
}

LoopNodePtr LoopNode::create(GraphPtr g, NodePtr loopnode, GraphPtr body)
{
    auto node = std::make_shared<LoopNode>(g, loopnode, body);
    to_sgnode(node);
    return node;
}

LoopNodePtr LoopNode::create(SubGraphPtr sg, const std::string &name)
{
    auto node = std::make_shared<LoopNode>(sg, name);
    to_sgnode(node);
    return node;
}

void LoopNode::to_sgnode(LoopNodePtr node)
{
    std::set<TensorPtr> free_out_wlist;
    for (auto t : node->tail()->outputs())
    {
        free_out_wlist.insert(t);
    }
    node->m_sg->replace = true;
    SubGraph::to_node(node->m_sg, node, true, free_out_wlist);
    for (auto n : node->m_sg->m_nodes)
    {
        ADD_LOCATION_BY_NODE(node, n);
    }
    node->set_unified_view(false);
    node->set_attr(Constants::skip_unfold, true);
}
} //namespace aipubt
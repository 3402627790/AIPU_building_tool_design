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
#include "core/ifnode.h"
#include "core/constants.h"

namespace aipubt
{
IfNode::IfNode(GraphPtr g, NodePtr ifnode, GraphPtr then_sg, GraphPtr else_sg) : SubGraphNode(nullptr)
{
    m_name = ifnode->name();
    m_type = ifnode->type();
    m_params = std::move(ifnode->params());

    std::vector<NodePtr> sg_nodes;
    m_head = Node::create(g->get_valid_node_name(ifnode->name() + "/head"), OpType_IfHeader);
    sg_nodes.push_back(m_head);
    auto if_inputs = ifnode->inputs();
    for (auto t : if_inputs)
    {
        m_head->add_input(t);
        ifnode->remove_input(t);
    }
    auto call_back = std::make_shared<Tensor>(g->get_valid_tensor_name(ifnode->name() + "/call_back"),
                                              TensorShape(1), Tensor::INT32);
    auto cond = std::make_shared<Tensor>(*if_inputs[0]);
    cond->set_name(g->get_valid_tensor_name(ifnode->name() + "/cond"));
    cond->set_io_tensor(false);
    m_head->add_output(call_back);
    m_head->add_output(cond);
    for (auto t : then_sg->get_input_tensors())
    {
        t->set_io_tensor(false);
        m_head->add_output(t);
    }
    auto else_inputs = else_sg->get_input_tensors();
    for (auto t : else_inputs)
    {
        t->set_io_tensor(false);
        m_head->add_output(t);
    }

    auto then_outputs = then_sg->get_output_tensors();
    auto then_nodes = then_sg->nodes();
    for (auto n : then_nodes)
    {
        n->set_name(g->get_valid_node_name(ifnode->name() + "/then/" + n->name()));
        for (auto t : n->outputs())
        {
            t->set_name(g->get_valid_tensor_name(ifnode->name() + "/then/" + t->name()));
        }
        if (n->type() == OpType_Input)
        {
            n->remove_output(n->outputs()[0]);
            then_sg->remove_node(n);
            continue;
        }
        g->add_node(n);
        then_sg->remove_node(n, false);
        sg_nodes.push_back(n);
    }

    auto else_outputs = else_sg->get_output_tensors();
    auto else_nodes = else_sg->nodes();
    for (auto n : else_nodes)
    {
        n->set_name(g->get_valid_node_name(ifnode->name() + "/else/" + n->name()));
        for (auto t : n->outputs())
        {
            t->set_name(g->get_valid_tensor_name(ifnode->name() + "/else/" + t->name()));
        }
        if (n->type() == OpType_Input)
        {
            n->remove_output(n->outputs()[0]);
            else_sg->remove_node(n);
            continue;
        }
        g->add_node(n);
        else_sg->remove_node(n, false);
        sg_nodes.push_back(n);
    }

    m_sep = Node::create(g->get_valid_node_name(ifnode->name() + "/dummy"), OpType_Dummy);
    sg_nodes.push_back(m_sep);
    for (auto t : then_outputs)
    {
        t->set_io_tensor(false);
        m_sep->add_input(t);
    }
    for (auto t : else_inputs)
    {
        m_sep->add_input(t);
    }
    for (auto inp : m_sep->inputs())
    {
        auto out = std::make_shared<Tensor>(*inp);
        out->set_name(g->get_valid_tensor_name(inp->name() + "/dummy"));
        m_sep->add_output(out);
    }
    for (size_t i = then_outputs.size(); i < m_sep->outputs().size(); ++i)
    {
        auto t = else_inputs[i - then_outputs.size()];
        auto consumers = t->consumers();
        for (auto c : consumers)
        {
            if (c != m_sep)
            {
                c->replace_input_by(t, m_sep->outputs()[i]);
            }
        }
    }

    m_tail = Node::create(g->get_valid_node_name(ifnode->name() + "/tail"), OpType_IfEnder);
    sg_nodes.push_back(m_tail);
    m_tail->add_input(call_back);
    m_tail->add_input(cond);
    for (size_t i = 0; i < then_outputs.size(); ++i)
    {
        m_tail->add_input(m_sep->outputs()[i]);
    }
    for (auto t : else_outputs)
    {
        t->set_io_tensor(false);
        m_tail->add_input(t);
    }
    auto if_outputs = ifnode->outputs();
    for (auto t : if_outputs)
    {
        m_tail->add_output(t);
        ifnode->remove_output(t);
    }

    g->add_node(m_head);
    g->add_node(m_sep);
    g->add_node(m_tail);
    g->remove_node(ifnode);
    m_sg = SubGraph::create(g, sg_nodes);
    set_attr(".subgraph_color", "antiquewhite2");
}

IfNode::IfNode(SubGraphPtr sg, const std::string &name) : SubGraphNode(sg, name)
{
    m_type = OpType_If;
    NodePtr head = nullptr, tail = nullptr;
    for (auto &&n : sg->nodes())
    {
        if (n->type() == OpType_IfHeader)
        {
            head = n;
        }
        if (n->type() == OpType_IfEnder)
        {
            tail = n;
        }
    }
    if (!head || !tail)
    {
        ERROR() << "Missing header/ender for ifnode: " << m_name;
        return;
    }
    auto sep = tail->inputs()[2]->node();
    aipu_assert(sep->type() == OpType_Dummy);
    m_head = head;
    m_sep = sep;
    m_tail = tail;
}

IfNodePtr IfNode::create(GraphPtr g, NodePtr ifnode, GraphPtr then_sg, GraphPtr else_sg)
{
    auto node = std::make_shared<IfNode>(g, ifnode, then_sg, else_sg);
    to_sgnode(node);
    return node;
}

IfNodePtr IfNode::create(SubGraphPtr sg, const std::string &name)
{
    auto node = std::make_shared<IfNode>(sg, name);
    to_sgnode(node);
    return node;
}

void IfNode::to_sgnode(IfNodePtr node)
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

// traverse between begin and end, but not including begin and end
static void traverse(NodePtr n, NodePtr begin, NodePtr end, std::set<NodePtr> &res)
{
    if (n == begin || n == end)
    {
        return;
    }
    res.insert(n);
    for (auto c : n->children())
    {
        if (!res.count(c))
        {
            traverse(c, begin, end, res);
        }
    }
    for (auto p : n->parents())
    {
        if (!res.count(p))
        {
            traverse(p, begin, end, res);
        }
    }
}

void IfNode::get_branch_nodes(std::vector<NodePtr> &then_nodes, std::vector<NodePtr> &else_nodes) const
{
    size_t output_num = m_tail->outputs().size();
    std::set<NodePtr> then_set, else_set;
    for (size_t i = 0; i < output_num; ++i)
    {
        auto n = m_sep->inputs()[i]->node();
        traverse(n, m_sep, m_head, then_set);
    }
    for (size_t i = 2 + output_num; i < m_tail->inputs().size(); ++i)
    {
        auto n = m_tail->inputs()[i]->node();
        traverse(n, m_tail, m_sep, else_set);
    }

    auto sg = subgraph();
    size_t input_num = 0;
    for (auto n : sg->nodes())
    {
        if (n->type() == OpType_SubGraphInput)
        {
            input_num++;
        }
    }
    std::set<NodePtr> union_set;
    std::set_union(then_set.begin(), then_set.end(), else_set.begin(), else_set.end(),
                   std::inserter(union_set, union_set.begin()));
    aipu_assert(union_set.size() == then_set.size() + else_set.size());
    aipu_assert(input_num + 3 + union_set.size() == sg->nodes().size());

    then_nodes = std::vector<NodePtr>{then_set.begin(), then_set.end()};
    else_nodes = std::vector<NodePtr>{else_set.begin(), else_set.end()};
}
} //namespace aipubt
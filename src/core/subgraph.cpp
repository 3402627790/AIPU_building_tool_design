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
 * @Date: 2020-01-29 14:28:21
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-11-22 18:10:58
 */
#include "builder/lowering/lowering.h"
#include "builder/profiler.h"
#include "core.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/core.h"
#include "core/ifnode.h"
#include "core/loopnode.h"
#include "core/ops/ops.h"
#include "core/plugin_loader.h"
#include "utils/json.h"
#include "utils/string_utils.h"
#include <fstream>
#include <iostream>
namespace aipubt
{
typedef std::vector<DataLayout> LayoutList;
typedef std::vector<LayoutList> LayoutCombinations;

template <typename T>
static T get_tensor_by_name(const std::vector<T> &ts, const std::string &name)
{
    for (auto nt : ts)
    {
        if (name == nt->name())
        {
            return nt;
        }
    }
    return nullptr;
}

void SubGraph::update_global_tensors()
{
    // copy global tensors
    //  this->m_global_tensors = origin_graph()->global_tensors();
    for (auto kv : origin_graph()->global_tensors())
    {
        // workspace is private buffer, should each subgraph has independ one
        if (kv.first != Constants::workspace)
        {
            this->m_global_tensors[kv.first] = kv.second;
        }
    }
}

SubGraphPtr SubGraph::create(GraphPtr g)
{
    auto p = SubGraphPtr(new SubGraph(g));
    p->update_global_tensors();
    p->update_real_nodes();
    p->update_internal_tensors();
    return p;
}

SubGraphPtr SubGraph::create(GraphPtr g, const std::vector<NodePtr> &nodes)
{
    auto p = SubGraphPtr(new SubGraph(g));
    for (auto n : nodes)
    {
        p->add_node(n);
    }
    p->update_global_tensors();
    p->update_real_nodes();
    p->update_internal_tensors();
    return p;
}
SubGraphPtr SubGraph::create(GraphPtr g, const std::vector<NodePtr> &begin_nodes, const std::vector<NodePtr> &end_nodes)
{
    auto p = SubGraphPtr(new SubGraph(g));
    p->update_global_tensors();
    p->m_nodes = topological_sort(begin_nodes, end_nodes);
    p->reset_layer_id();
    p->update_real_nodes();
    p->update_internal_tensors();
    return p;
}

SubGraph::SubGraph(GraphPtr g)
{
    ori_graph = g;
}

void SubGraph::update_real_nodes()
{
    m_real_nodes.clear();
    for (auto n : m_nodes)
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            auto sgn = std::dynamic_pointer_cast<SubGraphNode>(n);
            if (sgn)
            {
                sgn->m_sg->update_real_nodes();
                for (auto nn : sgn->m_sg->m_real_nodes)
                {
                    if (!contains(m_real_nodes, nn))
                    {
                        m_real_nodes.push_back(nn);
                    }
                }
            }
        }
        else if (n->type() == OpType_SubGraphInput) // sginp is not a real node
        {
            continue;
        }
        else if (!contains(m_real_nodes, n))
        {
            m_real_nodes.push_back(n);
        }
    }
    for (auto n : m_real_nodes)
    {
        n->set_attr("subgraph_internal_node", true);
    }
}

GraphPtr SubGraph::origin_graph()
{
    return ori_graph.lock();
}

SubGraphNodePtr SubGraph::subgraphnode() const
{
    return m_sgn.lock();
}

std::set<TensorPtr> SubGraph::update_internal_tensors()
{
    std::set<TensorPtr> in_tensors;
    std::set<TensorPtr> out_tensors;
    for (auto n : m_nodes)
    {
        out_tensors.insert(n->outputs().begin(), n->outputs().end());
        in_tensors.insert(n->inputs().begin(), n->inputs().end());
    }
    for (auto t : out_tensors)
    {
        if (in_tensors.count(t))
        {
            m_internal_tensors.insert(t);
        }
    }
    return m_internal_tensors;
}

SubGraphNodePtr SubGraph::as_node(bool replace, bool add_sg_input)
{
    auto node = SubGraphNode::create(get_shared_ptr());
    this->replace = replace;
    if (!replace)
    {
        return node;
    }
    to_node(get_shared_ptr(), node, add_sg_input);
    for (auto snd : m_nodes)
        ADD_LOCATION_BY_NODE(node, snd);
    return node;
}

void SubGraph::to_node(SubGraphPtr sg, SubGraphNodePtr node, bool add_sg_input, const std::set<TensorPtr> &free_out_wlist)
{
    node->m_sg = sg;
    sg->m_sgn = SubGraphNodeRef(node);

    std::vector<std::string> in_tensor_name;
    std::vector<std::string> out_tensor_name;

    for (auto n : sg->nodes())
    {
        if (n->type() == OpType_Input ||
            n->type() == OpType_SubGraphInput ||
            n->type() == OpType_InternalConstant ||
            n->type() == OpType_Constant) // sginput/input/constant
        {
            for (auto t : n->outputs())
            {
                in_tensor_name.push_back(t->name());
            }
        }
        else
        {
            for (auto t : n->inputs())
            {
                if (!t->node() || !contains(sg->nodes(), t->node()))
                {
                    in_tensor_name.push_back(t->name());
                }
            }
        }

        for (auto t : n->outputs())
        {
            if (t->is_io_tensor())
            {
                out_tensor_name.push_back(t->name());
            }
            else
            {
                for (auto c : t->consumers())
                {
                    if (!contains(sg->nodes(), c))
                    {
                        out_tensor_name.push_back(t->name());
                        break;
                    }
                }
            }
        }
    }

    auto internal_tensors = sg->update_internal_tensors();
    for (auto inp_node : sg->get_input_nodes())
    {
        for (uint32_t i = 0; i < inp_node->inputs().size(); ++i)
        {
            auto inp = inp_node->inputs()[i];
            if (!internal_tensors.count(inp))
            {
                bool has_input_t = false;
                ProxyTensorPtr bk = nullptr;
                for (auto t : node->m_in_tensors)
                {
                    // we need keep the same tensor ptr for same tensor in subgraph
                    if (t->name() == inp->name())
                    {
                        inp_node->remove_input(i);
                        inp_node->add_input(t, i);
                        bk = t;

                        // bk = std::make_shared<ProxyTensor>(t);
                        has_input_t = true;
                        break;
                    }
                }
                if (!has_input_t)
                {
                    // copy the tensor for internal as the tensor is an external tensor
                    bk = std::make_shared<ProxyTensor>(inp);
                    inp_node->remove_input(i);
                    inp_node->add_input(bk, i);
                    if (add_sg_input)
                    {
                        auto sg_input_node = Node::create(inp_node->name() + "_sg_input_" + std::to_string(i), OpType_SubGraphInput);
                        sg_input_node->add_output(bk);
                        sg->add_node(sg_input_node);

                        LayoutCombinations out_layouts = {{bk->shape().layout()}};
                        sg_input_node->set_attr(Constants::valid_output_layouts, out_layouts);
                        sg_input_node->set_attr(Constants::keeping_layout, false);
                    }
                }
                node->m_in_tensors.push_back(bk);
                node->add_input(inp);
            }
        }
    }
    // keep the input tensor order with the _ordered_input_id if setted attr: _ordered_input_id
    auto n_in_tensors = node->inputs();
    bool changed = false;
    do
    {
        changed = false;
        for (uint32_t m = 0; m < node->m_in_tensors.size(); m++)
        {
            auto itensor = node->m_in_tensors[m];
            if (itensor->attrs().count(Constants::ordered_input_id))
            {
                uint32_t sorted_id = itensor->get_attr(Constants::ordered_input_id);
                if (sorted_id == m)
                {
                    continue;
                }
                auto bk = node->m_in_tensors[sorted_id];
                node->m_in_tensors[sorted_id] = node->m_in_tensors[m];
                node->m_in_tensors[m] = bk;
                changed = true;
                for (uint32_t i = 0; i < n_in_tensors.size(); i++)
                {
                    if (n_in_tensors[i]->name() == itensor->name())
                    {
                        auto bk = node->remove_input(sorted_id);
                        node->add_input(n_in_tensors[i], sorted_id);
                        node->remove_input(i);
                        node->add_input(bk, i);
                    }
                }
            }
        }
    } while (changed);

    std::set<TensorPtr> io_tensors;
    for (auto out_node : sg->m_nodes)
    {
        if (out_node->type() != OpType_SubGraphInput)
        {
            for (uint32_t i = 0; i < out_node->outputs().size(); ++i)
            {
                auto out = out_node->outputs()[i];
                if (out->is_io_tensor())
                {
                    io_tensors.insert(out);
                }
            }
        }
    }
    for (auto out_node : sg->get_output_nodes())
    {
        for (uint32_t i = 0; i < out_node->outputs().size(); ++i)
        {
            auto out = out_node->outputs()[i];
            auto has_outside_consumer = false;
            auto consumers = out->consumers();
            for (auto c : consumers)
            {
                if (!std::count(sg->m_nodes.begin(), sg->m_nodes.end(), c))
                {
                    has_outside_consumer = true;
                }
            }
            if (!sg->m_internal_tensors.count(out) || has_outside_consumer)
            {
                // skip map free output if not in whitelist
                if (consumers.empty() && free_out_wlist.size() && !free_out_wlist.count(out))
                {
                    continue;
                }
                // copy the tensor for internal as the tensor is an external tensor
                auto bk = std::make_shared<ProxyTensor>(out);
                out_node->add_output(bk, out_node->remove_output(out));
                node->m_out_tensors.push_back(bk);
                if (has_outside_consumer)
                {
                    for (auto c : consumers)
                    {
                        if (contains(sg->m_nodes, c))
                        {
                            c->replace_input_by(out, bk);
                        }
                    }
                }
                if (io_tensors.count(out))
                {
                    io_tensors.erase(out);
                }
                node->add_output(out);
            }
        }
    }
    for (auto out : io_tensors)
    {
        auto out_node = out->node();
        auto bk = std::make_shared<ProxyTensor>(out);
        out_node->add_output(bk, out_node->remove_output(out));
        node->m_out_tensors.push_back(bk);
        for (auto c : out->consumers())
        {
            if (contains(sg->m_nodes, c))
            {
                c->replace_input_by(out, bk);
            }
        }
        node->add_output(out);
    }

    for (auto name : in_tensor_name)
    {
        auto t = sg->get_tensor(name);
        sg->m_in_tensors.push_back(t);
    }

    for (auto name : out_tensor_name)
    {
        auto t = sg->get_tensor(name);
        sg->m_out_tensors.push_back(t);
    }

    // copy weights
    for (auto sn : sg->m_nodes)
    {
        auto &constants = node->constants();
        // rename the weight as node_name/weight_name
        for (auto cts : sn->constants())
        {
            auto t = std::make_shared<Tensor>(*cts.second);
            // cts.second->memory_offset()->set_base_offset(t->memory_offset());
            cts.second->memory_offset()->relative_offset = 0;
            constants[sn->name() + "/" + cts.first] = t;
            // use same pointer!!!
            //  constants[sn->name() + "/" + cts.first] = cts.second;
        }
        // node->constants().insert(sn->constants().begin(), sn->constants().end());
    }
    for (auto sn : sg->m_nodes)
    {
        sg->origin_graph()->remove_node(sn, false);
    }
    node->set_name(sg->origin_graph()->get_valid_node_name(node->name()));
    sg->origin_graph()->add_node(node);
}

SubGraphNodePtr SubGraphNode::create(SubGraphPtr subg, const std::string &name)
{
    auto node = SubGraphNodePtr(new SubGraphNode(subg, name));
    if (subg)
    {
        subg->m_sgn = SubGraphNodeRef(node);
    }
    return node;
}

SubGraphNode::SubGraphNode(SubGraphPtr subg, const std::string &name_) : m_sg(subg)
{
    if (name_.size())
    {
        m_name = name_;
    }
    else if (subg)
    {
        m_name = "subgraph_" + subg->nodes()[0]->name();
    }
    m_type = OpType_SubGraph;
}

SubGraphNodePtr SubGraphNode::from_nodes(GraphPtr g, std::vector<NodePtr> &nodes, const std::string &name_)
{
    SubGraphPtr sg = SubGraph::create(g, nodes);
    auto node = sg->as_node();
    if (name_.size())
    {
        node->set_name(name_);
    }
    return node;
}

bool SubGraphNode::is_subgraph_node(BaseNodePtr n)
{
    return n && n->type() >= OpType_SubGraph && n->type() < OpType_SubGraphMaxCount;
}

SubGraphNodePtr SubGraphNode::from_node(BaseNodePtr n)
{
    if (is_subgraph_node(n))
    {
        return std::dynamic_pointer_cast<SubGraphNode>(n);
    }
    return nullptr;
}

SubGraphNode::~SubGraphNode()
{
    // subgraph node just a proxy node, it produce nothing but the subgraph;
    //  m_outputs.clear();   //the sub nodes will free the outputs
    //  m_constants.clear(); // the sub nodes will free their constants;
    std::set<TensorPtr> release_constants;
    for (auto kv : m_constants)
    {
        release_constants.insert(kv.second);
    }
    for (auto n : m_sg->nodes())
    {
        auto csts = n->constants();
        for (auto cst : csts)
        {
            if (release_constants.count(cst.second))
            {
                n->constants().erase(cst.first);
            }
        }
    }
}

void SubGraphNode::flat(bool recurrently)
{
    // how to flat a multi-level embedding subgraph
    // 1. remove the subgraphnode from subgraph
    // 2. add back the real node
    // 3. fixup the connections
    std::vector<SubGraphNodePtr> backup;
    std::map<ProxyTensorPtr, TensorPtr> real_tensor_map;
    for (auto n : m_sg->m_nodes)
    {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn && !sgn->unified_view())
        {
            if (recurrently)
            {
                sgn->flat(recurrently);
            }
            if (sgn->type() == OpType_Loop ||
                sgn->type() == OpType_If)
            {
                continue;
            }
            backup.push_back(sgn);
            for (auto inp : sgn->m_in_tensors)
            {
                real_tensor_map[inp] = inp->proxied_tensor();
            }
            for (auto inp : sgn->m_out_tensors)
            {
                real_tensor_map[inp] = inp->proxied_tensor();
            }
        }
    }

    for (auto sgn : backup)
    {
        for (auto inp : sgn->m_in_tensors)
        {
            auto real = real_tensor_map[inp];
            for (auto c : inp->consumers())
            {
                c->replace_input_by(inp, real);
                if (inp->node() && inp->node()->type() == OpType_SubGraphInput)
                {
                    sgn->subgraph()->remove_node(inp->node());
                }
            }
        }
        uint32_t s = sgn->inputs().size();
        for (uint32_t i = 0; i < s; ++i)
        {
            sgn->Node::remove_input(0); // to prevent it overwrite the proxy tensor
        }
        for (auto out : sgn->m_out_tensors)
        {
            auto n = out->node();
            auto real = real_tensor_map[out];
            for (auto c : out->consumers())
            {
                c->replace_input_by(out, real);
            }
            n->add_output(real, n->remove_output(out));
        }

        s = sgn->outputs().size();
        for (uint32_t i = 0; i < s; ++i)
        {
            sgn->Node::remove_output(0); // to prevent it overwrite the proxy tensor
        }
    }
    for (auto sgn : backup)
    {
        m_sg->remove_node(sgn);
        for (auto n : sgn->subgraph()->nodes())
        {
            m_sg->add_node(n);
        }
    }
}

std::vector<NodePtr> SubGraphNode::unfold()
{
    // how to unfold a subgraph node
    // 1. replace inputs then outputs from proxy tensors on inter node by real tensors
    // 2. add inter node to upper graph
    // 3. remove this from upper graph
    std::vector<NodePtr> ret;
    auto upper = m_graph.lock();
    if (!upper) // this node not belong any upper graph, do nothing.
    {
        return ret;
    }
    std::set<ProxyTensorPtr> io_tensors(m_in_tensors.begin(), m_in_tensors.end());
    io_tensors.insert(m_out_tensors.begin(), m_out_tensors.end());
    std::set<TensorPtr> real_io_tensors;
    auto nodes = m_sg->nodes();
    for (auto &cts : m_constants) // restore relative offsets
    {
        for (auto n : nodes)
        {
            bool is_restored = false;
            for (auto &kv : n->constants())
            {
                if (cts.first == n->name() + "/" + kv.first)
                {
                    auto offset = cts.second->memory_offset()->relative_offset;
                    kv.second->memory_offset()->relative_offset = offset;
                    is_restored = true;
                    break;
                }
            }
            if (is_restored)
            {
                break;
            }
        }
    }
    for (auto n : nodes)
    {
        if (n->type() == OpType_SubGraphInput)
        {
            continue;
        }
        for (uint32_t i = 0; i < n->inputs().size(); ++i)
        {
            auto t = std::dynamic_pointer_cast<ProxyTensor>(n->inputs()[i]);
            if (t && io_tensors.count(t))
            {
                auto real_t = t->proxied_tensor();
                n->remove_input(i);
                n->add_input(real_t, i);
                if (n->inputs()[i]->node() && n->inputs()[i]->node()->type() == OpType_SubGraphInput)
                {
                    m_sg->remove_node(n->inputs()[i]->node());
                }
                real_io_tensors.insert(real_t);
            }
        }
        for (uint32_t i = 0; i < n->outputs().size(); ++i)
        {
            auto t = std::dynamic_pointer_cast<ProxyTensor>(n->outputs()[i]);
            if (t && io_tensors.count(t))
            {
                auto real_t = t->proxied_tensor();
                n->remove_output(i);
                n->add_output(real_t, i);
                real_io_tensors.insert(real_t);
            }
        }
        n->set_name(upper->get_valid_node_name(n->name()));
        for (auto t : n->outputs())
        {
            if (!real_io_tensors.count(t))
            {
                t->set_name(upper->get_valid_tensor_name(t->name()));
            }
        }
        upper->add_node(n);
        m_sg->remove_node(n, false);
        ret.push_back(n);
    }
    auto upper_in_tensors = upper->get_input_tensors();
    auto upper_out_tensors = upper->get_output_tensors();
    for (auto &&t : real_io_tensors)
    {
        if (!contains(upper_in_tensors, t) && !contains(upper_out_tensors, t))
        {
            t->set_io_tensor(false);
        }
    }
    upper->remove_node(get_shared_ptr());
    return ret;
}
const std::string SubGraphNode::dot(const std::string &color, const std::string &prefix)
{
    // the attr is on priority
    const std::vector<std::string> m_color = {"antiquewhite",
                                              "azure",
                                              "beige",
                                              "chartreuse",
                                              "coral",
                                              "cyan2",
                                              "darkolivegreen1",
                                              "gold",
                                              "gray",
                                              "greenyellow",
                                              "lightblue1",
                                              "purple",
                                              "rosybrown1"

    };
    if (m_unified_view)
    {
        return Node::dot(color, prefix);
    }
    std::ostringstream oss;
    auto name = utils::verify_name(m_name);
    std::string m_b_color = "lightgrey";
    if (m_attrs.count(".subgraph_color"))
    {
        m_b_color = get_attr<std::string>(".subgraph_color");
    }
    oss << "subgraph cluster_" << name + "_subgraph {\n"
        << "style=filled;"
        << "color=" << m_b_color << ";"
        << "node[style = filled, color = white];";
    // << "label = \"" << prefix << m_name;<<"\";";
    oss << "\n";
    oss << m_sg->dot(m_color, false);
    oss << "};\n";

    // add external edges use tensor
    std::map<std::string, std::set<std::string>> all_edges;
    for (uint32_t i = 0; i < outputs().size(); ++i)
    {
        auto out = outputs()[i];
        auto out_node = get_input_node_by_tensor(out, true);
        if (!out_node)
        {
            continue;
        }
        if (out_node->type() == OpType_SubGraphInput)
        {
            continue;
        }
        auto n1 = out_node->get_node_dot_name();
        for (auto consumer : out->consumers())
        {
            auto n2 = consumer->get_node_dot_name();
            auto sgn = SubGraphNode::from_node(consumer);
            if (SubGraphNode::is_subgraph_node(consumer) && sgn && !sgn->unified_view())
            {
                auto in_nodes = sgn->get_output_nodes_by_tensor(out, true);
                for (auto in_node : in_nodes)
                {
                    n2 = in_node->get_node_dot_name();
                    if (!all_edges[n1].count(n2))
                    {
                        oss << n1 << " -> " << n2 << ";// outedge\n";
                        all_edges[n1].insert(n2);
                    }
                }
            }
            else
            {
                if (!all_edges[n1].count(n2))
                {
                    oss << n1 << " -> " << n2 << ";// outedge\n";
                    all_edges[n1].insert(n2);
                }
            }
        }
    }

    // add external in edges
    std::set<std::string> not_repeat;
    for (uint32_t i = 0; i < inputs().size(); ++i)
    {
        if (i >= m_in_tensors.size())
        {
            DEBUG() << "m_in_tensors and inputs size mismatch.";
            continue;
        }
        auto inp = m_in_tensors[i];
        if (!inp || not_repeat.count(inp->name()))
        {
            continue;
        }
        not_repeat.insert(inp->name());
        // subgraph out edge has handled it.
        auto node = inp->node();
        if (!node || node->type() == OpType_SubGraphInput)
        {
            auto realinp = std::dynamic_pointer_cast<ProxyTensor>(inp);
            if (realinp)
            {
                node = realinp->proxied_tensor()->node();
            }
        }
        if (node->type() == OpType_SubGraphInput)
        {
            continue;
        }
        auto sgn = SubGraphNode::from_node(node);
        if (!node || (SubGraphNode::is_subgraph_node(node) && sgn && !sgn->unified_view()))
        {
            continue;
        }
        auto n1 = node->get_node_dot_name();
        auto consumers = get_output_nodes_by_tensor(inp, true);
        for (auto consumer : consumers)
        {
            auto n2 = consumer->get_node_dot_name();
            if (!all_edges[n1].count(n2))
            {
                oss << n1 << " -> " << n2 << ";//in edge\n";
                all_edges[n1].insert(n2);
            }
        }
    }
    return oss.str();
}

const std::string SubGraphNode::json()
{
    return AIPUBuilder::get_profiler()->json_from_node(this);
}

NodePtr SubGraphNode::clone() const
{
    // backup io tensor
    std::set<TensorPtr> io_tensors;
    for (auto t : m_out_tensors)
    {
        if (!t->is_io_tensor())
        {
            io_tensors.insert(t);
            t->set_io_tensor(true);
        }
    }

    auto sg = m_sg->clone();

    // restore io tensor
    for (auto t : io_tensors)
    {
        t->set_io_tensor(false);
    }

    std::map<std::string, NodePtr> sg_map;
    for (auto n : sg->nodes())
    {
        sg_map[n->name()] = n;
    }
    // remove subgraph_input nodes
    std::vector<NodePtr> nodes = sg->nodes();
    for (auto n : nodes)
    {
        if (n->type() == OpType_SubGraphInput)
        {
            for (auto c : n->outputs()[0]->consumers())
            {
                if (sg_map.count(c->name()))
                {
                    int idx = c->remove_input(n->outputs()[0]);
                    while (idx != INT32_MAX)
                    {
                        idx = c->remove_input(n->outputs()[0]);
                    }
                }
            }
            sg->remove_node(n);
            sg_map.erase(n->name());
        }
    }
    nodes.clear();
    for (auto n : m_sg->nodes()) // we must ensure the nodes order is same as original one, due to the input mapping requires the order
    {
        if (n->type() == OpType_SubGraphInput)
        {
            continue;
        }
        nodes.push_back(sg_map[n->name()]);
    }
    // clone input tensors
    std::map<NodePtr, std::vector<std::pair<TensorPtr, int>>> inputs_tobe_add;
    std::set<TensorPtr> inps;
    for (auto t : m_in_tensors)
    {
        if (inps.count(t))
        {
            continue;
        }
        inps.insert(t);
        auto nt = std::make_shared<Tensor>(*t);
        for (auto c : t->consumers())
        {
            auto n = sg_map.at(c->name());
            if (n)
            {
                for (uint32_t i = 0; i < c->inputs().size(); ++i)
                {
                    if (c->inputs()[i] == t)
                    {
                        inputs_tobe_add[n].push_back(std::make_pair(nt, i));
                    }
                }
            }
        }
    }

    for (auto &it : inputs_tobe_add)
    {
        auto node = it.first;
        auto tensors = it.second;

        std::stable_sort(tensors.begin(), tensors.end(), [](std::pair<TensorPtr, int> a, std::pair<TensorPtr, int> b) {
            return a.second < b.second;
        });

        for (auto &p : tensors)
        {
            node->add_input(p.first, p.second);
        }
    }

    //check if the input order the same!!!
    for (auto n : m_sg->nodes())
    {
        auto origin_input = n->inputs();
        if (!sg_map[n->name()])
        {
            continue;
        }
        auto cloned_input = sg_map[n->name()]->inputs();

        if (origin_input.size() != cloned_input.size())
        {
            ERROR() << "input size not equal " << n->name();
            continue;
        }

        for (size_t i = 0; i < origin_input.size(); ++i)
        {
            if (origin_input[i]->name() != cloned_input[i]->name())
            {
                ERROR() << " input not equal " << i << " " << origin_input[i]->name() << " " << cloned_input[i]->name();
            }
        }
    }

    SubGraphNodePtr node = nullptr;
    SubGraphPtr nsg = SubGraph::create(sg, nodes);
    if (m_type == OpType_Loop)
    {
        node = LoopNode::create(nsg, m_name);
    }
    else if (m_type == OpType_If)
    {
        node = IfNode::create(nsg, m_name);
    }
    else
    {
        node = nsg->as_node(true, !has_plugin());
    }

    // reorder the input/outputs order of node
    auto out_tensors = node->m_out_tensors;
    node->m_out_tensors.clear();
    for (auto t : m_out_tensors)
    {
        auto nt = get_tensor_by_name(out_tensors, t->name());
        aipu_assert(nt && "Clone failed!");
        for (auto iot : io_tensors)
        {
            if (nt->name() == iot->name())
            {
                nt->set_io_tensor(false);
            }
        }
        node->m_out_tensors.push_back(nt);
    }
    auto outs = node->outputs();
    node->clear_outputs();
    for (auto t : outputs())
    {
        node->add_output(get_tensor_by_name(outs, t->name()));
    }

    auto in_tensors = node->m_in_tensors;
    node->m_in_tensors.clear();
    for (auto t : m_in_tensors)
    {
        auto nt = get_tensor_by_name(in_tensors, t->name());
        aipu_assert(nt && "Clone failed!");
        node->m_in_tensors.push_back(nt);
    }
    auto ins = node->inputs();
    node->clear_inputs();
    for (auto t : inputs())
    {
        node->add_input(get_tensor_by_name(ins, t->name()));
    }

    if (m_name.size())
    {
        node->set_name(m_name);
    }
    node->m_unified_view = m_unified_view;
    node->m_tag = m_tag;
    for (auto kv : m_params)
    {
        node->m_params[kv.first] = kv.second;
    }
    for (auto kv : m_attrs)
    {
        // TODO: fixme!!! the attr may be tensor
        node->m_attrs[kv.first] = kv.second;
    }
    for (auto kv : m_constants)
    {
        if (kv.second)
        {
            node->m_constants[kv.first] = std::make_shared<Tensor>(*kv.second);
        }
        else
        {
            WARN() << "clone the constant " << kv.first << " of node " << m_name << " error, which is a nullptr.";
        }
    }
    for (auto t : m_placeholders)
    {
        node->m_placeholders.push_back(std::make_shared<Tensor>(*t));
    }
    if (m_loc != nullptr)
    {
        if (node->loc()->roots().size()) // the node loc should have `as_node` loc
        {
            node->loc()->roots()[0]->set_name(__func__);
        }

        node->loc()->set_original(m_loc->is_original());
    }
    return node;
}

void SubGraphNode::update_input_tensor_map()
{
    for (uint32_t i = m_inputs.size(); i < m_in_tensors.size(); ++i)
    {
        *m_in_tensors[i] = nullptr; // update proxied tensor
    }
    for (uint32_t i = 0; i < m_in_tensors.size() && i < m_inputs.size(); ++i)
    {
        *m_in_tensors[i] = m_inputs[i]; // update proxied tensor
    }
}
void SubGraphNode::update_output_tensor_map()
{
    for (uint32_t i = 0; i < m_out_tensors.size(); ++i)
    {
        if (i < m_outputs.size())
        {
            *m_out_tensors[i] = m_outputs[i]; // update proxied tensor
        }
        else
        {
            *m_out_tensors[i] = nullptr;
        }
    }
}
/**
 * The Subgraph Node input and output's number must be fixed!! which is decided by internal nodes.
 * so, to modify the input/output of a subgraph node, there is only one legal operation: replace a tensor.
 * here, the add/remove_input/output interface will not changed the number of tensors, and the order as well(we remove the shift operations).
 * if a tensor be removed, we expect that it will be added back soon.
 */
void SubGraphNode::add_output(TensorPtr t, int idx)
{
    Node::add_output(t, idx);
    update_output_tensor_map();
}

TensorPtr SubGraphNode::remove_output(int idx)
{
    auto ret = Node::remove_output(idx);
    update_output_tensor_map();

    return ret;
}

void SubGraphNode::add_input(TensorPtr t, int idx)
{
    Node::add_input(t, idx);
    update_input_tensor_map();
}

TensorPtr SubGraphNode::remove_input(int idx)
{
    auto ret = Node::remove_input(idx);
    update_input_tensor_map();
    return ret;
}

bool SubGraphNode::equal(const NodePtr other) const
{
    if (Node::equal(other))
    {
        // compare the sub node
        auto sgnode = from_node(other);
        if (m_sg->nodes().size() != sgnode->m_sg->nodes().size())
        {
            return false;
        }
        for (uint32_t i = 0; i < m_sg->nodes().size(); ++i)
        {
            // TODO this contrain is too force, it should only make sure the subgraph is same and the node is equal, the order is not must.
            if (!m_sg->nodes()[i]->equal(sgnode->m_sg->nodes()[i]))
            {
                return false;
            }
        }
    }
    else
    {
        return false;
    }
    return true;
}

NodePtr SubGraphNode::get_input_node_by_tensor(TensorPtr t, bool view)
{
    if (t->node().get() == this)
    {
        for (auto out : m_out_tensors)
        {
            if (out->name() == t->name())
            {
                auto node = out->node();
                if (!view)
                {
                    return node;
                }
                auto sgn = SubGraphNode::from_node(node);
                if (sgn && !sgn->unified_view())
                {
                    return SubGraphNode::from_node(node)->get_input_node_by_tensor(out, view);
                }
                return node;
            }
        }
    }
    return nullptr;
}

std::vector<NodePtr> SubGraphNode::get_output_nodes_by_tensor(TensorPtr t, bool view)
{
    std::vector<NodePtr> nodes;
    get_output_nodes_by_tensor(t, nodes, view);
    return nodes;
}

void SubGraphNode::get_output_nodes_by_tensor(TensorPtr t, std::vector<NodePtr> &nodes, bool view)
{
    TensorPtr real = nullptr;
    for (auto inp : m_in_tensors)
    {
        if (inp->name() == t->name())
        {
            real = inp;
            break;
        }
    }
    if (real)
    {
        for (auto node : real->consumers())
        {
            auto sgn = SubGraphNode::from_node(node);
            if (view && SubGraphNode::is_subgraph_node(node) && sgn && !sgn->unified_view())
            {
                sgn->get_output_nodes_by_tensor(real, nodes, view);
            }
            else if (!contains(nodes, node))
            {
                nodes.push_back(node);
            }
        }
    }
}

static inline void copy_attrs(GraphPtr from, GraphPtr to, const std::string &subset = std::string())
{
    if (subset.size() && from->attrs().count(subset))
    {
        std::map<std::string, Any> attrs = from->attrs()[subset];
        for (auto &kv : attrs)
        {
            to->set_attr(kv.first, kv.second);
        }
    }
    else
    {
        for (auto &kv : from->attrs())
        {
            to->set_attr(kv.first, kv.second);
        }
    }
}
static void deep_copy_attrs(GraphPtr from, GraphPtr to)
{
    for (auto tn : to->nodes())
    {
        if (tn->type() == OpType_Loop || tn->type() == OpType_If)
        {
            auto fn = (*from)[tn->name()];
            if (fn && fn->type() == tn->type())
            {
                auto fsg = SubGraphNode::from_node(fn)->subgraph();
                auto tsg = SubGraphNode::from_node(tn)->subgraph();
                copy_attrs(fsg, tsg);
                deep_copy_attrs(fsg, tsg);
            }
        }
    }
}
static void copy_subgraph_attrs(GraphPtr from, const std::vector<GraphPtr> &to)
{
    std::vector<std::string> subsets(to.size(), std::string());
    if (to.size() == 2 &&
        from->attrs().count("then_branch_attrs") &&
        from->attrs().count("else_branch_attrs"))
    {
        subsets = {"then_branch_attrs", "else_branch_attrs"};
    }
    for (size_t i = 0; i < to.size(); ++i)
    {
        copy_attrs(from, to[i], subsets[i]);
        deep_copy_attrs(from, to[i]);
    }
}
static void remove_subgraph_scope(GraphPtr sg, const std::string &scope)
{
    for (auto n : sg->nodes())
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            remove_subgraph_scope(SubGraphNode::from_node(n)->subgraph(), scope);
        }
        if (utils::startswith(n->name(), scope) && n->name().size() > scope.size())
        {
            n->set_name(n->name().substr(scope.size()));
        }
        for (auto t : n->outputs())
        {
            if (utils::startswith(t->name(), scope) && t->name().size() > scope.size())
            {
                t->set_name(t->name().substr(scope.size()));
            }
        }
    }
}
GraphPtr SubGraphNode::extract_graph(const std::string &graph_name, const std::vector<std::string> &m_in_names, const std::vector<std::string> &m_out_names) const
{
    auto sg = this->subgraph();
    auto nsg = sg->clone();
    std::vector<std::string> in_names = m_in_names;
    std::vector<std::string> out_names = m_out_names;
    if (in_names.empty())
    {
        for (auto t : this->m_in_tensors)
        {
            if (!contains(in_names, t->name()))
            {
                in_names.push_back(t->name());
            }
        }
    }
    std::vector<TensorPtr> input_tensors(in_names.size(), nullptr);
    for (auto node : sg->get_bottom_nodes())
    {
        if (node->type() == OpType_Input || node->type() == OpType_SubGraphInput)
        {
            auto pt = std::dynamic_pointer_cast<ProxyTensor>(node->outputs()[0]);
            auto n = (*nsg)[node->name()];
            auto t = n->outputs()[0];
            TensorPtr nt = t;
            if (pt)
            {
                nt = std::make_shared<Tensor>(*pt->proxied_tensor());
                std::vector<Expr> exprs;
                if (sg->is_dynamic())
                {
                    exprs = nt->shape().symbol();
                    for (auto expr : exprs)
                    {
                        if (expr->op() != ExprImpl::Op::Val && expr->op() != ExprImpl::Op::Constant)
                        {
                            exprs.clear();
                            break;
                        }
                    }
                }
                ops::input(nt, nsg, false, exprs);
                auto consumers = t->consumers();
                for (auto c : consumers)
                {
                    c->replace_input_by(t, nt);
                }
                nsg->remove_node(n);
            }
            for (size_t j = 0; j < in_names.size(); ++j)
            {
                if (in_names[j] == nt->name())
                {
                    input_tensors[j] = nt;
                    break;
                }
            }
        }
        else
        {
            for (size_t i = 0; i < node->inputs().size(); ++i)
            {
                auto pt = std::dynamic_pointer_cast<ProxyTensor>(node->inputs()[i]);
                if (pt)
                {
                    auto nt = std::make_shared<Tensor>(*pt->proxied_tensor());
                    ops::input(nt, nsg);
                    auto n = (*nsg)[node->name()];
                    n->remove_input(i);
                    n->add_input(nt, i);
                    for (size_t j = 0; j < in_names.size(); ++j)
                    {
                        if (in_names[j] == nt->name())
                        {
                            input_tensors[j] = nt;
                            break;
                        }
                    }
                }
            }
        }
    }
    nsg->record_input_tensors(input_tensors); // update input tensors

    if (out_names.empty())
    {
        for (auto t : this->m_out_tensors)
        {
            out_names.push_back(t->name());
        }
    }
    std::vector<TensorPtr> output_tensors(out_names.size(), nullptr);
    for (size_t i = 0; i < this->m_out_tensors.size(); ++i)
    {
        auto pt = this->m_out_tensors[i];
        auto nt = std::make_shared<Tensor>(*pt->proxied_tensor());
        auto n = (*nsg)[pt->node()->name()];
        for (size_t j = 0; j < n->outputs().size(); ++j)
        {
            if (n->outputs()[j]->name() == pt->name())
            {
                auto t = n->remove_output(j);
                n->add_output(nt, j);
                auto consumers = t->consumers();
                for (auto c : consumers)
                {
                    for (size_t k = 0; k < c->inputs().size(); ++k)
                    {
                        if (c->inputs()[k]->name() == t->name())
                        {
                            c->remove_input(k);
                            c->add_input(nt, k);
                        }
                    }
                }
                for (size_t k = 0; k < out_names.size(); ++k)
                {
                    if (out_names[k] == nt->name())
                    {
                        output_tensors[k] = nt;
                        break;
                    }
                }
            }
        }
    }
    nsg->record_output_tensors(output_tensors); // update output tensors

    nsg->topological_sort();
    nsg->set_name(graph_name);
    return nsg;
}

void SubGraphNode::serialize(std::map<char *, uint32_t> &data_map,
                             std::ostream &oss,
                             std::ostream &wt_oss,
                             std::ostream &sg_oss) const
{
    if (m_attrs.count("is_call") && m_attrs.at("is_call").cast<bool>())
    {
        std::string bin_path = graph()->get_attr("model_bin");
        std::string txt_path = graph()->get_attr("file_path");
        if (m_attrs.count("txt_path"))
        {
            txt_path = m_attrs.at("txt_path").cast<std::string>();
        }
        else
        {
            txt_path += "." + utils::verify_name(m_name) + ".def";
        }
        std::ofstream ofs;
        ofs.open(txt_path);
        auto nsg = extract_graph(m_name);
        nsg->serialize(ofs, wt_oss);
        Node::serialize(data_map, oss, wt_oss, sg_oss);
        oss << "\nsubgraph_name=" << (std::string)m_params.at("call_subgraph") << "\n";
        std::vector<std::string> types;
        for (auto t : inputs())
        {
            types.push_back(t->type_s());
        }
        oss << "subgraph_input_type="
            << "[" << utils::join_string(",", types) << "]"
            << "\n";
        types.clear();
        for (auto t : inputs())
        {
            types.push_back(t->shape().to_string());
        }
        oss << "subgraph_input_shape="
            << "[" << utils::join_string(",", types) << "]"
            << "\n";
        types.clear();
        for (auto t : outputs())
        {
            types.push_back(t->type_s());
        }
        oss << "subgraph_output_type="
            << "[" << utils::join_string(",", types) << "]"
            << "\n";
        types.clear();
        for (auto t : outputs())
        {
            types.push_back(t->shape().to_string());
        }
        oss << "subgraph_output_shape="
            << "[" << utils::join_string(",", types) << "]"
            << "\n";
        oss << "txt_path=" << txt_path << "\n";
        oss << "bin_path=" << bin_path << "\n";
        oss << "\n";
        return;
    }

    if (!BuilderConfig::get<bool>("dump_debug_IR"))
    {
        if (m_type == OpType_Loop)
        {
            auto sg = subgraph();
            auto nsg = sg->clone();
            nsg->topological_sort();
            auto sgn = dynamic_cast<const LoopNode *>(this);
            std::vector<NodePtr> body_nodes;
            for (auto n : nsg->nodes())
            {
                if (n->type() == OpType_SubGraphInput ||
                    n->type() == OpType_ForRangeHeader ||
                    n->type() == OpType_ForRangeEnder)
                {
                    continue;
                }
                body_nodes.push_back(n);
            }

            int depend_input_num = m_params.at("dependency_inputs_num");
            auto body_sgn = SubGraph::create(nsg, body_nodes)->as_node(true, true);
            body_sgn->set_unified_view(false);
            std::vector<std::string> in_names, out_names;
            for (size_t i = 4; i < sgn->head()->outputs().size(); ++i)
            {
                in_names.push_back(sgn->head()->outputs()[i]->name());
            }
            for (size_t i = 6 + depend_input_num; i < sgn->tail()->inputs().size(); ++i)
            {
                out_names.push_back(sgn->tail()->inputs()[i]->name());
            }
            std::string body_name = m_params.at("body");
            auto body_sg = body_sgn->extract_graph(body_name, in_names, out_names);

            copy_subgraph_attrs(sg, {body_sg});
            remove_subgraph_scope(body_sg, m_name + "/");
            body_sg->serialize(sg_oss, wt_oss, true);
            sg_oss << "\n";
            Node::serialize(data_map, oss, wt_oss, sg_oss);
            return;
        }
        else if (m_type == OpType_If)
        {
            auto sg = subgraph();
            auto nsg = sg->clone();
            nsg->topological_sort();
            auto sgn = dynamic_cast<const IfNode *>(this);
            std::vector<NodePtr> then_nodes, else_nodes;
            sgn->get_branch_nodes(then_nodes, else_nodes);
            std::vector<NodePtr> nsg_then_nodes, nsg_else_nodes;
            for (auto n : then_nodes)
            {
                nsg_then_nodes.push_back((*nsg)[n->name()]);
            }
            for (auto n : else_nodes)
            {
                nsg_else_nodes.push_back((*nsg)[n->name()]);
            }

            std::string then_sg_name = m_params.at("then_branch");
            int then_input_num = m_params.at("then_branch_inputs_num");
            auto then_sgn = SubGraph::create(nsg, nsg_then_nodes)->as_node(true, true);
            then_sgn->set_unified_view(false);
            std::vector<std::string> then_in_names, then_out_names;
            for (int i = 2; i < 2 + then_input_num; ++i)
            {
                then_in_names.push_back(sgn->head()->outputs()[i]->name());
            }
            size_t output_num = sgn->tail()->outputs().size();
            for (size_t i = 0; i < output_num; ++i)
            {
                then_out_names.push_back(sgn->sep()->inputs()[i]->name());
            }
            auto then_sg = then_sgn->extract_graph(then_sg_name, then_in_names, then_out_names);

            std::string else_sg_name = m_params.at("else_branch");
            auto else_sgn = SubGraph::create(nsg, nsg_else_nodes)->as_node(true, true);
            else_sgn->set_unified_view(false);
            std::vector<std::string> else_in_names, else_out_names;
            for (size_t i = output_num; i < sgn->sep()->outputs().size(); ++i)
            {
                else_in_names.push_back(sgn->sep()->outputs()[i]->name());
            }
            for (size_t i = 2 + output_num; i < sgn->tail()->inputs().size(); ++i)
            {
                else_out_names.push_back(sgn->tail()->inputs()[i]->name());
            }
            auto else_sg = else_sgn->extract_graph(else_sg_name, else_in_names, else_out_names);
            for (size_t i = output_num; i < sgn->sep()->inputs().size(); ++i)
            {
                auto raw_name = sgn->sep()->inputs()[i]->name();
                else_sg->get_input_tensors()[i - output_num]->set_name(raw_name);
            }

            copy_subgraph_attrs(sg, {then_sg, else_sg});
            remove_subgraph_scope(then_sg, m_name + "/then/");
            remove_subgraph_scope(else_sg, m_name + "/else/");
            then_sg->serialize(sg_oss, wt_oss, true);
            sg_oss << "\n";
            else_sg->serialize(sg_oss, wt_oss, true);
            sg_oss << "\n";
            Node::serialize(data_map, oss, wt_oss, sg_oss);
            return;
        }
    }

    // for subgraph, if this is single node subgraph, we just treat as the origin node
    // for mutil-node subgraph, a subgraph_scope key would be added.
    m_sg->update_real_nodes();
    if (m_sg->m_nodes.size() == 1)
    {
        auto node = m_sg->m_nodes[0];
        node->serialize(data_map, oss, wt_oss, sg_oss);
    }
    else
    {
        for (auto node : m_sg->m_real_nodes)
        {
            node->serialize(data_map, oss, wt_oss, sg_oss);
            oss << "subgraph_scope=" << m_name << "\n";
            oss << "\n";
        }
    }
}

void SubGraphNode::set_unified_view(bool unified)
{
    m_unified_view = unified;
}

GraphPtr SubGraphNode::subgraph() const
{
    return m_sg;
}

void SubGraphNode::add_child_helper(NodePtr child)
{
    if (!child)
    {
        return;
    }
    Node::add_child_helper(child);
}

void SubGraphNode::add_parent_helper(NodePtr parent)
{

    if (!parent)
    {
        return;
    }
    Node::add_parent_helper(parent);
}

void SubGraphNode::remove_parent_helper(NodePtr parent)
{
    if (!parent)
    {
        return;
    }
    Node::remove_parent_helper(parent);
}

void SubGraphNode::remove_child_helper(NodePtr child)
{
    if (!child)
    {
        return;
    }
    Node::remove_child_helper(child);
}
void SubGraphNode::set_graph(GraphPtr g)
{
    m_sg->ori_graph = g;
    Node::set_graph(g);
}

bool SubGraphNode::has_plugin() const
{
    return m_attrs.count(Constants::plugin_attrs_name);
}

static void update_layout_attributes(SubGraphNodePtr sgn)
{
    if (sgn->attrs().count(Constants::plugin_attrs_name))
    {
        return;
    }
    auto get_io_los = [sgn](TensorPtr inp, bool in_or_out /*true is in,false is out*/) {
        NodePtr node = nullptr;
        uint32_t idx = UINT32_MAX;
        for (auto n : sgn->subgraph()->get_input_nodes())
        {
            for (uint32_t i = 0; i < n->inputs().size(); ++i)
            {
                auto pt = std::dynamic_pointer_cast<ProxyTensor>(n->inputs()[i]);
                if (pt && pt->proxied_tensor() == inp)
                {
                    node = n;
                    idx = i;
                    break;
                }
            }
            if (node)
            {
                break;
            }
        }
        LayoutList inlo;
        if (!node)
        {
            return inlo;
        }
        if (node && node->type() == OpType_SubGraph)
        {
            auto sgnode = SubGraphNode::from_node(node);
            update_layout_attributes(sgnode);
        }
        auto key = Constants::valid_input_layouts;
        if (!in_or_out)
        {
            key = Constants::valid_output_layouts;
        }
        auto node_in = node->get_attr<LayoutCombinations>(key);
        for (auto los : node_in)
        {
            if (los.size() > idx)
            {
                inlo.push_back(los[idx]);
            }
            else if (los.size())
            {
                inlo.push_back(los[0]);
            }
            else // empty suppose support all
            {
                // TODO
                /* code */
            }
        }
        return inlo;
    };

    LayoutCombinations input_layout(1), output_layout(1);
    for (auto inp : sgn->inputs())
    {
        auto los = get_io_los(inp, true);
        LayoutCombinations in_los;
        for (auto list : input_layout)
        {
            for (auto lo : los)
            {
                auto m_list = list;
                m_list.push_back(lo);
                in_los.push_back(m_list);
            }
        }
        input_layout = std::move(in_los);
    }
    for (auto out : sgn->outputs())
    {
        auto los = get_io_los(out, false);
        LayoutCombinations out_los;
        for (auto list : output_layout)
        {
            for (auto lo : los)
            {
                auto m_list = list;
                m_list.push_back(lo);
                out_los.push_back(m_list);
            }
        }
        output_layout = std::move(out_los);
    }
    sgn->set_attr(Constants::valid_input_layouts, input_layout);
    sgn->set_attr(Constants::valid_output_layouts, output_layout);
}

void SubGraphNode::update_attrs()
{
    // update layout info
    update_layout_attributes(get_shared_ptr());
}
} // namespace aipubt

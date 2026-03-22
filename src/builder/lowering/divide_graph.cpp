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
 * @Date: 2024-04-11 15:54:19
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-04-11 15:54:19
 */
#include "builder/lowering/lowering.h"
#include "checker/checker.h"
#include "core/core.h"

namespace aipubt
{

static void search_subgraph(NodePtr n,
                            std::set<NodePtr> &ran_nodes,
                            std::vector<NodePtr> &subgraph,
                            bool is_dynamic)
{
    if (ran_nodes.count(n) || n->is_dynamic() != is_dynamic)
    {
        return;
    }
    ran_nodes.insert(n);
    subgraph.push_back(n);
    for (auto p : n->parents())
    {
        bool independent = true;
        for (auto c : p->children())
        {
            if (!ran_nodes.count(c))
            {
                independent = false;
                break;
            }
        }
        if (!independent)
        {
            continue;
        }
        search_subgraph(p, ran_nodes, subgraph, is_dynamic);
    }
}

static std::vector<std::vector<NodePtr>> search_subgraphs(GraphPtr g)
{
    std::vector<std::vector<NodePtr>> subgraphs;
    std::vector<NodePtr> nodes(g->nodes().rbegin(), g->nodes().rend());
    std::set<NodePtr> ran_nodes;
    for (auto n : nodes)
    {
        if (ran_nodes.count(n))
        {
            continue;
        }
        std::vector<NodePtr> sg;
        search_subgraph(n, ran_nodes, sg, n->is_dynamic());
        if (sg.size())
        {
            subgraphs.push_back(sg);
        }
    }
    return subgraphs;
}

static void set_subgraph_color(SubGraphNodePtr sgn)
{
    std::string bg_color = sgn->is_dynamic() ? "lightblue" : "lightgrey";
    sgn->set_attr(".subgraph_color", bg_color);
}

static void merge_subgraphs(GraphPtr g)
{
    while (true)
    {
        g->topological_sort();
        uint32_t merged_count = 0;
        for (auto &sg : search_subgraphs(g))
        {
            if (sg.size() > 1)
            {
                std::vector<NodePtr> sg_nodes;
                for (auto n : sg)
                {
                    auto nodes = SubGraphNode::from_node(n)->unfold();
                    sg_nodes.insert(sg_nodes.end(), nodes.begin(), nodes.end());
                }
                auto sgn = SubGraph::create(g, sg_nodes)->as_node(true, true);
                sgn->set_unified_view(false);
                set_subgraph_color(sgn);
                merged_count++;
            }
        }
        if (!merged_count)
        {
            break;
        }
    }
}

static void record_io_tensors(SubGraphNodePtr sgn)
{
    auto sg = sgn->subgraph();
    for (auto n : sg->nodes())
    {
        if (n->type() == OpType_Input ||
            n->type() == OpType_SubGraphInput)
        {
            n->outputs()[0]->set_io_tensor(true);
        }
    }
    for (auto t : sg->get_output_tensors())
    {
        t->set_io_tensor(true);
    }
}

void divide_graph(GraphPtr g)
{
    for (auto &sg : search_subgraphs(g))
    {
        auto sgn = SubGraph::create(g, sg)->as_node(true, true);
        sgn->set_unified_view(false);
        set_subgraph_color(sgn);
    }
    merge_subgraphs(g);
    for (auto n : g->nodes())
    {
        record_io_tensors(SubGraphNode::from_node(n));
    }
    if (!g->check_sg())
    {
        ERROR() << "check subgraph failed";
    }
    if (!is_DAG(g))
    {
        ERROR() << "check DAG failed";
    }
}

} // namespace aipubt

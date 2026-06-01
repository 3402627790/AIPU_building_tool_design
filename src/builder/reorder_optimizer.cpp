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
 * @Author: Kyle Huang
 * @Date: 2024-09-14 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2024-09-14 14:27:33
 */
#include "reorder_optimizer.h"
#include "builder/lowering/hash_constant.h"
#include "core/config.h"
#include "utils/json.h"
#include "utils/utils.h"
#include <fstream>
#include <stack>
#include <vector>

namespace aipubt
{
void ReorderOptimizer::set_graph(GraphPtr g)
{
    m_graph = g;
}
void ReorderOptimizer::set_levels(std::vector<std::vector<NodePtr>> &levels)
{
    m_levels = levels;
}
static NodePtr get_node(GraphPtr g, std::string name)
{
    NodePtr node;
    for (auto n : g->nodes())
    {
        if (n->name() == name)
        {
            node = n;
            break;
        }
    }
    return node;
}
void ReorderOptimizer::node_reorder(int strategy)
{
    std::string load_file = BuilderConfig::get<std::string>(Constants::load_reorder_cfg);
    if (load_file != "")
    {
        if (!utils::file_exists(load_file))
        {
            WARN() << "doesn't exist";
            return;
        }

        // INFO() << "before branch reorder";
        // for (size_t i = 0; i < m_levels.size(); i++)
        // {
        //     INFO() << " level " << i;
        //     // if(i == 3)
        //     {
        //         int sg = 0;
        //         for (auto n : m_levels[i])
        //         {
        //             INFO() << "new sg";
        //             int cnt = 0;
        //             auto sgn = SubGraphNode::from_node(n);
        //             if (sgn)
        //             {
        //                 auto nodes = sgn->subgraph()->nodes();
        //                 for (auto node : nodes)
        //                 {
        //                     INFO() << "sg " << sg << "--" << cnt << " " << node->name();
        //                     cnt++;
        //                 }
        //             }
        //             sg++;
        //         }
        //     }
        // }

        std::stringstream buffer;
        std::ifstream ifile(load_file);
        buffer << ifile.rdbuf();
        utils::JsonObject cfg_graph_info = utils::JsonObject::parse(buffer.str());
        auto level_obj = cfg_graph_info.obj();
        ifile.close();

        for (auto l_obj : level_obj)
        {
            auto sg_obj = l_obj.second.obj();
            for (auto s_obj : sg_obj)
            {
                std::string sg_name = s_obj.first;
                auto node_obj = s_obj.second.obj();
                NodePtr sg_node = get_node(m_graph, sg_name);
                auto sgn = SubGraphNode::from_node(sg_node);
                if (sgn)
                {
                    std::vector<NodePtr> new_order;
                    auto sg = sgn->subgraph();
                    for (auto n_obj : node_obj)
                    {
                        auto node = get_node(sg, n_obj.first);
                        new_order.push_back(node);
                    }
                    sg->sort_by_ordered(new_order);
                }
            }
        }

        // INFO() << "after branch reorder";
        // for (size_t i = 0; i < m_levels.size(); i++)
        // {
        //     INFO() << " level " << i;
        //     // if(i == 3)
        //     {
        //         int sg = 0;
        //         for (auto n : m_levels[i])
        //         {
        //             INFO() << "new sg";
        //             int cnt = 0;
        //             auto sgn = SubGraphNode::from_node(n);
        //             if (sgn)
        //             {
        //                 auto nodes = sgn->subgraph()->nodes();
        //                 for (auto node : nodes)
        //                 {
        //                     INFO() << "sg " << sg << "--" << cnt << " " << node->name();
        //                     cnt++;
        //                 }
        //             }
        //             sg++;
        //         }
        //     }
        // }
    }
    else if (strategy == -1)
    {
        std::vector<std::vector<NodePtr>> new_orders;
        for (auto n : m_graph->nodes())
        {
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                auto sg_order = sg_order_dfs(sgn->subgraph());
                new_orders.push_back(sg_order);
            }
        }
        m_graph->topological_sort(new_orders);
    }
    else if (strategy == 1)
    {
        std::vector<std::vector<NodePtr>> new_orders;
        for (auto n : m_graph->nodes())
        {
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                auto sg = sgn->subgraph();
                auto sg_order = sg_order_optimize(sg);
                new_orders.push_back(sg_order);
            }
        }
        m_graph->topological_sort(new_orders);
    }
    else if (strategy == 2)
    {
        std::vector<std::vector<NodePtr>> new_orders;
        // int cnt = 0;
        for (auto &level : m_levels)
        {
            level_branch_optimize(level);
            // cnt++;
        }
    }
    else if (strategy == 3)
    {
        std::vector<std::vector<NodePtr>> new_orders;
        // int cnt = 0;
        for (auto &level : m_levels)
        {

            level_stage_optimize(level);
            // cnt++;
        }
    }

    if (BuilderConfig::get<bool>(Constants::dump_reorder_cfg))
    {
        utils::JsonObject level_info;
        for (size_t i = 0; i < m_levels.size(); ++i)
        {
            utils::JsonObject sg_node_info;
            auto level = m_levels[i];
            for (size_t j = 0; j < level.size(); ++j)
            {
                utils::JsonObject node_info;
                auto node = level[j];
                SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
                if (sgnode)
                {
                    int op_id = 0;
                    for (auto n : *sgnode->subgraph())
                    {
                        node_info[n->name()] = op_id;
                        op_id++;
                    }
                }

                sg_node_info[node->name()] = node_info;
            }
            level_info["level" + std::to_string(i)] = sg_node_info;
        }
        std::ofstream os("reorder.json");
        if (os.is_open())
        {
            os << level_info;
            os << std::endl;
            os.close();
        }
    }
}

void ReorderOptimizer::update_w_map(GraphPtr g, std::vector<NodePtr> &skip_nodes, std::map<NodePtr, uint64_t> &w_map)
{
    std::set<NodePtr> candidates;
    std::set<NodePtr> run(skip_nodes.begin(), skip_nodes.end());
    for (auto n : g->nodes())
    {
        if (n->parents().size() == 0)
        {
            candidates.insert(n);
        }
    }

    while (candidates.size())
    {
        // INFO() << " new loop: " << candidates.size();
        NodePtr root;
        for (auto n : candidates)
        {
            bool ready = true;
            for (auto p : n->parents())
            {
                if (!contains(run, p))
                {
                    // INFO() << n->name() << " cannot run due to " << p->name();
                    ready = false;
                }
            }
            if (ready)
            {
                root = n;
                // INFO() << n->name() << " can run";
                break;
            }
        }
        if (!root)
        {
            ERROR();
        }

        if (contains(skip_nodes, root))
        {
            w_map[root] = 0;
        }
        else
        {
            uint32_t total_mem = 0;
            for (auto p : root->parents())
            {
                total_mem += w_map[p];
            }
            for (auto o : root->outputs())
            {
                total_mem += o->mem_size();
            }
            w_map[root] = total_mem;
            // INFO() << "mem for: " << root->name() << " is " << total_mem;
        }

        candidates.erase(root);
        run.insert(root);
        for (auto node : root->children())
        {
            candidates.insert(node);
        }
    }
    return;
}

void ReorderOptimizer::best_order(uint32_t index, NodePtr n, std::vector<NodePtr> &new_order, std::map<NodePtr, uint64_t> &w_map, GraphPtr g)
{
    if (contains(new_order, n))
    {
        return;
    }
    auto it = new_order.begin() + index;
    new_order.insert(it, n);
    // INFO() << "insert " << n->name() << " in " << index;

    std::vector<NodePtr> parents;
    uint32_t new_index = index;

    for (auto p : n->parents())
    {
        if (!contains(new_order, p))
        {
            parents.push_back(p);
        }
    }
    if (parents.size() > 1)
    {
        NodePtr last_node;
        NodePtr cur_node;
        while (parents.size())
        {
            std::stable_sort(parents.begin(), parents.end(), [&w_map](NodePtr a, NodePtr b) {
                return w_map.at(a) < w_map.at(b);
            });

            cur_node = parents[0];
            if (last_node)
            {
                auto it = std::find(new_order.begin(), new_order.end(), last_node);
                if (it != new_order.end())
                {
                    new_index = std::distance(new_order.begin(), it) + 1;
                }
                else
                {
                    ERROR();
                }
            }
            best_order(new_index, cur_node, new_order, w_map, g);
            parents.erase(parents.begin());
            last_node = cur_node;
        }
    }
    else if (parents.size() > 0)
    {
        best_order(new_index, parents[0], new_order, w_map, g);
    }
    else
    {
        update_w_map(g, new_order, w_map);
    }
}

static bool dfs(NodePtr node, std::set<NodePtr> &visited, std::set<NodePtr> &visiting, std::stack<NodePtr> &result)
{
    if (visiting.find(node) != visiting.end())
    {
        return false;
    }

    if (visited.find(node) != visited.end())
    {
        return true;
    }

    visiting.insert(node);

    for (NodePtr child : node->children())
    {
        if (!dfs(child, visited, visiting, result))
        {
            return false;
        }
    }

    visiting.erase(node);
    visited.insert(node);
    result.push(node);

    return true;
}

std::vector<NodePtr> ReorderOptimizer::sg_order_dfs(GraphPtr g)
{
    std::map<std::string, NodePtr> nodes;
    std::set<NodePtr> visited;
    std::set<NodePtr> visiting;
    std::stack<NodePtr> result;

    for (auto &node : g->nodes())
    {
        if (visited.find(node) == visited.end())
        {
            if (!dfs(node, visited, visiting, result))
            {
                return {};
            }
        }
    }

    std::vector<NodePtr> sorted;
    while (!result.empty())
    {
        sorted.push_back(result.top());
        result.pop();
    }

    return sorted;
}

std::vector<NodePtr> ReorderOptimizer::sg_order_optimize(GraphPtr g)
{

    std::vector<NodePtr> new_order;
    std::vector<NodePtr> bottom_nodes;
    for (auto n : g->nodes())
    {
        if (n->children().size() == 0)
        {
            bottom_nodes.push_back(n);
        }
    }
    std::map<NodePtr, uint64_t> w_map;
    update_w_map(g, new_order, w_map);
    if (bottom_nodes.size() > 1)
    {
        NodePtr last_node;
        NodePtr cur_node;
        uint32_t new_index = 0;
        while (bottom_nodes.size())
        {
            std::stable_sort(bottom_nodes.begin(), bottom_nodes.end(), [&w_map](NodePtr a, NodePtr b) {
                return w_map.at(a) < w_map.at(b);
            });

            cur_node = bottom_nodes[0];
            if (last_node)
            {
                auto it = std::find(new_order.begin(), new_order.end(), last_node);
                if (it != new_order.end())
                {
                    new_index = std::distance(new_order.begin(), it) + 1;
                }
                else
                {
                    ERROR();
                }
            }
            best_order(new_index, cur_node, new_order, w_map, g);
            bottom_nodes.erase(bottom_nodes.begin());
            last_node = cur_node;
        }
    }
    else if (bottom_nodes.size() > 0)
    {
        best_order(0, bottom_nodes[0], new_order, w_map, g);
    }
    else
    {
        ERROR();
    }
    DEBUG() << "branch reorder optimization new order:";
    for (auto n : new_order)
    {
        DEBUG() << n->name();
    }

    // g->set_order(new_order);
    return new_order;
}

std::vector<std::vector<NodePtr>> get_branch(GraphPtr g)
{
    std::vector<NodePtr> bottom_nodes;
    std::vector<std::vector<NodePtr>> branches;
    std::vector<NodePtr> branch;
    for (auto n : g->nodes())
    {
        branch.push_back(n);
        if (n->children().size() == 0)
        {
            branches.push_back(branch);
            branch.clear();
        }
    }
    return branches;
}

std::vector<int> find_common_branches(NodePtr n, std::vector<std::vector<NodePtr>> &branches)
{
    std::vector<int> ret;
    for (size_t i = 0; i < branches.size(); ++i)
    {
        auto branch = branches[i];
        if (contains(branch, n))
        {
            if (!contains(ret, i))
            {
                ret.push_back(i);
            }
            break;
        }
        for (auto p : n->parents())
        {
            if (p->type() == OpType_SubGraphInput)
            {
                continue;
            }
            if (contains(branch, p))
            {
                if (!contains(ret, i))
                {
                    ret.push_back(i);
                }
                break;
            }
        }
    }
    return ret;
}

std::vector<std::vector<std::vector<NodePtr>>> get_stage(GraphPtr g)
{
    std::vector<std::vector<std::vector<NodePtr>>> stages;
    // std::vector<NodePtr> input_nodes;
    std::vector<std::vector<NodePtr>> branches;
    std::set<NodePtr> start_n;
    for (auto n : g->nodes())
    {
        if (contains(start_n, n))
        {
            continue;
        }
        if (n->parents().size() == 0)
        {
            if (n->children().size() == 0)
            {
                start_n.insert(n);
                branches.push_back({n});
            }
            else
            {
                for (auto c : n->children())
                {
                    bool is_input_n = true;
                    for (auto p : c->parents())
                    {
                        if (p->parents().size() != 0)
                        {
                            is_input_n = false;
                            break;
                        }
                    }
                    if (is_input_n)
                    {
                        std::vector<NodePtr> branch;
                        for (auto p : c->parents())
                        {
                            if (!contains(start_n, p))
                            {
                                start_n.insert(p);
                                branch.push_back(p);
                            }
                        }
                        branch.push_back(c);
                        branches.push_back(branch);
                    }
                }
            }
        }
    }

    if (branches.size() == 0)
    {
        ERROR();
    }
    else if (branches.size() == 1)
    {
        branches[0] = g->nodes();
    }
    else
    {
        for (auto n : g->nodes())
        {
            // INFO() << "DEBUG " << n;
            if (n->name() == "subgraph_matmul_76")
            {
                INFO();
            }
            auto common_branches_idx = find_common_branches(n, branches);
            if (n->type() == OpType_SubGraphInput)
            {
                if (common_branches_idx.size() == 0)
                {
                    branches[0].push_back(n);
                }
                continue;
            }
            std::sort(common_branches_idx.begin(), common_branches_idx.end(), std::greater<int>());
            if (common_branches_idx.size() == 1)
            {
                auto idx = common_branches_idx[0];
                if (!contains(branches[idx], n))
                {
                    branches[idx].push_back(n);
                }
            }
            else if (common_branches_idx.size() > 1)
            {
                std::vector<std::vector<NodePtr>> common_branches;
                for (auto idx : common_branches_idx)
                {
                    common_branches.push_back(branches[idx]);
                    branches.erase(branches.begin() + idx);
                }
                branches.push_back({n}); //new branches
                stages.push_back(common_branches);
            }
            else
            {
                ERROR();
            }
        }
    }

    stages.push_back(branches);

    return stages;
}

int get_similar_score(std::vector<NodePtr> &branch1, std::vector<NodePtr> &branch2)
{
    int score = 0;
    std::map<std::string, std::string> constant_hex;
    for (auto n : branch1)
    {
        for (auto kv : n->constants())
        {
            if (!kv.second)
            {
                continue;
            }
            if (kv.second->attrs().count(Constants::hash))
            {
                std::string hex = kv.second->get_attr(Constants::hash);
                constant_hex[hex] = n->name();
            }
        }
    }
    int match = 0;
    for (auto n : branch2)
    {
        for (auto kv : n->constants())
        {
            if (!kv.second)
            {
                continue;
            }
            if (kv.second->attrs().count(Constants::hash))
            {
                std::string hex = kv.second->get_attr(Constants::hash);
                if (constant_hex.count(hex))
                {
                    match++;
                    // INFO() << "match " << constant_hex[hex] << " and " << n->name();
                }
            }
        }
    }

    score = constant_hex.size() ? match * 100 / constant_hex.size() : 0;

    return score;
}

std::vector<NodePtr> get_similar_branch(std::vector<NodePtr> &ref_branch, std::vector<std::vector<NodePtr>> &branches)
{
    std::vector<int> scores;
    for (size_t i = 0; i < branches.size(); ++i)
    {
        auto branch = branches[i];
        auto score = get_similar_score(ref_branch, branch);
        scores.push_back(score);
    }

    size_t maxIndex = 0;
    for (size_t i = 1; i < scores.size(); ++i)
    {
        if (scores[i] > scores[maxIndex])
        {
            maxIndex = i;
        }
    }
    auto result = branches[maxIndex];
    branches.erase(branches.begin() + maxIndex);
    return result;
}

void ReorderOptimizer::update_branch(NodePtr n, std::vector<std::vector<NodePtr>> &ref_branches)
{
    auto sgn = SubGraphNode::from_node(n);
    if (sgn)
    {
        auto sg = sgn->subgraph();
        auto branches = get_branch(sg);
        // if(m_show)
        // {
        //     INFO() << "get branch for level[1]";
        //     for(size_t i = 0; i < branches.size(); ++i)
        //     {
        //         INFO() << "branch " << i;
        //         for(auto n : branches[i])
        //         {
        //             INFO() << n->name();
        //         }

        //     }
        // }
        // INFO() << "branch size " << branches.size() << " ref branches size " << ref_branches.size();
        if (branches.size() != ref_branches.size())
        {
            return;
        }
        HashConstant()(sg);
        std::vector<NodePtr> new_branch;
        // if(m_show)
        // {
        //     INFO() << "updated branch for level[1]";
        // }
        // int cnt = 0;
        for (auto ref_branch : ref_branches)
        {
            auto branch = get_similar_branch(ref_branch, branches);
            // INFO() << "similar of branch";
            // for(auto n : ref_branch)
            // {
            //     INFO() << n->name();
            // }
            // INFO() << "is";
            // for(auto n : branch)
            // {
            //     INFO() << n->name();
            // }
            new_branch.insert(new_branch.end(), branch.begin(), branch.end());

            // if(m_show)
            // {

            //     INFO() << "branch " << cnt;
            //     for(auto n : branch)
            //     {
            //         INFO() << n->name();
            //     }
            //     cnt++;
            // }
        }

        // sg->topological_sort(new_branches);
        sg->sort_by_ordered(new_branch);
        // if(m_show)
        // {
        //     INFO() << "new branch for sg";
        //     auto sg_branches = get_branch(sg);
        //     for(size_t i = 0; i < sg_branches.size(); ++i)
        //     {
        //         INFO() << "branch " << i;
        //         for(auto n : sg_branches[i])
        //         {
        //             INFO() << n->name();
        //         }

        //     }
        // }
    }
}

void order_by_stage(GraphPtr g, std::vector<std::vector<std::vector<NodePtr>>> &stages)
{
    std::vector<NodePtr> new_branch;
    for (auto stage : stages)
    {
        for (auto branch : stage)
        {
            new_branch.insert(new_branch.end(), branch.begin(), branch.end());
        }
    }
    g->sort_by_ordered(new_branch);
}

void ReorderOptimizer::update_stage(NodePtr n, std::vector<std::vector<std::vector<NodePtr>>> &ref_stages)
{
    auto sgn = SubGraphNode::from_node(n);
    if (sgn)
    {
        auto sg = sgn->subgraph();
        auto stages = get_stage(sg);
        if (stages.size() != ref_stages.size())
        {
            return;
        }
        for (size_t i = 0; i < stages.size(); ++i)
        {
            auto branches = stages[i];
            auto ref_branches = ref_stages[i];
            if (branches.size() != ref_branches.size())
            {
                continue;
            }
            HashConstant()(sg);
            stages[i].clear();
            for (auto ref_branch : ref_branches)
            {
                auto branch = get_similar_branch(ref_branch, branches);
                stages[i].push_back(branch);
            }
        }
        // INFO() << "order by stage for " << n->name();
        order_by_stage(sg, stages);
    }
}

void ReorderOptimizer::level_branch_optimize(std::vector<NodePtr> &level)
{
    if (level.size() < 2)
    {
        return;
    }

    for (size_t i = 0; i < level.size(); ++i)
    {
        auto sgn = SubGraphNode::from_node(level[i]);
        if (sgn)
        {
            auto sg = sgn->subgraph();
            auto branches = get_branch(sg);
            // if(m_show)
            // {
            //     INFO() << "get branch for level[0]";
            //     for(size_t i = 0; i < branches.size(); ++i)
            //     {
            //         INFO() << "branch " << i;
            //         for(auto n : branches[i])
            //         {
            //             INFO() << n->name();
            //         }

            //     }
            // }
            if (branches.size() > 1)
            {
                HashConstant()(sg);
                for (size_t j = i + 1; j < level.size(); ++j)
                {
                    update_branch(level[j], branches);
                }
            }
        }
    }
}

void ReorderOptimizer::level_stage_optimize(std::vector<NodePtr> &level)
{
    if (level.size() < 2)
    {
        return;
    }

    for (size_t i = 0; i < level.size() - 1; ++i)
    {
        auto sgn = SubGraphNode::from_node(level[i]);
        if (sgn)
        {
            auto sg = sgn->subgraph();
            auto stages = get_stage(sg);

            if (stages.size() == 1 && stages[0].size() == 1)
            {
                continue;
            }

            // INFO() << "order by stage for " << level[i]->name();
            // order_by_stage(sg, stages);

            HashConstant()(sg);
            for (size_t j = i + 1; j < level.size(); ++j)
            {
                update_stage(level[j], stages);
            }
        }
    }
}

} // namespace aipubt
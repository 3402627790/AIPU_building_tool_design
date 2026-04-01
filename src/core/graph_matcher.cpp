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
 * @Date: 2020-01-29 14:27:49
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-10-15 17:54:45
 */
#include "core.h"
#include <algorithm>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
using namespace std;

namespace aipubt
{

GraphMatcher::GraphMatcher(vector<pair<string, int>> nodes, vector<pair<string, string>> edges)
{
    // build graph
    map<string, NodePtr> node_dict;
    for (auto n : nodes)
    {
        NodePtr node = Node::create(n.first, (OpType)n.second);
        m_nodes.push_back(node);
        node_dict[n.first] = node;
    }
    for (auto e : edges)
    {
        NodePtr n0 = node_dict[e.first];
        NodePtr n1 = node_dict[e.second];
        n0->add_child_helper(n1);
    }
    vector<NodePtr> in_nodes;
    for (auto n : m_nodes)
    {
        if (0 == n->parents().size())
        {
            in_nodes.push_back(n);
        }
    }
    // build path
    if (!in_nodes.size())
    {
        return;
    }
    m_path.push_back(in_nodes[0]);
    deep_first_search(m_path, m_path[0]);
    for (unsigned int i = 1; i < in_nodes.size(); ++i)
    {
        vector<NodePtr> sub_path;
        vector<int> distance;
        NodePtr node = in_nodes[i];
        deep_first_search(sub_path, node, &distance, 0);

        vector<NodePtr> connections;
        for (auto n : m_path)
        {
            if (contains(sub_path, n))
            {
                connections.push_back(n);
            }
        }
        if (!connections.size())
        {
            m_path.clear();
            ERROR() << "Error, this is not a complete subgraph, cannot matcher!";
            return;
            // throw invalid_argument("Error, this is not a complete subgraph");
        }
        vector<int> cd;
        int min_dis = INT32_MAX;
        NodePtr min_root = nullptr;
        for (auto n : connections)
        {
            for (unsigned int idx = 0; idx < sub_path.size(); idx++)
            {
                if (n == sub_path[idx] && distance[idx] < min_dis)
                {
                    min_dis = distance[idx];
                    min_root = n;
                }
            }
        }
        m_roots.push_back(min_root);
        root_path[min_root->name()] = {min_dis, node};

        m_path.push_back(node);
        deep_first_search(m_path, node);
    }

    // build node/path map
    for (auto n : m_nodes)
    {
        int idx = 0;
        for (auto pn : m_path)
        {
            if (pn == n)
            {
                break;
            }
            idx++;
        }
        m_node_path_map.push_back(idx);
    }
}

GraphMatcher::~GraphMatcher()
{
    m_nodes.clear();
}
void GraphMatcher::deep_first_search(vector<NodePtr> &path, NodePtr root, vector<int> *distance, int depth)
{
    for (auto n : root->children())
    {
        if (!contains(path, n))
        {
            path.push_back(n);
            if (distance)
            {
                distance->push_back(depth + 1);
            }
            deep_first_search(path, n, distance, depth + 1);
        }
    }
}

bool GraphMatcher::check_node_children(NodePtr ref, NodePtr n)
{
    if (0 == ref->children().size())
    {
        return true;
    }
    bool m_relaxed = m_mode & RelaxedMode;
    if ((m_relaxed && ref->children().size() > n->children().size()) || (!m_relaxed && ref->children().size() != n->children().size()))
    {
        return false;
    }
    std::map<OpType, int> counter;
    counter[OpType_NoOp] = 0;
    for (auto child : n->children())
    {
        OpType optype = get_single_real_op_type(child);
        if (!counter.count(optype))
        {
            counter[optype] = 0;
        }
        counter[optype]++;
    }
    for (auto child : ref->children())
    {
        OpType optype = get_single_real_op_type(child);
        if (!counter.count(optype))
        {
            counter[optype] = 0;
        }
        counter[optype]--;
    }
    // bool ret = true;
    int any_op = counter[OpType_NoOp];
    counter[OpType_NoOp] = 0;
    for (auto p : counter)
    {
        if (p.second < 0)
        {
            return false;
        }
        any_op += p.second;
    }
    return any_op == ((int)n->children().size() - (int)ref->children().size());
}

void GraphMatcher::back_search(vector<NodePtr> &path, NodePtr root, vector<NodePtr> &top, int max_depth)
{
    if (0 == max_depth)
    {
        top.push_back(root);
        return;
    }
    for (auto n : root->parents())
    {
        if (!contains(path, n))
        {
            path.push_back(root);
            back_search(path, n, top, max_depth - 1);
        }
    }
}

std::vector<NodePtr> GraphMatcher::get_input_with_root(NodePtr root, NodePtr ref)
{
    auto de_top = root_path[ref->name()];
    std::vector<NodePtr> top;
    std::vector<NodePtr> path;
    back_search(path, root, top, de_top.first);
    if (get_single_real_op_type(de_top.second) == OpType_NoOp)
    {
        return top;
    }
    std::vector<NodePtr> ret;
    for (auto n : top)
    {
        if (get_single_real_op_type(n) == get_single_real_op_type(de_top.second))
        {
            ret.push_back(n);
        }
    }
    return ret;
}

std::pair<bool, vector<NodePtr>> GraphMatcher::pattern_match(NodePtr root, vector<NodePtr> &path, vector<NodePtr> &match_g)
{
    bool ret = false;
    unsigned int idx = match_g.size();
    if (idx == path.size())
    {
        ret = true;
    }
    else if (idx < path.size() && (get_single_real_op_type(root) == path[idx]->type() || path[idx]->type() == OpType_NoOp))
    {
        if (!check_node_children(path[idx], root))
        {
            return {false, match_g};
        }
        match_g.push_back(root);
        std::vector<NodePtr> branches;
        if (path[idx]->children().size() != 0)
        {
            branches = root->children();
        }
        if (!branches.size())
        {
            if (path[idx]->children().size() != 0)
            {
                return {false, match_g};
            }
            return {true, match_g};
        }
        else
        {
            ret = backtracing(root, branches, path, match_g);
            if (ret && match_g.size() != path.size())
            {
                for (auto r : m_roots)
                {
                    idx = 0;
                    for (auto pn : path)
                    {
                        if (r == pn)
                        {
                            break;
                        }
                        idx++;
                    }
                    if (idx < match_g.size())
                    {
                        for (auto p : get_input_with_root(match_g[idx], r))
                        {
                            if (!contains(match_g, p))
                            {
                                pattern_match(p, path, match_g);
                            }
                        }
                    }
                }
            }
        }
    }

    return {ret, match_g};
}

bool GraphMatcher::backtracing(NodePtr root, vector<NodePtr> &branches, vector<NodePtr> &path, vector<NodePtr> &match_g)
{
    auto match_g_bk = match_g;
    auto need_to_try = branches;
    // std::vector<NodePtr > removed_branch;
    for (auto b : branches)
    {
        if (contains(match_g, b))
        {
            need_to_try.erase(std::remove(need_to_try.begin(), need_to_try.end(), b), need_to_try.end());
            if (0 == need_to_try.size())
            {
                return true;
            }
            return backtracing(root, need_to_try, path, match_g);
        }
        auto mg = pattern_match(b, path, match_g);
        if (mg.first)
        {
            need_to_try.erase(std::remove(need_to_try.begin(), need_to_try.end(), b), need_to_try.end());
            if (0 == need_to_try.size())
            {
                return true;
            }
            bool ok = backtracing(root, need_to_try, path, match_g);
            if (ok)
            {
                return true;
            }
        }
        match_g = match_g_bk;
    }
    return false;
}

std::vector<std::vector<NodePtr>> GraphMatcher::match(GraphPtr g)
{
    std::vector<std::vector<NodePtr>> ret;
    std::set<NodePtr> viewed;
    const bool withany = (m_mode & WithAnyMode);
    const bool greedy = m_mode & GreedyMode;
    for (auto n : g->nodes())
    {
        if ((get_single_real_op_type(n) == m_path[0]->type() || m_path[0]->type() == OpType_NoOp))
        {
            if (!greedy && contains(viewed, n))
            {
                continue;
            }
            std::vector<NodePtr> sg;
            auto m = pattern_match(n, m_path, sg);
            if (m.first && m.second.size() == m_nodes.size())
            {
                std::vector<NodePtr> msg;
                for (auto idx : m_node_path_map)
                {
                    if (m_path[idx]->type() != OpType_NoOp ||
                        (m_path[idx]->type() == OpType_NoOp && withany))
                    {
                        msg.push_back(m.second[idx]);
                        viewed.insert(m.second[idx]);
                    }
                }
                ret.push_back(msg);
            }
        }
    }
    std::sort(ret.begin(), ret.end(), [](const std::vector<NodePtr> &a, const std::vector<NodePtr> &b)
              {
        string a_name, b_name;
        for (auto n : a)
        {
            a_name += n->name();
        }
        for (auto n : b)
        {
            b_name += n->name();
        }
        return a_name < b_name; });
    return ret;
}

OpType GraphMatcher::get_single_real_op_type(NodePtr n)
{
    OpType t = n->type();
    if ((m_mode & SingleOpSubgraphMode) && (t == OpType_SubGraph))
    {
        SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(n);
        auto nodes = sgn->subgraph()->nodes();
        if (nodes.size() == 1)
        {
            t = get_single_real_op_type(nodes[0]);
        }
    }
    return t;
}

}; // namespace aipubt

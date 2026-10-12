//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-05-26 17:25:56
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-24 17:09:16
 */
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "transform/passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static bool is_expand_node(const NodePtr &n)
{
    std::vector<std::string> passes;
    if (n->loc() != nullptr)
        passes = n->loc()->all();

    if (contains(passes, "pre_tile"))
    {
        return true;
    }
    return false;
}

static int32_t check_real_equal(uint32_t i, std::vector<uint32_t> &vec, const std::vector<NodePtr> &nodes)
{
    auto node1 = nodes.at(i);
    for (auto j : vec)
    {
        auto node2 = nodes.at(j);
        if (node1->equal(node2))
        {
            return j;
        }
    }
    return -1;
}

inline void hash_combine(std::size_t &seed, const std::string &val)
{
    std::hash<std::string> hasher;
    seed = seed * 31 + hasher(val);
}

inline void hash_combine(std::size_t &seed, size_t val)
{
    std::hash<size_t> hasher;
    seed = seed * 31 + hasher(val);
}
static uint32_t get_hash(NodePtr node, std::map<NodePtr, uint32_t> &hash_map)
{
    if (hash_map.find(node) != hash_map.end())
    {
        return hash_map[node];
    }

    std::size_t seed = 0;
    hash_combine(seed, node->type());
    for (const auto &input : node->inputs())
    {
        hash_combine(seed, input->name());
    }
    if (node->constants().count("weights"))
    {
        std::vector<std::string> wt;
        TensorPtr weight = node->constants()["weights"];
        for (auto c : node->constants())
        {
            wt.push_back(c.second->name());
        }
        std::stable_sort(wt.begin(), wt.end());
        for (auto w : wt)
        {
            hash_combine(seed, w);
        }
    }
    for (auto o : node->outputs())
    {
        for (auto s : o->shape())
        {
            hash_combine(seed, s);
        }
    }

    hash_map[node] = seed;

    return seed;
}

static std::vector<NodePtrList> criteria(NodePtr n, std::map<NodePtr, uint32_t> &hash_map)
{
    std::vector<NodePtrList> ret;
    std::set<uint32_t> already_merged;
    bool enable_cse_tile_node = BuilderConfig::get<bool>(Constants::enable_cse_tile_node);
    std::map<uint32_t, std::vector<uint32_t>> hash_index_map;
    for (uint32_t i = 0; i < n->children().size(); ++i)
    {
        if (already_merged.count(i) || is_expand_node(n->children()[i]))
        {
            continue;
        }
        auto child_i = n->children()[i];
        std::string tsg_name_i = child_i->get_attr(Constants::tsg_name);
        if (child_i->type() == OpType_Reshape && child_i->outputs()[0]->is_io_tensor())
        {
            continue;
        }

        if (child_i->type() == OpType_DynamicQuantize &&
            child_i->outputs()[0]->consumers().size() == 1 &&
            child_i->outputs()[0]->consumers()[0]->type() == OpType_FullyConnected)
        {
            continue;
        }
        uint32_t hash = get_hash(child_i, hash_map);
        bool is_equal = hash_index_map.find(hash) != hash_index_map.end();

        if (is_equal)
        {
            auto vec = hash_index_map[hash];
            auto index = check_real_equal(i, vec, n->children());
            if (index == (int)i)
            {
                ERROR() << "same index";
            }
            if (index < 0)
            {
                hash_index_map[hash].push_back(i);
            }
            else
            {
                auto child_j = n->children()[index];
                bool not_equal = false;
                if (!enable_cse_tile_node)
                {
                    std::string tsg_name_j = child_j->get_attr(Constants::tsg_name);
                    if (tsg_name_i != tsg_name_j)
                    {
                        not_equal = true;
                    }
                }

                if (!not_equal)
                {
                    ret.push_back({child_j, child_i});
                    already_merged.insert(index);
                    already_merged.insert(i);
                }
            }
        }
        else
        {
            hash_index_map[hash].push_back(i);
        }
    }
    return ret;
}

// common subexpression elimination for eliminating common subnode
bool CSE_subnode(GraphPtr g)
{
    bool is_changed = false;
    std::map<NodePtr, uint32_t> hash_map;
    while (true)
    {
        bool is_merged = false;
        std::set<NodePtr> removed_node;
        auto nodes = g->nodes();
        for (auto node : nodes)
        {
            if (removed_node.count(node))
            {
                continue;
            }
            auto matched = criteria(node, hash_map);
            if (matched.size() == 0)
            {
                continue;
            }
            for (auto m : matched)
            {
                auto n0 = m[0];
                auto n1 = m[1];
                for (uint32_t i = 0; i < n1->outputs().size(); ++i)
                {
                    auto o = n1->outputs()[i];
                    if (o->is_io_tensor())
                    {
                        auto reshape_out = ops::reshape(n0->outputs()[i], n1->outputs()[i]->shape());
                        auto reshape_node = reshape_out->node();
                        reshape_node->remove_output(0);
                        reshape_node->add_output(o);

                        g->add_node(reshape_node);
                        ADD_LOCATION_BY_NODE(reshape_node, n1);
                    }
                    else
                    {
                        while (o->consumers().size())
                        {
                            for (auto c : o->consumers())
                            {
                                c->add_input(n0->outputs()[i], c->remove_input(o));
                            }
                        }
                    }
                }
                g->remove_node(n1);
                removed_node.insert(n1);
                is_changed = true;
                is_merged = true;
            }
        }
        if (!is_merged)
        {
            break;
        }
    }
    return is_changed;
}
}; // namespace aipubt

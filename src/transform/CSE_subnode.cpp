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
 * @Date: 2020-05-26 17:25:56
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-24 17:09:16
 */
 #include "core/config.h"
 #include "core/core.h"
 #include "core/ops/ops.h"
 #include "passes.h"
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
    
    static uint32_t hash_node(bool clean = false)
    {
        static uint32_t i = 1;
        if (clean)
        {
            i = 1;
        }
        return i++;
    }
    // suppose hash_map has all nodes
    static bool hash_equal(std::map<NodePtr, uint32_t> &hash_map, NodePtr node1, NodePtr node2)
    {
        auto id0 = hash_map.at(node1);
        auto id1 = hash_map.at(node2);
        if (id0 == id1)
        {
            return true;
        }
        if (node1->equal(node2))
        {
            //use small id as ref
            if (id0 < id1)
            {
                hash_map[node1] = id1;
            }
            else
            {
                hash_map[node2] = id0;
            }
            return true;
        }
        return false;
    }
    
    static std::vector<NodePtrList> criteria(NodePtr n, std::map<NodePtr, uint32_t> &hash_map)
    {
        std::vector<NodePtrList> ret;
        std::set<uint32_t> already_merged;
        bool enable_cse_tile_node = BuilderConfig::get<bool>(Constants::enable_cse_tile_node);
        if (n->children().size() > 1)
        {
            for (uint32_t i = 0; i < n->children().size(); ++i)
            {
                if (already_merged.count(i) || is_expand_node(n->children()[i]))
                {
                    continue;
                }
                auto child_i = n->children()[i];
                std::string tsg_name_i = child_i->get_attr(Constants::tsg_name);
                for (uint32_t j = i + 1; j < n->children().size(); ++j)
                {
                    if (already_merged.count(j))
                    {
                        continue;
                    }
                    auto child_j = n->children()[j];
                // avoid dead loop when full simplify is enabled and working with fuse_reshape
                if (child_j->type() == OpType_Reshape && child_j->outputs()[0]->is_io_tensor())
                {
                    continue;
                }
                std::string tsg_name_j = child_j->get_attr(Constants::tsg_name);
                bool is_equal = hash_equal(hash_map, n->children()[i], n->children()[j]);

                // default: don't merge nodes between tile slices
                // enable: merge it (e.g. multi crop)
                if (!enable_cse_tile_node)
                {
                    is_equal = is_equal && (tsg_name_i == tsg_name_j);
                }

                if (is_equal)
                {
                    for (uint32_t k = 0; k < n->children()[i]->inputs().size(); ++k)
                    {

                        if (n->children()[j]->inputs()[k] != n->children()[i]->inputs()[k])
                        {
                            is_equal = false;
                            break;
                        }
                    }
                }
                if (is_equal)
                {
                    ret.push_back({n->children()[i], n->children()[j]});
                    already_merged.insert(j);
                    already_merged.insert(i);
                    break;
                }
            }
        }
    }
    return ret;
}
// common subexpression elimination for eliminating common subnode
bool CSE_subnode(GraphPtr g)
{
    bool is_changed = false;
    std::map<NodePtr, uint32_t> hash_map;

    //init hash_map
    hash_node(true);
    for (auto node : g->nodes())
    {
        hash_map[node] = hash_node();
    }
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
                        hash_map[reshape_node] = hash_node();
                        ADD_LOCATION_BY_NODE(reshape_node, n1);
                    }
                    else
                    {
                        while (o->consumers().size())
                        {
                            for (auto c : o->consumers())
                            {
                                c->add_input(n0->outputs()[i], c->remove_input(o));
                                hash_map[c] = hash_node(); // Update hash_map for the modified consumer nodes
                            }
                        }
                    }
                }

                g->remove_node(n1);
                hash_map.erase(n1); // Remove the old node from hash_map
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

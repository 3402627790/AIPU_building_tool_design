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
 * @Date: 2020-02-09 12:15:07
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-11-01 11:29:55
 */
#ifndef CORE_GRAPH_MATCHER_H
#define CORE_GRAPH_MATCHER_H

#include "core/optype.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace aipubt
{
class Graph;
class Node;
using NodePtr = std::shared_ptr<Node>;

using GraphMatcherPattern = std::pair<std::vector<std::pair<std::string, int>>,          // nodes
                                      std::vector<std::pair<std::string, std::string>>>; // edges
class GraphMatcher
{
public:
    /**
     * @brief The match mode for graph matcher
     *
     * StrictMode: the pattern must strictly match, all internal edges must be same as pattern.
     * RelaxedMode: the graph is match the pattern, but with more internal edges. for example:
     *              pattern= n0->n1, n1->n2
     *              graph= n0->n1,n1->n2,n0->n2 this graph can match the pattern
     * WithAnyNode: the NoOp would regards as a Any type of op, it is a wildcard type.
     * GreedyMode: it would regardless the overlay between two matches, for example:
     *          pattern = Type0->Type1
     *          graph = n0(Type0)->n1(Type1), n0->n2(Type1), n3(Type0)->n1
     *      in this case, we will give tree matched: n0->n1, n0->n2, n3->n1.
     *      if not in greedy mode, the return: n0->n1 or (n0->n2,n3->n1), none of viewed node can be do second match.
     */
    enum MatchMode
    {
        StrictMode = 0,
        RelaxedMode = 0b1,
        WithAnyMode = 0b10,
        GreedyMode = 0b100,
        SingleOpSubgraphMode = 0b1000,

    };
    GraphMatcher(GraphMatcherPattern pattern) : GraphMatcher(pattern.first, pattern.second) {}
    GraphMatcher(std::vector<std::pair<std::string, int>> nodes, std::vector<std::pair<std::string, std::string>> edges);
    ~GraphMatcher();
    std::vector<std::vector<NodePtr>> match(GraphPtr g);

    std::vector<NodePtr> match_nodes(NodePtr root);

    // default is relaxed mode.
    // if relaxed mode: the ref's children can be < smaller than graph's node's children
    // if not: ref's children must strictly equal to the matched one.
    void set_mode(uint32_t b)
    {
        m_mode = b;
    }
    uint32_t mode()
    {
        return m_mode;
    }

private:
    void deep_first_search(std::vector<NodePtr> &path, NodePtr root, std::vector<int> *distance = nullptr, int depth = 0);
    bool backtracing(NodePtr root, std::vector<NodePtr> &branches, std::vector<NodePtr> &path, std::vector<NodePtr> &match_g);
    bool check_node_children(NodePtr ref, NodePtr node);
    void back_search(std::vector<NodePtr> &path, NodePtr root, std::vector<NodePtr> &top, int max_depth);
    std::vector<NodePtr> get_input_with_root(NodePtr root, NodePtr ref);
    std::pair<bool, std::vector<NodePtr>> pattern_match(NodePtr root, std::vector<NodePtr> &path, std::vector<NodePtr> &match_g);
    OpType get_single_real_op_type(NodePtr n);
    std::vector<NodePtr> m_nodes;
    std::vector<NodePtr> m_path;
    std::vector<NodePtr> m_roots;
    std::vector<int> m_node_path_map;
    std::map<std::string, std::pair<int, NodePtr>> root_path;
    uint32_t m_mode = StrictMode;
};
}; // namespace aipubt

#endif //! CORE_GRAPH_MATCHER_H
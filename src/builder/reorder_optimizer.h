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
#ifndef REORDER_OPTIMIZER_H
#define REORDER_OPTIMIZER_H

#include "core/core.h"
#include <set>
namespace aipubt
{

class ReorderOptimizer
{
public:
    ReorderOptimizer() {}
    ~ReorderOptimizer(){};

    void node_reorder(int strategy);
    void set_graph(GraphPtr g);
    void set_levels(std::vector<std::vector<NodePtr>> &levels);

private:
    GraphPtr m_graph;
    std::vector<std::vector<NodePtr>> m_levels;
    // int m_strategy;
    void update_w_map(GraphPtr g, std::vector<NodePtr> &skip_nodes, std::map<NodePtr, uint64_t> &w_map);
    void best_order(uint32_t index, NodePtr n, std::vector<NodePtr> &new_order, std::map<NodePtr, uint64_t> &w_map, GraphPtr g);
    std::vector<NodePtr> sg_order_optimize(GraphPtr g);
    std::vector<NodePtr> sg_order_dfs(GraphPtr g);
    void level_branch_optimize(std::vector<NodePtr> &level);
    void update_branch(NodePtr n, std::vector<std::vector<NodePtr>> &ref_branches);
    void level_stage_optimize(std::vector<NodePtr> &level);
    void update_stage(NodePtr n, std::vector<std::vector<std::vector<NodePtr>>> &ref_stages);
    // bool m_show = false;
};
} // namespace aipubt
#endif //!REORDER_OPTIMIZER_H
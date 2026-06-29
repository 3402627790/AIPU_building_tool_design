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
 * @Date: 2024-02-22 14:27:29 
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2024-02-22 11:16:43
 */
#ifndef BUILDER_LOAD_BALANCER_H
#define BUILDER_LOAD_BALANCER_H

#include "core/core.h"
#include <set>
namespace aipubt
{

class LoadBalancer
{
public:
    bool schedule();
    const std::vector<std::vector<NodePtr>> &get_levels();
    void set_graph(GraphPtr g);
    void set_core(int core);

protected:
    // merge_s split_s split_load_s
    std::vector<std::vector<uint32_t>> m_load_balance_strategy = {
        {0, 0, 0}, // 0: no split
        {0, 1, 0}, // 1: merge by core num, split output c
        {0, 1, 1}, // 1: merge by core num, split by tiling
        {0, 2, 2}, // 1: merge to 1 node, split batch
    };
    std::vector<std::vector<NodePtr>> m_levels;
    GraphPtr m_graph;
    int m_core;
};
};     // namespace aipubt
#endif //!BUILDER_MC_SCHEDULER_H

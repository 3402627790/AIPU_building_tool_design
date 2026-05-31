//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Ryan Lai
 * @Date: 2024-03-27 11:40:00
 * @Last Modified by: Ryan Lai
 * @Last Modified time: 2024-03-27 11:40:00
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>

/**
 * This pass base on:
 *  "multi transpose per switch axis" faster than
 *  "one transpose switch multi axis"
 *
 * e.g. swim_transformer have a transpose:
 * slow: perm[1,3,4,0,2]
 * fast: perm[1,3,0,2,4] + perm[0,1,4,2,3]
 *
 * PLEASE REMOVE ME after lib finish transpose update !
 * JIRA: CTL-8587
 * DOC: https://confluence.armchina.com/x/NV2LBw
 */

namespace aipubt
{

bool split_transpose_by_perm(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (OpType_Transpose != n->type())
            continue;

        std::vector<int> perm = n->params()["perm"];
        std::vector<int> perm_match = {1, 3, 4, 0, 2};

        if (perm != perm_match)
            continue;

        auto input = n->remove_input(0);
        auto out = n->remove_output(0);

        // transpose 1 don't swap channel
        auto transpose_1 = ops::transpose(input, {1, 3, 0, 2, 4})->node();
        g->add_node(transpose_1);
        ADD_LOCATION_BY_NODE(transpose_1, n);

        // transpose 2 swap channel only
        auto transpose_2 = ops::transpose(transpose_1->outputs()[0], {0, 1, 4, 2, 3})->node();
        g->add_node(transpose_2);
        ADD_LOCATION_BY_NODE(transpose_2, n);

        transpose_2->remove_output(0);
        transpose_2->add_output(out);
        g->remove_node(n);

        is_changed = true;
    }
    return is_changed;
}

} // namespace aipubt

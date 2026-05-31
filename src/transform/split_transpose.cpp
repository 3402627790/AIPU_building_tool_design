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
 * @Author: Chao Liu
 * @Date: 2022-05-24 11:25:43
 * @Last Modified by: Chao Liu
 * @Last Modified time: 2022-05-26 11:25:43
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>
namespace aipubt
{
static inline bool creteria(NodePtr n)
{
    if (n->outputs()[0]->is_io_tensor())
        return false;
    if (n->type() != OpType_Transpose)
        return false;
    std::vector<int> perm0 = n->params()["perm"];
    std::vector<int> const_perm = {0, 1, 3, 2};
    if (perm0 != const_perm)
    {
        return false;
    }
    return true;
}

bool split_transpose(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }
        /*origin [H,W,C1,C2]--->[H,W,C2,C1] perm=[0,1,3,2]

          new   [H,W,C1,C2]--->[C2,C1,H,W] perm=[3,2,0,1]
                [C2,C1,H,W]--->[1,C2*C1,H,W] reshape
                [1,C2*C1,H,W]--->[1,H,W,C2*C1] perm=[0,2,3,1]
                [1,H,W,C2*C1]--->[H,W,C2,C1] reshape
        */
        std::vector<int> inp_shape = n->inputs()[0]->shape();
        auto input = n->remove_input(0);
        auto out = n->remove_output(0);

        //add transpose node
        auto transpose_node0 = ops::transpose(input, {3, 2, 0, 1})->node();
        g->add_node(transpose_node0);
        ADD_LOCATION_BY_NODE(transpose_node0, n);

        //add reshape node0
        TensorShape reshape0_shape(1, inp_shape[3] * inp_shape[2], inp_shape[0], inp_shape[1]);
        reshape0_shape.set_layout(DataLayout_NHWC);
        auto reshape_node0 = ops::reshape(transpose_node0->outputs()[0], reshape0_shape)->node();
        g->add_node(reshape_node0);
        ADD_LOCATION_BY_NODE(reshape_node0, n);

        //add transpose node
        auto transpose_node1 = ops::transpose(reshape_node0->outputs()[0], {0, 2, 3, 1})->node();
        g->add_node(transpose_node1);
        ADD_LOCATION_BY_NODE(transpose_node1, n);

        TensorShape reshape1_shape(inp_shape[0], inp_shape[1], inp_shape[3], inp_shape[2]);
        reshape1_shape.set_layout(DataLayout_NHWC);
        auto reshape_node1 = ops::reshape(transpose_node1->outputs()[0], reshape1_shape)->node();
        g->add_node(reshape_node1);
        reshape_node1->remove_output(0);
        reshape_node1->add_output(out);
        ADD_LOCATION_BY_NODE(reshape_node1, n);
        g->remove_node(n);
        is_changed = true;
    }
    return is_changed;
}

} // namespace aipubt
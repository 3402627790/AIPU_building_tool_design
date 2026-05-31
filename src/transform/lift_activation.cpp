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
 * @Last Modified time: 2021-11-09 10:10:56
 */
#include "core/core.h"
#include "passes.h"
#include <algorithm>
#include <map>
#include <set>

namespace aipubt
{

static bool creteria(NodePtr node)
{
    // currently only lift when node has only one parent
    if (node->type() != OpType_Activation || node->parents().size() != 1 || node->inputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (node->parents()[0]->type() == OpType_Cast)
    {
        return false;
    }
    return true;
}

static bool lift_one_level(NodePtr node)
{
    auto parent = node->parents()[0];
    std::string method = node->params()["method"];
    if (method == Constants::act_prelu)
    {
        // prelu has some shape format:
        // 1. [c]
        // 2. [1,1,c] or [1,1,1,c]
        // 3. [h,w,c]
        // 4. [n,h,w,c]
        return false;
    }
    if (!creteria(node))
    {
        return false;
    }
    if (parent->attrs().count(Constants::view_op) && parent->get_attr<bool>(Constants::view_op) && parent->children().size() == 1 && parent->outputs().size() == 1)
    {
        auto inp = parent->inputs()[0];
        auto inter = parent->outputs()[0];
        auto out = node->outputs()[0];
        node->clear_inputs();
        node->clear_outputs();
        parent->clear_inputs();
        parent->clear_outputs();

        inter->shape() = inp->shape();
        inter->quantization() = out->quantization();
        inter->set_type(out->type());

        parent->add_input(inter);
        parent->add_output(out);

        node->add_input(inp);
        node->add_output(inter);
        return true;
    }
    return false;
}
/**
 * @brief lift_activation for a chance that it can be merged with conv/fc/etlwise etc.
 *
 * @param g
 * @return true if the graph changed
 * @return false
 */

bool lift_activation(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }
        while (lift_one_level(n))
        {
            is_changed = true;
        }
    }
    return is_changed;
}
}; // namespace aipubt

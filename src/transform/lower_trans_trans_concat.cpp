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
 * @Last Modified time: 2022-04-28 14:21:07
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>
#include <vector>

namespace aipubt
{

static bool criteria(const NodePtr n)
{
    if (n->type() == OpType_Concat)
    {
        for (auto inp : n->inputs())
        {
            if (inp->is_io_tensor())
            {
                return false;
            }
        }
        std::vector<int> perm;
        for (auto c : n->parents())
        {
            if (c->type() != OpType_Transpose || c->children().size() != 1)
            {
                return false;
            }
            if (perm.size() == 0)
            {
                perm = c->params()["perm"];
            }
            else
            {
                std::vector<int> p1 = c->params()["perm"];
                if (perm != p1)
                {
                    return false;
                }
            }
        }
        return true;
    }
    return false;
}
bool lower_trans_trans_concat(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto out = n->outputs()[0];
        auto inter = n->inputs()[0];
        auto trans = n->parents()[0];
        std::vector<TensorPtr> inputs;
        for (auto t : n->inputs())
        {
            inputs.push_back(t->node()->inputs()[0]);
        }
        while (n->parents().size() > 1)
        {
            g->remove_node(n->parents()[1]);
        }
        int axis = n->params()["axis"];
        if (axis < 0)
        {
            axis += out->shape().dim();
        }
        std::vector<int> perm = trans->params()["perm"];
        axis = perm[axis];
        for (uint32_t i = 0; i < perm.size(); ++i)
        {
            inter->shape()[perm[i]] = out->shape()[i];
        }
        n->params()["axis"] = axis;
        n->clear_inputs();
        n->add_input(inputs);
        n->clear_outputs();
        n->add_output(inter);
        n->infer_shape(true);
        MODIFY_NODE(n);
        trans->clear_inputs();
        trans->clear_outputs();
        trans->add_input(inter);
        trans->add_output(out);
        trans->infer_shape(true);
        MODIFY_NODE(trans);
        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt

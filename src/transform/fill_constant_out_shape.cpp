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
 * @Last Modified time: 2021-11-04 11:46:44
 */
#include "core/config.h"
#include "core/core.h"
#include "transform/passes.h"

namespace aipubt
{
static inline bool creteria(NodePtr n)
{
    if (n->is_dynamic())
    {
        return false;
    }

    if (n->type() != OpType_Constant)
    {
        return false;
    }

    if (!n->constants().count("weights"))
    {
        return false;
    }

    if (!n->constants()["weights"])
    {
        return false;
    }

    if (n->outputs()[0]->shape().dim() != 0 &&
        n->constants()["weights"]->shape().dim() != 0)
    {
        return false;
    }

    for (auto &&c : n->children())
    {
        if (c->type() == OpType_Gather)
        {
            return false;
        }
    }

    return true;
}

static void infer_children_shape(NodePtr n)
{
    for (auto c : n->children())
    {
        if (c->outputs().size() != 1)
            return;
        if (c->inputs().size() != 1)
            return;
        if (c->outputs()[0]->shape().dim() != 0)
            return;
        c->outputs()[0]->shape() = c->inputs()[0]->shape();

        return infer_children_shape(c);
    }
}

bool fill_constant_out_shape(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }

        TensorPtr weight = n->constants()["weights"];
        int elt_num = weight->mem_size() / Tensor::dtype_size(weight->type());

        if (weight->shape().dim() == 0)
        {
            weight->shape() = {elt_num};
        }

        if (n->outputs()[0]->shape().dim() == 0)
        {
            n->outputs()[0]->shape() = {elt_num};
        }

        infer_children_shape(n);
        is_changed = true;
    }
    if (is_changed)
    {
        graph_infer_shape(g);
    }
    return is_changed;
}
} // namespace aipubt

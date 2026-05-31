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
 * @Author: Lang Hu
 * @Date: 2025-02-10 13:59:45
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-02-10 13:59:45
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "transform/passes.h"

namespace aipubt
{
static const std::set<OpType> broadcast_ops = {
    OpType_Add,
    OpType_Sub,
    OpType_Mul,
    OpType_Div,
};

static inline bool creteria(NodePtr n)
{
    if (n->type() == OpType_Constant)
    {
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

        bool has_gather = false;
        bool has_broadcast_ops = false;
        for (auto &&c : n->children())
        {
            if (c->type() == OpType_Gather)
            {
                has_gather = true;
            }
            if (broadcast_ops.count(c->type()))
            {
                has_broadcast_ops = true;
            }
        }
        if (has_gather && has_broadcast_ops)
        {
            return true;
        }
    }

    return false;
}

// split scalar constant if shared by add/sub/mul/div and gather
bool split_constant(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }

        TensorPtr weight = n->constants()["weights"];
        auto out = n->outputs()[0];

        auto weight_size = weight->mem_size();
        auto new_weight = new char[weight_size];
        memcpy(new_weight, weight->data<char>(), weight_size);
        auto new_out = ops::constant(g, new_weight, weight->shape(), weight->type(), out->quantization());
        ADD_LOCATION_BY_NODE(new_out->node(), n);

        auto children = n->children();
        for (auto &&c : children)
        {
            if (broadcast_ops.count(c->type()))
            {
                c->add_input(new_out, c->remove_input(out));
            }
        }

        is_changed = true;
    }

    return is_changed;
}
} // namespace aipubt

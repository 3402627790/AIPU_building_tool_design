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
 * @Author: Barney Zhao
 * @Date: 2023-08-22 17:00:00
 * @Last Modified by: Barney Zhao
 * @Last Modified time: 2023-08-22 17:00:00
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>
#include <vector>

namespace aipubt
{

static bool criteria(const std::vector<NodePtr> &nodes)
{
    auto trans = nodes[0];
    auto elt = nodes[1];
    auto broadcastable = [](TensorShape inp_shape, TensorShape transpose_out_shape)
    {
        bool same_dim = false;
        for (int i = 0; i < inp_shape.dim(); ++i)
        {
            auto a = inp_shape[inp_shape.dim() - 1 - i];
            auto b = transpose_out_shape[transpose_out_shape.dim() - 1 - i];
            if (a == 1)
            {
                continue;
            }
            else
            {
                if (same_dim)
                {
                    return false;
                }
                if (a != b)
                {
                    return false;
                }
                else
                {
                    same_dim = true;
                }
            }
        }
        return true;
    };
    if (elt->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    auto elt_in1 = elt->inputs()[0] == trans->outputs()[0] ? elt->inputs()[1] : elt->inputs()[0];
    TensorShape any_shape = elt_in1->shape(); // no transpose input

    // only support [1...N..1]
    if (broadcastable(any_shape, trans->outputs()[0]->shape()) &&
        trans->outputs()[0]->shape().size() >= any_shape.size())
    {
        int axis = 0;
        for (int i = 0; i < any_shape.dim(); ++i)
        {
            if (any_shape[i] != 1)
            {
                axis = i;
            }
        }
        if (elt_in1->node()->is_dynamic())
        {
            for (int i = 0; i < any_shape.dim(); ++i)
            {
                if (i == axis)
                    continue;
                if (!any_shape.symbol()[i]->is_constant())
                {
                    return false;
                }
            }
        }
        std::vector<int> perm = trans->params()["perm"];
        if (perm[axis] == axis && any_shape.size() != 1)
        {
            return false;
        }
    }
    else
    {
        return false;
    }

    return true;
}

bool lower_trans_mul_or_add_or_sub(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns =
        {{{{"0", OpType_Transpose}, {"1", OpType_Mul}}, {{"0", "1"}}},
         {{{"0", OpType_Transpose}, {"1", OpType_Add}}, {{"0", "1"}}},
         {{{"0", OpType_Transpose}, {"1", OpType_Sub}}, {{"0", "1"}}},
         {{{"0", OpType_Transpose}, {"1", OpType_Add}}, {{"0", "1"}}},
         {{{"0", OpType_Transpose}, {"1", OpType_Mul}}, {{"0", "1"}}}};
    bool is_changed = false;
    for (auto pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_fused = false;
            for (auto &&m : matched)
            {
                if (!criteria(m))
                {
                    continue;
                }

                /**
                 * -------\             reshape--\
                 *         |--ops  ==>            |--ops--trans
                 * trans--/             ---------/
                 */
                auto trans = m[0];
                auto elt = m[1];
                auto trans_in = trans->inputs()[0];
                auto trans_out = trans->outputs()[0];
                auto elt_in1 = trans->outputs()[0] == elt->inputs()[0] ? elt->inputs()[1] : elt->inputs()[0];
                auto elt_out = elt->outputs()[0];

                // create reshape node
                std::vector<int> perm = trans->params()["perm"];
                std::vector<int> new_dim(trans_out->shape().dim(), 1);
                for (int32_t i = 0; i < elt_in1->shape().dim(); ++i)
                {
                    new_dim[perm[i]] = elt_in1->shape()[i];
                }
                std::vector<tongue::Expr> new_expr;
                if (elt_in1->node()->is_dynamic())
                {
                    new_expr.resize(trans_out->shape().dim(), 1);
                    for (int32_t i = 0; i < elt_in1->shape().symbol().dim(); ++i)
                    {
                        new_expr[perm[i]] = elt_in1->shape().symbol()[i];
                    }
                }
                TensorShape new_shape(new_dim);
                auto reshape_0_output = ops::reshape(elt_in1, new_shape, new_expr);
                auto reshape_0_node = reshape_0_output->node();
                g->add_node(reshape_0_node);
                ADD_LOCATION_BY_NODE(reshape_0_node, elt);

                // modify elt node
                elt->add_input(trans->inputs()[0], elt->remove_input(trans->outputs()[0]));
                elt->add_input(reshape_0_output, elt->remove_input(elt_in1));
                elt->clear_outputs();
                trans_out->shape() = trans_in->shape();
                trans_out->quantization() = elt_out->quantization();
                trans_out->set_type(elt_out->type());
                elt->add_output(trans_out);

                // modify transpose node
                trans->clear_inputs();
                trans->clear_outputs();
                trans->add_input(trans_out);
                trans->add_output(elt_out);

                is_changed = true;
                is_fused = true;
            }
            if (!is_fused)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}
}; // namespace aipubt

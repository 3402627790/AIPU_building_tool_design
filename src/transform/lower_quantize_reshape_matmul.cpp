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
 * =====================================================================================
 *
 *       Filename:  lower_quantize_reshape_matmul.cpp
 *
 *    Description:  In the subgraph Quantize->Reshape->Matmul, lift Reshape to be Reshape->Quantize->Matmul,
 *                  so that Quantize and Matmul can both use the same NC'HWC32 or NC'HWC16 layout
 *       Refer to:  https://jira.armchina.com/browse/CP-24267
 *
 *        Version:  1.0
 *        Created:  08/04/25 14:59:35
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "passes.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/config.h"
#include <vector>

namespace aipubt
{

static bool criteria(NodePtr n0, NodePtr n1, NodePtr n2)
{
    if (n0->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    if (n0->outputs()[0]->quantization().scales().size() != 1)
    {
        return false;
    }

    auto in_shape = n1->inputs()[0]->shape();
    auto out_shape = n1->outputs()[0]->shape();
    if (in_shape.dim() != 4 || out_shape.dim() != 4)
    {
        return false;
    }
    if (in_shape[0] * in_shape[1] != out_shape[0] * out_shape[1] ||
        in_shape[2] != out_shape[2] ||
        in_shape[3] != out_shape[3])
    {
        return false;
    }

    if (n0->outputs()[0]->consumers().size() > 1)
    {
        return false;
    }

    return true;
}

bool lower_quantize_reshape_matmul(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Quantize}, {"1", OpType_Reshape}, {"2", OpType_MatMul}},
        {{"0", "1"}, {"1", "2"}},
    };

    bool is_changed = false;
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);

    auto matched = matcher.match(g);
    while (matched.size())
    {
        bool is_lifted = false;
        for (auto &&m : matched)
        {
            if (!criteria(m[0], m[1], m[2]))
            {
                continue;
            }

            auto origin_reshape_in = m[1]->inputs()[0];
            auto origin_reshape_out = m[1]->outputs()[0];
            auto consumers = origin_reshape_out->consumers();
            auto shape = origin_reshape_out->shape();

            auto quantize_in = m[0]->inputs()[0];
            auto reshape_out = ops::reshape(quantize_in, shape);
            auto reshape_node = reshape_out->node();
            g->add_node(reshape_node);
            ADD_LOCATION_BY_NODE(reshape_node, m[0]);
            if (BuilderConfig::get<bool>(Constants::enable_ds))
            {
                std::vector<Expr> ds_shape = shape.symbol();
                reshape_node->params()["ds_shape"] = ds_shape;
            }
            reshape_node->infer_shape(true);
            m[0]->add_input(reshape_out, m[0]->remove_input(quantize_in));
            MODIFY_NODE(m[0]);

            auto n = m[0];
            auto out = n->outputs()[0];
            while (out != origin_reshape_out)
            {
                out->shape() = shape;
                n->infer_shape(true);
                MODIFY_NODE(n);
                n = out->consumers()[0];
                out = n->outputs()[0];
            }

            for (auto c : consumers)
            {
                c->add_input(origin_reshape_in, c->remove_input(origin_reshape_out));
                MODIFY_NODE(c);
            }

            g->remove_node(m[1]);

            is_lifted = true;
            is_changed = true;
        }

        if (!is_lifted)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
} // namespace aipubt

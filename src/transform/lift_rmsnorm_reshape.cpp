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
 *       Filename:  lift_rmsnorm_reshape.cpp
 *
 *    Description:  In the subgraph RMSNorm->Reshape->Conv, lift Reshape to be Reshape->RMSNorm->Conv,
 *                  so that RMSNorm and Conv can both use the same NC'HWC16 layout
 *       Refer to:  https://jira.armchina.com/browse/CP-14136
 *
 *        Version:  1.0
 *        Created:  12/25/24 16:39:08
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <vector>

namespace aipubt
{
static inline bool check_reshape(NodePtr n)
{
    auto in_shape = n->inputs()[0]->shape();
    auto out_shape = n->outputs()[0]->shape();
    if (in_shape.dim() != 4 || out_shape.dim() != 4)
    {
        return false;
    }
    if (in_shape[0] != out_shape[0] ||
        in_shape[1] * in_shape[2] != out_shape[1] * out_shape[2] ||
        in_shape[3] != out_shape[3])
    {
        return false;
    }

    auto reshape_out = n->outputs()[0];
    if (reshape_out->is_io_tensor())
    {
        return false;
    }

    auto consumers = reshape_out->consumers();
    for (auto c : consumers)
    {
        if (c->type() != OpType_Convolution)
        {
            return false;
        }
    }

    return true;
}

static bool criteria(NodePtr n0, NodePtr n1, NodePtr n2)
{
    auto &params = n0->params();
    if (params.find("axis") == params.end())
    {
        return false;
    }
    std::vector<int> axis_vec = params["axis"];
    if (axis_vec.size() != 1)
    {
        return false;
    }
    if (axis_vec[0] != -1 &&
        axis_vec[0] != n0->inputs()[0]->shape().dim() - 1)
    {
        return false;
    }

    auto rmsnorm_out = n0->outputs()[0];
    if (rmsnorm_out->is_io_tensor())
    {
        return false;
    }

    auto consumers = rmsnorm_out->consumers();
    for (auto c : consumers)
    {
        if (c->type() != OpType_Reshape &&
            c->type() != OpType_Quantize)
        {
            return false;
        }

        if (c->type() == OpType_Reshape) // check norm+reshape+conv
        {
            if (!check_reshape(c))
            {
                return false;
            }
            if (c->outputs()[0]->shape() != n1->outputs()[0]->shape())
            {
                return false;
            }
        }
        else // check norm+quantize+reshape+conv
        {
            if (c->outputs()[0]->is_io_tensor())
            {
                return false;
            }
            auto children = c->children();
            if (children.size() != 1)
            {
                return false;
            }
            auto cc = children[0];
            if (cc->type() != OpType_Reshape)
            {
                return false;
            }
            if (!check_reshape(cc))
            {
                return false;
            }
            if (cc->outputs()[0]->shape() != n1->outputs()[0]->shape())
            {
                return false;
            }
        }
    }

    return true;
}

bool lift_rmsnorm_reshape(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {
            {{"0", OpType_RMSNorm}, {"1", OpType_NoOp}, {"2", OpType_Reshape}, {"3", OpType_Convolution}},
            {{"0", "1"}, {"1", "2"}, {"2", "3"}},
        },
        {
            {{"0", OpType_RMSNorm}, {"1", OpType_Reshape}, {"2", OpType_Convolution}},
            {{"0", "1"}, {"1", "2"}},
        },
        {
            {{"0", OpType_LayerNorm}, {"1", OpType_NoOp}, {"2", OpType_Reshape}, {"3", OpType_Convolution}},
            {{"0", "1"}, {"1", "2"}, {"2", "3"}},
        },
        {
            {{"0", OpType_LayerNorm}, {"1", OpType_Reshape}, {"2", OpType_Convolution}},
            {{"0", "1"}, {"1", "2"}},
        },
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
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

                auto rmsnorm_in = m[0]->inputs()[0];
                auto rmsnorm_out = m[0]->outputs()[0];
                auto shape = m[1]->outputs()[0]->shape();

                auto reshape_out = ops::reshape(rmsnorm_in, shape);
                auto reshape_node = reshape_out->node();
                reshape_node->infer_shape(true);
                g->add_node(reshape_node);
                ADD_LOCATION_BY_NODE(reshape_node, m[0]);

                m[0]->add_input(reshape_out, m[0]->remove_input(rmsnorm_in));
                rmsnorm_out->shape() = shape;
                MODIFY_NODE(m[0]);

                auto consumers = rmsnorm_out->consumers();
                for (auto c : consumers)
                {
                    auto n = c;
                    if (c->type() != OpType_Reshape)
                    {
                        c->outputs()[0]->shape() = shape;
                        c->infer_shape(true);
                        MODIFY_NODE(c);
                        n = c->children()[0];
                    }
                    auto children = n->children();
                    for (auto cc : children)
                    {
                        cc->add_input(n->inputs()[0], cc->remove_input(n->outputs()[0]));
                        cc->infer_shape(true);
                        MODIFY_NODE(cc);
                    }
                    g->remove_node(n);
                }

                is_lifted = true;
                is_changed = true;
            }

            if (!is_lifted)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}
} // namespace aipubt

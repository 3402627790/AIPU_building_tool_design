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
 * @Date: 2024-11-05 16:41:46
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-11-05 16:41:46
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
static inline bool criteria(NodePtr n0, NodePtr n1)
{
    if (n0->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    int axis = 2;
    if (n0->params().count("axis"))
    {
        axis = n0->params()["axis"];
    }
    if (axis != 2)
    {
        return false;
    }
    std::vector<int> perm = n1->params()["perm"];
    if (perm != std::vector<int>({0, 1, 3, 2}))
    {
        return false;
    }
    return true;
}

void lift_cache_transpose(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_CacheUpdate}, {"1", OpType_Transpose}},
        {{"0", "1"}},
    };
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);

    auto matched = matcher.match(g);
    while (matched.size())
    {
        bool is_lifted = false;
        for (auto &&m : matched)
        {
            if (!criteria(m[0], m[1]))
            {
                continue;
            }
            auto inp = m[0]->inputs()[2];
            m[1]->clear_inputs();
            m[1]->add_input(inp);
            std::vector<int> shape_list = {
                inp->shape().n(),
                inp->shape().h(),
                inp->shape().c(),
                inp->shape().w(),
            };
            auto out = m[1]->outputs()[0];
            out->shape() = TensorShape(shape_list, inp->shape().layout());
            m[1]->infer_shape(true);
            MODIFY_NODE(m[1]);

            auto consumers = out->consumers();
            m[0]->add_input(out, m[0]->remove_input(inp));
            auto mid = m[0]->outputs()[0];
            shape_list = {
                mid->shape().n(),
                mid->shape().h(),
                mid->shape().c(),
                mid->shape().w(),
            };
            mid->shape() = TensorShape(shape_list, mid->shape().layout());
            auto shapes = m[0]->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
            shapes[0] = std::vector<Expr>({
                shapes[0][0],
                shapes[0][1],
                shapes[0][3],
                shapes[0][2],
            });
            m[0]->set_attr(Constants::output_symbol_shape, shapes);
            m[0]->params()["axis"] = 3;
            m[0]->infer_shape(true);
            MODIFY_NODE(m[0]);

            for (auto c : consumers)
            {
                c->add_input(mid, c->remove_input(out));
                c->infer_shape(true);
                MODIFY_NODE(c);
            }
            is_lifted = true;
        }
        if (!is_lifted)
        {
            break;
        }
        matched = matcher.match(g);
    }
}
} // namespace aipubt

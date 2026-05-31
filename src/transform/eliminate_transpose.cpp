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
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-11-05 15:50:52
 */
#include "core/core.h"
#include "core/config.h"
#include "passes.h"

namespace aipubt
{
static inline bool creteria(NodePtr n)
{
    if (n->type() == OpType_Transpose)
    {
        auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
        bool is_dynamic = n->is_dynamic();
        std::vector<int> perm = n->params()["perm"];
        auto inp = n->inputs()[0]->shape();
        auto check_perm = [](const std::vector<int> &perm) {
            for (uint32_t i = 1; i < perm.size(); ++i)
            {
                if (perm[i] <= perm[i - 1])
                {
                    return false;
                }
            }
            return true;
        };
        if (check_perm(perm))
        {
            return true;
        }
        else
        {
            std::vector<int> sqz, new_perm;
            for (int i = 0; i < inp.dim(); ++i)
            {
                if (enable_ds && is_dynamic)
                {
                    if (inp.symbol()[i]->is_constant() && inp[i] == 1)
                    {
                        sqz.push_back(i);
                    }
                }
                else if (inp[i] == 1)
                {
                    sqz.push_back(i);
                }
            }
            for (auto p : perm)
            {
                if (std::find(sqz.begin(), sqz.end(), p) == sqz.end())
                {
                    new_perm.push_back(p);
                }
            }
            return check_perm(new_perm);
        }
    }
    return false;
}

bool eliminate_transpose(GraphPtr g)
{
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }
        auto reshape = Node::create(g->get_valid_node_name(n->name()), OpType_Reshape);
        reshape->params()["shape"] = n->outputs()[0]->shape();

        ADD_LOCATION_BY_NODE(reshape, n);
        g->replace_node_by(n, reshape);
        if (enable_ds && reshape->is_dynamic())
        {
            std::vector<Expr> ds_shape = reshape->outputs()[0]->shape().symbol();
            reshape->params()["ds_shape"] = ds_shape;
        }
        if (reshape->is_dynamic())
        {
            reshape->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{reshape->outputs()[0]->shape().symbol()};
        }
        reshape->infer_shape(true);

        is_changed = true;
    }
    return is_changed;
}
} // namespace aipubt

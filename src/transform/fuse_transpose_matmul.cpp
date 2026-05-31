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
 * @Author: Matt Chen
 * @Date: 2023-12-21 15:25:43
 * @Last Modified by: Matt Chen
 * @Last Modified time: 2023-12-21 15:25:43
 *
 *   |       |             |       |
 *   |   transpose         |       |
 *    \      /       =>     \     /
 *     matmul                matmul(trans_b inverted)
 *      |                     |
 *
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/compute_tensor.h"
#include "passes.h"
#include <algorithm>
#include <set>
namespace aipubt
{
    static inline bool creteria(const std::vector<NodePtr> &nodes)
    {
        /*
        *  matmul(any, transpose(axis=[any,any,3,2])) or matmul(any, transpose(axis=[any,2,1]))
        *  fuse to matmul(any,any, trans_b=true)
        */
        auto trans = nodes[0];
        auto in_shape_dim = trans->inputs()[0]->shape().dim();
        std::vector<int> perm0 = trans->params()["perm"];
        if (in_shape_dim==3)
        {
            std::vector<int> perm3 = {0,2,1};
            if (perm0!=perm3)
            {
                return false;
            }
        }
        else if (in_shape_dim==4)
        {
            std::vector<int> perm4 = {0,1,3,2};
            if (perm0!=perm4)
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        if (trans->outputs()[0]->consumers().size() > 1)
        {
            return false;  // transpose only feed matmul
        }

        auto matmul = nodes[1];
        if (trans->outputs()[0] != matmul->inputs()[1] || trans->outputs()[0]->is_io_tensor())
        {
            return false;  // not in second arc and trans not output tensor
        }

        return true;
    }

    bool fuse_transpose_matmul(GraphPtr g)
    {
        std::vector<GraphMatcherPattern> patterns = {
            {{{"0", OpType_Transpose}, {"1",OpType_MatMul}},
            {{"0", "1"}}}
            };
        bool is_changed = false;
        for(auto &pattern : patterns)
        {
            GraphMatcher matcher(pattern);
            matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
            auto matched = matcher.match(g);
            while (matched.size())
            {
                bool is_fused = false;
                for (auto &&m : matched)
                {
                    if (!creteria(m))
                    {
                        continue;
                    }
                    auto trans = m[0];
                    auto matmul = m[1];
                    bool trans_b = matmul->params()["trans_b"];
                    matmul->params()["trans_b"] = !trans_b;
                    matmul->remove_input(1);
                    matmul->add_input(trans->inputs()[0],1);
                    MODIFY_NODE(matmul);

                    trans->clear_inputs();
                    trans->clear_outputs();
                    g->remove_node(trans);
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
}
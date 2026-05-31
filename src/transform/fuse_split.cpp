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
 * @Author: Samuel Han
 * @Date: 2022-04-11 14:24:01
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-22 11:08:59
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>

namespace aipubt
{

static bool criteria(std::vector<NodePtr> &nodes)
{
    auto n0 = nodes[0];
    auto n1 = nodes[1];
    if (n1->inputs()[0]->is_io_tensor() || n1->parents().size() > 1)
    {
        return false;
    }

    int32_t n0_axis = n0->params()["axis"];
    int32_t n1_axis = n1->params()["axis"];
    if (n0_axis != n1_axis)
    {
        return false;
    }
    return true;
}

bool fuse_split(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Split}, {"1", OpType_Split}}, {{"0", "1"}}},
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m))
                {
                    continue;
                }

                auto m0_outs = m[0]->outputs();
                auto m1_outs = m[1]->outputs();
                auto fuse_it = std::find(m0_outs.begin(), m0_outs.end(), m[1]->inputs()[0]);
                auto fuse_idx = std::distance(m0_outs.begin(), fuse_it);

                std::vector<int32_t> m0_ori_splits = m[0]->params()["splits"];
                std::vector<int32_t> m1_ori_splits = m[1]->params()["splits"];
                std::vector<int32_t> fuse_splits;
                fuse_splits.insert(fuse_splits.end(), m0_ori_splits.begin(), m0_ori_splits.begin() + fuse_idx);
                fuse_splits.insert(fuse_splits.end(), m1_ori_splits.begin(), m1_ori_splits.end());
                fuse_splits.insert(fuse_splits.end(), m0_ori_splits.begin() + fuse_idx + 1, m0_ori_splits.end());

                std::vector<TensorPtr> fuse_outputs;
                fuse_outputs.insert(fuse_outputs.end(), m0_outs.begin(), m0_outs.begin() + fuse_idx);
                fuse_outputs.insert(fuse_outputs.end(), m1_outs.begin(), m1_outs.end());
                fuse_outputs.insert(fuse_outputs.end(), m0_outs.begin() + fuse_idx + 1, m0_outs.end());
                if (m[0]->is_dynamic())
                {
                    std::vector<tongue::Expr> m0_ori_ds_splits = m[0]->params()["ds_splits"];
                    std::vector<tongue::Expr> m1_ori_ds_splits = m[1]->params()["ds_splits"];
                    std::vector<tongue::Expr> fuse_ds_splits;
                    fuse_ds_splits.insert(fuse_ds_splits.end(), m0_ori_ds_splits.begin(), m0_ori_ds_splits.begin() + fuse_idx);
                    fuse_ds_splits.insert(fuse_ds_splits.end(), m1_ori_ds_splits.begin(), m1_ori_ds_splits.end());
                    fuse_ds_splits.insert(fuse_ds_splits.end(), m0_ori_ds_splits.begin() + fuse_idx + 1, m0_ori_ds_splits.end());
                    m[0]->params()["ds_splits"] = fuse_ds_splits;
                }
                m[0]->params()["splits"] = fuse_splits;
                m[0]->clear_outputs();
                m[0]->add_output(fuse_outputs);
                g->remove_node(m[1]);
                is_changed = true;
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}

} // namespace aipubt
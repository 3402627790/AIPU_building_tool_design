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
 * @Author: Tom Gao
 * @Date: 2022-03-22 10:25:43
 * @Last Modified by: Aaron Gaoao
 * @Last Modified time: 2022-09-10 21:54:12
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr n1, NodePtr n2)
{
    std::vector<std::vector<int>> pads_0 = n1->params()["pads"];
    std::vector<std::vector<int>> pads_1 = n2->params()["pads"];
    if (pads_0.size() != pads_1.size())
    {
        return false;
    }
    std::string mode_n1 = n1->params()["mode"];
    std::string mode_n2 = n2->params()["mode"];
    if (mode_n1 == "CONSTANT" && mode_n2 == "CONSTANT")
    {
        int value_n1 = n1->params()["constant_value"];
        int value_n2 = n2->params()["constant_value"];
        if (value_n1 == value_n2 && !n1->outputs()[0]->is_io_tensor())
        {
            return true;
        }
    }
    return false;
}

bool fuse_pad(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Pad}, {"1", OpType_Pad}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!creteria(m[0], m[1]))
            {
                continue;
            }
            std::vector<std::vector<int>> pads_0 = m[0]->params()["pads"];
            std::vector<std::vector<int>> pads_1 = m[1]->params()["pads"];
            std::vector<std::pair<uint32_t, uint32_t>> pads;
            for (uint32_t i = 0; i < pads_0.size(); ++i)
            {
                pads.push_back({pads_0[i][0] + pads_1[i][0],
                                pads_0[i][1] + pads_1[i][1]});
            }
            std::vector<std::vector<tongue::Expr>> ds_pads;
            if (m[0]->params().count("ds_pads") || m[1]->params().count("ds_pads"))
            {
                std::vector<std::vector<tongue::Expr>> ds_pads_0;
                std::vector<std::vector<tongue::Expr>> ds_pads_1;
                if (m[0]->params().count("ds_pads"))
                {
                    ds_pads_0 = m[0]->params()["ds_pads"];
                }
                if (m[1]->params().count("ds_pads"))
                {
                    ds_pads_1 = m[1]->params()["ds_pads"];
                }
                for (uint32_t i = 0; i < pads_0.size(); ++i)
                {
                    tongue::Expr ds_pad_before;
                    tongue::Expr ds_pad_end;
                    if (ds_pads_0.size() != 0)
                    {
                        ds_pad_before = ds_pads_0[i][0];
                        ds_pad_end = ds_pads_0[i][1];
                    }
                    else
                    {
                        ds_pad_before = pads_0[i][0];
                        ds_pad_end = pads_0[i][1];
                    }
                    if (ds_pads_1.size() != 0)
                    {
                        ds_pad_before += ds_pads_1[i][0];
                        ds_pad_end += ds_pads_1[i][1];
                    }
                    else
                    {
                        ds_pad_before += tongue::Expr(pads_1[i][0]);
                        ds_pad_end += tongue::Expr(pads_1[i][1]);
                    }
                    ds_pads.push_back({ds_pad_before, ds_pad_end});
                }
            }
            auto inp = m[0]->remove_input(0);
            auto out = m[1]->remove_output(0);

            auto pad_out = ops::pad(inp, pads, ops::PadMode::Constant, (int32_t)m[0]->params()["constant_value"], ds_pads);
            auto pad_node = pad_out->node();
            pad_node->remove_output(0);
            pad_node->add_output(out);
            ADD_LOCATION_BY_NODE(pad_node, m[0], m[1]);

            g->remove_node(m[0]);
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
    return is_changed;
}
}; // namespace aipubt

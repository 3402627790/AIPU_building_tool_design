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
 * @Date: 2024-01-12 16:31:35
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-01-12 18:05:19
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"

namespace aipubt
{

static inline bool criteria(NodePtr n0, NodePtr n1)
{
    return !n0->outputs()[0]->is_io_tensor();
}

bool fuse_quantize_cast(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Quantize}, {"1", OpType_Cast}},
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
            if (!criteria(m[0], m[1]))
            {
                continue;
            }
            auto m1_output = m[1]->outputs()[0];
            m[0]->remove_output(0);
            m[0]->add_output(m1_output);
            m[1]->clear_inputs();
            m[1]->clear_outputs();
            g->remove_node(m[1]);

            if (m[0]->params().count("quantize_scale"))
            {
                m[0]->params()["quantize_scale"] = m1_output->quantization().scale();
            }
            if (m[0]->params().count("quantize_zp"))
            {
                m[0]->params()["quantize_zp"] = m1_output->quantization().offset();
            }

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

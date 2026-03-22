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
 * @Date: 2024-11-21 14:17:46
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-11-21 14:17:46
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
    if (n0->children().size() != 1)
    {
        return false;
    }
    if (!n0->params().count("ratio_y"))
    {
        return false;
    }
    float ratio_y = n0->params()["ratio_y"];
    std::vector<int> in_shape = n0->inputs()[0]->shape();
    std::vector<int> mid_shape = n0->outputs()[0]->shape();
    std::vector<int> out_shape = n1->outputs()[0]->shape();
    std::vector<std::vector<int>> crops = n1->params()["crops"];
    if (in_shape.size() != 4 ||
        in_shape[0] != mid_shape[0] ||
        in_shape[3] != mid_shape[3])
    {
        return false;
    }
    if ((int)std::floor((in_shape[1] - 1) * ratio_y) != out_shape[1])
    {
        return false;
    }
    if (mid_shape[0] != out_shape[0] ||
        crops[1][0] != 0 ||
        mid_shape[2] != out_shape[2] ||
        mid_shape[3] != out_shape[3])
    {
        return false;
    }
    return true;
}

void fuse_resize_crop(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Resize}, {"1", OpType_Crop}},
        {{"0", "1"}},
    };
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);

    auto matched = matcher.match(g);
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (!criteria(m[0], m[1]))
            {
                continue;
            }

            auto out = m[1]->outputs()[0];
            m[1]->clear_inputs();
            m[1]->clear_outputs();
            m[0]->clear_outputs();
            m[0]->add_output(out);
            MODIFY_NODE(m[0]);
            g->remove_node(m[1]);

            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(g);
    }
}
} // namespace aipubt

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
 * @Date: 2022-06-30 15:13:08
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-06-30 17:57:33
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"

namespace aipubt
{
static inline bool criteria(NodePtr n1, NodePtr n2)
{
    bool ret = true;
    std::vector<std::vector<int32_t>> crops = n1->params()["crops"];
    ret = ret && (!n1->outputs()[0]->is_io_tensor());
    for (auto crop : crops)
    {
        ret = ret && (crop[0] == 0);
    }
    auto n2_in_shape_backup = TensorShape(n2->inputs()[0]->shape());
    n2->inputs()[0]->shape() = n1->inputs()[0]->shape();
    auto n2_out_shape_new = n2->infer_shape();
    n2->inputs()[0]->shape() = n2_in_shape_backup;
    // INFO() << "n2_out_shape_new: " << n2_out_shape_new[0];
    // INFO() << "n2_out_shape: " << n2->outputs()[0]->shape();
    ret = ret && (n2_out_shape_new[0] == n2->outputs()[0]->shape());

    return ret;
}

bool fuse_crop_conv_pool(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Crop}, {"1", OpType_Pooling}}, {{"0", "1"}}},
        // {{{"0", OpType_Crop}, {"1", OpType_Pooling3D}}, {{"0", "1"}}},
        {{{"0", OpType_Pad}, {"1", OpType_Convolution}}, {{"0", "1"}}},
        // {{{"0", OpType_Pad}, {"1", OpType_Convolution3D}}, {{"0", "1"}}},
    };
    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m[0], m[1]))
                {
                    continue;
                }
                auto inpt = m[0]->remove_input(0);
                m[1]->clear_inputs();
                m[1]->add_input(inpt);

                m[0]->clear_inputs();
                m[0]->clear_outputs();
                g->remove_node(m[0]);
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

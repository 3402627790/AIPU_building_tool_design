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
 * @Author: Xiaolong Zhang
 * @Date: 2022-12-09 11:19:07
 * @Last Modified by: Xiaolong Zhang
 * @Last Modified time: 2022-07-22 17:08:22
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <math.h>
namespace aipubt
{

static inline bool criteria(NodePtr n0, NodePtr n1)
{
    if (n0->children().size() > 1)
    {
        return false;
    }

    if ((int)n0->params()["block_size_x"] != (int)n0->params()["block_size_y"])
    {
        return false;
    }

    if ((int)n1->params()["dilation_y"] > 1 || (int)n1->params()["dilation_x"] > 1)
    {
        return false;
    }

    return true;
}

bool fuse_s2d_conv(GraphPtr g)
{
    // TODO: support s2b/d2s
    GraphMatcherPattern pattern = {
        {{"0", OpType_SpaceToDepth}, {"1", OpType_Convolution}},
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

            int block_size = m[0]->params()["block_size_x"]; // update params
            m[1]->params()["stride_y"] = (int)m[1]->params()["stride_y"] * block_size;
            m[1]->params()["stride_x"] = (int)m[1]->params()["stride_x"] * block_size;
            m[1]->params()["kernel_y"] = (int)m[1]->params()["kernel_y"] * block_size;
            m[1]->params()["kernel_x"] = (int)m[1]->params()["kernel_x"] * block_size;
            m[1]->params()["pad_left"] = (int)m[1]->params()["pad_left"] * block_size;
            m[1]->params()["pad_right"] = (int)m[1]->params()["pad_right"] * block_size;
            m[1]->params()["pad_top"] = (int)m[1]->params()["pad_top"] * block_size;
            m[1]->params()["pad_bottom"] = (int)m[1]->params()["pad_bottom"] * block_size;

            auto w0 = ComputeTensor::create(m[1]->constants()["weights"]); //NHWC
            auto N = w0->shape()[0];
            auto H = w0->shape()[1];
            auto W = w0->shape()[2];
            auto C = w0->shape()[3];
            int new_cin = C / pow((int)block_size, 2);
            int new_H = H * block_size;
            int new_W = W * block_size;
            auto t0 = w0.reshape(N, H, W, block_size, block_size, new_cin);
            auto t1 = t0.permute({0, 1, 3, 2, 4, 5}).reshape(N, new_H, new_W, new_cin);
            m[1]->constants()["weights"] = t1; // update D2S weights.

            auto in = m[0]->inputs()[0];
            m[1]->clear_inputs();
            m[1]->add_input(in);
            m[0]->clear_inputs();
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
    return is_changed;
}
}; // namespace aipubt

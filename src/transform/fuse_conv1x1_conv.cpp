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
 * @Last Modified time: 2022-04-22 12:05:31
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>

namespace aipubt
{
static bool criteria(NodePtr conv1x1, NodePtr conv)
{
    if (conv1x1->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (conv1x1->params()["kernel_x"] != 1 ||
        conv1x1->params()["kernel_y"] != 1 ||
        conv1x1->params()["stride_x"] != 1 ||
        conv1x1->params()["stride_y"] != 1 ||
        conv1x1->params()["dilation_x"] != 1 ||
        conv1x1->params()["dilation_y"] != 1 ||
        conv1x1->params()["group"] != 1 ||
        conv1x1->params()["pad_left"] != 0 ||
        conv1x1->params()["pad_right"] != 0 ||
        conv1x1->params()["pad_top"] != 0 ||
        conv1x1->params()["pad_bottom"] != 0 ||
        (std::string)conv1x1->params()["with_activation"] != "NONE")
    {
        return false;
    }
    if (conv->params()["pad_left"] != 0 ||
        conv->params()["pad_right"] != 0 ||
        conv->params()["pad_top"] != 0 ||
        conv->params()["pad_bottom"] != 0 ||
        conv->params()["dilation_x"] != 1 ||
        conv->params()["dilation_y"] != 1 ||
        conv->params()["group"] != 1)
    {
        return false;
    }
    auto w0 = conv1x1->constants()["weights"];
    auto w1 = conv->constants()["weights"];
    if (w0->type() != Tensor::Dtype::FP32 || w1->type() != Tensor::Dtype::FP32 || !w0->data<void>() || !w1->data<void>())
    {
        return false;
    }
    return true;
}

bool fuse_conv1x1_conv(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Convolution}, {"1", OpType_Convolution}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    std::set<NodePtr> removed;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (!criteria(m[0], m[1]))
            {
                continue;
            }
            if (removed.count(m[0]) || removed.count(m[1]))
            {
                continue;
            }
            m[1]->clear_inputs();
            m[1]->add_input(m[0]->inputs());
            auto out = m[0]->outputs()[0];
            auto w0 = ComputeTensor::create(m[0]->constants()["weights"]);
            auto b0 = ComputeTensor::create(m[0]->constants()["biases"]);
            auto w1 = ComputeTensor::create(m[1]->constants()["weights"]);
            auto b1 = ComputeTensor::create(m[1]->constants()["biases"]);
            auto mc = w0->shape()[0];
            auto ic = w0->shape()[3];
            auto oc = w1->shape()[0];
            auto kh = w1->shape()[1];
            auto kw = w1->shape()[2];
            auto w00 = w0.reshape(mc, ic).permute({1, 0});
            auto w10 = w1.permute({3, 0, 1, 2}).reshape(mc, oc * kh * kw);
            auto w2 = w00.matmul(w10).reshape(ic, oc, kh, kw).astype(Tensor::Dtype::FP64);
            auto new_w = w2.permute({1, 2, 3, 0});

            auto wb1 = w1 * b0.astype(Tensor::Dtype::FP64);
            auto new_b = b1 + (wb1.sum({1, 2, 3})).astype(Tensor::Dtype::FP64);
            auto input = m[0]->inputs()[0];
            m[1]->clear_inputs();
            m[1]->add_input(input);
            m[0]->clear_inputs();
            m[1]->constants()["weights"] = new_w.astype(Tensor::Dtype::FP32);
            m[1]->constants()["biases"] = new_b.astype(Tensor::Dtype::FP32);

            g->remove_node(m[0]);
            is_changed = true;
            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}

} // namespace aipubt

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
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-10 21:52:49
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr conv, NodePtr reshape, NodePtr trans)
{
    if (reshape->outputs()[0]->is_io_tensor() || conv->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    auto inp = reshape->inputs()[0]->shape();
    auto out = reshape->outputs()[0]->shape();
    if (out.dim() == 6 && inp.dim() == 4)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (inp[i] != out[i])
            {
                return false;
            }
        }
        if (out[4] != out[5]) // d2s only support block_x=block_y
        {
            return false;
        }
        if (out[4] > 16)
        {
            return false;
        }
        std::vector<int> perm = trans->params()["perm"];
        if (inp[0] == 1) // the batch dim is ignore
        {
            std::vector<int> _p;
            _p.push_back(0);
            for (auto i : perm)
            {
                if (i != 0)
                {
                    _p.push_back(i);
                }
            }
            perm = _p;
        }
        if (perm == std::vector<int>({0, 1, 4, 2, 5, 3}))
        {
            return true;
        }
    }
    return false;
}

static void interlevel_conv_out_c(NodePtr conv, int block)
{
    ComputeTensor w = conv->constants()["weights"];
    std::vector<int> shape = w->shape();
    auto c = shape[0];
    auto nc = c / block / block;
    std::vector<int> new_shape = {nc, block, block};
    for (uint32_t i = 1; i < shape.size(); ++i)
    {
        new_shape.push_back(shape[i]);
    }
    w = w.reshape(new_shape).permute({1, 2, 0, 3, 4, 5}).reshape(shape);
    conv->constants()["weights"] = w;
    std::vector<int> bias_shape = {nc, block, block};
    ComputeTensor b = conv->constants()["biases"];
    b = b.reshape(bias_shape).permute({1, 2, 0}).reshape(c);
    conv->constants()["biases"] = b;
    if (conv->constants().count("scale"))
    {
        ComputeTensor b = conv->constants()["scale"];
        b = b.reshape(bias_shape).permute({1, 2, 0}).reshape(c);
        conv->constants()["scale"] = b;
    }
    if (conv->constants().count("shift"))
    {
        ComputeTensor b = conv->constants()["shift"];
        b = b.reshape(bias_shape).permute({1, 2, 0}).reshape(c);
        conv->constants()["shift"] = b;
    }
    if (conv->constants().count("negative_slope"))
    {
        ComputeTensor b = conv->constants()["negative_slope"];
        b = b.reshape(bias_shape).permute({1, 2, 0}).reshape(c);
        if (b)
        {
            conv->constants()["negative_slope"] = b;
        }
    }
}
bool conv_transpose_to_d2s(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Convolution}, {"1", OpType_Reshape}, {"2", OpType_Transpose}},
        {{"0", "1"}, {"1", "2"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!creteria(m[0], m[1], m[2]))
            {
                continue;
            }
            auto internal = m[1]->outputs()[0]->shape();
            auto block = internal[4];
            interlevel_conv_out_c(m[0], block);
            MODIFY_NODE(m[0]);
            m[1]->clear_inputs();
            auto conv_out = m[0]->outputs()[0];
            auto d2s_tensor = ops::depth_to_space(conv_out, block);
            auto out = m[2]->outputs()[0];
            auto rs_tensor = ops::reshape(d2s_tensor, out->shape());
            auto rs = rs_tensor->node();
            m[2]->clear_outputs();
            rs->clear_outputs();
            rs->add_output(out);
            ADD_LOCATION_BY_NODE(d2s_tensor->node(), m[1], m[2]);
            ADD_LOCATION_BY_NODE(rs, m[1], m[2]);
            g->remove_node(m[1]);
            g->remove_node(m[2]);

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

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
 * @Author: Kyle Huang
 * @Date: 2020-05-19 17:00:00
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2020-05-19 17:00:00
 */
#include "builder/lowering/lowering.h"
#include "core/compute_tensor.h"
#include "core/core.h"
#include <algorithm>
#include <set>

namespace aipubt
{
void crop_conv_kernel(GraphPtr g)
{
    auto criterion = [](NodePtr n) -> bool {
        if (n->type() != OpType_Convolution)
        {
            return false;
        }
        return true;
    };

    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }
        int i_h = n->inputs()[0]->shape().h();
        int i_w = n->inputs()[0]->shape().w();
        int k_y = n->params()["kernel_y"];
        int k_x = n->params()["kernel_x"];
        int p_t = n->params()["pad_top"];
        int p_b = n->params()["pad_bottom"];
        int p_l = n->params()["pad_left"];
        int p_r = n->params()["pad_right"];

        bool crop_t, crop_b, crop_l, crop_r;
        crop_t = (k_y > p_t) && (p_t > 0);
        crop_b = (k_y > n->inputs()[0]->shape().h() + p_t) && (p_b > 0);
        crop_l = (k_x > p_l) && (p_l > 0);
        crop_r = (k_x > n->inputs()[0]->shape().w() + p_l) && (p_r > 0);

        bool crop_h, crop_w;
        crop_h = (crop_t || crop_b) && (n->outputs()[0]->shape().h() == 1) && (n->params()["dilation_y"] == 1);
        crop_w = (crop_l || crop_r) && (n->outputs()[0]->shape().w() == 1) && (n->params()["dilation_x"] == 1);

        if (crop_h)
        {
            int new_k_y = (k_y - p_t) > i_h ? i_h : (k_y - p_t);
            n->params()["kernel_y"] = new_k_y;
            n->params()["pad_top"] = 0;
            n->params()["pad_bottom"] = 0;
            auto weight = n->constants()["weights"];
            auto weight_shape = weight->shape();
            weight_shape[1] = new_k_y;
            auto new_weight = ComputeTensor::zeros(weight_shape, weight->type());
            auto ori_weight = ComputeTensor::create(weight);
            ori_weight = ori_weight.reshape(weight->shape());
            new_weight[{{0, -1}, {0, -1}, {0, -1}, {0, -1}}] = ori_weight[{{0, -1}, {p_t, p_t + new_k_y}, {0, -1}, {0, -1}}];
            new_weight->make_memory_continuous();
            n->constants()["weights"] = new_weight;
        }
        if (crop_w)
        {
            int new_k_x = (k_x - p_l) > i_w ? i_w : k_x - p_l;
            n->params()["kernel_x"] = new_k_x;
            n->params()["pad_left"] = 0;
            n->params()["pad_right"] = 0;
            auto weight = n->constants()["weights"];
            auto weight_shape = weight->shape();
            weight_shape[2] = new_k_x;
            auto new_weight = ComputeTensor::zeros(weight_shape, weight->type());
            auto ori_weight = ComputeTensor::create(weight);
            ori_weight = ori_weight.reshape(weight->shape());
            new_weight[{{0, -1}, {0, -1}, {0, -1}, {0, -1}}] = ori_weight[{{0, -1}, {0, -1}, {p_l, p_l + new_k_x}, {0, -1}}];
            new_weight->make_memory_continuous();
            n->constants()["weights"] = new_weight;
        }
    }
}
}; // namespace aipubt

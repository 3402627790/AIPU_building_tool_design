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
 * @Author: Kyle Huang
 * @Date: 2023-10-12
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2023-10-12
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static inline bool creteria(NodePtr n1, NodePtr n2)
{
    if (n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    return true;
}

bool fuse_constant_pad(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Constant}, {"1", OpType_Pad}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    std::set<NodePtr> removed;
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
            std::vector<std::vector<int32_t>> pads = m[1]->params()["pads"];
            uint32_t pad_value = m[1]->params()["constant_value"];
            ComputeTensor weight = m[0]->constants()["weights"];

            ComputeTensor new_weight = ComputeTensor::zeros(m[1]->outputs()[0]->shape(), weight->type());
            if (weight->type() == Tensor::Dtype::UINT8)
            {
                new_weight = new_weight + (uint8_t)pad_value;
            }
            else if (weight->type() == Tensor::Dtype::INT8)
            {
                new_weight = new_weight + (int8_t)pad_value;
            }
            else
            {
                WARN() << "fuse_constant_pad doesn't support this data type";
                continue;
            }

            std::vector<ComputeTensor::Slice> new_slices;
            std::vector<ComputeTensor::Slice> slices;
            for (int i = 0; i < m[1]->outputs()[0]->shape().dim(); ++i)
            {
                auto pad = pads[i];
                ComputeTensor::Slice new_slice(pad[0], pad[0] + m[0]->outputs()[0]->shape()[i]);
                new_slices.push_back(new_slice);
                ComputeTensor::Slice slice(0, m[0]->outputs()[0]->shape()[i]);
                slices.push_back(slice);
            }
            new_weight[new_slices] = weight[slices];
            new_weight->monopolize_data();
            m[0]->constants()["weights"] = new_weight;
            auto t = m[1]->outputs()[0];
            m[1]->clear_outputs();
            m[0]->clear_outputs();
            m[0]->add_output(t);

            g->remove_node(m[1]);

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

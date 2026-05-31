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
    if (n0->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (n0->children().size() > 1)
    {
        return false;
    }
    if (n0->outputs()[0]->type() != Tensor::FP8E5M2 &&
        n0->outputs()[0]->type() != Tensor::FP8E4M3FN)
    {
        return false;
    }
    if (n1->outputs()[0]->type() != Tensor::FP16 &&
        n1->outputs()[0]->type() != Tensor::BFP16)
    {
        return false;
    }

    if (!(n0->params().count("scale_value") ^ n0->constants().count("scale")))
    {
        return false;
    }
    if (n1->inputs()[0]->quantization().scales().size() > 1)
    {
        return false;
    }
    return true;
}

bool fuse_fp8_dequantize(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {
            {{"0", OpType_Convolution}, {"1", OpType_DeQuantize}},
            {{"0", "1"}},
        },
        {
            {{"0", OpType_FullyConnected}, {"1", OpType_DeQuantize}},
            {{"0", "1"}},
        },
        {
            {{"0", OpType_MatMul}, {"1", OpType_DeQuantize}},
            {{"0", "1"}},
        },
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
                auto m0_output = m[0]->outputs()[0];
                auto m1_output = m[1]->outputs()[0];
                m[0]->remove_output(0);
                m[0]->add_output(m1_output);
                m[1]->clear_inputs();
                m[1]->clear_outputs();
                g->remove_node(m[1]);

                if (m[0]->params().count("scale_value"))
                {
                    float scale_value = m[0]->params()["scale_value"];
                    scale_value = scale_value / ((float)m0_output->quantization().scale());
                    m[0]->params()["scale_value"] = scale_value;
                }
                else if (m[0]->constants().count("scale"))
                {
                    auto scale = ComputeTensor::create(m[0]->constants()["scale"]);
                    scale = scale / ((float)m0_output->quantization().scale());
                    m[0]->constants()["scale"] = scale.astype(Tensor::FP32);
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
    }
    return is_changed;
}
}; // namespace aipubt

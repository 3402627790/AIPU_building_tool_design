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
 * @Author: Samuel Han
 * @Date: 2023-06-02 10:57:50
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-06-11 16:31:10
 */

#include "core/core.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

static bool criteria(std::vector<aipubt::NodePtr> &nodes)
{
    auto cast_node = nodes[0];
    if (cast_node->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    auto dsize = aipubt::Tensor::dtype_size(cast_node->inputs()[0]->type());
    auto is_float = aipubt::Tensor::is_float(cast_node->inputs()[0]->type());
    if (is_float || dsize >= 4)
    {
        return false;
    }

    // layerlib CAST doesn't support fp->quant
    if (cast_node->inputs()[0]->type() >= aipubt::Tensor::Dtype::FP16 &&
        cast_node->inputs()[0]->type() <= aipubt::Tensor::Dtype::BFP16)
    {
        return false;
    }
    return true;
}

void fuse_cast_quantize(aipubt::GraphPtr g)
{
    std::vector<aipubt::GraphMatcherPattern> patterns = {
        {{{"0", aipubt::OpType_Cast}, {"1", aipubt::OpType_Quantize}}, {{"0", "1"}}},
    };
    for (auto &pattern : patterns)
    {
        aipubt::GraphMatcher matcher(pattern);
        matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m))
                {
                    continue;
                }
                auto cast_node = m[0];
                auto quantized_node = m[1];

                cast_node->clear_outputs();
                cast_node->add_output(quantized_node->outputs()[0]);
                cast_node->params()["to_dtype"] = quantized_node->outputs()[0]->type();
                cast_node->params()["ignore_scale_zp"] = false;
                cast_node->params()["clip_mode"] = "SATURATION";

                quantized_node->clear_inputs();
                quantized_node->clear_outputs();
                g->remove_node(quantized_node);
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
}

} // namespace qtlibn

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
 * @Last Modified time: 2026-04-15 15:47:19
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "dtypes/op_dtypes.h"
#include "passes.h"
#include "quantization_transform/common.h"
#include "quantization_transform/ops/no_quantization.h"

namespace aipubt
{

static inline bool criteria(NodePtrList nodes)
{
    auto n = nodes[0]; //matmul node
    if (n->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (n->children().size() != 1)
    {
        return false;
    }
    auto cast_node = nodes[1];
    if (!cast_node->params().count("scale_value") || !n->params().count("scale_value"))
    {
        return false;
    }

    return true;
}

bool fuse_matmul_cast(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_MatMul}, {"1", OpType_Cast}},
        {{"0", "1"}},
    };
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);

    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (!criteria(m))
            {
                continue;
            }
            auto n = m[0];
            auto cast_node = m[1];
            auto cast_out = cast_node->outputs()[0];
            if (cast_node->params().count("scale_value") && n->params().count("scale_value"))
            {
                // this graph has quantized, so we can get scale/zp;
                auto in_scale0 = n->inputs()[0]->quantization().scale();
                auto in_scale1 = n->inputs()[1]->quantization().scale();
                auto out_scale = cast_node->outputs()[0]->quantization().scale();

                int32_t cast_scale_value = cast_node->params()["scale_value"];
                auto multi_bits = dtype_utils::range_to_bits(0, cast_scale_value, false);
                auto total_scale = out_scale / in_scale0 / in_scale1;
                float scale;
                int8_t shift;
                quant_utils::get_scale_approximation_params(total_scale, scale, shift, multi_bits);
                n->params()["scale_value"] = static_cast<uint32_t>(scale);
                n->params()["scale_type"] = dtype_utils::bits_to_dtype(multi_bits, false);
                n->params()["shift_value"] = shift;
                n->params()["shift_type"] = Tensor::INT8;

                n->outputs()[0]->set_type(cast_out->type());
                n->outputs()[0]->quantization() = cast_out->quantization();
                n->remove_output(0);
                n->add_output(cast_out);
                cast_node->clear_inputs();
                cast_node->clear_outputs();
                g->remove_node(cast_node);
                is_changed = true;
                is_fused = true;
            }
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}; // namespace aipubt

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
 * @Date: 2022-03-30 13:38:32
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-30 14:21:45
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace aipubt
{

static inline bool criteria(NodePtr n0, NodePtr n1)
{
    return !n0->outputs()[0]->is_io_tensor();
}

bool fuse_cast(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Cast}, {"1", OpType_Cast}},
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
            /*
              case0: n0 and n1 both donot have quantization info, fused_cast will has no quantization info;
              case1: n0  or n1 has quantization info, fused_cast has quantization info;
              if n1.params['ignore_scale_zp'] == true, will not re-generate the scale_value/shift_value
            */
            if ((m[0]->params().count("scale_value") || m[1]->params().count("scale_value")) && !m[1]->params()["ignore_scale_zp"])
            {
                // this graph has quantized, so we can get scale/zp;
                auto in_scale = m[0]->inputs()[0]->quantization().scale();
                auto out_scale = m[1]->outputs()[0]->quantization().scale();
                auto total_scale = out_scale / in_scale;
                Tensor::Dtype to_dtype = m[1]->params()["to_dtype"];
                auto qbits = Tensor::dtype_bits(to_dtype);
                float scale;
                int8_t shift;
                quant_utils::get_scale_approximation_params(total_scale, scale, shift, qbits);
                m[1]->params()["scale_value"] = static_cast<uint32_t>(scale);
                m[1]->params()["scale_type"] = Tensor::dtype_to_unsigned(to_dtype);
                m[1]->params()["shift_value"] = shift;
                m[1]->params()["shift_type"] = Tensor::INT8;
            }
            auto in = m[0]->inputs()[0];
            m[1]->remove_input(0);
            m[0]->remove_input(0);
            m[1]->add_input(in);
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
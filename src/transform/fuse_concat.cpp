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
 * @Last Modified time: 2023-03-21 12:49:06
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"
#include <algorithm>

namespace aipubt
{

constexpr int32_t SHIFT_MAX_BITS = 15;

static bool _is_same_quantization(aipubt::NodePtr n)
{
    // check two part: scale * 0.5 ** shift == 1, and input.scale == output.scale, input.zp == output.zp
    std::vector<float> scales = n->params()["scale_value"];
    std::vector<int32_t> shifts = n->params()["shift_value"];

    bool ret = true;
    for (int32_t i = 0; i < (int32_t)scales.size(); i++)
    {
        float ts = scales[i] * std::pow(0.5, shifts[i]);
        if (ts != 1.0)
        {
            ret = ret && false;
            break;
        }
        if (n->inputs()[i]->quantization() != n->outputs()[0]->quantization())
        {
            ret = ret && false;
            break;
        }
    }
    return ret;
}

static bool _quantization_criteria(std::vector<NodePtr> &nodes)
{
    auto n0 = nodes[0];
    auto n1 = nodes[1];
    auto param0 = n0->params();
    auto param1 = n1->params();
    bool ret = true;
    // if one concat has quantized scale/shift and another does not have,
    // it is hard to judge which has higher performance between fuse concat and not fuse concat, so we donot fuse these concat.
    // but the concat with scale/shift has  scale * 0.5 ** shift == 1.0 in all branch, like scales=[4096,4096],shifts=[12,12],
    // we think this concat is also a without quantized scale/shift node.
    if (param0.count("scale_value") && !param1.count("scale_value"))
    {
        ret = ret && _is_same_quantization(n0);
    }
    else if ((!param0.count("scale_value") && param1.count("scale_value")))
    {
        ret = ret && _is_same_quantization(n1);
    }
    // else
    // {
    //     //1.float; 2.quantized in directly memory concat
    // }

    return ret;
}

static bool criteria(std::vector<NodePtr> &nodes)
{
    auto n0 = nodes[0];
    auto n1 = nodes[1];
    if (n0->outputs()[0]->is_io_tensor() || n0->children().size() > 1)
    {
        return false;
    }
    int32_t n0_axis = n0->params()["axis"];
    int32_t n1_axis = n1->params()["axis"];
    if (n0_axis != n1_axis)
    {
        return false;
    }
    bool ret = _quantization_criteria(nodes);
    if (!ret)
    {
        return false;
    }
    return true;
}

static void _quantization_update(aipubt::NodePtr pre_node, aipubt::NodePtr post_node, int32_t fuse_idx)
{
    auto param0 = pre_node->params();
    auto param1 = post_node->params();
    if (param0.count("scale_value") && param1.count("scale_value"))
    {
        std::vector<int32_t> post_scales = param1["scale_value"];
        std::vector<int32_t> post_shifts = param1["shift_value"];
        std::vector<aipubt::Tensor::Dtype> post_scale_types = param1["scale_type"];

        int32_t multiplier_bits = Tensor::dtype_bits(post_node->outputs()[0]->type());
        if (post_node->attrs().count("multiplier_bits"))
        {
            multiplier_bits = post_node->get_attr("multiplier_bits");
        }
        else
        {
            int32_t post_max_scale = *std::max_element(post_scales.begin(), post_scales.end());
            int32_t post_max_scale_in_bits = floor(std::log2(post_max_scale));
            post_max_scale_in_bits = post_max_scale_in_bits > SHIFT_MAX_BITS ? SHIFT_MAX_BITS : post_max_scale_in_bits;
            multiplier_bits = post_max_scale_in_bits;
        }
        aipubt::Tensor::Dtype scale_dtype = dtype_utils::bits_to_dtype(multiplier_bits, false);
        int32_t fuse_post_scale = post_scales[fuse_idx];
        int8_t fuse_post_shift = post_shifts[fuse_idx];
        std::vector<int32_t> fused_scales;
        std::vector<int32_t> fused_shifts;
        std::vector<aipubt::Tensor::Dtype> fused_scale_types;
        fused_scales.insert(fused_scales.end(), post_scales.begin(), post_scales.begin() + fuse_idx);
        fused_shifts.insert(fused_shifts.end(), post_shifts.begin(), post_shifts.begin() + fuse_idx);
        fused_scale_types.insert(fused_scale_types.end(), post_scale_types.begin(), post_scale_types.begin() + fuse_idx);
        float ts = fuse_post_scale * std::pow(0.5, fuse_post_shift);
        for (int32_t i = 0; i < (int32_t)pre_node->inputs().size(); i++)
        {
            float scale;
            int8_t shift;
            float fused_ts = ts * param0["scale_value"][i] * std::pow(0.5, param0["shift_value"][i]);
            quant_utils::get_scale_approximation_params(fused_ts, scale, shift, multiplier_bits);
            fused_scales.push_back((int32_t)scale);
            fused_shifts.push_back((int32_t)shift);
            fused_scale_types.push_back(scale_dtype);
        }
        fused_scales.insert(fused_scales.end(), post_scales.begin() + fuse_idx + 1, post_scales.end());
        fused_shifts.insert(fused_shifts.end(), post_shifts.begin() + fuse_idx + 1, post_shifts.end());
        fused_scale_types.insert(fused_scale_types.end(), post_scale_types.begin() + fuse_idx + 1, post_scale_types.end());
        post_node->params()["scale_value"] = fused_scales;
        post_node->params()["shift_value"] = fused_shifts;
        post_node->params()["scale_type"] = fused_scale_types;
        post_node->params()["shift_type"] = std::vector<aipubt::Tensor::Dtype>(fused_scales.size(), aipubt::Tensor::INT8);
    }
    else if (!param0.count("scale_value") && param1.count("scale_value"))
    {
        // in pre_node, concat is like directly memory concat, post node has scale * 0.5 ** shift == 1.0, and input.zp == output.zp,
        // so we think, post node is like directly memory concat, so we delete the scale_value/shift_value, and pre_node inputs insert to
        // post_node inputs.
        param1.erase("scale_value");
        param1.erase("shift_value");
    }
}

bool fuse_concat(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Concat}, {"1", OpType_Concat}}, {{"0", "1"}}},
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);
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

                std::vector<TensorPtr> fuse_inputs;
                auto m0_out = m[0]->outputs()[0];
                auto m0_inps = m[0]->inputs();
                auto m1_inps = m[1]->inputs();
                auto fuse_it = std::find(m1_inps.begin(), m1_inps.end(), m0_out);
                auto fuse_idx = std::distance(m1_inps.begin(), fuse_it);

                fuse_inputs.insert(fuse_inputs.end(), m1_inps.begin(), m1_inps.begin() + fuse_idx);
                fuse_inputs.insert(fuse_inputs.end(), m0_inps.begin(), m0_inps.end());
                fuse_inputs.insert(fuse_inputs.end(), m1_inps.begin() + fuse_idx + 1, m1_inps.end());

                _quantization_update(m[0], m[1], fuse_idx);

                m[1]->clear_inputs();
                m[1]->add_input(fuse_inputs);
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
    }

    return is_changed;
}

} // namespace aipubt

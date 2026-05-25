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
 * @Author: Matt Chen
 * @Date: 2022-04-06 19:22:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-18 16:06:11
 */

#include "quantization_transform/ops/clip.h"
#include <algorithm>

namespace qtlibn
{
void ClipOp::quantize(aipubt::NodePtr node)
{
    float clip_max = param_utils::get_param(node, "clip_max");
    float clip_min = param_utils::get_param(node, "clip_min");
    if (clip_min > clip_max)
    {
        INFO() << "The 'clip_min' must not greater than 'clip_max', please check IR.";
    }
    // float out_min = node->outputs()[0]->quantization().mins()[0];
    // float out_max = node->outputs()[0]->quantization().maxs()[0];
    // out_min = std::min(std::max(out_min, clip_min), clip_max);
    // out_max = std::max(std::min(out_max, clip_max), clip_min);
    // node->outputs()[0]->quantization().set_mins({out_min});
    // node->outputs()[0]->quantization().set_maxs({out_max});

    // per-channel quantization unsupported yet
    bool in_qinv = node->inputs()[0]->quantization().qinvariant();
    node->outputs()[0]->quantization().set_qinvariant(in_qinv);
    // Clip node may come from Eltwise/Convs activation
    bool inputs_qinvariant = std::all_of(node->inputs().begin(), node->inputs().end(), [](aipubt::TensorPtr t) { return t->quantization().qinvariant(); });
    if (inputs_qinvariant)
    {
        node->outputs()[0]->quantization().set_scale(1);
        node->outputs()[0]->quantization().set_offset(0);
        int in_bits = node->inputs()[0]->quantization().bits();
        node->outputs()[0]->quantization().set_bits(in_bits);

        auto in_dtype = node->inputs()[0]->quantization().aipu_dtype();
        node->outputs()[0]->quantization().set_aipu_dtype(in_dtype);
    }
    else
    {
        bool o_signed = clip_min >= 0 ? false : true;
        if (!aipubt::Tensor::is_signed(node->inputs()[0]->type()))
        {
            o_signed = false;
        }
        dtype_utils::set_aipu_dtype(node->outputs()[0], o_signed);
        Op::quantize(node); // set: scale,zp,qbit,qinv
    }

    float local_rescale = node->outputs()[0]->quantization().scale() / node->inputs()[0]->quantization().scale();
    float do_scales;
    int8_t do_shifts;
    int q_out_bits = node->outputs()[0]->quantization().bits();
    int32_t multiplier_bits = q_out_bits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }
    quant_utils::get_scale_approximation_params(local_rescale, do_scales, do_shifts, multiplier_bits, node->name());
    node->params()["scale_value"] = static_cast<uint32_t>(do_scales);
    node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
    node->params()["shift_value"] = do_shifts;
    node->params()["shift_type"] = aipubt::Tensor::INT8;

    if (!node->attrs().count(Constants::params_quantized) || !node->get_attr(Constants::params_quantized))
    {
        float cscale = node->outputs()[0]->quantization().scale();
        int czerop = node->outputs()[0]->quantization().offset();
        if (q_out_bits <= 8)
        {
            cscale = cscale * 256;
            czerop = 0;
        }
        float t_qmin = float(dtype_utils::bits_to_range_min(32, true));
        float t_qmax = float(dtype_utils::bits_to_range_max(32, true));
        float cmax = quant_utils::linear_quantize_clip(clip_max, cscale, czerop, t_qmin, t_qmax);
        float cmin = quant_utils::linear_quantize_clip(clip_min, cscale, czerop, t_qmin, t_qmax);

        node->params()["clip_max"] = int(std::max(t_qmin, std::min(t_qmax, cmax)));
        node->params()["clip_min"] = int(std::max(t_qmin, std::min(t_qmax, cmin)));
    }
}

REGISTER_QUANTIZATION_OP(Clip);

}; // namespace qtlibn

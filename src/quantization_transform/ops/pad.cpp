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
 * @Date: 2022-04-07 21:42:54
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-11-22 13:59:18
 */

#include "quantization_transform/ops/pad.h"

namespace qtlibn
{

void PadOp::quantize(aipubt::NodePtr node)
{
    NoQuantization::quantize(node);

    if (node->params().count("constant_value"))
    {
        float constant_value = node->params()["constant_value"];
        // pad_quantized attribute is used to distinguish the quantized constant value or the float constant value,
        // and the float value needs to quantized to aipu quantized value.
        if (!node->attrs().count(Constants::params_quantized) || !node->get_attr(Constants::params_quantized))
        {
            auto o_scale = node->outputs()[0]->quantization().scale();
            auto o_zerop = node->outputs()[0]->quantization().offset();
            auto o_aipu_dtype = node->outputs()[0]->quantization().aipu_dtype();
            auto o_dtype = node->outputs()[0]->type();
            if (o_aipu_dtype != aipubt::Tensor::VOID && !aipubt::Tensor::is_float(o_aipu_dtype))
            {
                o_dtype = o_aipu_dtype;
            }
            auto qmin = dtype_utils::dtype_to_range_min(o_dtype);
            auto qmax = dtype_utils::dtype_to_range_max(o_dtype);
            float p_constant_value = quant_utils::linear_quantize_clip(constant_value, o_scale, o_zerop, qmin, qmax);
            node->params()["constant_value"] = static_cast<int32_t>(p_constant_value);
        }
    }
}

REGISTER_QUANTIZATION_OP(Pad);

} // namespace qtlibn

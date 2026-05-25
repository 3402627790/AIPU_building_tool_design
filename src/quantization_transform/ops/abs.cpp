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
 * @Date: 2022-04-03 19:28:30
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-20 15:09:44
 */

#include "quantization_transform/ops/abs.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{
void AbsOp::quantize(aipubt::NodePtr node)
{
    auto inp_t = node->inputs()[0];
    auto out_t = node->outputs()[0];
    if (node->attrs().count("tflite_quantization") && (bool)node->get_attr("tflite_quantization"))
    {
        Op::quantize(node);
        dtype_utils::set_aipu_dtype(out_t, false);
        float out_scale = out_t->quantization().scale();
        float inp_scale = inp_t->quantization().scale();
        if (out_scale != inp_scale)
        {
            float doscale;
            int8_t doshift;
            int32_t multiplier_bits = out_t->quantization().bits();
            if (node->attrs().count("multiplier_bits"))
            {
                multiplier_bits = node->get_attr("multiplier_bits");
            }
            quant_utils::get_scale_approximation_params(out_scale / inp_scale, doscale, doshift, multiplier_bits);
            auto scale_type = dtype_utils::bits_to_dtype(multiplier_bits, false);
            node->params()["scale_value"] = static_cast<int32_t>(doscale);
            node->params()["scale_type"] = scale_type;
            node->params()["shift_value"] = doshift;
            node->params()["shift_type"] = aipubt::Tensor::INT8;
        }
    }
    else
    {
        NoQuantization::quantize_transparent(out_t, inp_t);
    }
}

REGISTER_QUANTIZATION_OP(Abs);

}; // namespace qtlibn

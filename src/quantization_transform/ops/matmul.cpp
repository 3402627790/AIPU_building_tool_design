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
 * @Date: 2022-04-06 18:49:50
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-11 11:22:26
 */

#include "quantization_transform/ops/matmul.h"

namespace qtlibn
{
void MatMulOp::quantize(aipubt::NodePtr node)
{
    auto inp0 = node->inputs()[0];
    auto inp1 = node->inputs()[1];
    auto out = node->outputs()[0];
    auto out_bits = out->quantization().bits();
    bool out_sign = aipubt::Tensor::is_signed(inp0->quantization().aipu_dtype()) ||
                    aipubt::Tensor::is_signed(inp1->quantization().aipu_dtype());

    dtype_utils::set_aipu_dtype(out, out_sign);

    Op::quantize(node);
    if (inp0->quantization().qinvariant() && inp1->quantization().qinvariant())
    {
        out->quantization().set_scales({1.0});
        out->quantization().set_offsets({0});
    }

    auto inp0_scale = inp0->quantization().scale();
    auto inp1_scale = inp1->quantization().scale();
    auto out_scale = out->quantization().scale();
    float ts = out_scale / inp0_scale / inp1_scale;
    if (node->params().count("fused_multiplier"))
    {
        float fused_multiplier = node->params()["fused_multiplier"];
        ts *= fused_multiplier;
    }
    float do_scale;
    int8_t do_shift;
    int32_t multiplier_bits = out_bits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }

    quant_utils::get_scale_approximation_params(ts, do_scale, do_shift, multiplier_bits);
    auto unsigned_output_dtype = dtype_utils::bits_to_dtype(multiplier_bits, false);
    node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
    node->params()["scale_type"] = unsigned_output_dtype;
    node->params()["shift_value"] = static_cast<uint32_t>(do_shift);
    node->params()["shift_type"] = aipubt::Tensor::INT8;
}

REGISTER_QUANTIZATION_OP(MatMul);

}; // namespace qtlibn
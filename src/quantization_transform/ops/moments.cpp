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

#include "quantization_transform/ops/moments.h"

namespace qtlibn
{
void MomentsOp::quantize(aipubt::NodePtr node)
{
    auto inp = node->inputs()[0];
    auto out0 = node->outputs()[0];
    auto out1 = node->outputs()[1];
    out0->quantization().set_bits(inp->quantization().bits());
    bool out_sign = aipubt::Tensor::is_signed(inp->quantization().aipu_dtype());
    out1->quantization().set_qinvariant(false);

    dtype_utils::set_aipu_dtype(out0, out_sign);
    dtype_utils::set_aipu_dtype(out1, false);

    Op::quantize(node);

    float mean_scale = out0->quantization().scale() / inp->quantization().scale();
    float do_scale0;
    int8_t do_shift0;
    quant_utils::get_scale_approximation_params(mean_scale, do_scale0, do_shift0, out0->quantization().bits());
    auto unsigned_output_dtype = dtype_utils::bits_to_dtype(out1->quantization().bits(), false);
    node->params()["input_scale_value"] = static_cast<uint32_t>(do_scale0);
    node->params()["input_scale_type"] = unsigned_output_dtype;
    node->params()["input_shift_value"] = static_cast<int32_t>(do_shift0);
    node->params()["input_shift_type"] = aipubt::Tensor::INT8;

    float var_scale = out1->quantization().scale() / (out0->quantization().scale() * out0->quantization().scale());
    float do_scale1;
    int8_t do_shift1;
    quant_utils::get_scale_approximation_params(var_scale, do_scale1, do_shift1, out1->quantization().bits());
    node->params()["var_scale_value"] = static_cast<uint32_t>(do_scale1);
    node->params()["var_scale_type"] = unsigned_output_dtype;
    node->params()["var_shift_value"] = static_cast<int32_t>(do_shift1);
    node->params()["var_shift_type"] = aipubt::Tensor::INT8;
}

REGISTER_QUANTIZATION_OP(Moments);

}; // namespace qtlibn
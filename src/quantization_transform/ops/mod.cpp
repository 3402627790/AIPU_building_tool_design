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
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-05-12 17:16:34
 */

#include "quantization_transform/ops/mod.h"

namespace qtlibn
{
void ModOp::quantize(aipubt::NodePtr node)
{
    //set qbit, scale, zerop, qmin, qmax, dtype
    auto inpq1 = node->inputs()[0]->quantization();
    auto inpq2 = node->inputs()[1]->quantization();
    dtype_utils::set_aipu_dtype(node->outputs()[0], aipubt::Tensor::is_signed(inpq1.aipu_dtype()));
    Op::quantize(node);

    float scale0 = 1, scale1 = 1;
    int8_t shift0 = 0, shift1 = 0;
    auto outq = node->outputs()[0]->quantization();
    int q_out_bits = node->outputs()[0]->quantization().bits();
    if (inpq1.qinvariant() && inpq2.qinvariant())
    {
        node->outputs()[0]->quantization().set_qinvariant(true);
        node->outputs()[0]->quantization().set_scale(1);
        node->outputs()[0]->quantization().set_offset(0);
    }
    else if (inpq1.qinvariant())
    {
        node->outputs()[0]->quantization().set_qinvariant(false);
        float local_rescale = outq.scale() / inpq1.scale();
        quant_utils::get_scale_approximation_params(local_rescale, scale0, shift0, q_out_bits);
    }
    else if (inpq2.qinvariant())
    {
        node->outputs()[0]->quantization().set_qinvariant(false);
        float local_rescale = outq.scale() / inpq2.scale();
        quant_utils::get_scale_approximation_params(local_rescale, scale1, shift1, q_out_bits);
    }
    else
    {
        node->outputs()[0]->quantization().set_qinvariant(false);
        float local_rescale = outq.scale() / inpq1.scale();
        quant_utils::get_scale_approximation_params(local_rescale, scale0, shift0, q_out_bits);
        local_rescale = outq.scale() / inpq2.scale();
        quant_utils::get_scale_approximation_params(local_rescale, scale1, shift1, q_out_bits);
    }

    auto o_dtype = dtype_utils::bits_to_dtype(q_out_bits, false);
    node->params()["scale_value"] = {static_cast<uint32_t>(scale0), static_cast<uint32_t>(scale1)};
    node->params()["scale_type"] = {o_dtype, o_dtype};
    node->params()["shift_value"] = {static_cast<uint32_t>(shift0), static_cast<uint32_t>(shift1)};
    node->params()["shift_type"] = {aipubt::Tensor::INT8, aipubt::Tensor::INT8};
}

REGISTER_QUANTIZATION_OP(Mod);

}; // namespace qtlibn

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
 * @Date: 2023-10-27 14:02:05
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-27 14:03:29
 */

#include "quantization_transform/ops/col2im.h"

namespace qtlibn
{
void Col2ImOp::quantize(aipubt::NodePtr node)
{
    bool is_signed = aipubt::Tensor::is_signed(node->inputs()[0]->quantization().aipu_dtype());
    dtype_utils::set_aipu_dtype(node->outputs()[0], is_signed);

    Op::quantize(node);

    float local_rescale = node->outputs()[0]->quantization().scales() / node->inputs()[0]->quantization().scales();
    float do_scale;
    int8_t do_shift;
    int q_out_bits = node->outputs()[0]->quantization().bits();
    int32_t multiplier_bits = q_out_bits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }
    quant_utils::get_scale_approximation_params(local_rescale, do_scale, do_shift, multiplier_bits, node->name());
    node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
    node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
    node->params()["shift_value"] = do_shift;
    node->params()["shift_type"] = aipubt::Tensor::INT8;
}

REGISTER_QUANTIZATION_OP(Col2Im);

}; // namespace qtlibn

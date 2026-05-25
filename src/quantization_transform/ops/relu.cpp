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
 * @Date: 2022-03-23 19:22:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-03-08 16:18:41
 */

#include "quantization_transform/ops/relu.h"

namespace qtlibn
{
void RELUOp::quantize(aipubt::NodePtr node)
{
    // infer output sign
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);

    Op::quantize(node);

    auto q_out_bits = node->outputs()[0]->quantization().bits();
    auto aipu_input_scale = node->inputs()[0]->quantization().scale();
    auto aipu_output_scale = node->outputs()[0]->quantization().scale();

    float do_scale;
    int8_t do_shift;
    float ts = aipu_output_scale / aipu_input_scale;

    int32_t multiplier_bits = q_out_bits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }
    quant_utils::get_scale_approximation_params(ts, do_scale, do_shift, multiplier_bits, node->name());
    node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
    node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
    node->params()["shift_value"] = do_shift;
    node->params()["shift_type"] = aipubt::Tensor::INT8;
}

REGISTER_QUANTIZATION_OP(RELU);

}; // namespace qtlibn

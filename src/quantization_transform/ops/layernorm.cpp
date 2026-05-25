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
 * @Last Modified time: 2024-04-28 15:43:57
 */

#include "quantization_transform/ops/layernorm.h"
#include "quantization_transform/ops/groupnorm.h"

namespace qtlibn
{

// void LayerNormOp::quantize(aipubt::NodePtr node)
// {
//     // bool sign = aipubt::Tensor::is_signed(node->inputs()[0]->quantization().aipu_dtype());
//     // if (node->outputs()[0]->quantization().mins().size() > 0)
//     // {
//     //     auto mins = node->outputs()[0]->quantization().mins()[0];
//     //     if (mins >= 0)
//     //     {
//     //         sign = false;
//     //     }
//     // }
//     // dtype_utils::set_aipu_dtype(node->outputs()[0], sign);
//     // node->placeholders()[0]->quantization().set_bits(node->outputs()[0]->quantization().bits());
//     // node->placeholders()[1]->quantization().set_bits(node->outputs()[0]->quantization().bits());
//     // dtype_utils::set_aipu_dtype(node->placeholders()[0], false, false);
//     // dtype_utils::set_aipu_dtype(node->placeholders()[1], sign, false);
//     // dtype_utils::set_aipu_dtype(node->constants()["weights"], true, false);
//     // dtype_utils::set_aipu_dtype(node->constants()["biases"], true, false);
//     // Op::quantize(node);
//     // quant_utils::aipu_quantize(node->constants()["weights"]);
//     // quant_utils::aipu_quantize(node->constants()["biases"]);
//     // groupnorm_utils::groupnorm_quantize(node);

//     GroupNormOp::quantize(node);
// }

REGISTER_QUANTIZATION_OP(LayerNorm);
}; // namespace qtlibn

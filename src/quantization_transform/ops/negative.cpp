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
 * @Last Modified by: Matt Chen
 * @Last Modified time: 2022-04-06 19:47:51
 */

#include "quantization_transform/ops/negative.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{
void NegativeOp::quantize(aipubt::NodePtr node)
{
    NoQuantization::quantize_transparent(node->outputs()[0], node->inputs()[0]);
    if (!aipubt::Tensor::is_signed(node->outputs()[0]->quantization().aipu_dtype()))
    {
        node->outputs()[0]->quantization().set_aipu_dtype(aipubt::Tensor::VOID);
        dtype_utils::set_aipu_dtype(node->outputs()[0], true);
    }
    // if (node->outputs()[0]->quantization().scheme() == aipubt::Quantization::Scheme::Asymmetric)
    {
        int zerop = node->inputs()[0]->quantization().offset();
        bool in_signed = aipubt::Tensor::is_signed(node->inputs()[0]->quantization().aipu_dtype());
        if (in_signed)
        {
            node->outputs()[0]->quantization().set_offset(-1 * zerop + 1);
        }
        else
        {
            auto out_dtype = node->outputs()[0]->quantization().aipu_dtype();
            int qmax = dtype_utils::dtype_to_range_max(out_dtype);
            node->outputs()[0]->quantization().set_offset(-1 * zerop - qmax);
        }
    }
}
REGISTER_QUANTIZATION_OP(Negative);
}; // namespace qtlibn

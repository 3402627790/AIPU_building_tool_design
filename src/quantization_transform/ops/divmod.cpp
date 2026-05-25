
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
 * @Date: 2023-10-17 10:35:11
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-18 15:53:42
 */

#include "quantization_transform/ops/divmod.h"

namespace qtlibn
{
void DivModOp::quantize(aipubt::NodePtr node)
{
    auto inp_t0 = node->inputs()[0];
    auto inp_t1 = node->inputs()[1];

    if (!(inp_t0->quantization().qinvariant() && inp_t1->quantization().qinvariant()))
    {
        ERROR() << node << ": two inputs should be qinvariant when wants to quantization.";
    }
    else
    {
        if (inp_t0->quantization().aipu_dtype() != inp_t1->quantization().aipu_dtype())
        {
            WARN() << node << ": two inputs have different quantization dtype: inp0.dtype = "
                   << inp_t0->quantization().aipu_dtype() << ", inp1.dtype = "
                   << inp_t1->quantization().aipu_dtype()
                   << ", which maybe cause quantization acc issue.";
        }

        auto inp0_dtype = inp_t0->quantization().aipu_dtype();
        dtype_utils::set_aipu_dtype(node->outputs()[0], aipubt::Tensor::is_signed(inp0_dtype));
        dtype_utils::set_aipu_dtype(node->outputs()[1], aipubt::Tensor::is_signed(inp0_dtype));
        for (auto &out : node->outputs())
        {
            out->quantization().set_scales({1.0});
            out->quantization().set_offsets({0});
            out->quantization().set_qinvariant(true);
        }
    }
}

REGISTER_QUANTIZATION_OP(DivMod);

}; // namespace qtlibn

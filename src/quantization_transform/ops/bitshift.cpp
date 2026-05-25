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
 * @Date: 2022-05-10 20:23:53
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-02 20:27:35
 */

#include "quantization_transform/ops/bitshift.h"

namespace qtlibn
{

void BitShiftOp::quantize(aipubt::NodePtr node)
{
    auto inp0_qinvariant = node->inputs()[0]->quantization().qinvariant();
    auto inp1_qinvariant = node->inputs()[1]->quantization().qinvariant();

    if (!inp0_qinvariant || !inp1_qinvariant)
    {
        ERROR() << "BitShift op needs two qinvariant inputs when quantizing.";
    }
    else
    {
        auto inp0_dtype = node->inputs()[0]->quantization().aipu_dtype();
        auto out_sign = aipubt::Tensor::is_signed(inp0_dtype);
        dtype_utils::set_aipu_dtype(node->outputs()[0], out_sign);
        node->outputs()[0]->quantization().set_scales({1.0});
        node->outputs()[0]->quantization().set_offsets({0});
        node->outputs()[0]->quantization().set_qinvariant(true);
    }
}

REGISTER_QUANTIZATION_OP(BitShift);

} // namespace qtlibn

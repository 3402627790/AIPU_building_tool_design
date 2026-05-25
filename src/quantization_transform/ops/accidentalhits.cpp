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
 * @Last Modified by: Ruodi Lu
 * @Last Modified time: 2022-04-21 10:44:20
 */

#include "quantization_transform/ops/accidentalhits.h"

namespace qtlibn
{
void AccidentalHitsOp::quantize(aipubt::NodePtr node)
{
    //output0 & output1
    for (int i = 0; i < 2; i++)
    {
        int32_t out_num = node->inputs()[i]->shape()[0];
        int32_t out_bits = dtype_utils::range_to_bits(0, out_num, false);
        node->outputs()[i]->quantization().set_scales({1.0});
        node->outputs()[i]->quantization().set_offsets({0});
        int32_t max_bits = std::max(16, out_bits);
        node->outputs()[i]->quantization().set_bits(max_bits);
        dtype_utils::set_aipu_dtype(node->outputs()[i], false);
        node->outputs()[i]->quantization().set_qinvariant(true);
    }

    //output2
    int32_t o_bits = node->outputs()[2]->quantization().bits();
    auto out2 = node->outputs()[2];
    out2->quantization().set_scales({1.0});
    out2->quantization().set_offsets({0});
    int32_t max_bits = std::max(16, o_bits);
    out2->quantization().set_bits(max_bits);
    dtype_utils::set_aipu_dtype(out2, false);
    out2->quantization().set_qinvariant(true);
}

REGISTER_QUANTIZATION_OP(AccidentalHits);

}; // namespace qtlibn

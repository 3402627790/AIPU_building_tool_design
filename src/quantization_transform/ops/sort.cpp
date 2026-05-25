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
 * @Date: 2022-05-20 09:55:11
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-20 10:13:04
 */

#include "quantization_transform/ops/sort.h"

namespace qtlibn
{
void SortOp::quantize(aipubt::NodePtr node)
{
    auto inp0_t = node->inputs()[0];
    auto out0_t = node->outputs()[0];
    NoQuantization::quantize_transparent(out0_t, inp0_t);

    auto out1_t = node->outputs()[1];
    auto o1_bits = out1_t->quantization().bits();
    o1_bits = std::max(o1_bits, 16);
    out1_t->quantization().set_bits(o1_bits);
    dtype_utils::set_aipu_dtype(out1_t, false);
    out1_t->quantization().set_scales({1.0});
    out1_t->quantization().set_offsets({0});
    out1_t->quantization().set_qinvariant(true);
}

REGISTER_QUANTIZATION_OP(Sort);

} //namespace qtlibn

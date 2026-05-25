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
 * @Date: 2022-04-11 09:59:38
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-11 10:33:54
 */

#include "quantization_transform/ops/ctcgreedydecoder.h"

namespace qtlibn
{

void CTCGreedyDecoderOp::quantize(aipubt::NodePtr node)
{
    auto out = node->outputs()[0];
    out->quantization().set_scales({1.0});
    out->quantization().set_offsets({0});
    int32_t max_bits = std::max(16, out->quantization().bits());
    out->quantization().set_bits(max_bits);
    dtype_utils::set_aipu_dtype(out, false);
    out->quantization().set_qinvariant(true);
}

REGISTER_QUANTIZATION_OP(CTCGreedyDecoder);

} // namespace qtlibn

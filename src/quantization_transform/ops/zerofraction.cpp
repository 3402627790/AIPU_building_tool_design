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
 * @Date: 2022-05-20 00:12:53
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-20 08:45:40
 */
#include "quantization_transform/ops/zerofraction.h"

namespace qtlibn
{
void ZeroFractionOp::quantize(aipubt::NodePtr node)
{
    auto o_bits = node->outputs()[0]->quantization().bits();
    float enlarge_scale = pow(2, o_bits) - 1;
    node->params()["output_scale"] = static_cast<int32_t>(enlarge_scale);

    auto out_t = node->outputs()[0];
    dtype_utils::set_aipu_dtype(out_t, false);
    out_t->quantization().set_scales({enlarge_scale});
    out_t->quantization().set_offsets({0});
    out_t->quantization().set_qinvariant(false);
}

REGISTER_QUANTIZATION_OP(ZeroFraction);

} // namespace qtlibn

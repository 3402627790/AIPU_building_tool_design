//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-05-17 19:44:54
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-17 20:13:22
 */

#include "quantization_transform/ops/intopk.h"

namespace qtlibn
{
void InTopKOp::quantize(aipubt::NodePtr node)
{
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);
    node->outputs()[0]->quantization().set_scales({1.0});
    node->outputs()[0]->quantization().set_offsets({0});
    node->outputs()[0]->quantization().set_qinvariant(true);
}

REGISTER_QUANTIZATION_OP(InTopK);

} // namespace qtlibn

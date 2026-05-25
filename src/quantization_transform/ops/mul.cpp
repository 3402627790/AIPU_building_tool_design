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
 * @Date: 2022-05-10 20:04:22
 * @Last Modified by: Ruodi Lu
 * @Last Modified time: 2022-07-14 11:43:32
 */

#include "quantization_transform/ops/mul.h"

namespace qtlibn
{
void MulOp::quantize(aipubt::NodePtr node)
{
    //set method in param
    node->params()["method"] = "MUL";
    if (!node->params().count("with_activation"))
    {
        node->params()["with_activation"] = "NONE";
    }
    BaseEltwiseOp::quantize(node);
    node->params().erase("method");
}

REGISTER_QUANTIZATION_OP(Mul);

}; // namespace qtlibn

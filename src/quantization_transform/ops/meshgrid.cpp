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
 * @Date: 2023-10-23 14:54:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-23 15:03:35
 */

#include "quantization_transform/ops/meshgrid.h"

namespace qtlibn
{

void MeshgridOp::quantize(aipubt::NodePtr node)
{
    for (int32_t i = 0; i < (int)node->inputs().size(); i++)
    {
        auto inp = node->inputs()[i];
        auto out = node->outputs()[i];
        if (!inp->quantization().qinvariant())
        {
            WARN() << node << ": currently only support int input and qinvariant is true.";
        }

        NoQuantization::quantize_transparent(out, inp);
    }
}

REGISTER_QUANTIZATION_OP(Meshgrid);
}; // namespace qtlibn

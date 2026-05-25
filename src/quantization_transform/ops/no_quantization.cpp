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
 * @Date: 2022-04-03 19:38:36
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 22:16:19
 */

#include "quantization_transform/ops/no_quantization.h"
#include "quantization_transform/ops/op_register.h"

namespace qtlibn
{

void NoQuantization::quantize(aipubt::NodePtr node)
{
    auto adaptor = get_adaptor();
    if (adaptor)
    {
        Op::quantize(node);
    }
    // default update inputs[0] quantization to output
    auto inp_t = node->inputs()[0];
    for (auto ot : node->outputs())
    {
        if (adaptor && !ot->quantization().empty() && ot->quantization() != inp_t->quantization())
        {
            WARN() << "node: " << node
                   << " is noquantization in qtlib, but its input.quantization() != output.quantization()(:"
                   << ot << ")";
        }
        quantize_transparent(ot, inp_t);
    }
}

void NoQuantization::quantize_transparent(aipubt::TensorPtr dst_t, aipubt::TensorPtr src_t)
{
    dst_t->quantization() = src_t->quantization();
    dst_t->set_type(src_t->type());
}

#define REGISTER_NO_QUANTIZATION_OP(class_type)  \
    static NoQuantization register_##class_type; \
    static OpRegisterer g_creator_##class_type(#class_type, &register_##class_type)

REGISTER_NO_QUANTIZATION_OP(IsNaN);
REGISTER_NO_QUANTIZATION_OP(IsInf);

}; // namespace qtlibn

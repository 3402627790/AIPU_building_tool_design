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
 * @Author: Kejia Qiang
 * @Date: 2022-11-10 20:24:00
 * @Last Modified by: Kejia Qiang
 * @Last Modified time: 2022-11-10 20:25:55
 */
#include "quantization_transform/ops/bitwise.h"

namespace qtlibn
{

void BitwiseOp::quantize(aipubt::NodePtr node)
{
    size_t inp_num = node->inputs().size();
    for (unsigned int i = 0; i < inp_num; i++)
    {
        if (!node->inputs()[i]->quantization().qinvariant())
        {
            WARN() << "Bitwise op needs all qinvariant inputs when quantizing.";
        }
    }
    /* it is hard for qtlib to check the input qinvariant of bitwise.
     * when input0 + input1 -> bitwise, if input0 and input1 is quantized, these
     * two inputs should set qinvariant = true. but if input0 and input1, qtlib is
     * difficult to set qinvariant=true in input op. so now in bitwise, we only give
     * a warning, and continue to quantize the output tensor.
     */
    auto output_dtype = node->outputs()[0]->type();
    auto out_sign = aipubt::Tensor::is_signed(output_dtype);
    dtype_utils::set_aipu_dtype(node->outputs()[0], out_sign);
    node->outputs()[0]->quantization().set_scales({1.0});
    node->outputs()[0]->quantization().set_offsets({0});
    node->outputs()[0]->quantization().set_qinvariant(true);
}

REGISTER_QUANTIZATION_OP(Bitwise);

} // namespace qtlibn

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
 * @Date: 2022-04-06 19:22:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-12-28 11:40:42
 */

#include "quantization_transform/ops/constant.h"

namespace qtlibn
{
void ConstantOp::quantize(aipubt::NodePtr node)
{
    if (!node->constants().count("weights"))
    {
        ERROR() << "Constant layer must have weights constants!";
        return;
    }
    aipubt::TensorPtr output = node->outputs()[0];
    aipubt::TensorPtr weights = node->constants()["weights"];
    if (output->quantization().qinvariant())
    {
        output->quantization().set_aipu_dtype(output->type());
        weights->quantization().set_aipu_dtype(weights->type());
        weights->quantization() = output->quantization();
        output->quantization().set_qinvariant(false);
    }
    else if (weights->quantization().qinvariant() || weights->quantization().data_unchanged())
    {
        Op::quantize(node);

        if (!weights->quantization().data_unchanged())
        {
            weights->quantization().set_qinvariant(true);
        }
        output->quantization() = weights->quantization();
        dtype_utils::set_aipu_dtype(output, aipubt::Tensor::is_signed(output->type()));
    }
    else
    {
        bool out_signed = aipubt::Tensor::is_signed(node->outputs()[0]->type());
        if (weights->type() == aipubt::Tensor::FP32)
        {
            float mins = weights->min<float_t>();
            if (mins >= 0.)
            {
                out_signed = false;
            }
        }
        dtype_utils::set_aipu_dtype(output, out_signed);
        dtype_utils::set_aipu_dtype(weights, out_signed, false);
        Op::quantize(node);
        //set output quantization info to weights_constant
        weights->quantization() = output->quantization();
        quant_utils::aipu_quantize(node->constants()["weights"]);
    }
    if (node->params().count("weights_scale"))
    {
        node->params()["weights_scale"] = node->constants()["weights"]->quantization().scales();
    }
    if (node->params().count("weights_zp"))
    {
        node->params()["weights_zp"] = node->constants()["weights"]->quantization().offsets();
    }
}

REGISTER_QUANTIZATION_OP(Constant);

}; // namespace qtlibn

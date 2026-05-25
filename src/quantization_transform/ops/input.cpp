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
 * @Date: 2020-05-08 15:11:55
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-12 20:56:01
 */

#include "quantization_transform/ops/input.h"

namespace qtlibn
{
template <typename Dtype>
static void forward_cast(aipubt::NodePtr node)
{
    for (aipubt::TensorPtr ot : node->outputs())
    {
        if (ot->quantization().quantized())
        {
            float_t *ot_fdata = ot->data<float_t>();
            aipubt::TensorShape ot_shape = ot->shape();
            Dtype *ot_qdata = new Dtype[ot_shape.size()];
            for (int i = 0; i < ot_shape.size(); ++i)
            {
                *(ot_qdata + i) = (Dtype)quant_utils::linear_quantize_clip(*(ot_fdata + i), ot->quantization().scale(), ot->quantization().offset(), std::numeric_limits<Dtype>::min(), std::numeric_limits<Dtype>::max());
            }
            ot->set_data(ot_qdata, ot_shape.size() * sizeof(Dtype));
        }
    }
}
static std::map<aipubt::Tensor::Dtype, std::function<void(aipubt::NodePtr)>> DTYPE_TO_INPUT_FORWARD_MAP{
    {aipubt::Tensor::UINT8, forward_cast<uint8_t>},
    {aipubt::Tensor::INT8, forward_cast<int8_t>},
    {aipubt::Tensor::UINT16, forward_cast<uint16_t>},
    {aipubt::Tensor::INT16, forward_cast<int16_t>},
    {aipubt::Tensor::FP32, forward_cast<float_t>},
};
void InputOp::forward(aipubt::NodePtr node)
{
    aipubt::Tensor::Dtype dtype = node->outputs()[0]->type();
    DTYPE_TO_INPUT_FORWARD_MAP[dtype](node);
}
void InputOp::quantize(aipubt::NodePtr node)
{
    bool out_signed = node->outputs()[0]->type() == aipubt::Tensor::BOOL ? false : aipubt::Tensor::is_signed(node->outputs()[0]->type());
    if (node->outputs()[0]->quantization().mins().size() > 0)
    {
        auto mins = node->outputs()[0]->quantization().mins()[0];
        out_signed = mins < 0;
    }
    dtype_utils::set_aipu_dtype(node->outputs()[0], out_signed);

    Op::quantize(node);
    // TODO: inp inqvariant
}

REGISTER_QUANTIZATION_OP(Input);
}; // namespace qtlibn
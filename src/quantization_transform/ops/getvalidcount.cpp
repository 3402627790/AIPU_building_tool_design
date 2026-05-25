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
 * @Date: 2022-11-11 11:03:20
 * @Last Modified by: Kejia Qiang
 * @Last Modified time: 2022-11-11 11:05:32
 */

#include "quantization_transform/ops/getvalidcount.h"

namespace qtlibn
{

void GetValidCountOp::quantize(aipubt::NodePtr node)
{
    /*
    IN1 -> [batch, anchors, 6(or other)]
    OUT1 -> [count]
    OUT2 -> [batch, anchors, 6(or other)] (rearranged)
    OUT3 -> [batch, anchors] (index)
    */
    float in_scale, in_zerop, qmin, qmax;
    float score_thresh = node->params()["score_threshold"];
    aipubt::Tensor::Dtype aipu_input_dtype = node->inputs()[0]->quantization().aipu_dtype();
    auto out1 = node->outputs()[0];
    auto out2 = node->outputs()[1];
    auto out3 = node->outputs()[2];
    bool sign = aipubt::Tensor::is_signed(node->inputs()[0]->type());

    node->params().erase(node->params().find("score_threshold"));
    out1->quantization().set_bits(16);
    out2->quantization().set_bits(node->inputs()[0]->quantization().bits());
    out3->quantization().set_bits(16);
    dtype_utils::set_aipu_dtype(node->outputs()[1], sign);
    dtype_utils::set_aipu_dtype(node->outputs()[0], false);
    dtype_utils::set_aipu_dtype(node->outputs()[2], false);
    Op::quantize(node);

    in_scale = node->inputs()[0]->quantization().scale();
    in_zerop = node->inputs()[0]->quantization().offset();
    qmin = dtype_utils::dtype_to_range_min(aipu_input_dtype);
    qmax = dtype_utils::dtype_to_range_max(aipu_input_dtype);
    score_thresh = quant_utils::linear_quantize_clip(score_thresh, in_scale, in_zerop, qmin, qmax);
    node->params()["score_threshold_value"] = static_cast<int>(score_thresh);
    node->params()["score_threshold_type"] = node->inputs()[0]->type();

    out1->quantization().set_scale(1);
    out1->quantization().set_offset(0);
    out1->quantization().set_qinvariant(true);

    out2->quantization() = node->inputs()[0]->quantization();

    out3->quantization().set_scale(1);
    out3->quantization().set_offset(0);
    out3->quantization().set_qinvariant(true);
}

REGISTER_QUANTIZATION_OP(GetValidCount);

} // namespace qtlibn

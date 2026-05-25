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
 * @Date: 2022-04-15 13:51:00
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-17 23:27:07
 */

#include "quantization_transform/ops/div.h"

namespace qtlibn
{

void DivOp::quantize(aipubt::NodePtr node)
{
    // infer output sign
    auto inp0_t = node->inputs()[0];
    auto inp1_t = node->inputs()[1];
    auto inp0_signed = aipubt::Tensor::is_signed(inp0_t->quantization().aipu_dtype());
    auto inp1_signed = aipubt::Tensor::is_signed(inp1_t->quantization().aipu_dtype());
    auto out_signed = inp0_signed || inp1_signed;
    dtype_utils::set_aipu_dtype(node->outputs()[0], out_signed);

    Op::quantize(node);

    auto inp1_bits = inp1_t->quantization().bits();
    auto lut_in_bits = inp1_bits;
    /*after input1+zp1, the range of input1 + zp1 is [-255, 255],
     *but in forward will abs(input1+zp1), which the range is [0, 255],
     *so the lut input bits is 8bits.
    */
    // if (inp1_t->quantization().offset() != 0)
    // {
    //     lut_in_bits += 1;
    // }
    auto lut_in_qmin = dtype_utils::bits_to_range_min(lut_in_bits, false);
    auto lut_in_qmax = dtype_utils::bits_to_range_max(lut_in_bits, false);

    auto lut_zoom_value = pow(2, 16) - 1;
    int32_t lut_len = pow(2, inp1_bits);
    if (node->attrs().count(Constants::lut_items_in_bits))
    {
        int32_t bits = node->get_attr<aipubt::NodeParamValue>(Constants::lut_items_in_bits);
        lut_len = pow(2, std::min(inp1_bits, bits));
    }
    auto step = (lut_in_qmax - lut_in_qmin) / (lut_len - 1);

    int32_t lut_idx = 0;
    //16bits lut table
    uint16_t *lut = new uint16_t[lut_len];
    for (float idx = lut_in_qmin; idx <= lut_in_qmax; idx += step, lut_idx++)
    {
        float value = lut_zoom_value / idx;
        value = quant_utils::clip<float, uint16_t>(round(value), 0, lut_zoom_value);
        lut[lut_idx] = static_cast<uint16_t>(value);
    }
    std::string lut_name = "div_lut_tensor";
    aipubt::TensorPtr lut_tensor = std::make_shared<aipubt::Tensor>(lut_name, aipubt::TensorShape(lut_len), aipubt::Tensor::UINT16);
    lut_tensor->set_data(lut, lut_len * sizeof(uint16_t));
    lut_tensor->quantization().set_bits(16);
    lut_tensor->quantization().set_aipu_dtype(aipubt::Tensor::UINT16);

    node->constants()["lut"] = lut_tensor;

    auto inp0_scale = inp0_t->quantization().scale();
    auto inp1_scale = inp1_t->quantization().scale();
    auto out_scale = node->outputs()[0]->quantization().scale();
    auto out_bits = node->outputs()[0]->quantization().bits();
    float ts = (out_scale * inp1_scale) / (inp0_scale * lut_zoom_value);
    float do_scale;
    int8_t do_shift;
    int32_t multiplier_bits = out_bits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }
    auto unsigned_output_dtype = dtype_utils::bits_to_dtype(multiplier_bits, false);
    quant_utils::get_scale_approximation_params(ts, do_scale, do_shift, multiplier_bits);
    node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
    node->params()["shift_value"] = do_shift;
    node->params()["scale_type"] = unsigned_output_dtype;
    node->params()["shift_type"] = aipubt::Tensor::INT8;
}

REGISTER_QUANTIZATION_OP(Div);

}; // namespace qtlibn
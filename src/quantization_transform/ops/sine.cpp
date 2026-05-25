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
 * @Date: 2022-04-15 13:50:13
 * @Last Modified by: Ruodi Lu
 * @Last Modified time: 2022-04-21 12:57:16
 */

#include "quantization_transform/ops/sine.h"

namespace qtlibn
{

static aipubt::ComputeTensor cal_sine_func(aipubt::ComputeTensor value, std::map<std::string, float> cal_param = {})
{
    return value.sin();
}

void SineOp::quantize(aipubt::NodePtr node)
{
    // infer output sign
    dtype_utils::set_aipu_dtype(node->outputs()[0], true);
    int qbits = (int)node->get_attr("q_bits_activation");
    int lut_bits = 10;
    if (node->attrs().count("lut_items_in_bits"))
    {
        lut_bits = node->get_attr("lut_items_in_bits");
    }
    int multiplier_bits = qbits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }
    int lsteps = std::min(1 << lut_bits, 1 << qbits);

    Op::quantize(node);
    if (node->outputs()[0]->quantization().bits() <= 8)
    {
        node->constants()["lut"] = generate_lut(node, cal_sine_func);
    }
    else
    {
        float_t qmax = (float_t)dtype_utils::bits_to_range_max(qbits, true);
        auto output = node->outputs()[0];
        output->quantization().set_scale(qmax);
        output->quantization().set_offset(0);
        output->quantization().set_min(-qmax);
        output->quantization().set_max(qmax);
        dtype_utils::set_aipu_dtype(node->outputs()[0], true);

        auto lut = aipubt::ComputeTensor::linspace(0, 2 * 3.1415926f, lsteps);
        lut = lut.sin();
        quant_utils::linear_quantize_clip(lut, aipubt::ComputeTensor(output->quantization().scale()),
                                          aipubt::ComputeTensor(output->quantization().offset()),
                                          output->quantization().mins()[0], output->quantization().maxs()[0]);
        node->constants()["lut"] = lut.astype(output->type());

        float_t local_scale = ((float_t)lsteps / 2 / 3.1415926f) /
                              node->inputs()[0]->quantization().scale();
        float_t do_scale;
        int8_t do_shift;
        quant_utils::get_scale_approximation_params(local_scale, do_scale, do_shift,
                                                    multiplier_bits);
        node->params()["scale_value"] = static_cast<uint32_t>(do_scale);
        node->params()["scale_type"] = dtype_utils::bits_to_dtype(qbits, false);
        node->params()["shift_value"] = do_shift;
        node->params()["shift_type"] = aipubt::Tensor::INT8;
    }
}

REGISTER_QUANTIZATION_OP(Sine);

}; // namespace qtlibn
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
 * @Date: 2022-11-03 19:28:20
 * @Last Modified by: Kejia Qiang
 * @Last Modified time: 2022-11-18 14:28:50
 */

#include "quantization_transform/ops/cumulate.h"

static uint32_t convert_less_mbit(int32_t &data, int32_t threshold_min, int32_t threshold_max)
{
    uint32_t left_shift = 0;
    while ((data > threshold_max) || (data < threshold_min))
    {
        data >>= 1;
        left_shift += 1;
    }
    return left_shift;
}

namespace qtlibn
{

void CumulateOp::quantize(aipubt::NodePtr node)
{
    std::string method = node->params()["method"];
    std::vector<int32_t> axes = node->params()["axis"];
    bool exclusive = node->params()["exclusive"];

    int32_t obits = node->outputs()[0]->quantization().bits();
    int32_t multiplier_bits = obits;
    bool sign = aipubt::Tensor::is_signed(node->inputs()[0]->type());

    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }

    dtype_utils::set_aipu_dtype(node->outputs()[0], sign);
    Op::quantize(node);

    float in_scale = node->inputs()[0]->quantization().scale();
    float out_scale = node->outputs()[0]->quantization().scale();
    float local_scale;

    if (method == "SUM")
    {
        multiplier_bits = std::max(multiplier_bits, 15);
        float do_scale;
        int8_t do_shift;
        local_scale = out_scale / in_scale;
        quant_utils::get_scale_approximation_params(local_scale, do_scale, do_shift, multiplier_bits);
        node->params()["shift_value"] = (int)do_shift;
        node->params()["shift_type"] = aipubt::Tensor::INT8;
        node->params()["scale_value"] = (int)do_scale;
        node->params()["scale_type"] = dtype_utils::bits_to_dtype(multiplier_bits, false);
    }
    else // PROD
    {
        static float inp_scale, oup_scale;
        static int8_t inp_shift, oup_shift;
        int32_t current_scale = 1.0;
        int32_t current_shift = 0.0;
        int32_t left_shift = 0;
        float_t *do_scale_table;
        float_t *do_shift_table;
        int32_t scale_min = dtype_utils::bits_to_range_min(15, false);
        int32_t scale_max = dtype_utils::bits_to_range_max(15, false);

        in_scale = 1 / in_scale;
        local_scale = out_scale;
        auto input_shape = node->inputs()[0]->shape();
        uint32_t step = 1;
        for (auto &i : axes)
        {
            step *= input_shape[i];
        }

        do_scale_table = new float_t[step]();
        do_shift_table = new float_t[step]();

        quant_utils::get_scale_approximation_params(in_scale, inp_scale, inp_shift, multiplier_bits);
        quant_utils::get_scale_approximation_params(local_scale, oup_scale, oup_shift, multiplier_bits);

        if (exclusive)
        {
            do_scale_table[0] = oup_scale;
            do_shift_table[0] = oup_shift;
        }
        else
        {
            current_scale = inp_scale;
            current_shift = inp_shift;
            int32_t tmp_scale = current_scale * oup_scale;
            int32_t out_left_shift;
            out_left_shift = convert_less_mbit(tmp_scale, scale_min, scale_max);
            do_scale_table[0] = tmp_scale;
            do_shift_table[0] = current_shift + oup_shift - out_left_shift;
        }

        int32_t shift_min = (int)do_shift_table[0];
        int32_t shift_max = (int)do_shift_table[0];

        for (auto s = 1u; s < step; s++)
        {
            current_scale *= inp_scale;
            current_shift += inp_shift;
            int32_t inp_left_shift;
            inp_left_shift = convert_less_mbit(current_scale, scale_min, scale_max);
            left_shift += inp_left_shift;
            int32_t tmp_scale = current_scale * oup_scale;
            int32_t out_left_shift;
            out_left_shift = convert_less_mbit(tmp_scale, scale_min, scale_max);
            do_scale_table[s] = tmp_scale;
            do_shift_table[s] = current_shift + oup_shift - left_shift - out_left_shift;
            if (shift_min > do_shift_table[s])
                shift_min = (int)do_shift_table[s];
            if (shift_max < do_shift_table[s])
                shift_max = (int)do_shift_table[s];
        }

        aipubt::TensorPtr scale_tensor_ptr = std::make_shared<aipubt::Tensor>("scale_tensor",
                                                                              aipubt::TensorShape(step), aipubt::Tensor::UINT16);
        quant_utils::convert_data_to_tensor<float_t>(scale_tensor_ptr, do_scale_table, step);
        aipubt::TensorPtr shift_tensor_ptr = std::make_shared<aipubt::Tensor>("shift_tensor",
                                                                              aipubt::TensorShape(step), aipubt::Tensor::INT32);
        quant_utils::convert_data_to_tensor<float_t>(shift_tensor_ptr, do_shift_table, step);

        node->constants()["scale"] = scale_tensor_ptr;
        node->constants()["shift"] = shift_tensor_ptr;
    }
}

REGISTER_QUANTIZATION_OP(Cumulate);
} // namespace qtlibn

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
 * @Date: 2022-11-03 19:28:30
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-06-08 11:15:21
 */

#include "quantization_transform/ops/dilation2d.h"

namespace qtlibn
{

void DilationOp::quantize(aipubt::NodePtr node)
{
    aipubt::TensorPtr input = node->inputs()[0];
    aipubt::TensorPtr output = node->outputs()[0];
    aipubt::TensorPtr weight = node->constants()["weights"];
    int32_t obits = output->quantization().bits();
    int32_t multiplier_bits = obits;
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }
    bool w_sign = true;
    if (weight->quantization().mins().size() == 0)
    {
        if (aipubt::Tensor::is_float(weight->type()))
        {
            w_sign = aipubt::Tensor::is_signed(weight->type());
        }
    }
    else
    {
        auto mins = weight->quantization().mins();
        w_sign = mins[0] < 0;
    }
    dtype_utils::set_aipu_dtype(node->constants()["weights"], w_sign, false);
    bool out_signed = aipubt::Tensor::is_signed(input->type()) || w_sign;
    if (node->type() == aipubt::OpType_Erosion)
    {
        out_signed = true;
    }
    dtype_utils::set_aipu_dtype(output, out_signed);

    Op::quantize(node);
    quant_utils::aipu_quantize(node->constants()["weights"]);

    uint32_t g_eltwise_scale_bits = 8;
    float_t inp0_scale = input->quantization().scale();
    float_t inp1_scale = weight->quantization().scale();
    float_t out_scale = output->quantization().scale();
    float_t inp_scale_max = std::max<float>(inp0_scale, inp1_scale);
    uint32_t clip_max = dtype_utils::dtype_to_range_max(aipubt::Tensor::INT16);
    float_t proof_min_ration = (1 << g_eltwise_scale_bits) / clip_max;
    if (inp0_scale / inp1_scale < proof_min_ration)
    {
        inp0_scale = std::min<float>(std::max<float>(inp0_scale, 1.0 / (float)clip_max), clip_max);
        inp1_scale = std::min<float>(std::max<float>(inp1_scale, 1.0 / (float)clip_max), clip_max);
        inp_scale_max = std::max<float>(inp0_scale, inp1_scale);
    }

    float_t local_scale = out_scale / inp_scale_max;
    float_t scale0, scale1;
    scale0 = (inp_scale_max / inp0_scale) * (1 << g_eltwise_scale_bits);
    scale1 = (inp_scale_max / inp1_scale) * (1 << g_eltwise_scale_bits);

    while ((int)scale0 > (int)clip_max || (int)scale1 > (int)clip_max)
    {
        if (scale0 == 1.0)
        {
            WARN() << "layer_type=" << node->type() << ", layer_name = " << node->name()
                   << ", the weight scale=" << scale1 << " has out range [0, " << clip_max
                   << "], please attention.";
        }
        if (scale1 == 1.0)
        {
            WARN() << "layer_type=" << node->type() << ", layer_name = " << node->name()
                   << ", the input scale=" << scale0 << " has out range [0, " << clip_max
                   << "], please attention.";
        }
        scale0 = std::max<float>(round(scale0 / 2), 1.0);
        scale1 = std::max<float>(round(scale1 / 2), 1.0);
        g_eltwise_scale_bits--;
    }

    float do_scale;
    int8_t do_shift;
    auto ts = local_scale * pow(0.5, g_eltwise_scale_bits);
    quant_utils::get_scale_approximation_params(ts, do_scale, do_shift, multiplier_bits);

    node->params()["shift_value"] = (int)do_shift;
    node->params()["shift_type"] = aipubt::Tensor::INT8;
    node->params()["scale_value"] = std::vector<int>{(int)do_scale, (int)scale0, (int)scale1};
    node->params()["scale_type"] = std::vector<aipubt::Tensor::Dtype>{
        dtype_utils::bits_to_dtype(multiplier_bits, false),
        aipubt::Tensor::UINT16,
        aipubt::Tensor::UINT16};
}

REGISTER_QUANTIZATION_OP(Dilation);
} // namespace qtlibn

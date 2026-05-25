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
 * @Date: 2022-05-20 13:55:16
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-20 14:04:11
 */

#include "quantization_transform/ops/pyramidroialign.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{
void PyramidROIAlignOp::quantize(aipubt::NodePtr node)
{
    bool proposal_normalized = param_utils::get_param(node, "proposal_normalized");
    if (proposal_normalized)
    {
        auto inp1_t = node->inputs()[1];
        auto out_t = node->outputs()[0];
        NoQuantization::quantize_transparent(out_t, inp1_t);

        auto inp0_scale = node->inputs()[0]->quantization().scale();
        auto box_input_qvalue = round(log2(inp0_scale));
        node->params()["box_input_qvalue"] = static_cast<int32_t>(box_input_qvalue);
    }
    else
    {
        auto sample = param_utils::get_param(node, "sample");
        int32_t sample_h = sample[0], sample_w = sample[1];
        std::vector<float> spatial_scale = param_utils::get_param(node, "spatial_scale_value");
        int32_t image_height = param_utils::get_param(node, "image_height");
        int32_t image_width = param_utils::get_param(node, "image_width");

        int32_t image_size_max = std::max(image_height, image_width);
        float spatial_max_scale = *max_element(spatial_scale.begin(), spatial_scale.end());
        int32_t spatial_shift = std::floor(std::log2(65535 / (image_size_max * spatial_max_scale)));

        int input_num = node->inputs().size();
        bool out_signed = false;
        for (int i = 1; i < input_num; i++)
            out_signed = out_signed || aipubt::Tensor::is_signed(node->inputs()[i]->type());

        dtype_utils::set_aipu_dtype(node->outputs()[0], out_signed);
        Op::quantize(node);

        int32_t output_bits = node->outputs()[0]->quantization().bits();
        float input0_scale = node->inputs()[0]->quantization().scale();
        std::vector<int> input_scales, output_scales, input_shifts, output_shifts;
        std::vector<aipubt::Tensor::Dtype> input_scales_types, output_scales_types, input_shifts_types, output_shifts_types;
        for (int i = 1; i < input_num; i++)
        {
            auto inp_quantization = node->inputs()[i]->quantization();
            float local_nor_scale = spatial_scale[i - 1] * std::pow(2, spatial_shift) / input0_scale;
            float local_out_scale = 1;
            if ((sample_h > 0) && (sample_w > 0))
                local_out_scale = float(node->outputs()[0]->quantization().scale()) / (inp_quantization.scale() * sample_h * sample_w);
            else
                local_out_scale = float(node->outputs()[0]->quantization().scale()) / inp_quantization.scale();
            float nor_do_scale, out_do_scale;
            int8_t nor_do_shift, out_do_shift;
            quant_utils::get_scale_approximation_params(local_nor_scale, nor_do_scale, nor_do_shift, 16);
            quant_utils::get_scale_approximation_params(local_out_scale, out_do_scale, out_do_shift, output_bits);

            input_scales.push_back(static_cast<uint32_t>(nor_do_scale));
            output_scales.push_back(static_cast<uint32_t>(out_do_scale));
            input_shifts.push_back(nor_do_shift);
            output_shifts.push_back(out_do_shift);

            input_scales_types.push_back(dtype_utils::bits_to_dtype(16, false));
            output_scales_types.push_back(dtype_utils::bits_to_dtype(output_bits, false));
            input_shifts_types.push_back(dtype_utils::bits_to_dtype(8, true));
            output_shifts_types.push_back(dtype_utils::bits_to_dtype(8, true));
        }

        int L0 = int((std::pow(112, 2) * std::pow(input0_scale, 2)) + 0.5);
        int L1 = int((std::pow(224, 2) * std::pow(input0_scale, 2)) + 0.5);
        int L2 = int((std::pow(448, 2) * std::pow(input0_scale, 2)) + 0.5);
        std::vector<int> levels = {L0, L1, L2};

        node->params()["shift_value"] = output_shifts;
        node->params()["shift_type"] = output_shifts_types;
        node->params()["scale_value"] = output_scales;
        node->params()["scale_type"] = output_scales_types;

        node->params()["input_scale"] = input_scales;
        node->params()["input_scale_type"] = input_scales_types;
        node->params()["input_shift"] = input_shifts;
        node->params()["input_shift_type"] = input_shifts_types;

        node->params()["spatial_shift"] = spatial_shift;

        node->params()["levels"] = levels;
    }
}

REGISTER_QUANTIZATION_OP(PyramidROIAlign);

} // namespace qtlibn

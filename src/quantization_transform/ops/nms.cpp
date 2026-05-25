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
 * @Author: Ruodi Lu
 * @Date: 2022-07-26 19:22:58
 * @Last Modified by: Ruodi Lu
 * @Last Modified time: 2022-07-26 11:47:51
 */

#include "quantization_transform/ops/nms.h"

namespace qtlibn
{
static std::pair<std::vector<float>, std::vector<int8_t>> generate_gussi_lut(uint8_t in_qmin, uint8_t in_qmax, float soft_nms_sigma)
{
    // uint16_t step = in_qmax - in_qmin + 1;
    int32_t lut_idx = 0;
    std::vector<float> lut;
    float idx;
    for (idx = (float)in_qmin / in_qmax; idx <= 1.0; idx += (float)1 / in_qmax, lut_idx++)
    {
        lut.push_back(soft_nms_sigma * idx * idx);
    }
    std::vector<float> do_scale;
    std::vector<int8_t> do_shift;
    quant_utils::get_scale_approximation_params(lut, do_scale, do_shift, 16);
    std::pair<std::vector<float>, std::vector<int8_t>> scale_shift_pair(do_scale, do_shift);
    return scale_shift_pair;
}
void NMSOp::quantize(aipubt::NodePtr node)
{
    float iou_threshold = node->params()["iou_threshold"];
    node->params().erase("iou_threshold");
    float center_point_box = 0.0;
    if (node->params().count("center_point_box"))
    {
        center_point_box = node->params()["center_point_box"];
    }
    // Op::quantize(node);
    float box_encoding_scale = node->inputs()[0]->quantization().scale();
    // int8_t box_encoding_zp = node->inputs()[0]->quantization().offset();
    // float box_per_class_scale = node->inputs()[1]->quantization().scale();
    // int8_t box_per_class_zp = node->inputs()[1]->quantization().offset();
    // float total_class_scale = node->inputs()[2]->quantization().scale();
    // int8_t total_class_zp = node->inputs()[2]->quantization().offset();
    float class_scale = node->inputs()[3]->quantization().scale();
    float class_zp = node->inputs()[3]->quantization().offset();
    // get the height and width of the input image
    float height;
    float width;
    if (node->params().count("image_height") && node->params().count("image_width") && center_point_box)
    {
        height = node->params()["image_height"];
        width = node->params()["image_width"];
    }
    else
    {
        height = box_encoding_scale;
        width = box_encoding_scale;
    }
    int8_t box_shift = 15; //fixed value
    float max_h_w = std::max(height, width);
    float box_scale;
    if (max_h_w <= box_encoding_scale || node->inputs()[0]->quantization().maxs()[0] <= 1.0)
    {
        quant_utils::get_scale_approximation_params(max_h_w / box_encoding_scale, box_scale, box_shift, 16);
        box_encoding_scale = max_h_w;
    }
    else
    {
        quant_utils::get_scale_approximation_params(1.0 / box_encoding_scale, box_scale, box_shift, 16);
        box_encoding_scale = 1.0;
    }
    aipubt::Tensor::Dtype box_scale_type = dtype_utils::bits_to_dtype(16, false);
    aipubt::Tensor::Dtype box_shift_type = dtype_utils::range_to_dtype((float)box_shift, (float)box_shift, true);
    std::string method = node->params()["method"];
    if (method != "HARD" && method != "GAUSSIAN")
    {
        WARN() << "NMS only support hard or gaussian method, but now the method is " << method;
    }
    if (method == "GAUSSIAN")
    {
        float soft_nms_sigma = node->params()["soft_nms_sigma"];
        soft_nms_sigma = -0.5 / soft_nms_sigma;
        int8_t soft_nms_sigma_in_shift = 8;
        uint8_t in_qmin = 0;
        uint8_t in_qmax = pow(2, soft_nms_sigma_in_shift) - 1;
        std::vector<float> do_scale;
        std::vector<int8_t> do_shift;
        do_scale = generate_gussi_lut(in_qmin, in_qmax, soft_nms_sigma).first;
        do_shift = generate_gussi_lut(in_qmin, in_qmax, soft_nms_sigma).second;
        aipubt::TensorShape t_shape(do_scale.size());
        aipubt::TensorPtr gaussian_scale_lut = std::make_shared<aipubt::Tensor>("soft_nms_sigma_do_scale", t_shape, aipubt::Tensor::UINT16);
        aipubt::TensorPtr gaussian_shift_lut = std::make_shared<aipubt::Tensor>("soft_nms_sigma_do_shift", t_shape, aipubt::Tensor::UINT8);
        quant_utils::convert_data_to_tensor<float_t>(gaussian_scale_lut, &do_scale[0], do_scale.size());
        quant_utils::convert_data_to_tensor<int8_t>(gaussian_shift_lut, &do_shift[0], do_shift.size());
        node->constants()["gaussian_scale_lut"] = gaussian_scale_lut;
        node->constants()["gaussian_shift_lut"] = gaussian_shift_lut;
        node->params()["soft_nms_sigma_in_shift"] = soft_nms_sigma_in_shift;
    }
    else if (method == "HARD")
    {
        int16_t iou_thresh_scale = 256;
        int8_t iou_thresh_shift = 8;
        node->params()["iou_thresh_shift"] = iou_thresh_shift;
        node->params()["iou_threshold"] = static_cast<int32_t>(iou_threshold * iou_thresh_scale);
    }

    if (!node->attrs().count(Constants::params_quantized) || !node->get_attr(Constants::params_quantized))
    {
        float score_threshold = 0.0;
        if (node->params().count("score_threshold"))
        {
            score_threshold = node->params()["score_threshold"];
        }
        int32_t score_q_min = dtype_utils::bits_to_range_min(node->inputs()[3]->quantization().bits(), false);
        int32_t score_q_max = dtype_utils::bits_to_range_max(node->inputs()[3]->quantization().bits(), false);
        node->params()["score_threshold"] = (int)quant_utils::linear_quantize_clip(score_threshold,
                                                                                   class_scale,
                                                                                   class_zp,
                                                                                   score_q_min,
                                                                                   score_q_max);
    }

    //set output scale / zp
    node->outputs()[0]->quantization().set_scale(box_encoding_scale);
    node->outputs()[0]->quantization().set_offset(0);
    node->outputs()[1]->quantization().set_scale(1.0);
    node->outputs()[1]->quantization().set_offset(0);
    node->outputs()[2]->quantization().set_scale(class_scale);
    node->outputs()[2]->quantization().set_offset(class_zp);
    node->outputs()[3]->quantization().set_scale(1.0);
    node->outputs()[3]->quantization().set_offset(0);
    //set dtype and qbits
    node->outputs()[0]->quantization().set_aipu_dtype(node->inputs()[0]->quantization().aipu_dtype());
    node->outputs()[0]->quantization().set_bits(node->inputs()[0]->quantization().bits());
    node->outputs()[0]->quantization().set_qinvariant(node->inputs()[0]->quantization().qinvariant());
    node->outputs()[2]->quantization().set_aipu_dtype(node->inputs()[3]->quantization().aipu_dtype());
    node->outputs()[2]->quantization().set_bits(node->inputs()[3]->quantization().bits());
    node->outputs()[2]->quantization().set_qinvariant(node->inputs()[3]->quantization().qinvariant());
    node->outputs()[1]->quantization().set_aipu_dtype(aipubt::Tensor::UINT16);
    node->outputs()[1]->quantization().set_bits(16);
    node->outputs()[1]->quantization().set_qinvariant(true);
    node->outputs()[3]->quantization().set_aipu_dtype(aipubt::Tensor::UINT16);
    node->outputs()[3]->quantization().set_bits(16);
    node->outputs()[3]->quantization().set_qinvariant(true);

    //int IR params
    node->params()["areas_shift"] = 13;
    node->params()["scale_value"] = (int32_t)box_scale;
    node->params()["scale_type"] = box_scale_type;
    node->params()["shift_type"] = box_shift_type;
    node->params()["shift_value"] = box_shift;
}
REGISTER_QUANTIZATION_OP(NMS);
}; // namespace qtlibn

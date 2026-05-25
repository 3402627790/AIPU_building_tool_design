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
 * @Date: 2022-11-10 20:04:16
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-06-08 15:11:42
 */

#include "quantization_transform/ops/multibox_transform_loc.h"

namespace qtlibn
{

aipubt::TensorPtr generate_expr_lut(float in_scale, float in_zerop,
                                    float out_scale, float out_zerop,
                                    float var,
                                    aipubt::Tensor::Dtype dtype,
                                    int lut_size_bits,
                                    int clamp_min, int clamp_max)
{
    int32_t lut_len = 1 << lut_size_bits;
    uint32_t in_qmax = dtype_utils::dtype_to_range_max(dtype);
    int32_t in_qmin = dtype_utils::dtype_to_range_min(dtype);
    float lut;
    int16_t *lut_int = new int16_t[lut_len];
    float step = float((in_qmax - in_qmin) / float(lut_len - 1));
    float dequant_value = in_qmin;
    for (int i = 0; i < lut_len; i++, dequant_value += step)
    {
        lut = quant_utils::linear_dequantize(dequant_value, in_scale, in_zerop);
        lut = lut / var;
        lut = exp(lut);
        lut = quant_utils::linear_quantize_clip(lut, out_scale, out_zerop, clamp_min, clamp_max);
        lut_int[i] = static_cast<int16_t>(lut);
    }
    aipubt::TensorShape lut_shape(lut_len);
    aipubt::TensorPtr lut_tensor = std::make_shared<aipubt::Tensor>(lut_shape, aipubt::Tensor::INT16);
    lut_tensor->set_data((void *)lut_int, lut_len * sizeof(aipubt::Tensor::INT16));
    return lut_tensor;
}

void calculate_quantization(aipubt::NodePtr node, aipubt::TensorPtr rois,
                            aipubt::TensorPtr delta, std::vector<float> &std_div)
{
    bool is_sym = rois->quantization().scheme() == aipubt::Quantization::Symmetric
                      ? true
                      : false;
    bool is_full = rois->quantization().quantized_range() == aipubt::Quantization::Full_Range
                       ? true
                       : false;
    float batch_inp_rois_scales = rois->quantization().scale();
    float batch_inp_deltas_scales = delta->quantization().scale();
    int32_t batch_inp_deltas_zp = delta->quantization().offset();

    aipubt::Tensor::Dtype delta_dtype = rois->type();
    int32_t delta_qbits = delta->quantization().bits();
    int32_t rois_qbits = delta->quantization().bits();

    float scale_ty_out = batch_inp_deltas_scales * std_div[0];
    float scale_tx_out = batch_inp_deltas_scales * std_div[1];
    int64_t delta_qmax = dtype_utils::bits_to_range_max(delta_qbits, true);

    float max_delta_th = log(3276.0) * std_div[2];
    float max_delta_tw = log(3276.0) * std_div[3];

    if ((delta_qmax + batch_inp_deltas_zp) > batch_inp_deltas_scales * max_delta_th ||
        (delta_qmax + batch_inp_deltas_zp) > batch_inp_deltas_scales * max_delta_tw)
    {
        batch_inp_deltas_scales = (float)delta_qmax / std::max<float>(max_delta_th, max_delta_tw);
    }

    std::vector<float> th_max, th_min, tw_max, tw_min;
    th_min.push_back(0.0);
    tw_min.push_back(0.0);
    if (node->params().count("max_exp_thtw"))
    {
        th_max.push_back(node->params()["max_exp_thtw"]);
        tw_max.push_back(node->params()["max_exp_thtw"]);
    }
    else
    {
        th_max.push_back(exp((float)delta_qmax / batch_inp_deltas_scales / std_div[2]));
        tw_max.push_back(exp((float)delta_qmax / batch_inp_deltas_scales / std_div[3]));
    }
    std::vector<float> th_scales, tw_scales;
    std::vector<int32_t> th_zp, tw_zp;

    quant_utils::get_quant_params_from_statistic_value(th_max, th_min, 16, true, is_sym, is_full, th_scales, th_zp);
    quant_utils::get_quant_params_from_statistic_value(tw_max, tw_min, 16, true, is_sym, is_full, tw_scales, tw_zp);

    int32_t lut_size_bits = delta_qbits;
    if (node->attrs().count("lut_items_in_bits"))
    {
        lut_size_bits = std::min<int32_t>(lut_size_bits, node->get_attr("lut_items_in_bits"));
    }

    aipubt::TensorPtr th_lut = generate_expr_lut(batch_inp_deltas_scales, batch_inp_deltas_zp,
                                                 th_scales[0],
                                                 th_zp[0],
                                                 std_div[2], delta_dtype, lut_size_bits,
                                                 0, 32767);
    aipubt::TensorPtr tw_lut = generate_expr_lut(batch_inp_deltas_scales, batch_inp_deltas_zp,
                                                 tw_scales[0],
                                                 tw_zp[0],
                                                 std_div[3], delta_dtype, lut_size_bits,
                                                 0, 32767);
    node->constants()["th_lut"] = th_lut;
    node->constants()["tw_lut"] = tw_lut;
    float y_scale, x_scale, h_scale, w_scale, anchor_scale;
    int8_t y_shift, x_shift, h_shift, w_shift, anchor_shift;
    quant_utils::get_scale_approximation_params(1.0 / scale_ty_out, y_scale, y_shift, 15);
    quant_utils::get_scale_approximation_params(1.0 / scale_tx_out, x_scale, x_shift, 15);
    quant_utils::get_scale_approximation_params(1.0 / th_scales[0], h_scale, h_shift, 15);
    quant_utils::get_scale_approximation_params(1.0 / tw_scales[0], w_scale, w_shift, 15);
    h_shift++; // for half calculate
    w_shift++;
    // convert input box scale(batch_inp_rois_scales) to output box (32767)
    quant_utils::get_scale_approximation_params(32767. / batch_inp_rois_scales, anchor_scale, anchor_shift, 14);
    std::vector<int32_t> box_scale_value = {(int)y_scale, (int)x_scale, (int)h_scale, (int)w_scale, (int)anchor_scale};
    node->params()["box_scale_value"] = box_scale_value;
    node->params()["box_scale_type"] = std::vector<aipubt::Tensor::Dtype>(box_scale_value.size(), aipubt::Tensor::UINT16);
    std::vector<int8_t> box_shift_value = {y_shift, x_shift, h_shift, w_shift, anchor_shift};
    node->params()["box_shift_value"] = box_shift_value;
    node->params()["box_shift_type"] = std::vector<aipubt::Tensor::Dtype>(box_shift_value.size(), aipubt::Tensor::INT8);

    int32_t delta_shift = rois_qbits + delta_qbits + 17 - 32;
    if (batch_inp_deltas_zp != 0)
    {
        delta_shift++;
    }
    node->params()["delta_shift"] = delta_shift;
}

void MultiboxTransformLocOp::quantize(aipubt::NodePtr node)
{
    aipubt::TensorPtr score = node->inputs()[0];
    aipubt::TensorPtr delta = node->inputs()[1];
    aipubt::TensorPtr rois = node->inputs()[2];
    aipubt::TensorPtr out_loc = node->outputs()[0];
    aipubt::TensorPtr count = node->outputs()[1];

    std::vector<float> var;
    for (int i = 0; i < 4; i++)
    {
        var.push_back(1.0 / (float)(node->params()["variance"][i]));
    }
    std::vector<float> std_div = {var[1], var[0], var[3], var[2]};

    dtype_utils::set_aipu_dtype(out_loc, true);
    dtype_utils::set_aipu_dtype(count, false);
    Op::quantize(node);

    calculate_quantization(node, rois, delta, std_div);

    float score_threshold = node->params()["score_threshold"];
    float clip_min, clip_max;
    node->params().erase("score_threshold");
    clip_min = dtype_utils::dtype_to_range_min(score->type());
    clip_max = dtype_utils::dtype_to_range_max(score->type());
    int score_threshold_value = (int)quant_utils::linear_quantize_clip(score_threshold,
                                                                       score->quantization().scale(),
                                                                       score->quantization().offset(),
                                                                       clip_min,
                                                                       clip_max);
    score_threshold_value = score_threshold_value > 0 ? score_threshold_value : 1;
    node->params()["score_threshold_value"] = score_threshold_value;
    node->params()["score_threshold_type"] = score->type();

    float score_scale;
    int8_t score_shift;
    quant_utils::get_scale_approximation_params(32767 / score->quantization().scale(),
                                                score_scale, score_shift, 15);
    node->params()["score_scale_value"] = static_cast<int32_t>(score_scale);
    node->params()["score_scale_type"] = dtype_utils::range_to_dtype(0, score_scale, false);
    node->params()["score_shift_value"] = score_shift;
    node->params()["score_shift_type"] = aipubt::Tensor::INT8;

    if (node->params().count("std_div"))
    {
        node->params().erase("std_div");
    }

    out_loc->quantization().set_scale(32767.0);
    out_loc->quantization().set_offset(0);
    out_loc->set_type(aipubt::Tensor::INT16);
    out_loc->quantization().set_bits(16);
    count->quantization().set_scale(1.0);
    count->quantization().set_offset(0);
    count->set_type(aipubt::Tensor::UINT16);
    count->quantization().set_bits(16);
}

REGISTER_QUANTIZATION_OP(MultiboxTransformLoc);
} // namespace qtlibn
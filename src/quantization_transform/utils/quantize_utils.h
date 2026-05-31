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
 * @Date: 2022-03-13 22:09:43
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-02 15:56:58
 */

#ifndef __QTLIB_QUANTIZATE_TOOL_H__
#define __QTLIB_QUANTIZATE_TOOL_H__

#include "core/compute_tensor.h"
#include "quantization_transform/common.h"
#include "quantization_transform/utils/calibrate_utils.h"
#include "quantization_transform/utils/data_type.h"
#include "quantization_transform/utils/permute_utils.h"

#define DEQUANTIZE(x, s, z) ((x) - (z)) * (s)
#define AIPU_QUANTIZE(x, s, z) ((x) * (s) - (z))
#define AIPU_DEQUANTIZE(x, s, z) (((x) + (z)) / (s))

namespace quant_utils
{

const float EPSINON = 0.00001;
typedef int32_t zp_qdtype;
using aipubt_dtype = aipubt::Tensor::Dtype;
using scale_zp_pair = std::pair<float, zp_qdtype>;
using scale_zp_dtype_pair = std::pair<scale_zp_pair, aipubt_dtype>;
using formula_func = std::function<float(float, float, int32_t)>;

//primitive
float dequantize_formula_android(float v, float s, int32_t z);
float dequantize_formula_aipu(float v, float s, int32_t z);
void dequantize_from_primitive_quant_info(const aipubt_dtype &src_dtype,
                                          const scale_zp_pair &src_scale_zp,
                                          float &float_max,
                                          float &float_min,
                                          formula_func &dequantize_formula);
//primitive
void dequantize_from_primitive_quant_info(const aipubt_dtype &src_dtype,
                                          const std::vector<float> &src_scales,
                                          const std::vector<zp_qdtype> &src_zerops,
                                          std::vector<float> &statistic_max,
                                          std::vector<float> &statistic_min,
                                          formula_func &dequantize_formula);

//primitive
void statistic_max_min(aipubt::TensorPtr tensor,
                       std::vector<float> &statistic_max,
                       std::vector<float> &statistic_min,
                       int32_t dim = 0);

//primitive
void primitive_dequantize(aipubt::TensorPtr tensor, formula_func &dequantize_formula);

//aipu
void get_quant_params_from_statistic_value(const std::vector<float> &max_val,
                                           const std::vector<float> &min_val,
                                           const int8_t q_bits,
                                           const bool is_sign,
                                           const bool is_sym_quant,
                                           const bool is_full_range,
                                           std::vector<float> &scales,
                                           std::vector<zp_qdtype> &zerops);
//aipu
void get_scale_approximation_params(const float &total_scale,
                                    float &do_scale,
                                    int8_t &do_shift,
                                    const int32_t mult_bits,
                                    const std::string &msg = "",
                                    const bool limit = false,
                                    const int32_t mult_bits_ceil = 15,
                                    const int32_t shift_bits_ceil = 31,
                                    bool force_positive_shift = true);

void get_scale_approximation_params(const std::vector<float> &total_scale,
                                    std::vector<float> &do_scale,
                                    std::vector<int8_t> &do_shift,
                                    const int32_t mult_bits,
                                    const std::string &msg = "",
                                    const bool limit = false,
                                    const int32_t mult_bits_ceil = 15,
                                    const int32_t shift_bits_ceil = 31,
                                    bool force_positive_shift = true);
const int32_t optimized_target_level(std::string target_name);
void aipu_quantize(aipubt::TensorPtr tensor);
//aipu
template <typename Ori_T>
void convert_data_to_tensor(aipubt::TensorPtr tensor,
                            Ori_T *data,
                            const int32_t len);

void calculate_scale_zp(aipubt::NodePtr node, bool force_to_recalculate = false);
void calculate_scale_zp(aipubt::TensorPtr t, bool force_to_recalculate = false);

template <typename DataType, typename RangeType>
DataType clip(const DataType &value, const RangeType &min_val, const RangeType &max_val)
{
    if (max_val < min_val)
    {
        ERROR() << "please make sure min_val <= max_val";
    }
    return value < min_val ? min_val : max_val < value ? max_val : value;
}

float linear_quantize(const float value, const float scale, const float zerop);
float linear_quantize_clip(const float value, const float scale, const float zerop, const float clip_min, const float clip_max);
void linear_quantize_clip(aipubt::ComputeTensor &x, aipubt::ComputeTensor &scale,
                          aipubt::ComputeTensor &zerop, float clip_min, float clip_max);
void linear_quantize_clip(float *dataptr, const size_t size,
                          const float scales, const float zerops,
                          const float clip_min, const float clip_max);

float linear_dequantize(const float value, const float scale, const float zerop);
std::vector<aipubt::Tensor::Dtype> preferred_fp_dtype(aipubt::NodePtr node);
void update_q_scheme(aipubt::TensorPtr t, aipubt::TensorPtr src_t);

}; // namespace quant_utils
#endif
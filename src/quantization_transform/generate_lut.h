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
 * @Date: 2022-03-17 22:23:52
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-23 16:39:42
 */

#ifndef __QTLIB_GENERATE_LUT_H__
#define __QTLIB_GENERATE_LUT_H__

#include "quantization_transform/common.h"
#include "quantization_transform/utils/data_type.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace qtlibn
{

using LUTFunc = std::function<aipubt::ComputeTensor(aipubt::ComputeTensor, std::map<std::string, float>)>;

// std::function<aipubt::TensorPtr(uint32_t, std::vector<quant_utils::scale_zp_dtype_pair>, LUTPt, std::map<std::string, float>)> generate_lut(aipubt::Tensor::Dtype dtype);
aipubt::TensorPtr generate_lut(aipubt::NodePtr node,
                               LUTFunc cal_lut_func,
                               std::map<std::string, float> cal_param,
                               int32_t input_idx,
                               int32_t output_idx);
aipubt::TensorPtr generate_lut(aipubt::NodePtr node,
                               LUTFunc cal_lut_func,
                               std::map<std::string, float> cal_param = {});

void generate_lut_fp(aipubt::NodePtr node, LUTFunc cal_lut_func, std::map<std::string, float> cal_param);

struct ApproxLut
{
    bool is_perf_mode(const std::string &min_target);
    std::string get_min_target(aipubt::NodePtr node);
    bool is_dynamic_lut(aipubt::NodePtr node, const int32_t &idx = 0);
    void get_min_max(aipubt::TensorPtr t, float &base_min, float &base_max);
    int32_t get_aiff_lut_items_in_bits(aipubt::NodePtr node);
};

bool is_aiff_perf_mode(aipubt::NodePtr node);
bool is_dynamic_lut(aipubt::NodePtr node, const int32_t &idx = 0);
void get_min_max(aipubt::TensorPtr t, float &base_min, float &base_max);
int32_t get_aiff_lut_items_in_bits(aipubt::NodePtr node);
}; // namespace qtlibn

// };
#endif
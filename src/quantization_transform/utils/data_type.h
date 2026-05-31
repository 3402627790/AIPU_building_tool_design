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
 * @Date: 2022-03-13 22:16:06
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-16 14:23:53
 */

#ifndef __DATA_TYPE_H__
#define __DATA_TYPE_H__

#include "quantization_transform/common.h"

namespace dtype_utils
{
using aipubt_dtype = aipubt::Tensor::Dtype;

double dtype_to_range_min(const aipubt_dtype &dtype);
double dtype_to_range_max(const aipubt_dtype &dtype);

int64_t bits_to_range_min(const int32_t bits, const bool is_signed);
uint64_t bits_to_range_max(const int32_t bits, const bool is_signed);

uint32_t range_to_bits(const float mins, const float maxs, const bool is_signed);
aipubt_dtype range_to_dtype(const float mins, const float maxs, const bool is_signed);

const size_t dtype_size(const aipubt_dtype &dtype);

aipubt_dtype bits_to_dtype(const int32_t bits, const bool is_signed);
int32_t dtype_to_bits(aipubt_dtype dtype);

aipubt::Tensor::Dtype bytes_to_signed_qdtype(uint8_t bytes);
aipubt::Tensor::Dtype bytes_to_unsigned_qdtype(uint8_t bytes);

aipubt::ComputeTensor to_fp24(aipubt::ComputeTensor t);

void set_aipu_dtype(aipubt::TensorPtr tensor, bool is_signed, bool is_change_tensor_type = true);

}; // namespace dtype_utils

#endif
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
 * @Date: 2020-05-08 14:56:54
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-11-01 14:36:13
 */

#ifndef __QUANTIZATION_COMMON_H__
#define __QUANTIZATION_COMMON_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "builder/target.h"
#include "core/any.h"
#include "core/compute_tensor.h"
#include "core/graph.h"
#include "core/logger.h"
#include "core/node.h"
#include "core/ops/ops.h"
#include "core/optype.h"
#include "core/quantization.h"
#include "core/subgraph.h"
#include "core/tensor.h"
#include "utils/string_utils.h"
#include <dtypes/op_data_types.h>
#include <dtypes/op_dtypes.h>

namespace Constants
{
constexpr const char *aipu_input_scale = "aipu_input_scale";
constexpr const char *aipu_output_scale = "aipu_output_scale";
constexpr const char *aipu_input_zp = "aipu_input_zp";
constexpr const char *aipu_output_zp = "aipu_output_zp";

constexpr const char *original_scale = "original_scale";
constexpr const char *original_zp = "original_zp";
constexpr const char *adjust_quantization_dim = "adjust_quantization_dim";
constexpr const char *lut_items_in_bits = "lut_items_in_bits";
constexpr const char *fasym_flag = "fasym_flag";
constexpr const char *params_quantized = "params_quantized";
constexpr const char *tflite_quantization = "tflite_quantization";
constexpr const char *trigger_float_op = "trigger_float_op";

}; // namespace Constants
#endif
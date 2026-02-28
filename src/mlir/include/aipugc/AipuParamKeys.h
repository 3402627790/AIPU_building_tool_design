//===- AipuParamKeys.h - AIPU Node params key-----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is used to save the keywords that need to be parsed.
//
//===----------------------------------------------------------------------===//

#ifndef AIPUGC_PARAM_KEYS_H
#define AIPUGC_PARAM_KEYS_H

namespace mlir {
namespace aipugc {

static const std::set<std::string> paramsKeys = {
    "with_activation", "group",           "num_output",  "scale_type",
    "scale_value",     "shift_type",      "shift_value", "is_perf_mode",
    "clip_mode",       "ignore_scale_zp", "constant_value",
    "interp_shift_type", "interp_shift_value",
    "quantize_scale", "quantize_zp", "round_mode", "unquantifiable", /* QuantizeOp */
    "activation_quantization_axis", "layer_top_range",
};
static const std::set<std::string> quantConstantKeys = {
    "scale",    "shift",   "lut",     "diff_shifts",   "sqrt_lut",   "log2_lut",
    "pow2_lut", "lut_log", "lut_exp", "weights_scale", "weights_zp",
};
} // namespace aipugc
} // namespace mlir

#endif // AIPUGC_MLIRGEN_H

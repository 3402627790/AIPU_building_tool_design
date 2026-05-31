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
 * @Date: 2020-08-20 11:11:22
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2026-04-08 16:13:03
 */

#ifndef __QTLIB_PASSES_H__
#define __QTLIB_PASSES_H__

#include "quantization_transform/insertop.h"
#include "quantization_transform/qinfo.h"
#include "quantization_transform/utils/permute_utils.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace qtlibn
{
void pad_scale_zp(aipubt::GraphPtr g);
void backup_primitive_scale_zp(aipubt::GraphPtr g);
void backup_primitive_scale_zp(aipubt::NodePtr node);
void resume_primitive_scale_zp(aipubt::GraphPtr g);
void resume_primitive_scale_zp(aipubt::NodePtr node);
void set_quantized_flag(aipubt::GraphPtr g);
void convtranspose_prepass(aipubt::GraphPtr g);
void convtranspose_postpass(aipubt::GraphPtr g);
void update_tensor_group_and_granularity(aipubt::GraphPtr g);
void update_tensor_group_and_granularity(aipubt::NodePtr n);
void basiclstm_concat_weights(aipubt::GraphPtr g);
void gru_weights_biases_preprocess(aipubt::GraphPtr g);
void set_unquantifiable_default(aipubt::GraphPtr g);
void convert_to_16bit(aipubt::GraphPtr g);
void convert_relu6_to_clip(aipubt::GraphPtr g);
void fuse_float_op_between_dequantize_and_quantize(aipubt::GraphPtr g);
void eliminate_cast(aipubt::GraphPtr g);
void fuse_constant_eltwise_to_leakyrelu(aipubt::GraphPtr g);
void update_instance_quantization_info(aipubt::GraphPtr g);
void change_data_type_for_conv_relu(aipubt::GraphPtr g);
void basiclstm_placeholder_loading_preprocess(aipubt::GraphPtr g);
void merge_placeholder_scales_and_offsets(aipubt::GraphPtr g);
void label_avgpool_from_reducemean(aipubt::GraphPtr g);
void trigger_float_op(aipubt::GraphPtr g);
void insert_cast_for_dequantize_and_quantize(aipubt::GraphPtr g);
void update_min_max_range_to_tensor(aipubt::GraphPtr g);
void compress_bias(aipubt::GraphPtr g);
void fuse_cast_quantize(aipubt::GraphPtr g);
void fuse_lstm_cell(aipubt::GraphPtr g);
void convert_resize_to_convolution(aipubt::GraphPtr g);
void set_param_quantized(aipubt::GraphPtr g);
void set_trigger_float_op(aipubt::GraphPtr g);
void simplify_dequantize_quantize(aipubt::GraphPtr g);
void set_node_default_attr(aipubt::GraphPtr g);
void insert_cast_in_io_tensor(aipubt::GraphPtr g);
void flash_min_max_to_quantized(aipubt::GraphPtr g);
void pool_and_reduce_preprocess(aipubt::GraphPtr g);
void fuse_fp_add_constant_to_bn(aipubt::GraphPtr g);
void fuse_mul_to_square(aipubt::GraphPtr g);
void fuse_algo_nodes_to_layernorm(aipubt::GraphPtr g);
void fuse_matmul_mul(aipubt::GraphPtr g);
void trans_op_to_qdq(aipubt::GraphPtr g);
void dequantize_preprocess(aipubt::GraphPtr g);

void pre_passes(aipubt::GraphPtr g);
void post_passes(aipubt::GraphPtr g);

void pre_quantize_passes(aipubt::GraphPtr g);
void post_quantize_passes(aipubt::GraphPtr g);

void pre_forward_passes(aipubt::GraphPtr g);
void fuse_dq_n_fp_op_q(aipubt::GraphPtr g);
}; // namespace qtlibn

#endif

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
 * @Last Modified time: 2026-04-08 16:13:27
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
void pre_passes(aipubt::GraphPtr g)
{
    insert_cast_in_io_tensor(g);
    set_node_default_attr(g);
    dequantize_preprocess(g);
    simplify_dequantize_quantize(g);
    trans_op_to_qdq(g);
    set_param_quantized(g);
    fuse_lstm_cell(g);
    backup_primitive_scale_zp(g);
    pad_scale_zp(g);
    convtranspose_prepass(g);
    update_tensor_group_and_granularity(g);
    basiclstm_concat_weights(g);
    gru_weights_biases_preprocess(g);
    set_unquantifiable_default(g);
    convert_to_16bit(g);

    fuse_cast_quantize(g);
    fuse_float_op_between_dequantize_and_quantize(g);
    fuse_dq_n_fp_op_q(g);
    fuse_fp_add_constant_to_bn(g);
    fuse_algo_nodes_to_layernorm(g);
    fuse_matmul_mul(g);
    set_node_default_attr(g);

    trigger_float_op(g);

    convert_resize_to_convolution(g);
    eliminate_cast(g);
    change_data_type_for_conv_relu(g);
    basiclstm_placeholder_loading_preprocess(g);
    label_avgpool_from_reducemean(g);
    pool_and_reduce_preprocess(g);
    fuse_mul_to_square(g);
    g->topological_sort();
    g->reset_layer_id();
}

static void insert_cast(aipubt::GraphPtr g)
{
    auto inserter = InsertCastOp();
    inserter.insert(g);
}

void post_passes(aipubt::GraphPtr g)
{
    convtranspose_postpass(g);
    insert_cast_for_dequantize_and_quantize(g);
    flash_min_max_to_quantized(g);
    insert_cast(g);
    g->topological_sort();
    g->reset_layer_id();
}

void pre_quantize_passes(aipubt::GraphPtr g)
{
    backup_primitive_scale_zp(g);
    update_tensor_group_and_granularity(g);
    basiclstm_concat_weights(g);
    gru_weights_biases_preprocess(g);
    update_min_max_range_to_tensor(g);
}
void post_quantize_passes(aipubt::GraphPtr g)
{
    convtranspose_postpass(g);
    convert_relu6_to_clip(g);
    compress_bias(g);
}
void pre_forward_passes(aipubt::GraphPtr g)
{
    set_quantized_flag(g);
}
}; // namespace qtlibn

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
 * @Author: Neo Wang
 * @Date: 2020-01-29 14:25:12
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-11-07 16:07:15
 */
#ifndef BUILDER_LOWERING_H
#define BUILDER_LOWERING_H
#include "core/basepass.h"
#include "core/optype.h"
#include "core/tensor.h"
#include <functional>
#include <memory>

#ifdef PASS_FUNCTION
#undef PASS_FUNCTION
#endif
#ifdef STATIC_PASS_FUNCTION
#undef STATIC_PASS_FUNCTION
#endif
#ifdef DYNAMIC_PASS_FUNCTION
#undef DYNAMIC_PASS_FUNCTION
#endif
#define PASS_FUNCTION(f)                             \
    void f(GraphPtr g);                              \
    class FPass_##f : public VFuncPass<FPass_##f, f> \
    {                                                \
    };
#define STATIC_PASS_FUNCTION(f)                                       \
    void f(GraphPtr g);                                               \
    class FPass_##f : public VFuncPass<FPass_##f, f, PassType_Static> \
    {                                                                 \
    };
#define DYNAMIC_PASS_FUNCTION(f)                                       \
    void f(GraphPtr g);                                                \
    class FPass_##f : public VFuncPass<FPass_##f, f, PassType_Dynamic> \
    {                                                                  \
    };
#define PASS_FUNCTION2(f)                           \
    bool f(GraphPtr g);                             \
    class FPass_##f : public FuncPass<FPass_##f, f> \
    {                                               \
    };
#define STATIC_PASS_FUNCTION2(f)                                     \
    bool f(GraphPtr g);                                              \
    class FPass_##f : public FuncPass<FPass_##f, f, PassType_Static> \
    {                                                                \
    };
#define DYNAMIC_PASS_FUNCTION2(f)                                     \
    bool f(GraphPtr g);                                               \
    class FPass_##f : public FuncPass<FPass_##f, f, PassType_Dynamic> \
    {                                                                 \
    };
namespace aipubt
{
/*
 * ===  FUNCTION  ======================================================================
 *         Name:  get_v1_in_c_alignment
 *  Description:  Get the in_c alignment to make sure HWA dc_times == 0 in v1
 * =====================================================================================
 */
int get_v1_in_c_alignment(int in_c);
int get_alignment(const std::vector<TensorPtr> &tensors);
int get_input_alignment(NodePtr node);
int get_output_alignment(NodePtr node);
int get_fc_in_c_base(Tensor::Dtype dtype);
int get_channel_base(const std::vector<TensorPtr> &tensors);
int get_input_channel_base(NodePtr node);
int get_output_channel_base(NodePtr node);
int get_aiff_max_h();
int get_aiff_max_w();
int get_aiff_max_in_c(bool is_fc = false, Tensor::Dtype dtype = Tensor::INT8);
int get_aiff_max_out_c();
int get_aiff_mtp_num_per_core();

void get_pe_spec(int &pe_w, int &pe_h);
bool try_optimize_aiff_reduce_shape(int &h, int &w, bool is_surface_reduce, int hw, const std::string &method);

void reshape_consts_of_norm(const TensorShape &shape, NodePtr n);

bool check_hwa_matmul(NodePtr node);
bool need_matmul_to_fc(NodePtr node);
std::vector<NodePtr> split_conv_output_c_(GraphPtr g, NodePtr n, const std::vector<int> &split_c_vec);

bool check_z3_aiff_global_avg_pool(NodePtr node);
bool check_x3_aiff_global_max_min_reduce(NodePtr node);
bool check_aiff_spec_pool3d(NodePtr node);

bool is_split_matmul_enabled();
bool is_adjust_matmul_for_large_w_enabled();
bool is_matmul_to_fc_enabled();

void update_disable_passes();
void show_all_passes_name();

void lower_before_all(GraphPtr g);
void lower_before_tiling(GraphPtr g);
void lower_before_optimize(GraphPtr g);
void lower_after_unfold(GraphPtr g);
void lower_before_plugin(GraphPtr g);
void lower_after_optimize(GraphPtr g);
void passes_after_build(GraphPtr g);

OpType get_real_op_type(NodePtr n);
NodePtr get_real_node(NodePtr n);
void get_all_graphs(GraphPtr g, std::vector<GraphPtr> &result);
std::vector<GraphPtr> get_all_graphs(GraphPtr g);
bool find_pass_by_name(const std::string &name, std::function<bool(GraphPtr)> &func);

bool check_plugin_option(const std::string &opt);
bool is_btconv_supported(NodePtr node);

void tile_graph_n_criteria(GraphPtr g, std::function<bool(NodePtr)> criteria, int step = 1);

PASS_FUNCTION(find_large_kernel_conv);
PASS_FUNCTION(datalayout_adaptor);
PASS_FUNCTION(transpose_gridsample);
PASS_FUNCTION(transpose_cumulate);
PASS_FUNCTION(resize_to_dwconv_d2s);
PASS_FUNCTION(resize_to_dwconv);

// The following is the before_hwa_passes
PASS_FUNCTION(record_original_param);
// PASS_FUNCTION(rename_activation);
PASS_FUNCTION(interp_to_upsample);
STATIC_PASS_FUNCTION(split_avg_pool_kernel);
PASS_FUNCTION(hwa_gru_lstm_pad);
PASS_FUNCTION(hwa_bilstm_opt);
PASS_FUNCTION(hwa_bn_inc_outc_adaptor);
PASS_FUNCTION(hwa_fc_outc_adaptor);
PASS_FUNCTION(hwa_fc_inc_adaptor);
STATIC_PASS_FUNCTION(abs_to_leakyrelu);
STATIC_PASS_FUNCTION(cast_to_leakyrelu);
STATIC_PASS_FUNCTION(hwa_relu_c_adaptor);
PASS_FUNCTION(crop_conv_kernel);
PASS_FUNCTION(split_conv_pad);
PASS_FUNCTION(split_pool3d_pad);
PASS_FUNCTION(hwa_eltwise_c_adaptor);
PASS_FUNCTION(hwa_add_mul_sub_c_adaptor);
PASS_FUNCTION(split_conv_output_c);
PASS_FUNCTION(split_bn_inc_outc);
PASS_FUNCTION(deconv_unit_stride_to_one_conv);
PASS_FUNCTION(deconv_non_unit_stride_to_conv_d2s);
PASS_FUNCTION(deconv_non_unit_stride_to_conv);
PASS_FUNCTION(deconv_unit_stride_to_conv);
PASS_FUNCTION(offset_crop_pad_with_elt);
PASS_FUNCTION(general_transpose_to_transpose4d);
PASS_FUNCTION(rename_depthwise_conv);
PASS_FUNCTION(hwa_group_conv_c_adaptor);
PASS_FUNCTION(hwa_multiplier_dwconv_c_adaptor);
PASS_FUNCTION(hwa_conv_inc_adaptor);
PASS_FUNCTION(split_conv_kernel);
PASS_FUNCTION(dilated16_to_dilated8);
PASS_FUNCTION(atrous_conv_to_conv);
PASS_FUNCTION(hwa_conv_pad_outc_for_last_layer);
// high_precision will use int16 for intermediate result, which may increase memory footprint
//  and degrade the performance
PASS_FUNCTION(split_conv_input_c);
PASS_FUNCTION(split_conv_input_hw);
PASS_FUNCTION(hwa_depthwise_conv_c_adaptor);
PASS_FUNCTION(gemm_to_matmul);
PASS_FUNCTION(offset_crop_pad);
PASS_FUNCTION(batch_FC_to_conv);
PASS_FUNCTION(rgb_to_yuv);
PASS_FUNCTION(yuv_to_rgb);
PASS_FUNCTION(hwa_bn_batch_process);
PASS_FUNCTION(hwa_conv_outc_adaptor);
PASS_FUNCTION(calculate_node_ops_count);
STATIC_PASS_FUNCTION(stride_kernel_size_adaptor);
PASS_FUNCTION(remove_crop);
PASS_FUNCTION(patch_internal_crop);
PASS_FUNCTION(patch_internal_split);
PASS_FUNCTION(patch_internal_concat);
PASS_FUNCTION(remove_concat);
PASS_FUNCTION(hwa_pooling_c_adaptor);
PASS_FUNCTION(hwa_resize_c_adaptor);
PASS_FUNCTION(tile_to_upsample);
PASS_FUNCTION(merge_pool1x1_conv_1x1);
STATIC_PASS_FUNCTION(conv_full_tiling);
PASS_FUNCTION(global_avg_pool_to_reduce);
STATIC_PASS_FUNCTION(reduce_to_pool);
PASS_FUNCTION(reduce_to_reshape);
PASS_FUNCTION(adapt_boundingbox_in_type);
PASS_FUNCTION(adapt_eltwise_in_type);
PASS_FUNCTION(adapt_eltwise_out_type);
PASS_FUNCTION(merge_redundant_pad);
PASS_FUNCTION(space2depth_to_batch2depth);
PASS_FUNCTION(remove_fakequant);
PASS_FUNCTION(asym_matmul_to_sym_matmul);
PASS_FUNCTION(matmul_shape_adaptor);
PASS_FUNCTION(swap_hwa_matmul_inputs);
STATIC_PASS_FUNCTION(split_matmul_for_large_outc);
PASS_FUNCTION(store_matmul_origin_shapes);
PASS_FUNCTION(adjust_matmul_for_large_w);
STATIC_PASS_FUNCTION(adjust_matmul_w_for_mtp_util);
PASS_FUNCTION(matmul_c_adaptor);
PASS_FUNCTION(adjust_matmul_inputs);
PASS_FUNCTION(matmul_to_fc);
PASS_FUNCTION(softmax_shape_adaptor);
STATIC_PASS_FUNCTION(layernorm_shape_adaptor);
STATIC_PASS_FUNCTION(normalization_shape_adaptor);
PASS_FUNCTION(add_address_to_dot);
PASS_FUNCTION(merge_permute_b2s);
PASS_FUNCTION(adapt_tensor_strides);
PASS_FUNCTION(adapt_slice_strides);
PASS_FUNCTION(adapt_concat_strides);
PASS_FUNCTION(remove_dummy_slice);
PASS_FUNCTION(remove_crop_with_strides);
PASS_FUNCTION(virtual_node_memory_remapping);
PASS_FUNCTION(deconv_stride_shrink);
PASS_FUNCTION(add_profiler_placeholder_to_dot);
PASS_FUNCTION(global_max_pool_to_reduce);
PASS_FUNCTION(pooling_to_reshape);
PASS_FUNCTION(tile_graph);
// PASS_FUNCTION(split_layernorm_channel);
PASS_FUNCTION(tnode_to_sg);
PASS_FUNCTION(PFAnode_to_sg);
PASS_FUNCTION(adapt_arithmetic_op);
PASS_FUNCTION(duplicate_chw_transpose);
STATIC_PASS_FUNCTION(conv_to_batch_fc);
PASS_FUNCTION(remove_split);
PASS_FUNCTION(optimize_transformer_conv_multi_head);
PASS_FUNCTION(optimize_transformer_fc_multi_head);
PASS_FUNCTION(remove_transformer_multi_head_layoutconvertor);
PASS_FUNCTION(split_special_concat);
PASS_FUNCTION(fuse_crop_tile_to_conv);
PASS_FUNCTION(replace_crop_to_conv);
PASS_FUNCTION(reset_all_tensors_stride);
PASS_FUNCTION2(remove_constant);
STATIC_PASS_FUNCTION2(split_maxpool_kernel);
PASS_FUNCTION(const_add_sub_mul_to_eltwise);
PASS_FUNCTION(add_sub_mul_to_eltwise);
PASS_FUNCTION(init_symbol_shape);
PASS_FUNCTION(infer_symbol_shape);
DYNAMIC_PASS_FUNCTION(pooling_to_globalpool);
DYNAMIC_PASS_FUNCTION(update_dynamic_max_shape);
DYNAMIC_PASS_FUNCTION(eliminate_dynamic_matmul);
DYNAMIC_PASS_FUNCTION(eltwise_input_dim_adaptor);
DYNAMIC_PASS_FUNCTION(eliminate_dynamic_fc);
DYNAMIC_PASS_FUNCTION(fc_batch_adaptor);
DYNAMIC_PASS_FUNCTION(divide_graph);
DYNAMIC_PASS_FUNCTION(lift_cache_pad);
PASS_FUNCTION(lift_cache_reshape);
DYNAMIC_PASS_FUNCTION(lift_cache_transpose);
DYNAMIC_PASS_FUNCTION(lift_reshape);
DYNAMIC_PASS_FUNCTION(lift_transpose);
DYNAMIC_PASS_FUNCTION(adapt_cache_stride);
PASS_FUNCTION(cache_c_adaptor);
PASS_FUNCTION(max_batch_adaptor);
DYNAMIC_PASS_FUNCTION(hwa_add_mul_sub_h_adaptor);
DYNAMIC_PASS_FUNCTION(hwa_softmax_c_adaptor);
DYNAMIC_PASS_FUNCTION(eliminate_broadcast_matmul);
PASS_FUNCTION(merge_switch_dummy);
PASS_FUNCTION(remove_switch);

PASS_FUNCTION2(apply_plugins);
PASS_FUNCTION2(apply_internal_plugins);
PASS_FUNCTION2(fuse_transpose_reshape_matmul);

PASS_FUNCTION(decompose_layernorm);
PASS_FUNCTION(decompose_instancenorm);
PASS_FUNCTION(decompose_hwc_layernorm);
PASS_FUNCTION(decompose_groupnorm_aiff);
PASS_FUNCTION(decompose_groupnorm);
PASS_FUNCTION(decompose_mvn);
PASS_FUNCTION(decompose_mvn_aiff);
STATIC_PASS_FUNCTION(decompose_rmsnorm);
PASS_FUNCTION(decompose_fp16_conv_acc16);
PASS_FUNCTION(split_softmax);
PASS_FUNCTION(split_deq_softmax_quat_matmul);
PASS_FUNCTION(split_attention_blocks);
PASS_FUNCTION(split_attention_blocks_V1);
PASS_FUNCTION(split_attention_blocks_V3);
PASS_FUNCTION(split_attention_blocks_V4);
PASS_FUNCTION(split_multi_head_block);
PASS_FUNCTION(split_fc_activation);
PASS_FUNCTION(eliminate_crop_pad_pair);
PASS_FUNCTION(record_gm_tensor);
PASS_FUNCTION(unfold_graph);
PASS_FUNCTION(apply_tune_strategy);

PASS_FUNCTION2(add_tensor_size_to_dot);
PASS_FUNCTION2(add_estimate_info_to_dot);
PASS_FUNCTION2(add_mem_info_to_dot);
PASS_FUNCTION2(actg);

PASS_FUNCTION(eliminate_reduce_shape_dim);
PASS_FUNCTION(reshape_aiff_reduce);
PASS_FUNCTION(split_reduce);
PASS_FUNCTION(adjust_reduce_axis);
PASS_FUNCTION(adjust_tpc_reduce_axis);

PASS_FUNCTION(eliminate_norm_shape_dim);
PASS_FUNCTION(reshape_aiff_norm);
PASS_FUNCTION(adjust_norm_axis);

PASS_FUNCTION(eliminate_concat1by1);
PASS_FUNCTION2(remove_layout_convertor);
STATIC_PASS_FUNCTION(fuse_resize_crop);
STATIC_PASS_FUNCTION(lower_d2s_d2s_eltwise);
STATIC_PASS_FUNCTION(split_global_pool2D);
PASS_FUNCTION(remove_pad);
PASS_FUNCTION(unify_transpose_layout);
} // namespace aipubt

#endif //! BUILDER_LOWERING_H

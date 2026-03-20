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
 * @Date: 2020-01-29 14:25:16
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-10-23 13:40:37
 */
#include "builder/lowering/lowering.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/logger.h"
#include "core/node.h"
#include "core/pass_manager.h"
#include "datalayout_schedule.h"
#include "transform/passes.h"
#include "utils/utils.h"
#include <mutex>

namespace aipubt
{
static std::set<std::string> disabled_passes_rt;
static std::set<std::string> disabled_passes = {
    "eliminate_concat_crop", // REMOVE ME: after tile mix axis ready
    "unify_transpose_layout",
    "split_global_pool2D", // for dped_instance
    "hwa_bilstm_opt",      // it has discard
    "swap_bn_transpose",
    "tile_graph",
    "tnode_to_sg",
    "PFAnode_to_sg",
    "swap_reshape_transpose",
    // "split_layernorm_channel",
    // "transpose_gridsample", // for AIAE-114, CP-1157; default open for x2
    //"resize_to_dwconv", // default enable for ai benchmark: deeplab_v3+,esrgan,vsr
    "fuse_mul_add_to_bn",            // for ai benchmark: dped,punet,pynet, L1 error worse a litter, thus still disable
    "fuse_add_mul_to_bn",            // for ai benchmark: srgan, L1 error worse a litter, thus still disable
    "crop_conv_kernel",              // for ai benchmark: mobilenet_v2, it cause mismatch when tiling is enabled
    "split_softmax",                 // for stable difussion.
    "split_deq_softmax_quat_matmul", // for sd fp16 softmax
    "split_multi_head_block",        // for sd int16 softmax
    "split_attention_blocks",
    "split_attention_blocks_V1",
    "split_attention_blocks_V3",
    "split_attention_blocks_V4",
    "split_fc_activation",     // for sd fp16 softmax block
    "eliminate_crop_pad_pair", // for sd fp16 softmax block
    "split_fc_transpose",      // for swim_transformer;
    "decompose_layernorm",
    "decompose_instancenorm",
    "decompose_hwc_layernorm",
    "decompose_groupnorm_aiff",
    "decompose_mvn",
    "decompose_mvn_aiff",
    "decompose_rmsnorm",
    "decompose_fp16_conv_acc16",
    "fuse_transpose_layernorm",
    "lower_gather",
    "split_to_crops", // for onnx_yolo_v8_seg
    "eliminate_dense_crop",
    "eliminate_dynamic_matmul",
    "eltwise_input_dim_adaptor",
    "eliminate_dynamic_fc",
    "eliminate_concat1by1",
    "fuse_multi_scalars",
};

void update_disable_passes()
{
    auto disable = BuilderConfig::get<std::vector<std::string>>("disable-pass");
    auto enable = BuilderConfig::get<std::vector<std::string>>("enable-pass");
    disabled_passes_rt = disabled_passes;
    disabled_passes_rt.insert(disable.begin(), disable.end());
    for (auto &p : enable)
    {
        disabled_passes_rt.erase(p);
    }
}

std::set<std::string> &get_disable_passes()
{
    static std::once_flag get_disable_passes_flag;
    std::call_once(get_disable_passes_flag, update_disable_passes);
    return disabled_passes_rt;
}

bool is_split_matmul_enabled()
{
    auto &disabled = get_disable_passes();
    return disabled.find("split_matmul_for_large_outc") == disabled.end();
}

bool is_adjust_matmul_for_large_w_enabled()
{
    auto &disabled = get_disable_passes();
    return disabled.find("adjust_matmul_for_large_w") == disabled.end();
}

bool is_matmul_to_fc_enabled()
{
    auto &disabled = get_disable_passes();
    return disabled.find("matmul_to_fc") == disabled.end();
}

OpType get_real_op_type(NodePtr n)
{
    OpType t = n->type();
    if (SubGraphNode::is_subgraph_node(n))
    {
        SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(n);
        auto nodes = sgn->subgraph()->nodes();
        if (nodes.size() == 1)
        {
            t = nodes[0]->type();
        }
    }
    return t;
}

NodePtr get_real_node(NodePtr n)
{
    if (SubGraphNode::is_subgraph_node(n))
    {
        SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(n);
        auto nodes = sgn->subgraph()->nodes();
        if (nodes.size() == 1)
        {
            return nodes[0];
        }
    }
    return n;
}

static auto other_passes = make_pass_tuple<
    FPass_fuse_concat_reshape_splits>();

static auto passes_before_all = make_pass_tuple<
    FPass_record_original_param>();

static auto passes_before_optimize = make_pass_tuple<
    FPass_fill_constant_out_shape,
    FPass_tile_graph,
    FPass_eliminate_concat_crop, // whether eliminate concat & corp after tile difference axis
    FPass_global_avg_pool_3d_to_2d,
    FPass_split_global_pool2D, // for dped_instance
    FPass_eliminate_concat1by1,
    FPass_swap_bn_transpose,      // for codeformer
    FPass_swap_reshape_transpose, // for codeformer
    FPass_eliminate_dense_crop,
    FPass_decompose_hwc_layernorm,  // for codeformer
    FPass_fuse_transpose_layernorm, // for codeformer (discarded)
    FPass_lower_gather,
    // FPass_split_layernorm_channel,
    FPass_transpose_gridsample, // for AIAE-114, CP-1157
    FPass_transpose_cumulate,
    FPass_resize_to_dwconv_d2s,
    FPass_resize_to_dwconv,
    FPass_rename_depthwise_conv,
    FPass_remove_fakequant,
    FPass_eliminate_reshape,
    FPass_pooling_to_reshape,
    FPass_slice_to_crop,
    FPass_split_to_crops,
    FPass_fuse_crop_tile_to_conv,
    FPass_replace_crop_to_conv,

    FPass_split_multi_head_block,
    FPass_split_attention_blocks,
    FPass_split_attention_blocks_V1,
    FPass_split_attention_blocks_V3,
    FPass_split_attention_blocks_V4,
    FPass_split_deq_softmax_quat_matmul,
    // NOTE: fuse_constant_matmul_to_fc must be before other matmul passes
    FPass_fuse_constant_matmul_to_fc,
    // NOTE: gemm_to_matmul must be before asym_matmul_to_sym_matmul
    FPass_gemm_to_matmul,
    // NOTE: asym_matmul_to_sym_matmul must be before fuse_tile_eltwise
    FPass_asym_matmul_to_sym_matmul,
    // NOTE: use asym_matmul_to_sym_matmul pass once more, since there might be some matmul nodes
    // which share the same input nodes, and one of these inputs might have been changed by the
    // previous asym_matmul_to_sym_matmul pass.
    FPass_asym_matmul_to_sym_matmul,
    FPass_eliminate_broadcast_matmul,

    // FPass_fuse_transpose_layernorm,
    //  NOTE: decompose_layernorm and decompose_instancenorm must be before fuse_tile_eltwise
    FPass_decompose_layernorm,
    FPass_decompose_instancenorm,

    FPass_decompose_rmsnorm,
    FPass_eliminate_norm_shape_dim,
    FPass_adjust_norm_axis,
    FPass_reshape_aiff_norm,

    FPass_decompose_fp16_conv_acc16,

    FPass_reduce_to_pool,
    FPass_global_max_pool_to_reduce,
    FPass_reduce_to_reshape,
    FPass_eliminate_reduce_shape_dim,
    FPass_split_reduce,
    FPass_reduce_to_reshape,
    FPass_eliminate_reduce_shape_dim,
    FPass_adjust_reduce_axis,
    FPass_reshape_aiff_reduce,
    FPass_adjust_tpc_reduce_axis,

    FPass_const_add_sub_mul_to_eltwise,
    FPass_fuse_tile_eltwise,
    FPass_fuse_mul_add_to_bn,
    FPass_fuse_add_mul_to_bn,
    FPass_fuse_mul_constant_to_bn, // for esrgan, mobilenet_v3, mv3_depth, srgan
    FPass_fuse_add_constant_to_bn, // for mobilenet_v3, efficient_b4, dped_instance, mv3_depth, srgan
    FPass_add_sub_mul_to_eltwise,
    FPass_tile_to_upsample,
    FPass_merge_pool1x1_conv_1x1,
    FPass_hwa_group_conv_c_adaptor,
    FPass_hwa_multiplier_dwconv_c_adaptor,
    FPass_hwa_depthwise_conv_c_adaptor,
    FPass_split_avg_pool_kernel,
    FPass_reduce_to_pool,
    FPass_reduce_to_reshape,

    FPass_global_avg_pool_to_reduce,
    FPass_split_maxpool_kernel,

    FPass_merge_permute_b2s,
    FPass_split_fc_transpose,
    // NOTE: conv_to_batch_fc must be before batch_FC_to_conv
    FPass_conv_to_batch_fc,
    FPass_fc_batch_adaptor,
    FPass_batch_FC_to_conv,
    FPass_eliminate_dynamic_fc,
    FPass_rgb_to_yuv,
    FPass_yuv_to_rgb,
    FPass_hwa_fc_inc_adaptor,
    FPass_hwa_fc_outc_adaptor,
    FPass_hwa_gru_lstm_pad,
    // FPass_hwa_bilstm_opt, //disabled
    FPass_split_bn_inc_outc,
    FPass_hwa_bn_batch_process,
    FPass_dilated16_to_dilated8,
    FPass_atrous_conv_to_conv,
    FPass_deconv_unit_stride_to_one_conv,
    FPass_deconv_non_unit_stride_to_conv_d2s,
    FPass_deconv_unit_stride_to_conv,
    FPass_deconv_non_unit_stride_to_conv,
    // FPass_merge_conv_output_crop, // it may fail after "hwa_conv_outc_adaptor" when conv c align + crop any channel
    // why need hwa_conv_inc_adaptor and hwa_conv_outc_adaptor here?
    //    we want to pad or crop conv firstly than split conv kernel or input c or outputc
    //    But in order to ensure that the modification does not lead to additional bugs, we still keep the original position
    FPass_fuse_pad_conv_pool,
    FPass_hwa_conv_inc_adaptor,
    FPass_hwa_conv_outc_adaptor,
    FPass_hwa_pooling_c_adaptor,
    FPass_hwa_resize_c_adaptor,
    FPass_crop_conv_kernel,
    FPass_split_conv_kernel,
    FPass_split_conv_pad,
    FPass_split_pool3d_pad,
    FPass_stride_kernel_size_adaptor,
    FPass_split_conv_input_c,
    FPass_hwa_conv_inc_adaptor,
    FPass_split_conv_output_c,
    FPass_hwa_conv_outc_adaptor,
    FPass_split_to_crops,
    FPass_adapt_eltwise_in_type,
    FPass_transpose_to_d2s,
    FPass_abs_to_leakyrelu,
    FPass_cast_to_leakyrelu,
    FPass_hwa_bn_inc_outc_adaptor,
    FPass_eltwise_input_dim_adaptor,
    FPass_hwa_eltwise_c_adaptor,
    FPass_hwa_add_mul_sub_c_adaptor,
    FPass_hwa_add_mul_sub_h_adaptor,
    FPass_hwa_relu_c_adaptor,
    FPass_hwa_softmax_c_adaptor,
    // FPass_conv1d_tiling,
    FPass_conv_full_tiling,
    FPass_fuse_crop,
    FPass_fuse_crop_pad,
    FPass_fuse_crop_pad_to_crop,
    FPass_fuse_pad_crop,
    FPass_offset_crop_pad,
    FPass_offset_crop_pad_with_elt,
    FPass_merge_redundant_pad,
    FPass_fuse_transpose_reshape_matmul,
    FPass_matmul_shape_adaptor,
    // NOTE: swap_hwa_matmul_inputs must be before matmul_c_adaptor and adjust_matmul_inputs
    FPass_swap_hwa_matmul_inputs,
    // NOTE: adjust_matmul_w_for_mtp_util must be before store_matmul_origin_shapes
    FPass_adjust_matmul_w_for_mtp_util,
    // NOTE: adjust_matmul_for_large_w must be before store_matmul_origin_shapes
    FPass_adjust_matmul_for_large_w,
    // NOTE: split_matmul_for_large_inc must be before adjust_matmul_inputs
    FPass_split_matmul_for_large_inc,
    // NOTE: split_matmul_for_large_outc must be before adjust_matmul_inputs
    // NOTE: split_matmul_for_large_outc must be after swap_hwa_matmul_inputs
    FPass_split_matmul_for_large_outc,
    FPass_store_matmul_origin_shapes,
    // NOTE: matmul_c_adaptor must be before adjust_matmul_inputs
    FPass_eliminate_dynamic_matmul,
    FPass_matmul_c_adaptor,
    FPass_full_shape_add_sub_mul_to_eltwise,

    FPass_adjust_matmul_inputs,
    FPass_matmul_to_fc,
    FPass_lift_cache_pad,
    FPass_lift_cache_reshape,
    FPass_lift_cache_transpose,
    FPass_adapt_cache_stride,
    FPass_cache_c_adaptor,
    FPass_offset_crop_pad,
    FPass_offset_crop_pad_with_elt,
    FPass_eliminate_crop,
    FPass_eliminate_pad,
    FPass_fuse_transpose,
    FPass_eliminate_transpose,
    FPass_fuse_transpose,
    // FPass_interp_to_upsample,
    FPass_adapt_boundingbox_in_type,
    FPass_adapt_eltwise_out_type,
    // FPass_remove_concat,
    FPass_patch_internal_crop,
    FPass_patch_internal_split,
    FPass_datalayout_adaptor,
    FPass_find_large_kernel_conv,

    FPass_CSE_subnode,
    FPass_patch_internal_concat,
    FPass_adapt_arithmetic_op,
    FPass_lower_reshape_combine,
    FPass_layernorm_shape_adaptor,
    FPass_fuse_reshape,
    FPass_lift_rmsnorm_reshape,
    FPass_softmax_shape_adaptor,
    FPass_normalization_shape_adaptor,
    // NOTE: softmax_shape_adaptor should be after lower_reshape_combine
    // FPass_fuse_reshape,
    // FPass_eliminate_reshape,
    FPass_fuse_transpose_with_reshape,
    FPass_fuse_transpose,
    FPass_eliminate_transpose,
    FPass_eliminate_cast,

    FPass_decompose_groupnorm_aiff,
    FPass_decompose_groupnorm,
    FPass_decompose_mvn,
    FPass_decompose_mvn_aiff,
    // groupnorm

    // it should after adapt_arithmetic_op, fuse_constant_matmul_to_fc, batch_FC_to_conv... before eliminate_reshape or fuse_reshape
    FPass_eliminate_crop_pad_pair,
    FPass_split_multi_head_block,
    FPass_split_fc_activation,  // for sd
    FPass_hwa_bn_batch_process, // must after lower reshape for BN(both in gsim and lowering)
    FPass_fuse_reshape,
    FPass_eliminate_reshape,
    FPass_duplicate_chw_transpose, // must after any fuse same transpose pass

    FPass_optimize_transformer_conv_multi_head,
    FPass_optimize_transformer_fc_multi_head,
    FPass_fuse_split_concat,
    FPass_fuse_constant_pad,
    FPass_split_special_concat,
    // FPass_split_transpose,
    FPass_split_transpose_by_perm,
    FPass_split_to_crops,
    FPass_split_softmax,
    FPass_apply_tune_strategy,
    FPass_fuse_resize_crop,
    FPass_lower_d2s_d2s_eltwise,
    FPass_remove_constant,
    FPass_CSE_subnode,
    FPass_gatherelements_to_gather,
    FPass_merge_switch_dummy,
    // it muse be at last
    FPass_reset_all_tensors_stride>();

static auto passes_after_unfold = make_pass_tuple<
    FPass_lift_reshape,
    FPass_lower_quantize_reshape_matmul,
    FPass_lift_transpose,
    FPass_eliminate_transpose,
    FPass_fuse_reshape>();

static auto passes_before_plugin = make_pass_tuple<
    FPass_fuse_multi_scalars>();

static auto passes_after_optimize = make_pass_tuple<
    FPass_actg,
    FPass_calculate_node_ops_count,
    DataLayoutSchedule,
    FPass_remove_transformer_multi_head_layoutconvertor,
    FPass_remove_dummy_slice,
    // FPass_adapt_tensor_strides,
    // FPass_adapt_slice_strides,
    FPass_remove_crop,
    // FPass_remove_crop_with_strides,
    FPass_remove_split,
    FPass_remove_concat,
    FPass_remove_pad,
    FPass_remove_switch,
    FPass_unify_transpose_layout,
    FPass_adapt_concat_strides,
    FPass_record_gm_tensor,
    FPass_max_batch_adaptor,
    FPass_virtual_node_memory_remapping>();

// This is used to find passes that are not called by lowring functions.
static auto pass_uncalled = make_pass_tuple<
    FPass_apply_plugins,
    FPass_apply_internal_plugins>();

void lower_before_all(GraphPtr g)
{
    passes_before_all->set_recursive(true);
    passes_before_all->run_on_graph(g);
}

void lower_before_tiling(GraphPtr g)
{
    auto passes_before_tiling = PassManager::create("passes_before_tiling");
    for (const auto &name : get_pass_before_tiling())
    {
        bool found = false;
        if (get_disable_passes().count(name))
        {
            DEBUG() << "Enable pass " << name << " before configure --pass-before-tiling.";
            continue;
        }

        for (auto &&p : passes_before_optimize->passes())
        {
            if (name == p->name() ||
                (name == "all" && !get_disable_passes().count(p->name()) && p->name() != "tile_graph"))
            {
                DEBUG() << "The pass " << p->name() << " will be executed before tiling.";
                passes_before_tiling->add(p);
                found = true;
                if (name != "all")
                {
                    break;
                }
            }
        }

        if (!found)
        {
            DEBUG() << "The pass " << name << " provided is invalid.";
        }
    }
    passes_before_tiling->set_recursive(true);
    passes_before_tiling->run_on_graph(g);
}

void lower_before_optimize(GraphPtr g)
{
    passes_before_optimize->set_recursive(true);
    passes_before_optimize->run_on_graph(g);
}

void lower_after_unfold(GraphPtr g)
{
    passes_after_unfold->run_on_graph(g);
}

void lower_before_plugin(GraphPtr g)
{
    passes_before_plugin->set_recursive(true);
    passes_before_plugin->run_on_graph(g);
}

void lower_after_optimize(GraphPtr g)
{
    passes_after_optimize->set_recursive(true);
    passes_after_optimize->run_on_graph(g);
}

void passes_after_build(GraphPtr g)
{
    static auto passes = make_pass_tuple<
        FPass_add_address_to_dot,
        FPass_add_tensor_size_to_dot,
        FPass_add_mem_info_to_dot,
        FPass_add_estimate_info_to_dot,
        FPass_add_profiler_placeholder_to_dot>();
    passes->run_on_graph(g);
}

void show_all_passes_name()
{
    std::cout << "passes before merge layers:\n";
    std::cout << "***************************\n";
    std::cout << passes_before_all->name() << "\n";
    std::cout << passes_before_optimize->name() << "\n";
    std::cout << passes_after_unfold->name() << "\n";
    std::cout << "passes after merge layers:\n";
    std::cout << "**************************\n";
    std::cout << passes_after_optimize->name() << "\n";
}

bool find_pass_by_name(const std::string &name, std::function<bool(GraphPtr)> &func)
{
    static std::map<std::string, std::function<bool(GraphPtr)>> passes;

    if (passes.empty())
    {
        for (auto pm : {passes_before_all, passes_before_optimize, passes_after_unfold, passes_after_optimize, pass_uncalled, other_passes})
        {
            for (auto p : pm->passes())
            {
                passes[p->name()] = [p](GraphPtr g) -> bool { return p->operator()(g); };
            }
        }
    }

    if (passes.count(name))
    {
        func = passes[name];
        return true;
    }

    return false;
}
} // namespace aipubt

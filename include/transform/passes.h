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
 * @Date: 2021-10-29 15:30:33
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-07-02 10:14:30
 */
#ifndef TRANSFORM_PASSES_H
#define TRANSFORM_PASSES_H
#include "core/basepass.h"
#include "core/core.h"
#include <algorithm>
#include <set>
#ifdef PASS_FUNCTION
#undef PASS_FUNCTION
#endif
#ifdef STATIC_PASS_FUNCTION
#undef STATIC_PASS_FUNCTION
#endif
#ifdef DYNAMIC_PASS_FUNCTION
#undef DYNAMIC_PASS_FUNCTION
#endif
#define PASS_FUNCTION(f)                            \
    bool f(GraphPtr g);                             \
    class FPass_##f : public FuncPass<FPass_##f, f> \
    {                                               \
    };
#define STATIC_PASS_FUNCTION(f)                                      \
    bool f(GraphPtr g);                                              \
    class FPass_##f : public FuncPass<FPass_##f, f, PassType_Static> \
    {                                                                \
    };
#define DYNAMIC_PASS_FUNCTION(f)                                      \
    bool f(GraphPtr g);                                               \
    class FPass_##f : public FuncPass<FPass_##f, f, PassType_Dynamic> \
    {                                                                 \
    };
/**
 * @brief This file is all passes apply for graph optimization, which is data-type-free.
 *
 */

namespace aipubt
{
PASS_FUNCTION(graph_infer_shape);
PASS_FUNCTION(fuse_transpose);
PASS_FUNCTION(fuse_reshape);
PASS_FUNCTION(slice_to_crop);
PASS_FUNCTION(eliminate_transpose);
PASS_FUNCTION(eliminate_reshape);
PASS_FUNCTION(eliminate_resize);
STATIC_PASS_FUNCTION(fuse_transpose_with_reshape);
STATIC_PASS_FUNCTION(fuse_pad);
STATIC_PASS_FUNCTION(fuse_crop);
PASS_FUNCTION(fuse_cast);
PASS_FUNCTION(fuse_constant_cast);
PASS_FUNCTION(eliminate_cast);
PASS_FUNCTION(fuse_activation);
PASS_FUNCTION(fuse_pad_crop);
PASS_FUNCTION(fuse_split);
PASS_FUNCTION(fuse_concat);
PASS_FUNCTION(fuse_concat_split);
PASS_FUNCTION(fuse_concat_splits);
PASS_FUNCTION(fuse_concat_reshape_splits);
PASS_FUNCTION(concat_split_with_arithmetic);
PASS_FUNCTION(lower_trans_trans_eltwise);
PASS_FUNCTION(lower_trans_mul_or_add_or_sub);
PASS_FUNCTION(rotate_trans);
PASS_FUNCTION(swap_trans_eltwise);
PASS_FUNCTION(CSE_subnode);
PASS_FUNCTION(sqrt_div_to_rsqrt_mul);
PASS_FUNCTION(fuse_concat_constant_to_pad);
PASS_FUNCTION(fuse_pad_conv_pool);
PASS_FUNCTION(fuse_crop_conv_pool);
PASS_FUNCTION(swap_pad_crop);
PASS_FUNCTION(eliminate_split);
PASS_FUNCTION(fuse_tile_eltwise);
PASS_FUNCTION(fuse_mul_add_to_bn);
PASS_FUNCTION(fuse_add_mul_to_bn);
STATIC_PASS_FUNCTION(fuse_mul_constant_to_bn);
STATIC_PASS_FUNCTION(fuse_add_constant_to_bn);
PASS_FUNCTION(lower_transpose);
PASS_FUNCTION(fuse_quantize_and_dequantize);
PASS_FUNCTION(fuse_fp8_dequantize);
PASS_FUNCTION(eliminate_concat_crop);
PASS_FUNCTION(eliminate_crop);
PASS_FUNCTION(eliminate_pad);
PASS_FUNCTION(fuse_pooling);
PASS_FUNCTION(fuse_clamp);
PASS_FUNCTION(lift_activation);
PASS_FUNCTION(merge_conv_output_crop);
PASS_FUNCTION(fuse_conv_outc);
PASS_FUNCTION(fuse_conv1x1_conv);
PASS_FUNCTION(fuse_constant_eltwise_to_leakyrelu);
PASS_FUNCTION(fuse_sigmoid_mul_to_silu);
PASS_FUNCTION(fuse_split_concat);
PASS_FUNCTION(reduce_const_eltwise);
PASS_FUNCTION(eliminate_channel_shuffle);
PASS_FUNCTION(lower_trans_trans_concat);
PASS_FUNCTION(fuse_s2d_conv);
PASS_FUNCTION(eliminate_global_transpose);
PASS_FUNCTION(fuse_constant_matmul_to_fc);
PASS_FUNCTION(conv_transpose_to_d2s);
PASS_FUNCTION(remove_batch);
PASS_FUNCTION(lower_reshape);
PASS_FUNCTION(duplicate_reshape);
PASS_FUNCTION(duplicate_transpose);
STATIC_PASS_FUNCTION(lower_reshape_combine);
STATIC_PASS_FUNCTION(split_fc_transpose);
PASS_FUNCTION(lower_reshape_reshape_eltwise);
PASS_FUNCTION(lower_reshape_eltwise);
PASS_FUNCTION(lower_reshape_reshape_add);
PASS_FUNCTION(lower_reshape_add);
// PASS_FUNCTION(eliminate_constant_add);
PASS_FUNCTION(fuse_slice);
PASS_FUNCTION(fuse_slice_concat);
PASS_FUNCTION(fuse_tile_squareddiff);
PASS_FUNCTION(simplify_name);
PASS_FUNCTION(fuse_fc_bn);
PASS_FUNCTION(full_shape_add_sub_mul_to_eltwise);
PASS_FUNCTION(fuse_conv_transpose);
PASS_FUNCTION(fuse_transpose_conv);
PASS_FUNCTION(fuse_transpose_layernorm);
PASS_FUNCTION(swap_gather_trans);
PASS_FUNCTION(eliminate_reshape_reshape);
PASS_FUNCTION(eliminate_transpose_transpose);
PASS_FUNCTION(move_activation);
PASS_FUNCTION(split_transpose);
PASS_FUNCTION(split_transpose_by_perm);
PASS_FUNCTION(split_to_single_crop);
PASS_FUNCTION(split_to_crops);
PASS_FUNCTION(lower_gather);
STATIC_PASS_FUNCTION(fuse_crop_pad);
PASS_FUNCTION(fuse_constant_pad);
PASS_FUNCTION(eliminate_concat_denseblock);
PASS_FUNCTION(eliminate_crop_denseblock);
PASS_FUNCTION(eliminate_dense_concat);
PASS_FUNCTION(eliminate_dense_crop);
PASS_FUNCTION(div_const_to_mul);
STATIC_PASS_FUNCTION(fuse_crop_pad_to_crop);
PASS_FUNCTION(set_layer_id);
PASS_FUNCTION(eliminate_eltwise_constant);
PASS_FUNCTION(swap_reshape_transpose);
PASS_FUNCTION(swap_bn_transpose);
PASS_FUNCTION(fuse_fc_crop);
PASS_FUNCTION(lift_crop);
PASS_FUNCTION(fuse_transpose_matmul);
PASS_FUNCTION(fuse_quantize_cast);
PASS_FUNCTION(fuse_matmul_cast);
PASS_FUNCTION(fuse_reshape_transpose_conv);
STATIC_PASS_FUNCTION(transpose_to_d2s);
PASS_FUNCTION(fuse_same_constant);
STATIC_PASS_FUNCTION(gather_to_slice);
PASS_FUNCTION(fuse_equal_not);
STATIC_PASS_FUNCTION(split_matmul_for_large_inc);
STATIC_PASS_FUNCTION(lift_rmsnorm_reshape);
PASS_FUNCTION(fill_constant_out_shape);
PASS_FUNCTION(split_constant);
PASS_FUNCTION(binary_op_broadcast);
PASS_FUNCTION(lower_quantize_reshape_matmul);
PASS_FUNCTION(gatherelements_to_gather);
STATIC_PASS_FUNCTION(global_avg_pool_3d_to_2d);
PASS_FUNCTION(fuse_multi_scalars);
} // namespace aipubt

#endif //! TRANSFORM_PASSES_H

//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2021-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire !ice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author Tom.Gao
 * @Date 2021-01-25 170621
 * @Last Modified by Tom Gao
 * @Last Modified time 2022-07-19 154143
 */
#include "gsim/gsim.h"
#include "core/graph.h"
#include "transform/passes.h"
#include "utils/utils.h"
namespace aipubt
{

GSim::GSim()
{
    add(make_pass_tuple<
        FPass_split_constant,
        FPass_fill_constant_out_shape,
        FPass_CSE_subnode,
        FPass_fuse_fp8_dequantize,
        FPass_eliminate_split,
        FPass_fuse_split,
        FPass_split_to_single_crop,
        FPass_gather_to_slice,
        FPass_slice_to_crop,
        FPass_fuse_cast,
        FPass_eliminate_cast,
        FPass_fuse_quantize_and_dequantize,
        FPass_transpose_to_d2s,
        FPass_conv_transpose_to_d2s,
        FPass_swap_gather_trans,
        FPass_lower_transpose,
        FPass_fuse_transpose_with_reshape,
        FPass_fuse_transpose,
        FPass_fuse_reshape,
        FPass_eliminate_reshape,
        FPass_eliminate_crop,
        FPass_binary_op_broadcast,
        FPass_div_const_to_mul,
        FPass_fuse_constant_eltwise_to_leakyrelu,
        FPass_fuse_sigmoid_mul_to_silu,
        FPass_fuse_constant_matmul_to_fc,
        FPass_gatherelements_to_gather>());
    add("lower_and_fuse_transpose",
        [](GraphPtr g) -> bool
        {
            bool ret = false;
            // CSE will fuse failed duplicated node
            // atomic flag in it will stop when -f
            ret = duplicate_transpose(g) || ret;
            ret = lower_transpose(g) || ret;
            ret = lower_trans_trans_concat(g) || ret;
            ret = fuse_transpose(g) || ret;
            ret = lower_trans_mul_or_add_or_sub(g) || ret;
            ret = lower_trans_trans_eltwise(g) || ret;

            if (swap_trans_eltwise(g))
            {
                bool is_opt = false;
                is_opt = fuse_transpose(g);
                lower_transpose(g);
                lower_trans_trans_concat(g);
                while (lower_trans_trans_eltwise(g))
                {
                    is_opt = true;
                    if (!rotate_trans(g))
                    {
                        break;
                    }
                }
                is_opt = fuse_transpose(g);
                rotate_trans(g);

                while (lower_trans_trans_eltwise(g))
                {
                    is_opt = true;
                    if (!rotate_trans(g))
                    {
                        break;
                    }
                }

                is_opt = fuse_transpose(g);
                ret = is_opt || ret;
            }

            ret = lower_transpose(g) || ret;
            ret = lower_trans_trans_concat(g) || ret;

            while (fuse_transpose_with_reshape(g))
            {
                ret = true;
                ret = lower_transpose(g) || ret;
                if (!fuse_transpose(g) && !fuse_reshape(g))
                {
                    break;
                }
            }

            while (lower_transpose(g))
            {
                ret = true;
                ret = fuse_transpose(g) || ret;
                ret = lower_trans_trans_concat(g) || ret;
            }
            ret = fuse_transpose(g) || ret;
            return ret;
        });
    add(make_pass_tuple<
            FPass_conv_transpose_to_d2s,
            FPass_fuse_concat_constant_to_pad,
            FPass_fuse_pad,
            FPass_fuse_crop,
            FPass_fuse_crop_pad,
            FPass_fuse_crop_pad_to_crop,
            FPass_fuse_pad_crop,
            FPass_fuse_pad_conv_pool>(),
        "fuse_all_pad",
        [](GraphPtr g)
        {
            bool ret = false;
            while (swap_pad_crop(g))
            {
                ret = true;
                ret = fuse_pad_conv_pool(g) || ret;
            }
            return ret;
        });
    add(make_pass_tuple<
            FPass_fuse_split,
            FPass_fuse_slice,
            FPass_fuse_concat,
            FPass_fuse_concat_split>(),
        "concat_split_with_arithmetic",
        [](GraphPtr g)
        {
            bool ret = false;
            while (concat_split_with_arithmetic(g))
            {
                ret = true;
            }
            return ret;
        });
    add("fuse_slice_concat",
        [](GraphPtr g)
        {
            if (fuse_slice_concat(g))
            {
                fuse_transpose(g);
                return true;
            }
            return false;
        });
    add(make_pass_tuple<
        FPass_fuse_constant_matmul_to_fc,
        FPass_lower_reshape_combine,
        FPass_fuse_reshape,
        FPass_eliminate_reshape,
        FPass_eliminate_eltwise_constant,
        FPass_fuse_fc_bn>());
    add(
        "fuse_activation",
        [](GraphPtr g)
        {
            if (!GSim::is_float_ir(g))
            {
                return fuse_activation(g);
            }
            return false;
        });
    add(
        make_pass_tuple<
            FPass_eliminate_resize,
            FPass_eliminate_channel_shuffle,
            FPass_eliminate_transpose,
            FPass_lower_reshape_combine,
            FPass_fuse_reshape,
            FPass_eliminate_reshape,
            FPass_CSE_subnode,
            FPass_sqrt_div_to_rsqrt_mul,
            FPass_reduce_const_eltwise,
            FPass_fuse_tile_eltwise,
            FPass_fuse_tile_squareddiff,
            FPass_fuse_pooling,
            FPass_fuse_clamp,
            FPass_merge_conv_output_crop,
            FPass_fuse_conv1x1_conv,
            FPass_fuse_conv_outc,
            FPass_lift_activation,
            FPass_fuse_transpose_matmul>(),
        "fuse_activation",
        [](GraphPtr g)
        {
            if (!GSim::is_float_ir(g))
            {
                return fuse_activation(g);
            }
            return false;
        },
        make_pass_tuple<
            FPass_fuse_s2d_conv,
            FPass_fuse_reshape_transpose_conv,
            FPass_eliminate_global_transpose>());
    add(
        make_pass_tuple<FPass_fuse_cast,
                        FPass_fuse_matmul_cast,
                        FPass_fuse_quantize_cast,
                        FPass_lower_reshape_combine,
                        FPass_fuse_reshape,
                        FPass_eliminate_reshape,
                        FPass_lift_crop,
                        FPass_fuse_fc_crop>(),
        "fuse_quantize_cast_pattern",
        [](GraphPtr g)
        {
            bool ret = false;
            while (fuse_quantize_cast(g))
            {
                ret = true;
                ret = fuse_cast(g) || ret;
            }
            while (fuse_cast(g))
            {
                ret = true;
                ret = fuse_quantize_cast(g) || ret;
            }
            ret = eliminate_cast(g) || ret;
            ret = fuse_constant_cast(g) || ret;
            return ret;
        });
    add(make_pass_tuple<
        FPass_fuse_same_constant,
        FPass_CSE_subnode,
        FPass_fuse_equal_not,
        FPass_split_matmul_for_large_inc,
        FPass_lift_rmsnorm_reshape,
        FPass_global_avg_pool_3d_to_2d,
        FPass_binary_op_broadcast,
        FPass_where_broadcast>());
}

bool GSim::is_float_ir(GraphPtr g)
{
    for (auto inp : g->get_input_tensors())
    {
        if (inp->type() >= Tensor::Dtype::FP32 && inp->type() <= Tensor::Dtype::FP64)
        {
            return true;
        }
    }
    return false;
}
} // namespace aipubt

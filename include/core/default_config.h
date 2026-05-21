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
 * @Date: 2020-03-13 17:31:09
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-01-06 17:51:08
 */
#pragma once
#include <core/any.h>
#include <core/constants.h>
#include <core/constexpr_utils.h>
#include <map>
#include <string>
#include <vector>

namespace aipubt
{
constexpr auto get_default_config_dict()
{
    constexpr auto default_config = cexpr::make_constexpr_dict(
        std::make_pair(Constants::target, "X3_1304"),
        std::make_pair(Constants::prof_unit, "default"),
        std::make_pair(Constants::enable_sparse_opt, false),
        std::make_pair(Constants::enable_sparse_opt, false),
        std::make_pair(Constants::enable_separate_gm, false),
        std::make_pair(Constants::tcm_size, 4 * 1024 * 1024),
        std::make_pair(Constants::gm_size, 4 * 1024 * 1024),
        std::make_pair(Constants::alloc_output_on_extsram, false),
        std::make_pair(Constants::ddr_data_gather, true),
        std::make_pair(Constants::disable_gm, false),
        std::make_pair(Constants::disable_input_buffer_reuse, false),
        std::make_pair(Constants::disable_mmu, false),
        std::make_pair(Constants::disable_output_buffer_reuse, false),
        std::make_pair(Constants::disable_cross_buffer_reuse, false),
        std::make_pair(Constants::disable_schedule_optimization, false),
        std::make_pair(Constants::disable_spsram_optimization, false),
        std::make_pair(Constants::dump_all_tensors, false),
        std::make_pair(Constants::dump_rt_tensor_id, -1),
        std::make_pair(Constants::dump_rt_input_tensors, false),
        std::make_pair(Constants::dump_rt_output_tensors, false),
        std::make_pair(Constants::enable_custom_smmu, false),
        std::make_pair(Constants::enable_gm_enhance, false),
        std::make_pair(Constants::enable_dynamic_gm, false),
        std::make_pair(Constants::enable_x2_sm_opt, false),
        std::make_pair(Constants::gm_enhance_strategy, 0),
        std::make_pair(Constants::dump_gm_enhance_cfg, false),
        std::make_pair(Constants::load_gm_enhance_cfg, ""),
        std::make_pair(Constants::dump_prior_in_gm, false),
        std::make_pair(Constants::load_prior_in_gm, ""),
        std::make_pair(Constants::dump_schedule_cfg, false),
        std::make_pair(Constants::load_schedule_cfg, ""),
        std::make_pair(Constants::enable_slc_ocm_mode, false),
        std::make_pair(Constants::enable_weight_align, false),
        std::make_pair(Constants::individual_inputs, false),
        std::make_pair(Constants::insert_init_reg, false),
        std::make_pair(Constants::load_balance_strategy, 1),
        std::make_pair(Constants::enable_ds, false),
        std::make_pair(Constants::enable_cl_opt, false),
        std::make_pair(Constants::printf_fifo_size, 16384u),
        std::make_pair(Constants::put_weight_gm, false),
        std::make_pair(Constants::serialize_scale_zp, true),
        std::make_pair(Constants::show_raw_graph, false),
        std::make_pair(Constants::show_short_name, false),
        std::make_pair(Constants::single_core_opt, false),
        std::make_pair(Constants::sort_after_each_pass, false),
        std::make_pair(Constants::stack_size, 263168u),
        std::make_pair(Constants::switch_for_debug, false),
        std::make_pair(Constants::wdc_rate, 0.9),
        std::make_pair(Constants::skip_simulator, false),
        std::make_pair(Constants::native_float_simulator, false),
        std::make_pair(Constants::dump_addr_cfg, false),
        std::make_pair(Constants::load_addr_cfg, ""),
        std::make_pair(Constants::max_wt_size, 3500ul * 1024ul * 1024ul),
        std::make_pair(Constants::profile, false),
        std::make_pair(Constants::datalayout_schedule_debug, false),
        std::make_pair(Constants::check_tensor_data_size, true),
        std::make_pair(Constants::break_descriptor_chain, false),
        std::make_pair(Constants::put_desc_gm, false),
        std::make_pair(Constants::put_ws_gm, false),
        std::make_pair(Constants::use_amat_scheduler, false),
        std::make_pair(Constants::branch_reorder_strategy, 0),
        std::make_pair(Constants::softmax_prior_in_gm, false),
        std::make_pair(Constants::disable_dependency, false),
        std::make_pair(Constants::soms_gm_loop, 100),
        std::make_pair(Constants::set_l2_flush, false),
        std::make_pair(Constants::prefetch_desc, false),
        std::make_pair(Constants::dump_tune_strategy, false),
        std::make_pair(Constants::load_tune_strategy, ""),
        std::make_pair(Constants::dump_reorder_cfg, false),
        std::make_pair(Constants::load_reorder_cfg, ""),
        std::make_pair(Constants::ws_mem_opt, false),
        std::make_pair(Constants::closer_sg_opt, false),
        std::make_pair(Constants::tile_split_output_c_conv, false),
        std::make_pair(Constants::remove_origin_constant, false),
        std::make_pair(Constants::enable_cse_tile_node, false),
        std::make_pair(Constants::specify_wt_gm, ""),
        std::make_pair(Constants::set_profile_option, ""),
        std::make_pair(Constants::show_soms_bottleneck, false),
        std::make_pair(Constants::split_wt_act, false),
        std::make_pair(Constants::random_init, 0),
        std::make_pair(Constants::max_sample_point_num, 0ul),
        std::make_pair(Constants::max_aiff_sample_point_num, 0ul),
        std::make_pair(Constants::max_dma_sample_point_num, 0ul),
        std::make_pair(Constants::max_custom_sample_point_num, 0ul),
        std::make_pair(Constants::asid_perf, "123,0"),
        std::make_pair(Constants::show_ds_shape, false),
        std::make_pair(Constants::print_emitter_var, false),
        std::make_pair(Constants::dump_constant_info, false),
        std::make_pair(Constants::dump_lora_info, false));
    return default_config;
}

constexpr auto get_default_config_value(const char *key)
{
    return get_default_config_dict().at(key);
}

template <typename T>
constexpr T get_default_config(const char *key)
{
    return get_default_config_value(key).get<T>();
}

inline Any constexpr_value_to_any(const cexpr::ConstexprValue &value)
{
    auto type = value.type();
    switch (type)
    {
    case cexpr::ConstexprValue::Type::INT:
        return Any(value.get<long long>());
    case cexpr::ConstexprValue::Type::UINT:
        return Any(value.get<unsigned long long>());
    case cexpr::ConstexprValue::Type::FLOAT:
        return Any(value.get<float>());
    case cexpr::ConstexprValue::Type::DOUBLE:
        return Any(value.get<double>());
    case cexpr::ConstexprValue::Type::BOOL:
        return Any(value.get<bool>());
    case cexpr::ConstexprValue::Type::STRING:
        return Any(value.get<const char *>());
    }
    return Any();
}

inline auto get_default_config(const char *key)
{
    auto v = get_default_config_value(key);
    return constexpr_value_to_any(v);
}

}; // namespace aipubt

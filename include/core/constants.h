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
 * @Date: 2020-08-21 15:57:42
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-07-15 16:17:03
 */
#ifndef CORE_CONSTANS_H
#define CORE_CONSTANS_H
namespace aipubt
{
namespace Constants
{
    // environment variables name
    constexpr static const char *LIB_PATH = "AIPULIB_PATH";
    constexpr const char *PASS_PERF = "AIPUBT_ENBALE_PASS_PERF";
    constexpr const char *PLUGIN_PATH = "AIPUPLUGIN_PATH";
    constexpr const char *LOGLEVEL = "AIPUBUILDER_LOG";
    constexpr const char *SUBGRAPH_VIEW = "AIPU_SUBGRAPH_VIEW";
    constexpr const char *AIPU_EXTERNAL_SRAM = "AIPU_EXTERNAL_SRAM";
    constexpr const char *ENABLE_ASID = "AIPU_ENABLE_ASID";
    constexpr const char *DISABLE_BACKTRACE = "AIPU_DISABLE_BACKTRACE";
    constexpr const char *AIPUOAS = "AIPUOAS";
    constexpr const char *AIPUOLD = "AIPUOLD";
    constexpr const char *AIPUOCC = "AIPUOCC";

    constexpr const char *CPSLD = "CPSLD";
    constexpr const char *CPSCC = "CPSCC";
    constexpr const char *CPS_PATH = "CPS_PATH";
    constexpr const char *CPSHOSTCC = "CPSHOSTCC";
    constexpr const char *CPSDEVKIT_PATH = "CPSDEVKIT_PATH";

    // switch keys/global config
    constexpr const char *random_init = "random-init";
    constexpr const char *enable_sparse_opt = "enable_sparse_opt";
    constexpr const char *serialize_scale_zp = "serialize_scale_zp";
    constexpr const char *show_location = "show_location";
    constexpr const char *dump_debug_IR = "dump_debug_IR";

    constexpr const char *disable_mmu = "disable_mmu";
    constexpr const char *show_info = "show_info";
    constexpr const char *sort_after_each_pass = "sort_after_each_pass";
    constexpr const char *disable_input_buffer_reuse = "disable_input_buffer_reuse";
    constexpr const char *disable_output_buffer_reuse = "disable_output_buffer_reuse";
    constexpr const char *disable_cross_buffer_reuse = "disable_cross_buffer_reuse";
    constexpr const char *alloc_output_on_extsram = "alloc_output_on_extsram";
    constexpr const char *disable_spsram_optimization = "disable_spsram_optimization";
    constexpr const char *dump_all_tensors = "dump_all_tensors";
    constexpr const char *dump_rt_tensor_id = "dump_rt_tensor_id";
    constexpr const char *dump_rt_input_tensors = "dump_rt_input_tensors";
    constexpr const char *dump_rt_output_tensors = "dump_rt_output_tensors";
    constexpr const char *individual_inputs = "individual-inputs";
    constexpr const char *switch_for_debug = "switch_for_debug";
    constexpr const char *show_raw_graph = "show_raw_graph";
    constexpr const char *show_short_name = "show_short_name";
    constexpr const char *USE_SLOTS = "USE_SLOTS";
    constexpr const char *single_core_opt = "single_core_opt";
    constexpr const char *disable_separate_gm = "disable_separate_gm";
    constexpr const char *enable_separate_gm = "enable_separate_gm";
    // constexpr const char *mem_opt = "mem_opt";
    constexpr const char *tile_cfg = "tile_cfg";
    constexpr const char *dump_tile_cfg = "dump_tile_cfg";
    constexpr const char *enable_cse_tile_node = "enable_cse_tile_node";
    constexpr const char *enable_debug_info = "enable_debug_info";
    constexpr const char *disable_plugins = "disable-plugins";
    constexpr const char *datalayout_schedule_debug = "datalayout_schedule_debug";

    constexpr const char *warn_all = "warn-all";

    constexpr const char *target = "target";
    constexpr const char *ocm_size = "ocm_size";
    constexpr const char *ocm_addr = "ocm_addr";
    constexpr const char *gm_size = "gm_size";
    constexpr const char *tcm_size = "tcm_size";
    constexpr const char *enable_tcm = "enable_tcm";
    constexpr const char *disable_schedule_optimization = "disable_schedule_optimization";
    constexpr const char *ddr_data_gather = "ddr_data_gather";
    constexpr const char *enable_custom_smmu = "enable_custom_smmu";
    constexpr const char *ddr_data_mode = "ddr_data_mode";
    constexpr const char *disable_gm = "disable_gm";
    constexpr const char *enable_gm_enhance = "enable_gm_enhance";
    constexpr const char *enable_dynamic_gm = "enable_dynamic_gm";
    constexpr const char *enable_x2_sm_opt = "enable_x2_sm_opt";
    constexpr const char *gm_enhance_strategy = "gm_enhance_strategy";
    constexpr const char *dump_gm_enhance_cfg = "dump_gm_enhance_cfg";
    constexpr const char *load_gm_enhance_cfg = "load_gm_enhance_cfg";
    constexpr const char *load_prior_in_gm = "load_prior_in_gm";
    constexpr const char *dump_prior_in_gm = "dump_prior_in_gm";
    constexpr const char *softmax_prior_in_gm = "softmax_prior_in_gm";
    constexpr const char *insert_init_reg = "insert_init_reg";
    constexpr const char *multi_batch = "multi_batch";
    constexpr const char *load_balance_strategy = "load_balance_strategy";
    constexpr const char *perf_json = "perf_json";
    constexpr const char *perf_bin = "perf_bin";
    constexpr const char *global_cwd = "global_cwd";
    constexpr const char *enable_weight_align = "enable_weight_align";
    constexpr const char *enable_slc_ocm_mode = "enable_slc_ocm_mode";
    constexpr const char *put_weight_gm = "put_weight_gm";
    constexpr const char *set_asid = "set_asid";
    constexpr const char *enable_ds = "enable_ds";
    constexpr const char *input_shape = "input_shape";
    constexpr const char *max_input_shape = "max_input_shape";
    constexpr const char *default_input_shape = "default_input_shape";
    constexpr const char *dynamic_input_dim = "dynamic_input_dim";
    constexpr const char *skip_simulator = "skip_simulator";
    constexpr const char *native_float_simulator = "native_float_simulator";
    constexpr const char *dump_schedule_cfg = "dump_schedule_cfg";
    constexpr const char *load_schedule_cfg = "load_schedule_cfg";
    constexpr const char *dump_reorder_cfg = "dump_reorder_cfg";
    constexpr const char *load_reorder_cfg = "load_reorder_cfg";
    constexpr const char *max_wt_size = "max_wt_size";
    constexpr const char *max_act_size = "max_act_size";
    constexpr const char *split_wt_act = "split_wt_act";
    constexpr const char *disable_icache = "disable_icache";
    constexpr const char *put_desc_gm = "put_desc_gm";
    constexpr const char *put_ws_gm = "put_ws_gm";
    constexpr const char *disable_dependency = "disable_dependency";
    constexpr const char *soms_gm_loop = "soms_gm_loop";
    constexpr const char *set_l2_flush = "set_l2_flush";
    constexpr const char *prefetch_desc = "prefetch_desc";
    constexpr const char *ws_mem_opt = "ws_mem_opt";
    constexpr const char *virtual_core_id = "virtual_core_id";
    constexpr const char *show_soms_bottleneck = "show_soms_bottleneck";
    constexpr const char *show_ds_shape = "show_ds_shape";
    constexpr const char *print_emitter_var = "print_emitter_var";
    constexpr const char *dump_constant_info = "dump_constant_info";
    constexpr const char *dump_lora_info = "dump_lora_info";
    constexpr const char *mq_save_ir = "mq_save_ir";
    constexpr const char *mq_load_ir = "mq_load_ir";
    constexpr const char *mq_ir_stage = "mq_ir_stage";

    constexpr const char *load_addr_cfg = "load_addr_cfg";
    constexpr const char *dump_addr_cfg = "dump_addr_cfg";
    constexpr const char *load_wt_update_cfg = "load_wt_update_cfg";
    constexpr const char *multi_core = "multi_core";
    constexpr const char *multi_tec = "multi_tec";
    constexpr const char *target_name = "target_name";
    constexpr const char *profile = "profile";
    constexpr const char *set_profile_option = "set_profile_option";
    // profiler options
    constexpr const char *max_sample_point_num = "max_sample_point_num";
    constexpr const char *max_aiff_sample_point_num = "max_aiff_sample_point_num";
    constexpr const char *max_dma_sample_point_num = "max_dma_sample_point_num";
    constexpr const char *max_custom_sample_point_num = "max_custom_sample_point_num";
    constexpr const char *asid_perf = "asid_perf";

    constexpr const char *mov_w_to_gm = "mov_w_to_gm";
    constexpr const char *mov_w_to_gm_on = "mov_w_to_gm_on";
    constexpr const char *mov_w_to_gm_wait = "mov_w_to_gm_wait";
    constexpr const char *wait_w_in_front = "wait_w_in_front";
    constexpr const char *mov_w_gm_offset = "mov_w_gm_offset";
    constexpr const char *mov_w_gm_size = "mov_w_gm_size";
    constexpr const char *ld_w_from_gm = "ld_w_from_gm";
    constexpr const char *ld_w_gm_offset = "ld_w_gm_offset";
    constexpr const char *ro_end = "ro_end";
    constexpr const char *same_w = "same_w";
    constexpr const char *same_w_same_level = "same_w_same_level";
    constexpr const char *mov_d_to_ddr = "mov_d_to_ddr";
    constexpr const char *mov_d_to_ddr_wait = "mov_d_to_ddr_wait";
    constexpr const char *mov_d_to_ddr_on = "mov_d_to_ddr_on";
    constexpr const char *wr_out_to_gm = "wr_out_to_gm";
    constexpr const char *wr_out_to_gm_idx = "wr_out_to_gm_idx";
    constexpr const char *mem_clock = "mem_clock";
    constexpr const char *node_level = "node_level";
    constexpr const char *proxy_node = "proxy_node";
    constexpr const char *check_tensor_data_size = "check_tensor_data_size";
    constexpr const char *break_descriptor_chain = "break_descriptor_chain";
    constexpr const char *use_amat_scheduler = "use_amat_scheduler";
    constexpr const char *use_single_mem = "use_single_mem";
    constexpr const char *branch_reorder_strategy = "branch_reorder_strategy";
    constexpr const char *warmup_len = "warmup_len";
    constexpr const char *preserve_gm_size = "preserve_gm_size";
    constexpr const char *load_tune_strategy = "load_tune_strategy";
    constexpr const char *dump_tune_strategy = "dump_tune_strategy";
    constexpr const char *closer_sg_opt = "closer_sg_opt";
    constexpr const char *reuse_buffer_expand = "reuse_buffer_expand";
    constexpr const char *tile_split_output_c_conv = "tile_split_output_c_conv";
    constexpr const char *remove_origin_constant = "remove_origin_constant";
    constexpr const char *specify_wt_gm = "specify_wt_gm";
    constexpr const char *set_mq_num = "set_mq_num";

    // attributes keys
    constexpr const char *random_reg = "random_reg";
    constexpr const char *dumpspace = "dumpspace";
    constexpr const char *workspace = "workspace";
    constexpr const char *localspace = "localspace";
    constexpr const char *parambase = "parambase";
    constexpr const char *param_bases = "param_bases";
    constexpr const char *stream_bases = "stream_bases";
    constexpr const char *stream_id = "stream_id";
    constexpr const char *remap_table = "remap_table";
    constexpr const char *descriptorbase = "descriptorbase";
    constexpr const char *mmu_reserve = "mmu_reserve";
    constexpr const char *mmu_support = "support_mmu";
    constexpr const char *sram_free = "sram_free";
    constexpr const char *valid_input_layouts = "valid_input_layouts";
    constexpr const char *valid_output_layouts = "valid_output_layouts";
    constexpr const char *deny_layouts_comb = "deny_layouts_comb";
    constexpr const char *keeping_layout = "keeping_layout";
    constexpr const char *next_descriptor_length = "next_descriptor_length";
    constexpr const char *next_descriptor_offset = "next_descriptor_offset";
    constexpr const char *next_descriptor_eof = "next_descriptor_eof";
    constexpr const char *first_descriptor_length = "first_descriptor_length";
    constexpr const char *first_descriptor_offset = "first_descriptor_offset";
    constexpr const char *previous_descriptor_eof = "previous_descriptor_eof";
    constexpr const char *descriptor_act = "descriptor_act";
    constexpr const char *descriptor_act_base = "descriptor_act_base";
    constexpr const char *descriptor_param = "descriptor_param";
    constexpr const char *descriptor_param_base = "descriptor_param_base";
    constexpr const char *descriptor_dt = "descriptor_dt";
    constexpr const char *descriptor_dt_base = "descriptor_dt_base";
    constexpr const char *pre_config_layouts = "pre_config_layouts";
    constexpr const char *use_in_tensor = "use_in_tensor";
    constexpr const char *use_out_tensor = "use_out_tensor";
    constexpr const char *is_workspace_tensor = "is_workspace_tensor";
    constexpr const char *numa = ".numa";

    constexpr const char *printf_fifo = "printf_fifo";
    constexpr const char *printf_fifo_size = "printf_fifo_size";
    constexpr const char *stack_size = "stack_size";
    constexpr const char *op_cnt = "op_cnt";
    constexpr const char *error_code = "error_code";
    constexpr const char *preallocated_soc_sram = "preallocated-soc-sram";
    constexpr const char *prof_unit = "prof_unit";
    constexpr const char *fast_perf = "fast_perf";
    constexpr const char *enable_dma_perf = "enable_dma_perf";
    constexpr const char *max_perf_sample_num = "max_perf_sample_num";
    constexpr const char *max_aiff_perf_sample_num = "max_aiff_perf_sample_num";
    constexpr const char *max_dma_perf_sample_num = "max_dma_perf_sample_num";
    constexpr const char *max_custom_perf_sample_num = "max_custom_perf_sample_num";
    constexpr const char *ro_offset = ".ro_offset";
    constexpr const char *ro_size = ".ro_size";
    constexpr const char *real_ro_size = ".real_ro_size";
    constexpr const char *desc_offset = ".desc_offset";
    constexpr const char *desc_size = ".desc_size";
    constexpr const char *real_desc_size = ".real_desc_size";
    constexpr const char *schedule_level = ".schedule_level";
    constexpr const char *schedule_gid = ".schedule_gid";
    constexpr const char *kernel_func = ".kernel_func";
    constexpr const char *codetree = ".codetree";
    constexpr const char *performance_info = ".performance_info";
    constexpr const char *bss_id = "bss_id";
    constexpr const char *n_align = ".n_align";
    constexpr const char *h_align = ".h_align";
    constexpr const char *w_align = ".w_align";
    constexpr const char *c_align = ".c_align";
    constexpr const char *remapping_node = ".remapping_node";
    constexpr const char *max_batch_offset = ".max_batch_offset";
    constexpr const char *max_batch_size = "max_batch_size";
    constexpr const char *kernel_loop_count = ".kernel_loop_count";
    constexpr const char *temp_param_base = ".temp_param_base";
    constexpr const char *tensor_shape_base = ".tensor_shape_base";
    constexpr const char *output_shape_offset = ".output_shape_offset";
    constexpr const char *output_shape_size = ".output_shape_size";
    constexpr const char *output_symbol_shape = ".output_symbol_shape";
    constexpr const char *virtual_multi_core = "virtual_multi_core";
    constexpr const char *virtual_core_num = "virtual_core_num";
    constexpr const char *virtual_core_idx = "virtual_core_idx";
    constexpr const char *virtual_core_node = "virtual_core_node";
    constexpr const char *lora_node = "lora_node";
    constexpr const char *lora_info = "lora_info";

    constexpr const char *build_error_log = "build_error_log";
    constexpr const char *build_warn_log = "build_warn_log";

    constexpr const char *dot_sections = ".dot_sections";
    constexpr const char *dot_address = ".dot_address";
    constexpr const char *dot_for_profiler = ".dot_for_profiler";
    constexpr const char *plugin_attrs_name = "_plugin";
    constexpr const char *plugin_impl = "plugin_impl";
    constexpr const char *kernel_infos = "kernel_infos";
    constexpr const char *node_perf = "node_perf";
    constexpr const char *operator_id = "operator_id";
    constexpr const char *opid_counter = "opid_counter";
    constexpr const char *tsg_name = "tsg_name";

    constexpr const char *ordered_input_id = "_ordered_input_id";
    constexpr const char *zerocopy_cst = ".zerocopy_cst";
    constexpr const char *large_kernel = ".large_kernel";
    constexpr const char *hash = ".hash";
    constexpr const char *private_buffers = ".private_buffers";
    constexpr const char *expr_offset = ".expr_offset";
    constexpr const char *output_tensor_id = ".output_tensor_id";
    constexpr const char *is_dynamic = ".is_dynamic";
    constexpr const char *movwtosm = "_movwtosm";
    constexpr const char *ldwfromsm = "_ldwfromsm";

    constexpr const char *subgraphid = ".subgraphid";
    constexpr const char *compress_info = "_cmpdinfo";
    constexpr const char *compress_frame_info = "_compressframeinfo";
    constexpr const char *skip_unfold = ".skip_unfold";

    // node builtin attrs
    constexpr const char *layout_aware = "_la";
    constexpr const char *elementwise_type = "_ew";
    constexpr const char *virtual_mem = "_vm";
    constexpr const char *view_op = "_vo";

    // profiler instrumentation
    constexpr const char *profiler_code = "profiler_code";
    constexpr const char *profiler_kernel = "profiler_kernel";

    // actication type
    constexpr const char *act_none = "NONE";
    constexpr const char *act_relu = "RELU";
    constexpr const char *act_relu1 = "RELU1";
    constexpr const char *act_relu6 = "RELU6";
    constexpr const char *act_leakyrelu = "LEAKYRELU";
    constexpr const char *act_clip = "CLIP";
    constexpr const char *act_prelu = "PRELU";

    // op_plugin options
    constexpr const char *op_plugin_options = "op_plugin_options";
    constexpr const char *wdc_rate = "wdc_rate";

    constexpr const char *arg_map = "arg_map";
    constexpr const char *print_errornum = "print_errornum";

    // specific params
    constexpr const char *layout = "layout";

    // for scheduling
    constexpr const char *keep_isolate_when_scheduling = "keep_isolate_when_scheduling";

    // for lib config
    constexpr const char *gsize = "gsize";
    constexpr const char *base_oc = "base_oc";

} // namespace Constants
} // namespace aipubt
#endif //! CORE_CONSTANS_H

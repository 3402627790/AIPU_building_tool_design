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
 * @Date: 2022-11-15 09:57:24
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-22 17:34:46
 */
#include <algorithm>
#include <builder/builder.h>
#include <builder/dumper/dumper_x.h>
#include <builder/dumper/dumper_x1.h>
#include <builder/lowering/lowering.h>
#include <builder/mcbuilder.h>
#include <builder/mqbuilder.h>
#include <builder/profiler.h>
#include <builder/target.h>
#include <builder/tools.h>
#include <core/any.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <core/pass_manager.h>
#include <core/plugin_loader.h>
#include <core/stub.h>
#include <dirent.h>
#include <fstream>
#include <gcompiler_api.h>
#include <iostream>
#include <regex>
#include <set>
#include <statistic/allocator_fp_estimation.h>
#include <statistic/perf_estimator.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <transform/passes.h>
#include <unistd.h>
#include <utils/utils.h>

namespace aipubt
{

bool actions_before_build(std::map<std::string, aipubt::Any> &arg_map, int &retcode)
{
    if (arg_map.count("output"))
    {
        aipubt::BuilderConfig::set("output", cast_any<std::string>(arg_map["output"]));
    }
    if (arg_map.count("aipubin"))
    {
        aipubt::BuilderConfig::set("aipubin", cast_any<std::string>(arg_map["aipubin"]));
    }
    if (arg_map.count("target"))
    {
        aipubt::set_target(cast_any<std::string>(arg_map["target"]));
    }
    if (arg_map.count("show_profile_events"))
    {
        std::cout << aipubt::AIPUBuilder::get_profiler()->show_event_description();
        std::cout << "currently support profile cycle count and other three events."
                  << std::endl;
        retcode = 0;
        return true;
    }

    bool enable_tile_cfg = false;
    if (arg_map.count("tiling"))
    {
        std::string tile_cfg = cast_any<std::string>(arg_map["tiling"]);
        std::vector<std::string> valid_tile_cfg = {
            "footprint",
            "fps",
            "fps_dev"};
        if (contains(valid_tile_cfg, tile_cfg) || utils::endswith(tile_cfg, ".json"))
        {
            aipubt::BuilderConfig::set(Constants::tile_cfg, tile_cfg);
            enable_tile_cfg = true;
        }
        else
        {
            WARN() << "invalid tiling parameter. Please check your build config";
        }
    }
    utils::register_atexit(Constants::print_errornum, []() { std::cerr << "Total errors: "
                                                                       << GCompilerLogger::get_error_count()
                                                                       << ",  warnings: "
                                                                       << GCompilerLogger::get_warn_count()
                                                                       << "\n"; });
    if (arg_map.count("DEBUG"))
    {
#if RELEASE_VERSION
        WARN() << " -D doesn't effect in Release Package";
#else
        aipubt::BuilderConfig::set("DEBUG", true);
#endif
    }
    if (arg_map.count("terminator"))
    {
        aipubt::BuilderConfig::set("terminator", cast_any<std::string>(arg_map["terminator"]));
    }
    if (arg_map.count("rand-reg"))
    {
        aipubt::BuilderConfig::set("rand-reg", true);
    }
    if (arg_map.count("reset_dma_per_layer"))
    {
        aipubt::BuilderConfig::set("reset_dma_per_layer", true);
    }
    if (arg_map.count("disable_aipu_cache"))
    {
        aipubt::BuilderConfig::set("disable_aipu_cache", true);
    }
    if (arg_map.count(Constants::disable_icache))
    {
        aipubt::BuilderConfig::set(Constants::disable_icache, true);
    }
    if (arg_map.count("enable_sparse_optimization"))
    {
        aipubt::BuilderConfig::set(Constants::enable_sparse_opt, true);
    }
    if (arg_map.count(Constants::show_location))
    {
        aipubt::BuilderConfig::set(Constants::show_location, true);
    }
    int seed = time(NULL);
    if (arg_map.count("rand-seed"))
    {
        seed = cast_any<int>(arg_map["rand-seed"]);
        INFO() << "Using random seed " << seed << " as for all random number generator.";
    }
    else
    {
        DEBUG() << "Randomize seed:" << seed;
    }
    aipubt::BuilderConfig::set("rand-seed", seed);
    if (arg_map.count("Lib"))
    {
        auto pathes = utils::split_string(cast_any<std::string>(arg_map["Lib"]), ':');
        aipubt::BuilderConfig::set("Lib", pathes);
    }
    uint64_t fifo_size = 0;
    if (arg_map.count("printf_fifo_size"))
    {
        fifo_size = cast_any<uint64_t>(arg_map["printf_fifo_size"]);
        fifo_size = fifo_size << 10;
        aipubt::BuilderConfig::set("printf_fifo_size", fifo_size);
    }
    if (arg_map.count("stack_size"))
    {
        uint32_t stack_size = cast_any<uint64_t>(arg_map["stack_size"]);
        stack_size = stack_size << 10;
        aipubt::BuilderConfig::set("stack_size", stack_size);
    }
    if (arg_map.count("number_for_debug"))
    {
        uint64_t number = cast_any<uint64_t>(arg_map["number_for_debug"]);
        // number = number << 10;
        aipubt::BuilderConfig::set("number_for_debug", number);
    }

    std::vector<std::string> enable_pass;
    if (arg_map.count("enable-pass"))
    {
        enable_pass = utils::split_string(cast_any<std::string>(arg_map["enable-pass"]), ',');
    }
    if (enable_tile_cfg)
    {
        std::vector<std::string> pass_for_mini_fp = {
            "tile_graph",
            "tnode_to_sg",
        };
        for (auto pass : pass_for_mini_fp)
        {
            if (!contains(enable_pass, pass))
            {
                enable_pass.push_back(pass);
            }
        }
        auto key = Constants::dump_tile_cfg;
        if (arg_map.count(key))
        {
            aipubt::BuilderConfig::set(key, true);
        }
    }
    if (arg_map.count(Constants::enable_cse_tile_node))
    {
        aipubt::BuilderConfig::set(Constants::enable_cse_tile_node, true);
    }
    if (enable_pass.size() > 0)
    {
        aipubt::BuilderConfig::set("enable-pass", enable_pass);
    }
    for (auto key :
         {"disable-pass", "disable-plugins",
          "strings_for_debug"})
    {
        if (arg_map.count(key))
        {
            aipubt::BuilderConfig::set(key, utils::split_string(cast_any<std::string>(arg_map[key]), ','));
        }
    }
    if (arg_map.count("serialize-before-pass"))
    {
        auto passes = utils::split_string(cast_any<std::string>(arg_map["serialize-before-pass"]), ',');
        aipubt::get_serialize_before_pass().insert(passes.begin(), passes.end());
    }
    if (arg_map.count("serialize-after-pass"))
    {
        auto passes = utils::split_string(cast_any<std::string>(arg_map["serialize-after-pass"]), ',');
        aipubt::get_serialize_after_pass().insert(passes.begin(), passes.end());
    }
    if (arg_map.count("svg-before-pass"))
    {
        auto passes = utils::split_string(cast_any<std::string>(arg_map["svg-before-pass"]), ',');
        aipubt::get_svg_before_pass().insert(passes.begin(), passes.end());
    }
    if (arg_map.count("svg-after-pass"))
    {
        auto passes = utils::split_string(cast_any<std::string>(arg_map["svg-after-pass"]), ',');
        aipubt::get_svg_after_pass().insert(passes.begin(), passes.end());
    }
    if (arg_map.count("pass-before-tiling"))
    {
        auto passes = utils::split_string(cast_any<std::string>(arg_map["pass-before-tiling"]), ',');
        aipubt::get_pass_before_tiling().insert(passes.begin(), passes.end());
    }

    auto enable_asid_str = getenv(Constants::ENABLE_ASID);
    if (enable_asid_str)
    {
        std::string enable_asid(enable_asid_str);
        utils::tolower(enable_asid);
        BuilderConfig::set<bool>("enable-asid", (enable_asid == "1" || enable_asid == "true"));
    }

    std::string target = cast_any<std::string>(arg_map["target"]);

    pid_t pid = getpid();
    INFO() << "BuildTool version: " << VERSION << ". Build for target " << target << " PID: " << pid;
    if (arg_map.count("rand-reg"))
    {
        DEBUG() << "Enable random initialize registers.";
    }

    if (arg_map.count("profile") && target != "Z1_0701")
    {
        BuilderConfig::set<bool>("profile", true);
        aipubt::AIPUBuilder::get_profiler()->enable_profile() = true;
        aipubt::AIPUBuilder::get_profiler()->unregister_all();
        std::string what_to_profile = "";
        std::string prof_unit = cast_any<std::string>(arg_map[Constants::prof_unit]);
        if (arg_map.count("profile_event"))
        {
            what_to_profile = cast_any<std::string>(arg_map["profile_event"]);
        }

        if (what_to_profile.empty())
        {
            INFO() << "using default profile events to profile " << prof_unit;
            aipubt::AIPUBuilder::get_profiler()->register_default_events(prof_unit);
        }
        else if (aipubt::AIPUBuilder::get_profiler()->register_events(what_to_profile) ==
                 false)
        {
            ERROR() << "Failed to register profile event: " << what_to_profile;
        }
        aipubt::BuilderConfig::set(Constants::prof_unit, prof_unit);
        if (arg_map.count("max_sample_point_num"))
        {
            uint64_t val = cast_any<uint64_t>(arg_map["max_sample_point_num"]);
            if (val > 0)
            {
                aipubt::BuilderConfig::set(Constants::max_perf_sample_num, val);
            }
        }

        if (arg_map.count("max_aiff_sample_point_num"))
        {
            uint64_t val = cast_any<uint64_t>(arg_map["max_aiff_sample_point_num"]);
            if (val > 0)
            {
                aipubt::BuilderConfig::set(Constants::max_aiff_perf_sample_num, val);
            }
        }

        if (arg_map.count("max_dma_sample_point_num"))
        {
            uint64_t val = cast_any<uint64_t>(arg_map["max_dma_sample_point_num"]);
            if (val > 0)
            {
                aipubt::BuilderConfig::set(Constants::max_dma_perf_sample_num, val);
            }
        }

        if (arg_map.count("max_custom_sample_point_num"))
        {
            uint64_t val = cast_any<uint64_t>(arg_map["max_custom_sample_point_num"]);
            if (val > 0)
            {
                aipubt::BuilderConfig::set(Constants::max_custom_perf_sample_num, val);
            }
        }

        if (arg_map.count("enable_dma_perf"))
        {
            BuilderConfig::set<bool>(Constants::enable_dma_perf, true);
        }

        if (arg_map.count("fast_perf"))
        {
            BuilderConfig::set<bool>(Constants::fast_perf, true);
        }

        if (arg_map.count("asid_perf"))
        {
            std::string asid_perf = cast_any<std::string>(arg_map["asid_perf"]);
            BuilderConfig::set<std::string>(Constants::asid_perf, asid_perf);
        }
        if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv6)
        {
            auto asid_perf = BuilderConfig::get<std::string>(Constants::asid_perf);
            auto fn = AIPUBuilder::get_profiler()->interface()->get_function("register_asid_perf");
            if (fn)
            {
                fn(asid_perf);
            }
        }
    }
    else
    {
        if (target == "Z1_0701")
        {
            WARN() << "Target Z1_0701 do not support profiler, disable it.";
        }
        aipubt::AIPUBuilder::get_profiler()->enable_profile() = false;
    }

    std::set<const char *> global_switches = {
        Constants::disable_mmu,
        Constants::disable_input_buffer_reuse,
        Constants::disable_output_buffer_reuse,
        Constants::disable_cross_buffer_reuse,
        Constants::disable_spsram_optimization,
        Constants::alloc_output_on_extsram,
        Constants::sort_after_each_pass,
        Constants::dump_all_tensors,
        Constants::dump_rt_input_tensors,
        Constants::dump_rt_output_tensors,
        Constants::individual_inputs,
        Constants::warn_all,
        Constants::disable_schedule_optimization,
        Constants::switch_for_debug,
        Constants::show_raw_graph,
        Constants::show_short_name,
        Constants::ddr_data_gather,
        Constants::enable_custom_smmu,
        Constants::disable_gm,
        Constants::enable_gm_enhance,
        Constants::enable_dynamic_gm,
        Constants::enable_x2_sm_opt,
        Constants::dump_gm_enhance_cfg,
        Constants::dump_prior_in_gm,
        Constants::dump_schedule_cfg,
        Constants::single_core_opt,
        Constants::insert_init_reg,
        Constants::enable_weight_align,
        Constants::enable_slc_ocm_mode,
        Constants::enable_debug_info,
        Constants::skip_simulator,
        Constants::native_float_simulator,
        Constants::dump_addr_cfg,
        Constants::disable_separate_gm,
        Constants::enable_separate_gm,
        Constants::datalayout_schedule_debug,
        Constants::put_desc_gm,
        Constants::put_ws_gm,
        Constants::put_weight_gm,
        Constants::random_init,
        Constants::use_amat_scheduler,
        Constants::softmax_prior_in_gm,
        Constants::disable_dependency,
        Constants::set_l2_flush,
        Constants::prefetch_desc,
        Constants::dump_tune_strategy,
        Constants::dump_reorder_cfg,
        Constants::ws_mem_opt,
        Constants::closer_sg_opt,
        Constants::reuse_buffer_expand,
        Constants::tile_split_output_c_conv,
        Constants::remove_origin_constant,
        Constants::show_soms_bottleneck,
        Constants::split_wt_act,
        Constants::print_emitter_var,
        Constants::dump_constant_info,
        Constants::dump_lora_info,
    };
    for (auto key : global_switches)
    {
        if (arg_map.count(key))
        {
            aipubt::BuilderConfig::set(key, cast_any<bool>(arg_map[key]));
        }
    }

    std::set<std::string> info_types;
    if (arg_map.count("show_buffer_index"))
    {
        WARN() << "Option --show_buffer_index is deprecated,"
               << " and will be removed in a future version, use \"--show_info mem\" instead.";
        info_types.insert("mem");
    }
    if (arg_map.count("show_tensor_size"))
    {
        WARN() << "Option --show_tensor_size is deprecated,"
               << " and will be removed in a future version, use \"--show_info tsize\" instead.";
        info_types.insert("tsize");
    }
    if (arg_map.count("show_info"))
    {
        std::set<std::string> valid_types = {"mem", "perf", "tsize", "all"};
        auto str = cast_any<std::string>(arg_map["show_info"]);
        for (auto &item : utils::split_string(str, ","))
        {
            if (valid_types.count(item))
            {
                if (item == "all")
                {
                    for (auto &type : valid_types)
                    {
                        if (type != "all")
                        {
                            info_types.insert(type);
                        }
                    }
                }
                else
                {
                    info_types.insert(item);
                }
            }
            else
            {
                auto types = std::vector<std::string>(valid_types.begin(), valid_types.end());
                WARN() << "Invalid type \"" << item << "\" given by option --show_info,"
                       << " only \"" << utils::join_string("\", \"", types) << "\" are supported.";
            }
        }
    }
    if (info_types.size())
    {
        aipubt::BuilderConfig::set(Constants::show_info, info_types);
    }
    if (arg_map.count("load_wt_update_cfg"))
    {
        aipubt::BuilderConfig::set(Constants::load_wt_update_cfg, cast_any<std::string>(arg_map["load_wt_update_cfg"]));
    }
    if (arg_map.count("load_addr_cfg"))
    {
        aipubt::BuilderConfig::set(Constants::load_addr_cfg, cast_any<std::string>(arg_map["load_addr_cfg"]));
    }
    if (arg_map.count("ddr_data_mode"))
    {
        aipubt::BuilderConfig::set(Constants::ddr_data_mode, cast_any<uint32_t>(arg_map["ddr_data_mode"]));
    }
    if (arg_map.count("multi_batch"))
    {
        aipubt::BuilderConfig::set(Constants::multi_batch, cast_any<uint32_t>(arg_map["multi_batch"]));
    }
    if (arg_map.count("perf_json"))
    {
        aipubt::BuilderConfig::set(Constants::perf_json, cast_any<std::string>(arg_map["perf_json"]));
    }
    if (arg_map.count("perf_bin"))
    {
        aipubt::BuilderConfig::set(Constants::perf_bin, cast_any<std::string>(arg_map["perf_bin"]));
    }
    if (arg_map.count("tcm_size"))
    {
        aipubt::BuilderConfig::set(aipubt::Constants::tcm_size, (uint32_t)cast_any<uint64_t>(arg_map["tcm_size"]) * 1024);
        if (target == "X1_1204")
        {
            BuilderConfig::set(Constants::enable_tcm, true);
        }
    }

    // reset dynamic options
    aipubt::BuilderConfig::set(Constants::enable_ds, false);
    aipubt::BuilderConfig::erase(Constants::input_shape);
    aipubt::BuilderConfig::erase(Constants::max_input_shape);
    aipubt::BuilderConfig::erase(Constants::default_input_shape);
    aipubt::BuilderConfig::erase(Constants::dynamic_input_dim);

    if (arg_map.count("enable_ds"))
    {
        bool status = cast_any<bool>(arg_map["enable_ds"]);
        if (aipubt::get_target()->isainfo().type < aipubt::ISA_ZYv5)
        {
            status = false;
        }
        aipubt::BuilderConfig::set(Constants::enable_ds, status);
    }
    for (auto &opt : {"input_shape", "max_input_shape", "default_input_shape"})
    {
        if (!arg_map.count(opt))
        {
            continue;
        }
        if (aipubt::BuilderConfig::get<bool>(Constants::enable_ds))
        {
            auto str = cast_any<std::string>(arg_map[opt]);
            for (auto &s : {"\'", "\""})
            {
                str.erase(0, str.find_first_not_of(s));
                str.erase(str.find_last_not_of(s) + 1);
            }
            aipubt::BuilderConfig::set(opt, str);
        }
        else
        {
            WARN() << "Option --" << opt << " is ignored because option --enable_ds is not enabled.";
        }
    }
    if (arg_map.count("dynamic_input_dim"))
    {
        if (aipubt::BuilderConfig::get<bool>(Constants::enable_ds))
        {
            std::set<std::string> dims;
            auto str = cast_any<std::string>(arg_map["dynamic_input_dim"]);
            for (auto &s : {"\'", "\""})
            {
                str.erase(0, str.find_first_not_of(s));
                str.erase(str.find_last_not_of(s) + 1);
            }
            if (str != "none")
            {
                for (auto d : utils::split_string(str, ","))
                {
                    auto s = utils::trim(d);
                    if (s.size())
                    {
                        dims.insert(s);
                    }
                }
            }
            if (str == "none" || dims.size())
            {
                aipubt::BuilderConfig::set(Constants::dynamic_input_dim, dims);
            }
        }
        else
        {
            WARN() << "Option --dynamic_input_dim is ignored because option --enable_ds is not enabled.";
        }
    }

    if (arg_map.count("gm_size"))
    {
        aipubt::BuilderConfig::set(aipubt::Constants::gm_size, (uint32_t)cast_any<uint64_t>(arg_map["gm_size"]) * 1024);
    }
    if (arg_map.count("warmup_len"))
    {
        aipubt::BuilderConfig::set(aipubt::Constants::warmup_len, (uint32_t)cast_any<uint64_t>(arg_map["warmup_len"]));
    }
    if (arg_map.count("preserve_gm_size"))
    {
        aipubt::BuilderConfig::set(aipubt::Constants::preserve_gm_size, (uint32_t)cast_any<uint64_t>(arg_map["preserve_gm_size"]) * 1024);
    }
    if (arg_map.count("ocm_size"))
    {
        aipubt::BuilderConfig::set(aipubt::Constants::ocm_size, (uint32_t)cast_any<uint64_t>(arg_map["ocm_size"]) * 1024);
    }
    if (arg_map.count("ocm_addr"))
    {
        aipubt::BuilderConfig::set(aipubt::Constants::ocm_addr, (uint64_t)cast_any<uint64_t>(arg_map["ocm_addr"]));
    }
    if (arg_map.count("set_asid"))
    {
        aipubt::BuilderConfig::set(Constants::set_asid, cast_any<std::string>(arg_map["set_asid"]));
    }
    if (arg_map.count("load_gm_enhance_cfg"))
    {
        aipubt::BuilderConfig::set(Constants::load_gm_enhance_cfg, cast_any<std::string>(arg_map["load_gm_enhance_cfg"]));
    }
    if (arg_map.count(Constants::load_prior_in_gm))
    {
        aipubt::BuilderConfig::set(Constants::load_prior_in_gm, cast_any<std::string>(arg_map["load_prior_in_gm"]));
    }
    if (arg_map.count("gm_enhance_strategy"))
    {
        aipubt::BuilderConfig::set(Constants::gm_enhance_strategy, cast_any<uint32_t>(arg_map["gm_enhance_strategy"]));
    }
    if (arg_map.count("load_schedule_cfg"))
    {
        aipubt::BuilderConfig::set(Constants::load_schedule_cfg, cast_any<std::string>(arg_map["load_schedule_cfg"]));
    }
    if (arg_map.count("load_balance_strategy"))
    {
        auto value = cast_any<float>(arg_map["load_balance_strategy"]);
        aipubt::BuilderConfig::set(Constants::load_balance_strategy, (int32_t)value);
    }
    else
    {
        if (arg_map.count("tiling"))
        {
            BuilderConfig::set(Constants::load_balance_strategy, 0);
        }
        else
        {
            BuilderConfig::set(Constants::load_balance_strategy, 1);
        }
    }
    if (arg_map.count("branch_reorder_strategy"))
    {
        auto value = cast_any<float>(arg_map["branch_reorder_strategy"]);
        aipubt::BuilderConfig::set(Constants::branch_reorder_strategy, (int32_t)value);
    }
    if (arg_map.count("max_wt_size"))
    {
        aipubt::BuilderConfig::set(Constants::max_wt_size, (uint64_t)cast_any<uint32_t>(arg_map["max_wt_size"]) * 1024 * 1024);
    }
    if (arg_map.count("max_act_size"))
    {
        aipubt::BuilderConfig::set(Constants::max_act_size, (uint64_t)cast_any<uint32_t>(arg_map["max_act_size"]) * 1024 * 1024);
    }
    if (arg_map.count("load_tune_strategy"))
    {
        aipubt::BuilderConfig::set(Constants::load_tune_strategy, cast_any<std::string>(arg_map["load_tune_strategy"]));
    }
    if (arg_map.count("load_reorder_cfg"))
    {
        aipubt::BuilderConfig::set(Constants::load_reorder_cfg, cast_any<std::string>(arg_map["load_reorder_cfg"]));
    }
    if (arg_map.count("mq_save_ir"))
    {
        aipubt::BuilderConfig::set(Constants::mq_save_ir, cast_any<std::string>(arg_map["mq_save_ir"]));
    }
    if (arg_map.count("mq_load_ir"))
    {
        aipubt::BuilderConfig::set(Constants::mq_load_ir, cast_any<std::string>(arg_map["mq_load_ir"]));
    }
    if (arg_map.count("mq_ir_stage"))
    {
        aipubt::BuilderConfig::set(Constants::mq_ir_stage, cast_any<std::string>(arg_map["mq_ir_stage"]));
    }
    if (arg_map.count("soms_gm_loop"))
    {
        aipubt::BuilderConfig::set(aipubt::Constants::soms_gm_loop, (uint32_t)cast_any<uint32_t>(arg_map["soms_gm_loop"]));
    }
    if (arg_map.count("specify_wt_gm"))
    {
        aipubt::BuilderConfig::set(Constants::specify_wt_gm, cast_any<std::string>(arg_map["specify_wt_gm"]));
    }
    if (arg_map.count("dump_rt_tensor_id"))
    {
        aipubt::BuilderConfig::set(Constants::dump_rt_tensor_id, cast_any<int32_t>(arg_map["dump_rt_tensor_id"]));
    }
    if (arg_map.count("set_profile_option"))
    {
        aipubt::BuilderConfig::set(Constants::set_profile_option, cast_any<std::string>(arg_map["set_profile_option"]));
    }
    if (arg_map.count("set_mq_num"))
    {
        aipubt::BuilderConfig::set(Constants::set_mq_num, cast_any<int32_t>(arg_map["set_mq_num"]));
    }
    // Enable TCM
    if (target == "X1_1204")
    {
        // add 4M tcm for x1 soc
        auto ptr = getenv("ENABLE_TCM");
        std::string enable_tcm;
        if (ptr)
        {
            enable_tcm = ptr;
        }
        bool valid_enable_tcm = !enable_tcm.empty() && (enable_tcm == "true" || enable_tcm == "TRUE" || enable_tcm == "1");
        if (valid_enable_tcm || enable_tile_cfg)
        {
            if (!aipubt::BuilderConfig::exist(aipubt::Constants::tcm_size))
                aipubt::BuilderConfig::set(aipubt::Constants::tcm_size, 4 * 1024 * 1024);
            BuilderConfig::set(Constants::enable_tcm, true);
        }
    }
    bool use_slots = utils::get_env("USE_SLOTS").size() && utils::get_env("USE_SLOTS") != "0";
    if (use_slots)
    {
        BuilderConfig::set(Constants::USE_SLOTS, true);
    }

    // for lib
    aipubt::ISAType arch = aipubt::get_target()->isainfo().type;
    //// common
    if (arg_map.count(Constants::op_plugin_options))
    {
        std::map<std::string, std::string> each_options = {};
        auto total_options = utils::split_string(cast_any<std::string>(arg_map[Constants::op_plugin_options]), ',');
        for (auto &to : total_options)
        {
            auto _tmp = utils::split_string(to, '=');
            if (_tmp.size() != 2)
            {
                ERROR() << "Invalid option: --" + std::string(Constants::op_plugin_options)
                        << ", Please see more info using --help-hidden";
                exit(-5);
            }
            each_options[_tmp[0]] = _tmp[1];
        }
        aipubt::BuilderConfig::set(Constants::op_plugin_options, each_options);
    }
    //// wdc control
    if (arg_map.count(Constants::wdc_rate))
    {
        Any wdc_rate = arg_map[Constants::wdc_rate];
        if (wdc_rate.get_type_id() == GET_ANY_TYPEID<int>() || wdc_rate.get_type_id() == GET_ANY_TYPEID<long>())
        {
            int tmp_value = (int)cast_any<int>(wdc_rate);
            aipubt::BuilderConfig::set(Constants::wdc_rate, (float)tmp_value);
        }
        else
        {
            aipubt::BuilderConfig::set(Constants::wdc_rate, (float)cast_any<float>(wdc_rate));
        }
    }
    else
    {
        aipubt::BuilderConfig::set(Constants::wdc_rate, (float)0.7);
        if ((target == "X1_1204") || (arch >= aipubt::ISA_ZYv5))
        {
            aipubt::BuilderConfig::set(Constants::wdc_rate, (float)0.9);
        }
    }
    // global_cwd
    std::string global_cwd = utils::get_temp_dir() + "/" + utils::get_uuid();
    if (arg_map.count("dump"))
    {
        global_cwd = "./";
    }
    else
    {
        utils::mkdir(global_cwd);
    }
    global_cwd = utils::get_abspath(global_cwd);
    INFO() << "global cwd: " << global_cwd;
    BuilderConfig::set(Constants::global_cwd, global_cwd);

    auto &version = Versioning::get_version();
    // collect all args
    std::string arg_str = "AIPUBuilder: version " + version.version + " (" + version.commit + ") ";
    for (auto it = arg_map.begin(); it != arg_map.end(); ++it)
    {
        static std::set<std::string> not_record = {"input", "output", "graphdef", "weight", "png", "svg"};
        if (not_record.count(it->first))
        {
            continue;
        }
        // key
        arg_str = arg_str + it->first;
        // value
        if (it->second.get_type_id() == GET_ANY_TYPEID<int>())
        {
            int tmp_value = (int)cast_any<int>(it->second);
            arg_str = arg_str + ":" + std::to_string(tmp_value) + ";";
        }
        else if (it->second.get_type_id() == GET_ANY_TYPEID<unsigned int>())
        {
            unsigned int tmp_value = (unsigned int)cast_any<unsigned int>(it->second);
            arg_str = arg_str + ":" + std::to_string(tmp_value) + ";";
        }
        else if (it->second.get_type_id() == GET_ANY_TYPEID<long unsigned int>())
        {
            long unsigned int tmp_value = (long unsigned int)cast_any<long unsigned int>(it->second);
            arg_str = arg_str + ":" + std::to_string(tmp_value) + ";";
        }
        else if (it->second.get_type_id() == GET_ANY_TYPEID<long int>())
        {
            long int tmp_value = (long int)cast_any<long int>(it->second);
            arg_str = arg_str + ":" + std::to_string(tmp_value) + ";";
        }
        else if (it->second.get_type_id() == GET_ANY_TYPEID<float>())
        {
            float tmp_value = (float)cast_any<float>(it->second);
            arg_str = arg_str + ":" + std::to_string(tmp_value) + ";";
        }
        else if (it->second.get_type_id() == GET_ANY_TYPEID<std::string>())
        {
            std::string tmp_value = (std::string)cast_any<std::string>(it->second);
            arg_str = arg_str + ":" + tmp_value + ";";
        }
        else if (it->second.get_type_id() == GET_ANY_TYPEID<bool>())
        {
            bool tmp_value = (bool)cast_any<bool>(it->second);
            std::string tmp_str = tmp_value ? "1" : "0";
            arg_str = arg_str + ":" + tmp_str + ";";
        }
        else
        {
            DEBUG() << "arg: " << it->first << " type: " << it->second.get_type_name() << " doesn't support!";
        }
    }
    aipubt::BuilderConfig::set(Constants::arg_map, (std::string)cast_any<std::string>(arg_str));

    return false;
}

void insert_node_shape_params(GraphPtr graph,
                              const std::map<std::string, std::vector<int>> &shapes)
{
    graph->walk_on_node([&shapes](NodePtr n) {
        if (n->type() == OpType_SubGraphInput ||
            SubGraphNode::is_subgraph_node(n))
        {
            return;
        }
        std::vector<std::vector<int>> in_shapes, out_shapes;
        for (auto inp : n->inputs())
        {
            auto name = inp->name();
            if (shapes.count(name))
            {
                in_shapes.push_back(shapes.at(name));
            }
        }
        for (auto out : n->outputs())
        {
            auto name = out->name();
            if (shapes.count(name))
            {
                out_shapes.push_back(shapes.at(name));
            }
        }
        if (in_shapes.size())
        {
            n->params()["real_bottom_shape"] = in_shapes;
        }
        std::vector<std::vector<Expr>> exprs;
        for (auto inp : n->inputs())
        {
            std::vector<Expr> s;
            for (int d = 0; d < inp->shape().dim(); ++d)
            {
                s.push_back(inp->shape().symbol()[d]);
            }
            exprs.push_back(s);
        }
        if (exprs.size())
        {
            n->params()["ds_bottom_shape"] = exprs;
        }
        std::vector<TensorPtr> tensors;
        if (n->type() == OpType_Input)
        {
            tensors = n->outputs();
        }
        else
        {
            tensors = n->inputs();
        }
        std::vector<std::string> dims;
        uint32_t index = 0;
        for (auto t : tensors)
        {
            for (int d = 0; d < t->shape().dim(); ++d)
            {
                if (!t->shape().symbol()[d]->is_constant())
                {
                    dims.push_back("d" + std::to_string(index));
                }
                index++;
            }
        }
        if (dims.size())
        {
            n->params()["dynamic_input_dim"] = dims;
        }
        if (out_shapes.size())
        {
            n->params()["real_top_shape"] = out_shapes;
        }
        exprs.clear();
        for (auto out : n->outputs())
        {
            std::vector<Expr> s;
            for (int d = 0; d < out->shape().dim(); ++d)
            {
                s.push_back(out->shape().symbol()[d]);
            }
            exprs.push_back(s);
        }
        if (exprs.size())
        {
            n->params()["ds_top_shape"] = exprs;
        } });
}

static void serialize_graph(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map)
{
    BuilderConfig::set("dump_debug_IR", true);
    BuilderConfig::set(Constants::show_location, true);
    bool serialize_scale_zp = g->get_attr("serialize_scale_zp");
    g->set_attr("serialize_scale_zp", true);
    g->serialize(cast_any<std::string>(arg_map["dumpIR"]) + ".def",
                 cast_any<std::string>(arg_map["dumpIR"]) + ".bin");
    g->set_attr("serialize_scale_zp", serialize_scale_zp);
}

bool infer_real_shape(GraphPtr graph,
                      const std::vector<int> &dump_shape_dims, // data-aware
                      const std::vector<std::vector<int>> &input_shapes)
{
    bool ret = true;
    auto &symbols = graph->symbols();
    std::vector<int> input_dims;
    for (auto &s : input_shapes)
    {
        if (s.empty())
        {
            input_dims.push_back(1);
            continue;
        }
        input_dims.insert(input_dims.end(), s.begin(), s.end());
    }
    std::map<std::string, std::vector<int>> shapes;

    auto eval_expr = [&](Expr expr) -> int {
        std::map<Symbol, Symbol::Value> args;
        for (auto &&var : expr->variables())
        {
            if (symbols.count(var.name()))
            {
                auto &sym = symbols.at(var.name());
                if (sym.dims.size())
                {
                    auto dim = sym.dims[0];
                    if (sym.type == 0 && dim < (int)input_dims.size())
                    {
                        args[var] = input_dims[dim];
                    }
                    if (sym.type == 1 && dim < (int)dump_shape_dims.size())
                    {
                        args[var] = dump_shape_dims[dim];
                    }
                }
            }
        }
        if (!expr->eval(args))
        {
            ret = false;
        }
        return static_cast<int>(expr->value());
    };

    graph->walk_on_node([&eval_expr, &shapes](NodePtr n) {
        if (n->type() == OpType_SubGraphInput ||
            SubGraphNode::is_subgraph_node(n))
        {
            return;
        }
        for (auto out : n->outputs())
        {
            std::vector<int> shape;
            for (int d = 0; d < out->shape().dim(); ++d)
            {
                shape.push_back(eval_expr(out->shape().symbol()[d]));
            }
            shapes[out->name()] = shape;
        }
        if (n->type() == OpType_Pad || n->type() == OpType_Crop ||
            n->type() == OpType_Reshape || n->type() == OpType_Tile)
        {
            std::string param_name;
            if (n->type() == OpType_Reshape || n->type() == OpType_Tile)
            {
                param_name = n->type() == OpType_Reshape ? "ds_shape" : "ds_repeats";
            }
            else
            {
                param_name = n->type() == OpType_Pad ? "ds_pads" : "ds_crops";
            }
            if (!n->params().count(param_name))
            {
                return;
            }
            std::vector<std::vector<Expr>> exprs;
            if (n->type() == OpType_Reshape || n->type() == OpType_Tile)
            {
                std::vector<Expr> s = n->params()[param_name];
                exprs.push_back(s);
            }
            else
            {
                exprs = n->params()[param_name];
            }
            std::vector<std::vector<int>> values(exprs.size());
            for (uint32_t i = 0; i < exprs.size(); ++i)
            {
                values[i].resize(exprs[i].size());
                for (uint32_t j = 0; j < exprs[i].size(); ++j)
                {
                    values[i][j] = eval_expr(exprs[i][j]);
                }
            }
            auto new_param_name = "real" + param_name.substr(2);
            if (n->type() == OpType_Reshape || n->type() == OpType_Tile)
            {
                n->params()[new_param_name] = values[0];
            }
            else
            {
                n->params()[new_param_name] = values;
            }
        } });

    insert_node_shape_params(graph, shapes);

    return ret;
}

GraphPtr infershape(GraphPtr g,
                    const std::set<std::string> &dynamic_dims,
                    const std::vector<std::vector<int>> &input_shapes,
                    const std::vector<int> &dump_dims,
                    bool clone_graph)
{
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    auto has_cfg_dims = BuilderConfig::exist(Constants::dynamic_input_dim);
    std::set<std::string> cfg_dims;
    if (has_cfg_dims)
    {
        cfg_dims = BuilderConfig::get<std::set<std::string>>(Constants::dynamic_input_dim);
    }
    auto dims = dynamic_dims;
    if (dynamic_dims.empty())
    {
        uint32_t index = 0;
        for (auto t : g->get_input_tensors())
        {
            auto n = t->node();
            auto shape = get_base_shape(t->shape());
            if (n->params().count("ds_output_shape"))
            {
                std::vector<std::vector<Expr>> ds_output_shape = n->params()["ds_output_shape"];
                if (ds_output_shape.size() == 1 && (int)ds_output_shape[0].size() == shape.dim())
                {
                    for (auto expr : ds_output_shape[0])
                    {
                        if (expr->op() == ExprImpl::Op::Constant)
                        {
                            dims.insert(expr->symbol().name());
                        }
                    }
                }
                else
                {
                    FATAL() << "Unknown ds_output_shape param from Input: " << n->name();
                }
                index += shape.dim();
            }
            else
            {
                for (int i = 0; i < shape.dim(); ++i)
                {
                    dims.insert("d" + std::to_string(index++));
                }
            }
        }
    }
    BuilderConfig::set(Constants::enable_ds, true);
    BuilderConfig::set(Constants::dynamic_input_dim, dims);

    GraphPtr new_g = g;
    if (clone_graph)
    {
        new_g = g->clone();
    }
    init_symbol_shape(new_g);
    infer_symbol_shape(new_g);

    if (input_shapes.size() || dump_dims.size())
    {
        if (!infer_real_shape(new_g, dump_dims, input_shapes))
        {
            ERROR() << "Infer real shapes of raw graph failed!";
        }
    }

    BuilderConfig::set(Constants::enable_ds, enable_ds);
    if (has_cfg_dims)
    {
        BuilderConfig::set(Constants::dynamic_input_dim, cfg_dims);
    }
    else
    {
        BuilderConfig::erase(Constants::dynamic_input_dim);
    }
    return new_g;
}

void actions_after_build(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map)
{
    uint64_t t0, t1;
    t0 = utils::get_time_in_us();
    passes_after_build(g);
    // estimation footprint optimization
    AllocatorFPEst afps;
    afps(g);
    auto get_size_string = [](uint64_t size, uint32_t precision = 2) -> std::string {
        return utils::get_human_readable_size(size, utils::DataUnit::DU_Undefined, precision);
    };

    if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
    {
        INFO() << "With GM optimization, DDR Footprint stastic(estimation):";
    }
    else
    {
        INFO() << "With SRAM optimization, DDR Footprint stastic(estimation):";
    }

    INFO() << "    Read and Write:" << get_size_string(afps.get_result());

    std::string temp_dot = "temp_g.dot";
    if (arg_map.count("dot") && g)
    {
        g->to_dot(cast_any<std::string>(arg_map["dot"]));
        temp_dot = cast_any<std::string>(arg_map["dot"]);
    }
    else if (arg_map.count("png") && g)
    {
        g->to_dot(temp_dot);
    }

    if (arg_map.count("png") && g)
    {
        INFO() << "drawing png ...";
        g->to_png(cast_any<std::string>(arg_map["png"]));
    }
    if (arg_map.count("svg") && g)
    {
        INFO() << "drawing svg ...";
        g->to_svg(cast_any<std::string>(arg_map["svg"]));
    }

    // serialize after dump if input shape specified
    if ((!BuilderConfig::exist(Constants::input_shape) || arg_map.count("skip_simulator")) &&
        arg_map.count("dumpIR") && g)
    {
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            insert_node_shape_params(g);
        }
        serialize_graph(g, arg_map);
    }

    if (AIPUBuilder::get_profiler()->enable_profile() && aipubt::get_target()->isainfo().type < aipubt::ISA_ZYv5)
    {
        if (!AIPUBuilder::get_profiler()->generate_graph_json(g, "temp.graph.json"))
        {
            WARN() << "failed to generate graph.json for profiler";
        }
    }

    if (BuilderConfig::get<bool>(Constants::dump_lora_info) &&
        g->attrs().count(Constants::lora_info))
    {
        std::string lora_info = g->get_attr(Constants::lora_info);
        std::ofstream ofs("temp.lora_info.json");
        if (ofs.is_open())
        {
            ofs << lora_info << std::endl;
            ofs.close();
        }
    }

    if ((!arg_map.count("dot")) && g)
    {
        utils::remove_file(temp_dot);
    }
    t1 = utils::get_time_in_us();
    INFO() << " -  draw graph time: " << (t1 - t0) / 1000 / 1000. << " s";
}
} // namespace aipubt

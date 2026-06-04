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
#include <launcher.h>
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
static std::string get_simulator_path(const std::string &simulator)
{
    std::string sim_path = simulator;
    auto cfg_dir = utils::get_home_dir() + "/.aipubuilder/";
    std::string cmd = "mkdir -p " + cfg_dir;
    system(cmd.c_str());
    auto cfg = cfg_dir + "config";
    std::string target_name = "ZYV1";
    if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZY)
    {
        target_name = "ZYV1";
    }
    else if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZYv2)
    {
        target_name = "ZYV2";
    }
    else if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZYv3)
    {
        target_name = "ZYV3";
        if (aipubt::get_target()->name() == "X1_1204")
        {
            target_name = "ZYX1";
        }
    }
    else if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZYv5)
    {
        target_name = "ZYX2";
        if (aipubt::get_target()->name() == "Z5_0901")
        {
            target_name = "ZYV5";
        }
    }
    else if (aipubt::get_target()->isainfo().type == aipubt::ISA_ZYv6)
    {
        target_name = "ZYX3";
        if (aipubt::get_target()->chipinfo().revision == 1)
        {
            target_name = "ZYX3P";
        }
        else if (aipubt::get_target()->chipinfo().revision == 2)
        {
            target_name = "ZYX3S";
        }
    }
    else
    {
        // TODO
    }
    if (!utils::file_exists(cfg))
    {
        std::ofstream ofile(cfg);
        utils::JsonObject json_sim;
        json_sim["simulator"] = utils::JsonObject(utils::JsonObject::OBJ);
        ofile << json_sim.to_string();
        ofile.close();
    }
    std::stringstream buffer;
    std::ifstream ifile(cfg);
    buffer << ifile.rdbuf();
    utils::JsonObject config = utils::JsonObject::parse(buffer.str());
    ifile.close();
    auto &json_sim = config["simulator"];
    if (simulator.empty())
    {
        if (ifile)
        {
            if (!json_sim.exist(target_name))
            {
                if (target_name != "ZYX2")
                {
                    ERROR() << "No target " << target_name << " simulator path found";
                }
            }
            else
            {
                std::string path = json_sim[target_name];
                sim_path = path;
            }
        }
    }
    else
    {
        if (json_sim.exist(target_name))
        {
            std::string sim = json_sim[target_name];
            if (sim == sim_path)
            {
                return sim_path;
            }
        }
        json_sim[target_name] = sim_path;
        std::ofstream ofile(cfg);
        if (ofile)
        {
            ofile << config.to_string();
            ofile.close();
        }
    }
    return sim_path;
}

static bool run_simulator(const std::string &cmd)
{
    INFO() << "run simulator:\n"
           << cmd;
    bool ret = system(cmd.c_str()) == 0;
    INFO() << "Simulator finished.";
    return ret;
}

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
    if (arg_map.count("enable_weight_align"))
    {
        aipubt::BuilderConfig::set(Constants::enable_weight_align, cast_any<uint32_t>(arg_map["enable_weight_align"]));
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
    aipubt::BuilderConfig::erase(Constants::show_divided_graph);

    if (arg_map.count("enable_ds"))
    {
        bool status = aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5;
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
    if (arg_map.count("show_divided_graph"))
    {
        if (aipubt::BuilderConfig::get<bool>(Constants::enable_ds))
        {
            aipubt::BuilderConfig::set(Constants::show_divided_graph, true);
        }
        else
        {
            WARN() << "Option --show_divided_graph is ignored because option --enable_ds is not enabled.";
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

static bool handle_op_counter(std::string name)
{
    std::ifstream op_cnt_file(name, std::ios::binary | std::ios::in);
    if (!op_cnt_file)
    {
        WARN() << "Cannot read op_counter.bin";
    }
    else
    {
        op_cnt_file.seekg(0, std::ios::end);
        size_t file_size = op_cnt_file.tellg();
        op_cnt_file.seekg(0);
        if (file_size < 4)
        {
            WARN() << "invalid size of op_counter.bin:" << file_size << " (expected 4)";
        }
        else
        {
            uint32_t counter;
            char *counter_ptr = (char *)(&counter);
            op_cnt_file.read(counter_ptr, 4);
            DEBUG() << "op_counter (Operator ID of last finished layer): " << counter;
        }
    }
    op_cnt_file.close();
    return true;
}

static bool handle_error_code(std::string name)
{
    static std::map<uint16_t, std::string> error_code_map = {
        {0x200, "RUNTIME_ERROR"},

        {0x300, "KEYBOARD_INTERRUPT"},
        {0x301, "SYSTEM_ERROR"},

        {0x380, "OUT_OF_SPEC"},
        {0x381, "OUT_OF_AIFF_SPEC"},
        {0x382, "OUT_OF_TPC_SPEC"},
        {0x383, "OUT_OF_DMA_SPEC"},

        {0x400, "OUT_OF_MEM_RANGE"},
        {0x401, "OUT_OF_SRAM_RANGE"},
        {0x402, "OUT_OF_LSRAM0_RANGE"},
        {0x403, "OUT_OF_LSRAM1_RANGE"},
        {0x404, "OUT_OF_LSRAM_RANGE"},
        {0x405, "OUT_OF_GSRAM0_RANGE"},
        {0x406, "OUT_OF_GSRAM1_RANGE"},
        {0x407, "OUT_OF_GSRAM_RANGE"},

        {0x480, "ARITHMETIC_ERROR"},
        {0x481, "FLOAT_POINT_ERROR"},
        {0x482, "UNDERFLOW_ERROR"},
        {0x483, "OVERFLOW_ERROR"},
        {0x484, "NOT_A_NUMBER_ERROR"},
        {0x485, "INFINITY_ERROR"},
        {0x486, "STR_LENGTH_ERROR"},
        {0x487, "ZERO_DIVISION_ERROR"},
    };
    std::ifstream error_code_file(name, std::ios::binary | std::ios::in);
    if (!error_code_file)
    {
        DEBUG() << "Cannot read error_code.bin";
    }
    else
    {
        error_code_file.seekg(0, std::ios::end);
        size_t file_size = error_code_file.tellg();
        error_code_file.seekg(0);
        if (file_size < 4)
        {
            WARN() << "invalid size of error_code.bin:" << file_size << " (expected 4)";
        }
        else
        {
            uint32_t error_code;
            char *error_code_ptr = (char *)(&error_code);
            error_code_file.read(error_code_ptr, 4);
            // error_code = (OpType_Resize << 16) + 0x123;
            if (error_code == 0)
            {
                DEBUG() << "error_code: " << error_code;
            }
            else if (error_code >> 31)
            {
                WARN() << "AIPU teminated with error_code: " << std::hex << error_code << std::dec;
            }
            else
            {
                OpType type = static_cast<OpType>((error_code & ~(1 << 31)) >> 16);
                uint16_t code = static_cast<uint16_t>(error_code & 0xFFFF);
                std::string msg = error_code_map.count(code) > 0 ? error_code_map[code] : "RUNTIME_ERROR";
                WARN() << "AIPU teminated with error_code: 0x" << std::hex << error_code << std::dec
                       << " (OpType_" << optype_to_string(type) << ", " << msg << ").";
            }
        }
    }
    error_code_file.close();
    return true;
}

static bool handle_multi_batch(std::vector<TensorPtr> &output_tensors, uint32_t batch, std::string output_file)
{
    for (uint32_t b = 0; b < batch; ++b)
    {
        std::string name = "";
        if (batch > 1)
        {
            name = "b" + std::to_string(b) + ".";
        }
        for (uint32_t i = 0; i < output_tensors.size(); ++i)
        {
            auto ofile = output_file;
            if (i != 0)
            {
                ofile += std::to_string(i);
            }
            auto out = output_tensors[i];
            out->set_data(new char[out->mem_size()], out->mem_size());
            std::ifstream ifs;
            ifs.open(name + ofile, std::ios::in | std::ios::binary);
            ifs.read(out->data<char>(), out->mem_size());
            ifs.close();
        }
    }
    return true;
}

static void insert_node_shape_params(GraphPtr graph,
                                     const std::map<std::string, std::vector<int>> &shapes = {})
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

static bool infer_real_shape(GraphPtr graph,
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

static bool is_empty_graph(GraphPtr g)
{
    size_t num_real_nodes = 0;
    g->walk_on_node([&num_real_nodes](NodePtr n) {
        if (SubGraphNode::is_subgraph_node(n) &&
            !n->attrs().count(Constants::plugin_attrs_name))
        {
            return;
        }
        if (n->attrs().count(Constants::ro_size) &&
            n->get_attr<uint32_t>(Constants::ro_size))
        {
            num_real_nodes++;
        } });
    return num_real_nodes == 0;
}

static bool eval_output_shape(GraphPtr g,
                              const std::vector<std::vector<int>> &in_shapes,
                              std::vector<int> &res)
{
    bool ret = true;
    std::vector<int> in_dims;
    if (in_shapes.size())
    {
        for (auto &s : in_shapes)
        {
            for (auto &d : s)
            {
                in_dims.push_back(d);
            }
        }
    }
    else
    {
        for (auto inp : g->get_input_tensors())
        {
            for (int d : inp->shape())
            {
                in_dims.push_back(d);
            }
        }
    }
    res.clear();
    auto &symbols = g->symbols();
    for (auto out : g->get_output_tensors())
    {
        for (int i = 0; i < out->shape().dim(); ++i)
        {
            auto expr = out->shape().symbol()[i];
            std::map<Symbol, Symbol::Value> args;
            for (auto &&var : expr->variables())
            {
                if (symbols.count(var.name()))
                {
                    auto &sym = symbols.at(var.name());
                    if (sym.type == 0 && sym.dims.size())
                    {
                        auto dim = sym.dims[0];
                        if (dim < (int)in_dims.size())
                        {
                            args[var] = in_dims[dim];
                        }
                    }
                }
            }
            if (!expr->eval(args))
            {
                WARN() << "Eval expr \"" << expr->to_string() << "\" failed!";
                ret = false;
            }
            res.push_back(expr->value());
        }
    }
    return ret;
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

GraphBuilder *run_gb(GraphPtr g, std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map)
{
    uint64_t t0, t1;
    t0 = utils::get_time_in_us();
    GraphBuilder *builder;
    std::vector<TensorPtr> outputs;
    std::vector<int32_t> dump_ids;
    if (arg_map.count("weight"))
    {
        auto weight_name = cast_any<std::string>(arg_map["weight"]);
        if (!utils::file_exists(weight_name))
        {
            ERROR() << "Cannot Open file:" << weight_name;
            return nullptr;
        }
        g->load_weight(weight_name);
    }
    if (arg_map.count("dump_tensors"))
    {
        auto dump_tensors = utils::split_string(cast_any<std::string>(arg_map["dump_tensors"]), ',');
        for (auto dump : dump_tensors)
        {
            int32_t id = Node::parse_param_value(dump);
            dump_ids.push_back(id);
        }
    }

    // auto cfg = BuilderConfig::get_instance();
    // INFO() << *cfg;

    if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
    {
        builder = new GraphELFBuilder(g);
    }
    else
    {
        builder = new GraphBuilder(g);
    }
    builder->build_aipu(input_tensors, outputs, dump_ids);
    auto optimized_g = builder->graph();
    t1 = utils::get_time_in_us();
    INFO() << " -  compile time: " << (t1 - t0) / 1000 / 1000. << " s";
    return builder;
}

bool run_simulator(std::string dump_file, std::map<std::string, aipubt::Any> &arg_map)
{
    std::string simulator;
    if (arg_map.count("simulator"))
    {
        simulator = cast_any<std::string>(arg_map["simulator"]);
    }
    simulator = get_simulator_path(simulator);
    bool ret = run_simulator(simulator + " " + dump_file);

    if (arg_map.count("save-simulator"))
    {
        get_simulator_path(simulator);
    }

    return ret;
}

static void get_elf_bss_info(const AIPUELFPtr elf,
                             std::vector<Tensor::Dtype> &in_dtypes,
                             std::vector<uint32_t> &out_sizes,
                             std::vector<uint32_t> &out_dims,
                             std::vector<Tensor::Dtype> &out_dtypes)
{
    for (auto &bss : elf->get_fm_list().fmdesc)
    {
        for (size_t i = 0; i < bss.num_buffer_descriptor; ++i)
        {
            auto &desc = bss.buffer_descriptor[i];
            for (size_t j = 0; j < desc.sub_section_cnt; ++j)
            {
                auto &subdesc = desc.sub_sections[j];
                if (subdesc.type == BssSubSectionDescriptor::input && subdesc.offset_in_ro_cnt != 0)
                {
                    if (in_dtypes.size() <= subdesc.id)
                    {
                        in_dtypes.resize(subdesc.id + 1);
                    }
                    in_dtypes[subdesc.id] = static_cast<Tensor::Dtype>(subdesc.dtype);
                }
                if (subdesc.type == BssSubSectionDescriptor::output && subdesc.offset_in_ro_cnt != 0)
                {
                    if (out_sizes.size() <= subdesc.id)
                    {
                        out_sizes.resize(subdesc.id + 1);
                        out_dtypes.resize(subdesc.id + 1);
                    }
                    out_sizes[subdesc.id] = subdesc.size;
                    out_dtypes[subdesc.id] = static_cast<Tensor::Dtype>(subdesc.dtype);
                }
                if (subdesc.type == BssSubSectionDescriptor::output_tensor_shape && subdesc.offset_in_ro_cnt != 0)
                {
                    if (out_dims.size() <= subdesc.id)
                    {
                        out_dims.resize(subdesc.id + 1);
                    }
                    out_dims[subdesc.id] = subdesc.size / sizeof(uint32_t);
                }
            }
        }
    }
}

static void update_simulator_cfg(const std::string &dump_prefix,
                                 const std::string &output_file,
                                 const std::vector<uint32_t> &output_sizes)
{
    auto dump_cfg = dump_prefix + ".cfg";
    std::ifstream ifs(dump_cfg, std::ios::in);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    ifs.close();
    utils::remove_file(dump_cfg);
    auto cfg_lines = utils::split_string(buffer.str(), "\n");
    for (uint32_t i = 0; i < output_sizes.size(); ++i)
    {
        auto ofile = output_file;
        if (i != 0)
        {
            ofile += std::to_string(i);
        }
        for (uint32_t j = 0; j < cfg_lines.size(); ++j)
        {
            auto line = cfg_lines[j];
            if (!utils::startswith(line, "FILE") || line.find_first_of("=") == std::string::npos)
            {
                continue;
            }
            auto splits = utils::split_string(line, "=");
            if (splits.size() == 2 && splits[1] == ofile)
            {
                auto dump_id = utils::replace(splits[0], "FILE", "");
                cfg_lines[j + 2] = "SIZE" + dump_id + "=" + utils::int_to_hex(output_sizes[i]);
                break;
            }
        }
    }
    std::ofstream ofs(dump_cfg, std::ios::out);
    auto cfg_str = utils::join_string("\n", cfg_lines);
    ofs.write(cfg_str.c_str(), cfg_str.size());
    ofs.close();
}

bool cpsdumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map)
{
    aipubt::rt::Launcher launcher;
    launcher.load_from_file(cast_any<std::string>(arg_map["aipubin"]));

    auto inputs = launcher.get_inputs();
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto &input = inputs[i];
        input.data = input_tensors[i]->data<char>();
        input.length = input_tensors[i]->mem_size();
        input.name = input_tensors[i]->name();
        input.shape = input_tensors[i]->shape();
    }
    INFO() << "fun with btrt:";
    uint64_t t0, t1;
    t0 = utils::get_time_in_us();
    launcher.run(inputs);
    t1 = utils::get_time_in_us();
    INFO() << "simulator done! total time: " << (t1 - t0) / 1000 / 1000. << " s";
    launcher.dump_outputs();

    return true;
}

bool elfdumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map, GraphELFBuilder *builder)
{
    for (auto arg : arg_map)
    {
        std::string str;

        if (arg.second.get_type_id() != GET_ANY_TYPEID<std::string>())
        {
            str = std::to_string(cast_any<int>(arg.second));
        }
        else
        {
            str = cast_any<string>(arg.second);
        }
        DEBUG() << arg.first << " " << str;
    }

    if (builder == nullptr)
    {
        builder = new GraphELFBuilder(cast_any<std::string>(arg_map["aipubin"]));
    }
    int core_num = arg_map.count(Constants::multi_core) ? cast_any<int>(arg_map[Constants::multi_core]) : 1;
    string target = Target::get_name_from_arch_number(builder->get_aipuelf()->compiler_msg().arch);
    if (core_num > 1)
    {
        target = target + "_MP" + std::to_string(core_num);
    }
    INFO() << "target: " << target;
    arg_map["target"] = target;

    auto ds_flag = builder->get_aipuelf()->compiler_msg().flag >> 6 & 0x1;

    if (ds_flag)
    {
        arg_map["enable_ds"] = true;
    }

    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return false;
    }

    std::vector<uint32_t> out_sizes, out_dims;
    std::vector<Tensor::Dtype> in_dtypes, out_dtypes;
    if (ds_flag)
    {
        auto str = BuilderConfig::get<std::string>(Constants::input_shape);
        if (str.find_first_of('=') != std::string::npos)
        {
            FATAL() << "assignment to symbols is not supported";
        }
        std::vector<std::vector<int>> in_shapes;
        for (auto &s : utils::split_string(str, "],", 1))
        {
            in_shapes.push_back(get_base_shape(TensorShape(s)));
        }
        if (in_shapes.size() != input_tensors.size())
        {
            ERROR() << "Number of input tensors " << input_tensors.size() << " is not equal to input shapes " << in_shapes.size();
            return false;
        }
        get_elf_bss_info(builder->get_aipuelf(), in_dtypes, out_sizes, out_dims, out_dtypes);
        for (uint32_t i = 0; i < input_tensors.size(); ++i)
        {
            input_tensors[i]->shape() = TensorShape(in_shapes[i]);
            input_tensors[i]->set_type(in_dtypes[i]);
        }
    }

    // auto cfg = BuilderConfig::get_instance();
    // INFO() << *cfg;

    Dumper *dumper;
    std::string dump_prefix = "temp";
    std::string output_file = cast_any<std::string>(arg_map["output"]);
    dumper = new DumperX(aipubt::BuilderConfig::get_instance(), builder->get_aipuelf(), Target::get_the_target());
    auto ret_dump = dumper->gen_dumper_files(input_tensors, dump_prefix, output_file);

    bool ret_sim = false;
    if (ret_dump && arg_map.count("simulator"))
    {
        ret_sim = run_simulator(dump_prefix + ".cfg", arg_map);
    }

    if (ret_sim && arg_map.count("input_shape"))
    {
        std::ifstream ifs;
        ifs.open("output_shape.bin", std::ios::in | std::ios::binary);
        std::vector<uint32_t> bin_sizes;
        bool update_cfg = false;
        for (uint32_t i = 0; i < out_sizes.size(); ++i)
        {
            TensorShape shape(std::vector<int>(out_dims[i], 0));
            uint32_t dim;
            for (uint32_t j = 0; j < out_dims[i]; ++j)
            {
                ifs.read((char *)(&dim), sizeof(uint32_t));
                shape[j] = dim;
            }

            auto ofile = output_file;
            if (i != 0)
            {
                ofile += std::to_string(i);
            }
            uint32_t bin_size = Tensor::dtype_size(out_dtypes[i]) * shape.size();
            if (bin_size < out_sizes[i])
            {
                auto data = new char[bin_size];
                std::ifstream ifs;
                ifs.open(ofile, std::ios::in | std::ios::binary);
                ifs.read(data, bin_size);
                ifs.close();

                utils::remove_file(ofile);
                std::ofstream ofs;
                ofs.open(ofile, std::ios::out | std::ios::binary);
                ofs.write(data, bin_size);
                ofs.close();
                delete[] data;
                update_cfg = true;
            }
            bin_sizes.push_back(bin_size);
        }
        ifs.close();
        if (update_cfg)
        {
            update_simulator_cfg(dump_prefix, output_file, bin_sizes);
        }
    }

    delete dumper;
    return ret_sim;
}

static bool show_aipubin_info(AIPUBin &aipubin)
{
    string s(aipubin.header.ident);
    if (s == "AIPUBIN_FILE_MAGIC_NUM")
    {
        std::cout << "AIPUBIN"
                  << "\n";
        std::cout << "Bin file version 0x" << std::hex << aipubin.header.version << "\n";
        std::cout << "Built for architecture: 0x" << std::hex << (aipubin.header.arch >> 20)
                  << ", ISA type: " << ((aipubin.header.arch >> 16) & 0xF) + 1
                  << ", target: " << std::dec << (aipubin.header.arch & 0xffff)
                  << "\n";
        std::cout << "Compiler version: "
                  << (aipubin.header.build_version & 0xff)
                  << "."
                  << ((aipubin.header.build_version >> 8) & 0xff)
                  << "."
                  << ((aipubin.header.build_version >> 16) & 0xffff)
                  << "\n";
        if (aipubin.header.flag & 0b11)
        {
            std::cout << "ASID enabled.\n";
        }
        else
        {
            std::cout << "ASID disabled.\n";
        }
    }
    return true;
}

bool aipudumper(std::vector<TensorPtr> &input_tensors, std::map<std::string, aipubt::Any> &arg_map)
{
    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return 0;
    }

    Dumper *dumper;
    std::string dump_prefix = "temp";
    std::string output_file = cast_any<std::string>(arg_map["output"]);

    std::map<BssSubSecType, std::vector<TensorPtr>> marked_tensors;
    auto aipubin = GraphBuilder::load(cast_any<std::string>(arg_map["aipubin"]));
    show_aipubin_info(aipubin);
    dumper = new DumperX1(aipubt::BuilderConfig::get_instance(), &aipubin, marked_tensors);

    auto ret_dump = dumper->gen_dumper_files(input_tensors, dump_prefix, output_file);

    // simulator
    bool ret_sim = false;
    if (ret_dump && arg_map.count("simulator"))
    {
        ret_sim = run_simulator(dump_prefix + ".cfg", arg_map);
    }

    if (arg_map.count("DEBUG"))
    {
        handle_op_counter("op_counter.bin");
    }
    handle_error_code("error_code.bin");

    delete dumper;
    return ret_sim;
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
    allocate_output_shape(new_g); // compute data-aware symbol offsets
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

// before_build --> build --> dump --> after_build --> simulator
GraphPtr aipurun(GraphPtr g,
                 std::vector<TensorPtr> &input_tensors,
                 std::map<std::string, aipubt::Any> &arg_map)
{
    // before_build
    auto error_cnt = GCompilerLogger::get_error_count();
    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return nullptr;
    }
    GraphPtr optimized_g;
    if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv7)
    {
        IBuilder *builder = new MqBuilder(g);

        builder->build();
        optimized_g = builder->get_graph();

        auto elf = builder->get_elf();
        std::stringstream ss;
        elf->save(ss);

        aipubt::rt::Launcher launcher;
        launcher.load_from_mem(ss.str().data(), ss.str().size());

        auto slot_addr = launcher.get_slot_addresses();
        optimized_g->set_attr("addr_vec_list", slot_addr);

        actions_after_build(optimized_g, arg_map);

        auto inputs = launcher.get_inputs();
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto &input = inputs[i];
            input.data = input_tensors[i]->data<char>();
            input.length = input_tensors[i]->mem_size();
            input.name = input_tensors[i]->name();
            input.shape = input_tensors[i]->shape();
        }
        INFO() << "fun with btrt:";
        uint64_t t0, t1;
        t0 = utils::get_time_in_us();
        launcher.run(inputs);
        t1 = utils::get_time_in_us();
        INFO() << "simulator done! total time: " << (t1 - t0) / 1000 / 1000. << " s";

        launcher.dump_outputs(cast_any<std::string>(arg_map["output"]));
        if (arg_map.count("dump"))
        {
            launcher.dump_temp(".");
        }
        if (BuilderConfig::get<bool>("dump_all_tensors"))
        {
            std::set<TensorPtr> dump_tensors;
            std::map<TensorPtr, int> layer_id_map;
            auto tensor_index_map = ((MqBuilder *)builder)->get_tensor_index_map();
            for (auto s : optimized_g->nodes())
            {
                auto sgn = SubGraphNode::from_node(s);
                if (sgn)
                {
                    auto sg = sgn->subgraph();
                    for (auto n : sg->nodes())
                    {
                        if (n->type() == OpType_Input || n->type() == OpType_SubGraphInput)
                        {
                            continue;
                        }
                        for (auto out : n->outputs())
                        {
                            auto tensor_p = std::dynamic_pointer_cast<ProxyTensor>(out);
                            if (tensor_p)
                            {
                                out = tensor_p->proxied_tensor(true);
                            }
                            dump_tensors.insert(out);
                            layer_id_map[out] = n->get_attr("layer_id");
                        }
                    }
                }
            }
            std::map<int, std::pair<std::string, int>> dump_map;
            for (auto t : dump_tensors)
            {
                if (!tensor_index_map.count(t))
                {
                    ERROR();
                }
                auto index = tensor_index_map[t];
                auto id = layer_id_map[t];
                auto name = "dumptensor_" + to_string(id) + "_" + utils::verify_name(t->name()) + ".bin";
                auto size = t->mem_size();
                dump_map[index] = make_pair(name, size);
            }
            launcher.dump_tensors(dump_map);
        }

        delete builder;
    }
    else
    {
        // build
        auto builder = run_gb(g, input_tensors, arg_map);
        optimized_g = builder->graph();
        if (GCompilerLogger::get_error_count() > error_cnt)
        {
            INFO() << "There are some errors in compiling time, Simulator would not run.";
            return nullptr;
        }

        // dump
        std::string global_cwd = BuilderConfig::get<std::string>(Constants::global_cwd);
        std::string dump_prefix = global_cwd + "/temp_" + utils::get_uuid();

        Dumper *dumper;
        if (arg_map.count("dump"))
        {
            dump_prefix = "temp";
        }
        dump_prefix = utils::get_abspath(dump_prefix);
        if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
        {
            std::string aipubin = dump_prefix + ".aipuelf";
            builder->write(aipubin);
            dumper = new DumperX(BuilderConfig::get_instance(), ((GraphELFBuilder *)builder)->get_aipuelf(), Target::get_the_target(), builder->graph(), builder->get_ro());
            optimized_g->attrs()["_elf"] = ((GraphELFBuilder *)builder)->get_aipuelf(); // attach elf into graph for later use
        }
        else
        {
            dumper = new DumperX1(BuilderConfig::get_instance(), builder->get_aipubin(), builder->get_marked_tensors());
        }

        std::vector<std::vector<int>> input_shapes;
        if (BuilderConfig::exist(Constants::input_shape))
        {
            input_shapes = parse_input_shape(optimized_g, Constants::input_shape);
            for (uint32_t i = 0; i < input_tensors.size(); ++i)
            {
                auto tensor = std::make_shared<Tensor>(*input_tensors[i]);
                auto shape = get_base_shape(tensor->shape());
                tensor->shape() = TensorShape(input_shapes[i], shape.layout());
                input_tensors[i] = tensor;
            }
        }

        auto output_file = cast_any<std::string>(arg_map["output"]);
        if (BuilderConfig::exist(Constants::multi_batch))
        {
            auto batch = BuilderConfig::get<uint32_t>(Constants::multi_batch);
            for (auto &input_t : input_tensors)
            {
                auto input_shape = input_t->shape();
                input_shape[0] = input_shape[0] * batch;
                input_t->shape() = TensorShape(input_shape, input_t->shape().layout());
            }
        }
        auto ret_dump = dumper->gen_dumper_files(input_tensors, dump_prefix, output_file);

        // after_build
        actions_after_build(optimized_g, arg_map);

        if (GCompilerLogger::get_error_count())
        {
            INFO() << "There are some errors in dumper, Simulator would not run.";
            return optimized_g;
        }

        std::vector<int> dump_shape_dims;

        // simulator
        if (ret_dump && !arg_map.count("skip_simulator"))
        {
            if (!run_simulator(dump_prefix + ".cfg", arg_map))
            {
                ERROR() << "run simulator failed!";
            }
            else
            {
                bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
                if (enable_ds)
                {
                    if (is_empty_graph(optimized_g))
                    {
                        eval_output_shape(optimized_g, input_shapes, dump_shape_dims);
                    }
                    else
                    {
                        auto ts_base = optimized_g->global_tensors()[Constants::tensor_shape_base];
                        uint32_t num_dims = ts_base->mem_size() / sizeof(uint32_t);
                        std::ifstream ifs;
                        ifs.open("output_shape.bin", std::ios::in | std::ios::binary);
                        for (uint32_t i = 0; i < num_dims; ++i)
                        {
                            uint32_t dim = 0;
                            ifs.read((char *)(&dim), sizeof(uint32_t));
                            dump_shape_dims.push_back(dim);
                        }
                        ifs.close();
                    }
                }

                bool update_cfg = false;
                std::vector<uint32_t> bin_sizes;
                uint32_t index = 0;
                auto out_tensors = builder->graph()->get_output_tensors();
                for (uint32_t i = 0; i < out_tensors.size(); ++i)
                {
                    auto ofile = output_file;
                    if (i != 0)
                    {
                        ofile += std::to_string(i);
                    }
                    auto out = out_tensors[i];
                    out->set_data(new char[out->mem_size()], out->mem_size());
                    std::ifstream ifs;
                    ifs.open(ofile, std::ios::in | std::ios::binary);
                    ifs.read(out->data<char>(), out->mem_size());
                    ifs.close();

                    if (enable_ds)
                    {
                        uint32_t shape_size = 1;
                        for (int j = 0; j < out->shape().dim(); ++j)
                        {
                            shape_size *= dump_shape_dims[index++];
                        }
                        auto bin_size = shape_size * Tensor::dtype_size(out->type());
                        if (bin_size < out->mem_size())
                        {
                            utils::remove_file(ofile);
                            std::ofstream ofs;
                            ofs.open(ofile, std::ios::out | std::ios::binary);
                            ofs.write(out->data<char>(), bin_size);
                            ofs.close();
                            update_cfg = true;
                        }
                        bin_sizes.push_back(bin_size);
                    }
                }
                if (update_cfg && arg_map.count("dump"))
                {
                    update_simulator_cfg(dump_prefix, output_file, bin_sizes);
                }
            }
        }

        if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
        {
            if (arg_map.count("multi_batch"))
            {
                auto output_tensors = builder->graph()->get_output_tensors();
                handle_multi_batch(output_tensors, cast_any<uint32_t>(arg_map["multi_batch"]), output_file);
            }
        }
        else
        {
            if (arg_map.count("DEBUG"))
            {
                handle_op_counter("op_counter.bin");
            }
            handle_error_code("error_code.bin");
        }

        // serialize real shape if input shape specified
        if (!arg_map.count("skip_simulator") && arg_map.count("dumpIR") && input_shapes.size())
        {
            if (!infer_real_shape(optimized_g, dump_shape_dims, input_shapes))
            {
                ERROR() << "Infer real shapes of optimized graph failed!";
            }
            serialize_graph(optimized_g, arg_map);

            std::set<std::string> dims;
            if (BuilderConfig::exist(Constants::dynamic_input_dim))
            {
                dims = BuilderConfig::get<std::set<std::string>>(Constants::dynamic_input_dim);
            }
            auto new_g = infershape(g, dims, input_shapes, dump_shape_dims, true);
            std::string prefix = arg_map["dumpIR"];
            arg_map["dumpIR"] = prefix + "_raw";
            serialize_graph(new_g, arg_map);
            arg_map["dumpIR"] = prefix;
        }

        // clean temp files
        if (!arg_map.count("dump"))
        {
            INFO() << "remove global cwd: " << global_cwd;
            utils::remove_dir(global_cwd);
        }
        delete builder;
        delete dumper;
    }

    return optimized_g;
}

// before_build --> build --> after_build
GraphPtr aipugb(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map)
{
    // before_build
    auto error_cnt = GCompilerLogger::get_error_count();
    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return nullptr;
    }

    GraphPtr optimized_g;
    if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv7)
    {

        IBuilder *builder = new MqBuilder(g);

        builder->build();
        optimized_g = builder->get_graph();

        actions_after_build(optimized_g, arg_map);

        auto elf = builder->get_elf();
        elf->save(cast_any<std::string>(arg_map["output"]));

        delete builder;
    }
    else
    {
        // build
        std::vector<TensorPtr> dummy_inputs = std::vector<TensorPtr>();
        auto builder = run_gb(g, dummy_inputs, arg_map);

        optimized_g = builder->graph();
        if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
        {
            optimized_g->attrs()["_elf"] = ((GraphELFBuilder *)builder)->get_aipuelf(); // attach elf into graph for later use
        }

        // after_build
        actions_after_build(optimized_g, arg_map);

        if (GCompilerLogger::get_error_count() > error_cnt)
        {
            if (arg_map.count("dump"))
            {
                builder->write(cast_any<std::string>(arg_map["output"]));
            }
            INFO() << "There are some errors in compiling time.";
            return nullptr;
        }

        builder->write(cast_any<std::string>(arg_map["output"]));
        // clean temp files
        if (!arg_map.count("dump"))
        {
            std::string global_cwd = BuilderConfig::get<std::string>(Constants::global_cwd);
            INFO() << "remove global cwd: " << global_cwd;
            utils::remove_dir(global_cwd);
        }

        delete builder;
    }
    return optimized_g;
}

GraphPtr aipuestimator(GraphPtr g, std::map<std::string, aipubt::Any> &arg_map)
{
    if (arg_map.count("svg"))
    {
        arg_map["show_info"] = "perf";
    }

    int retcode;
    if (actions_before_build(arg_map, retcode))
    {
        return nullptr;
    }

    PerfEstimator perf;
    perf(g);

    actions_after_build(g, arg_map);

    if (GCompilerLogger::get_error_count())
    {
        return nullptr;
    }

    auto global_cwd = BuilderConfig::get<std::string>(Constants::global_cwd);
    INFO() << "remove global cwd: " << global_cwd;
    utils::remove_dir(global_cwd);

    return g;
}

} // namespace aipubt

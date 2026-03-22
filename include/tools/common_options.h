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
 * @Author: Yuchou.Gan
 * @Date: 2020-06-18 15:38:29
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-01-06 17:34:58
 */
#ifndef COMMON_ARG_OPTION_H
#define COMMON_ARG_OPTION_H

#include <builder/builder.h>
#include <builder/lowering/lowering.h>
#include <builder/profiler.h>
#include <builder/target.h>
#include <core/any.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <core/default_config.h>
#include <core/plugin_loader.h>
#include <gcompiler_api.h>
#include <tools/argparser.h>
#include <utils/utils.h>
using namespace aipubt;

inline void get_common_argparser(ArgParser &args)
{
    args.add_argument("graphdef", "The input graph");
    args.add_argument("-v", "--version", "print the version.", true, true);
    args.add_argument("", "--plugin_version", "print the plugins' version.", true, true);
    args.add_argument("-w", "--weight", "The weight file for the graph", true,
                      false);
    args.add_argument("-h", "--help", "print the help message.", true, true);

    args.add_argument("--target",
                      "The target to be built. If the name ends with 'MPx', "
                      "it will build with multi-core optimization from X2 generation, "
                      "for example: X2_1204MP3 will optimize for 3 cores for X2_1204."
                      " Supports target code: " +
                          utils::join_string(",", Target::get_support_targets()),
                      string(get_default_config<const char *>(Constants::target)));
    args.add_argument("--png", "The graph in png format for vis.");
    args.add_argument("-L", "--Lib",
                      "The library search path for customize lib, use \":\" as "
                      "separator if have more than two lib pathes.",
                      false, false);
    args.add_argument("-p", "--profile", "enable profile counter.", true, true);
    args.add_argument("--prof_unit",
                      "Select hardware unit to profile, currently support \'AIFF\', \'DMA\' and \'TPC\'.",
                      string(get_default_config<const char *>(Constants::prof_unit)));
    args.add_argument("", "--fast_perf",
                      "Disable DMA/AIFF profile, which may cost much extra time. Only for X2", true, true);
    args.add_argument("--max_sample_point_num",
                      "Profiler max sample point number, only for X2", get_default_config<uint64_t>(Constants::max_sample_point_num));
    args.add_argument("--max_aiff_sample_point_num",
                      "Profiler max aiff sample point number, only for X2", get_default_config<uint64_t>(Constants::max_aiff_sample_point_num));
    args.add_argument("--max_dma_sample_point_num",
                      "Profiler max dma sample point number, only for X2", get_default_config<uint64_t>(Constants::max_dma_sample_point_num));
    args.add_argument("--max_custom_sample_point_num",
                      "Profiler max custom sample point number, only for X2", get_default_config<uint64_t>(Constants::max_custom_sample_point_num));
    args.add_argument("--asid_perf",
                      "Which asid footprint would like to perf, you can choose 2 items from [gm, asid0, asid1, asid2, asid3], only for X3", (std::string)get_default_config<const char *>(Constants::asid_perf));

    args.add_argument("--stack_size",
                      "The stack_size (KB) for AIPU, default is 257 (KB).",
                      (uint32_t)get_default_config<uint64_t>(Constants::stack_size) / 1024);
    args.add_argument("-Wall", "--warn-all",
                      "Show all warning messages.",
                      true, true);
    args.add_argument("", "--tcm_size",
                      "The dTCM size of X1_1204 SOC platform, default is 4MB, accept a number in kbytes.", true, false);
    args.add_argument("", "--gm_size",
                      "The Global Memory size of X2 or above SOC platform, accept a number in kbytes, default value will be same as hardware config: i.e. 4MB for X2_1204MP3; 1MB for X2_1204.", true, false);
    args.add_argument("", "--ocm_size",
                      "Setting the SOC SRAM size or On-Chip-Memory size, the allocator will try to use these memory,"
                      " it may cause a slight higher memory consumption, and the ocm or soc sram is no guarrented be used on"
                      " runtime, default is 0. Accept a number in KBytes.",
                      true, false);
    args.add_argument("", "--ocm_addr",
                      "set ocm address", true, false);
    args.add_argument("-g", "--enable_debug_info",
                      "add debug infomation into elf for >X2 targets.",
                      true, true);
    // for dev mode
    args.add_hidden_argument("", "--dump_all_tensors", "dump all tensors (including outputs) except inputs", true, true);
    args.add_hidden_argument("", "--dump_rt_tensor_id", "dump all real-time tensor output by id", true, false);
    args.add_hidden_argument("", "--dump_rt_input_tensors", "dump all real-time tensors (including inputs) except outputs", true, true);
    args.add_hidden_argument("", "--dump_rt_output_tensors", "dump all real-time tensors (including outputs) except inputs", true, true);
    args.add_hidden_argument("", "--dump_tensors",
                             "dump output tensors of specified layers (layer_id)."
                             "If you have several layers, use ',' to separate them"
                             "Internal tensors of subgraph and in/out tensors of graph would be skipped",
                             true, false);
    args.add_hidden_argument("", "--dump", "dump the ro/text/data file for emulator",
                             true, true);
    args.add_hidden_argument("", "--disable_mmu", "disable mmu optimize.", true, true);
    args.add_hidden_argument("", "--show_buffer_index", "Show buffer index in graph png/svg. (deprecated)", true, true);
    args.add_hidden_argument("", "--show_info", "Show additional information in graph png/svg,"
                                                " such as memory allocation, estimated performance data and tensor size, etc."
                                                " Optional types include \"mem\", \"perf\", \"tsize\" and \"all\","
                                                " use comma to separate if there are multiple types.",
                             true, false);
    args.add_hidden_argument("", "--disable_input_buffer_reuse", "disable the reusing of input buffer.", true, true);
    args.add_hidden_argument("", "--disable_output_buffer_reuse", "disable the reusing of output buffer.", true, true);
    args.add_hidden_argument("", "--disable_cross_buffer_reuse", "disable the reusing of output buffer.", true, true);
    args.add_hidden_argument("", "--alloc_output_on_extsram", "try allocate output on extsram.", true, true);
    args.add_hidden_argument("--svg", "The graph in svg format for vis.");
    args.add_hidden_argument("--dot", "The graph in dot format for vis.");
    args.add_hidden_argument("-D", "--DEBUG", "Building graph in debug mode.",
                             true, true);
    args.add_hidden_argument("", "--rand-reg",
                             "Enable random initial registers, a pice of extra code would be injected to randomize all register.",
                             true, true);
    args.add_hidden_argument("", "--reset_dma_per_layer", "Reset DMA when hwa is done per layer. (Z1 only).", true, true);
    args.add_hidden_argument("", "--disable_aipu_cache", "Disable both I-Cache and D-Cache for scalar of AIPU.", true, true);
    args.add_hidden_argument("", "--disable_icache", "Disable  I-Cache of AIPU.", true, true);
    args.add_hidden_argument("", "--disable_spsram_optimization", "Disable optimization for Single Port SRAM (SPSRAM).", true, true);
    args.add_hidden_argument("", "--rand-seed",
                             "Set random seed for random intial registers, default is rand",
                             true, false);
    args.add_hidden_argument("", "--show-all-passes",
                             "Show all passes for optimize graph.", true, true);
    args.add_hidden_argument("", "--sort_after_each_pass",
                             "do topological sort after each pass", true, true);
    args.add_hidden_argument("", "--dumpIR",
                             "Dump the optimized IR. a <DUMPIR>.def and "
                             "a<DUMPIR>.bin file would be generated.",
                             true, false);
    args.add_hidden_argument("", "--serialize-before-pass",
                             "Dump the optimized IR before a pass name. If you "
                             "have several passes, use ',' to separate them",
                             true, false);
    args.add_hidden_argument("", "--serialize-after-pass",
                             "Dump the optimized IR after a pass name. If you "
                             "have several passes, use ',' to separate them",
                             true, false);
    args.add_hidden_argument("", "--svg-before-pass",
                             "save graph to svg before a pass name. If you have "
                             "several passes, use ',' to separate them",
                             true, false);
    args.add_hidden_argument("", "--svg-after-pass",
                             "save graph to svg before a pass name. If you have "
                             "several passes, use ',' to separate them",
                             true, false);

    args.add_hidden_argument("", "--disable-pass",
                             "Disable pass(es) by name. If you have several "
                             "passes, use ',' to separate them",
                             true, false);
    args.add_hidden_argument("", "--enable-pass",
                             "Enable pass(es) by name. If you have several "
                             "passes, use ',' to separate them",
                             true, false);
    args.add_hidden_argument("", "--disable-plugins",
                             "Disable plugin(s) by name. If you have several "
                             "passes, use ',' to separate them",
                             true, false);
    args.add_hidden_argument("", "--pass-before-tiling",
                             "Add pass(es) executed before tiling. If you have several "
                             "passes, use ',' to separate them",
                             true, false);
    args.add_hidden_argument("", "--terminator",
                             "The last executed node name, exit the program after this node.",
                             true, false);

    args.add_hidden_argument("", "--show_profile_events",
                             "print supported profile event id and relative description",
                             true, true);
    args.add_hidden_argument("--profile_event", "select profile events");
    args.add_hidden_argument("", "--enable_dma_perf", "enable DMA profile, only for X2.", true, true);
    args.add_hidden_argument("--printf_fifo_size",
                             "The printf fifo limit size in KB, used for printf buffer, default is 16KB.",
                             (uint64_t)get_default_config<uint64_t>(Constants::printf_fifo_size) / 1024);
    args.add_hidden_argument("", "--tiling",
                             "Select the memory optimization mode. disabled by default."
                             "footprint. In this mode, we will try to minimize the footprint."
                             "fps. In this mode, we will try to improve the fps. (X2_1204_MP3 only)"
                             //  "balance. In this mode, we will try to balance the footprint and fps.",
                             //  "ddr_usage. In this mode, we will try to minimize the ddr_usage.",
                             "'*.json'. load tile_cfg which is a json file. GBuilder will tile graph according to this config",
                             //  "The following would be set in this mode:"
                             //  "ENABLE_TCM=1 (dtcm=4M by default)"
                             //  "enable-pass tile_graph, tnode_to_sg",
                             true, false);
    args.add_hidden_argument("", "--dump_tile_cfg",
                             "Dump the tile_cfg to tile_cfg_dump.json",
                             true, true);
    args.add_hidden_argument("", "--enable_cse_tile_node",
                             "Tiled node could be removed by pass CSE_subnodes",
                             true, true);
    args.add_hidden_argument("", "--strings_for_debug",
                             "Internal debug only. If you have "
                             "several strings, use ',' to separate them",
                             true, false);
    args.add_hidden_argument("--number_for_debug",
                             "Internal debug only.",
                             (uint64_t)16);
    args.add_hidden_argument("", "--switch_for_debug",
                             "Internal debug only",
                             true, true);
    args.add_hidden_argument("", "--show_tensor_size",
                             "Internal debug only. (deprecated)",
                             true, true);
    args.add_hidden_argument("", "--show_raw_graph",
                             "show the graph_raw.svg before lowering",
                             true, true);
    args.add_hidden_argument("", "--datalayout_schedule_debug",
                             "datalayout schedule dump layout info",
                             true, true);
    args.add_hidden_argument("", "--show_short_name",
                             "replace the name of node/tensor with a shorter one (type+layer_id)",
                             true, true);
    // for dev mode
    args.add_hidden_argument("", "--help-hidden", "Show this message", true,
                             true);
    args.add_hidden_argument("", "--individual-inputs", "Besides ro/text/data, dump temp.input0, temp.input1,..., for simulator/emulator", true, true);
    args.add_hidden_argument("", "--enable_sparse_optimization",
                             "Enable the sparse optimization for saving memory footprint, it only support Z3+ arch, please note it would have a lite performance drop in fps.",
                             true, true);
    args.add_hidden_argument("", "--disable_schedule_optimization", "disable mc_scheduler optimization.", true, true);
    args.add_hidden_argument("", "--single_core_opt", "enable single core optimization.", true, true);
    args.add_hidden_argument("", "--show_location",
                             "Save the location information to the <DUMPIR>.def file."
                             "Precondition: use --dumpIR",
                             true, true);
    args.add_hidden_argument("--wdc_rate",
                             "Set AIFF weight compression rate threshold for new WDC.");
    args.add_hidden_argument("--op_plugin_options",
                             "Set op plugins' options, use ',' and '=' to separate them, e.g. AIPU_32BIT_BIAS=1,AIPU_AVG_POOL_16BIT_SCALE=0");
    args.add_hidden_argument("", "--multi_batch",
                             "set multi batch number, default is single batch.", true, false);
    args.add_hidden_argument("", "--use_cump_for_gm",
                             "Use cump allocator for gm.", true, true);
    args.add_hidden_argument("", "--load_balance_strategy", "set load balance strategy", true, false);
    args.add_hidden_argument("", "--branch_reorder_strategy", "enable node execution reorder to improve performance, 1 for better memory usage 2 for better usage of L2 cache", true, false);
    args.add_hidden_argument("", "--gm_enhance_strategy", "set gm enhance strategy", true, false);
    args.add_hidden_argument("", "--perf_json", "set graph json path for cost model", true, false);
    args.add_hidden_argument("", "--perf_bin", "set profile data path for cost model", true, false);
    args.add_hidden_argument("", "--enable_ds", "enable dynamic shape", true, true);
    args.add_hidden_argument("", "--enable_cl_opt", "enable cl optimization for dynamic shape", true, true);
    args.add_hidden_argument("", "--max_input_shape",
                             "Set the largest input shape when dynamic shape is enabled.", true, false);
    args.add_hidden_argument("", "--default_input_shape",
                             "Set the preferred default input shape for better performance when dynamic shape is enabled.", true, false);
    args.add_hidden_argument("", "--dynamic_input_dim",
                             "Set which input dimensions of graph are dynamic when dynamic shape is enabled."
                             " Dynamic dimensions support custom strings from graph input nodes."
                             " By default, it starts with \"d\" and followed by a dimension number."
                             " For example, for a graph with two input tensors of shape [1,1,4096] and [128,1],"
                             " use --dynamic_input_dim \"d3\" if the first dimension of the second input tensor is dynamic."
                             " Use comma to separate if there are multiple dimensions,"
                             " and use \"none\" if there is no dynamic input dimension.",
                             true, false);
    args.add_hidden_argument("", "--show_divided_graph",
                             "Show divided graph before lowering in dynamic shape mode", true, true);
    args.add_hidden_argument("", "--disable_separate_gm", "disable average separate gm in cores.", true, true);
    args.add_hidden_argument("", "--enable_separate_gm", "enable average separate gm in cores.", true, true);
    args.add_hidden_argument("", "--max_wt_size", "split wt into two half, unit in MByte", true, false);
    args.add_hidden_argument("", "--max_act_size", "split act into two half, unit in MByte", true, false);
    args.add_hidden_argument("", "--dump_schedule_cfg", "dump gm config", true, true);
    args.add_hidden_argument("", "--load_schedule_cfg", "load gm config", true, false);
    args.add_hidden_argument("", "--softmax_prior_in_gm", "put softmax related tensor priority in gm config", true, true);
    args.add_hidden_argument("", "--warmup_len", "warmup length", true, false);
    args.add_hidden_argument("", "--use_amat_scheduler", "use AMAT scheduler for mem allocation", true, true);
    args.add_hidden_argument("", "--preserve_gm_size", "preserve gm size for other use unit in KByte", true, false);
    args.add_hidden_argument("", "--disable_dependency", "disable dependency and use core sync this may cause perf drop but decrease compile time", true, true);
    args.add_hidden_argument("", "--soms_gm_loop", "setup the loop to search best gm footprint, default is 100", true, false);
    args.add_hidden_argument("", "--prefetch_desc", "prefetch descriptor for x2+ targets", true, true);
    args.add_hidden_argument("", "--dump_tune_strategy", "dump load balance strategy", true, true);
    args.add_hidden_argument("", "--load_tune_strategy", "load load balance strategy", true, false);
    args.add_hidden_argument("", "--dump_reorder_cfg", "dump gm config", true, true);
    args.add_hidden_argument("", "--load_reorder_cfg", "load gm config", true, false);
    args.add_hidden_argument("", "--closer_sg_opt", "make closer nodes into one sg to reduce level.", true, true);
    args.add_hidden_argument("", "--tile_split_output_c_conv", "make convs with the same parent after split_conv_output_c_ in the same group when tiling", true, true);
    args.add_hidden_argument("", "--remove_origin_constant", "remove the original constant to reduce peak memory", true, true);
    args.add_hidden_argument("", "--dump_prior_in_gm", "dump priority in gm config", true, true);
    args.add_hidden_argument("", "--load_prior_in_gm", "load priority in gm config", true, false);
    args.add_hidden_argument("", "--show_soms_bottleneck", "show soms bottleneck", true, true);
    args.add_hidden_argument("", "--print_emitter_var", "print emitter variable", true, true);
    args.add_hidden_argument("", "--dump_constant_info", "dump constant information into constant_info.csv", true, true);
}

inline bool cmd_action(ArgParser &args, int &retcode)
{
    if (args.exists("help"))
    {
        cout << args.usage() << "\n";
        retcode = 0;
        return true;
    }
    if (args.exists("help-hidden"))
    {
        cout << args.full_usage() << "\n";
        retcode = 0;
        return true;
    }
    if (args.exists("show-all-passes"))
    {
        aipubt::show_all_passes_name();
        retcode = 0;
        return true;
    }
    if (args.exists("version"))
    {
        cout << std::left << ::setw(24) << args.program_name() << "\tversion: " << VERSION << std::endl;
        retcode = 0;
        return true;
    }
    if (args.exists("plugin_version"))
    {
        auto plugins = (*aipubt::PluginLoader::get_instance())[PluginType_BuilderOp];
        cout << "plugin versions\n"
             << aipubt::PluginLoader::get_version();

        retcode = 0;
        return true;
    }
    return false;
}

#endif //! COMMON_ARG_OPTION_H

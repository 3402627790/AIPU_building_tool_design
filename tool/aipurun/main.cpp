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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-01-29 14:29:04
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-01-06 17:25:37
 */
#include <algorithm>
#include <builder/tools.h>
#include <dirent.h>
#include <fstream>
#include <gcompiler_api.h>
#include <iostream>
#include <set>
#include <stdio.h>
#include <string>
#include <thread>
#include <tools/argparser.h>
#include <tools/common_options.h>
#include <utils/utils.h>

using namespace std;
using namespace aipubt;

DECLARE_VERSION(aipurun);
#define REPEAT 1

int main(int argc, const char **argv)
{
    auto arg_list = vector<string>(argv, argv + argc);
    ArgParser args;
    get_common_argparser(args);
    args.add_argument("-i", "--input", "The inputs file for the graph", false,
                      false);
    args.add_argument("-o", "--output", "The output result file.",
                      string("output.bin"));
    args.add_argument("--simulator", "The simulator executable path.");
    args.add_hidden_argument("", "--random-init", "Enable simulator random initial feature.", true, true);
    args.add_hidden_argument("", "--save-simulator", "Save the simulator path as default simulator path.", true, true);
    args.add_hidden_argument("", "--use-runtime", "use runtime to run instead of using the GBuilder internal runtime.", true, true);
    args.add_hidden_argument("", "--ddr_data_gather", "All data are put togather in ddr from 0x0", true, true);
    args.add_hidden_argument("", "--enable_custom_smmu", "Enable customized SMMU feature", true, true);
    args.add_hidden_argument("", "--ddr_data_mode", "0: Debug mode all data in seperate region, 1: output->0x8 2: output->0x6", true, false);
    args.add_hidden_argument("", "--disable_gm", "disable GM config", true, true);
    args.add_hidden_argument("", "--enable_gm_enhance", "enable GM enhance for puting weight & featuremap all into GM", true, true);
    args.add_hidden_argument("", "--enable_dynamic_gm", "enable dynamic GM for puting featuremap dynamically into GM", true, true);
    args.add_hidden_argument("", "--dump_gm_enhance_cfg", "dump gm config", true, true);
    args.add_hidden_argument("", "--load_gm_enhance_cfg", "load gm config", true, false);
    args.add_hidden_argument("", "--dump_prior_in_gm", "dump priority in gm config", true, true);
    args.add_hidden_argument("", "--load_prior_in_gm", "load priority in gm config", true, false);
    args.add_hidden_argument("", "--insert_init_reg", "insert register initialization task tcb before running.", true, true);
    args.add_hidden_argument("", "--enable_weight_align", "weight buffer will be aligned to 1:4kbyte 2:64byte", true, false);
    args.add_hidden_argument("", "--enable_slc_ocm_mode", "All data are put togather in ddr from 0x86000000", true, true);
    args.add_hidden_argument("", "--put_weight_gm", "put weight data to gm", true, true);
    args.add_hidden_argument("", "--set_asid", "set asid map address, must aligned to 4k, format: asid0=0x86000000,asid1=0x100000000", true, false);
    args.add_hidden_argument("", "--enable_x2_sm_opt", "enable x2 sm opt for puting weight into SM", true, true);
    args.add_hidden_argument("", "--skip_simulator", "skip simulator run", true, true);
    args.add_hidden_argument("", "--native_float_simulator", "enable cpu native floating point calculation for simulator", true, true);
    args.add_hidden_argument("", "--input_shape", "Set the input tensor shapes of the graph", true, false);
    args.add_hidden_argument("", "--put_desc_gm", "put descriptor into gm", true, true);
    args.add_hidden_argument("", "--put_ws_gm", "put workspace into gm", true, true);
    args.add_hidden_argument("", "--set_l2_flush", "set l2 cache flush", true, true);
    args.add_hidden_argument("", "--dump_addr_cfg", "Dump default address config file.", true, true);
    args.add_hidden_argument("", "--ws_mem_opt", "optimize workspace memory", true, true);
    args.add_hidden_argument("", "--specify_wt_gm", "specify the weight slot into gm. Can specify multiple slots, for example 3,7,9", true, false);
    args.add_hidden_argument("", "--set_profile_option", "set profile option for simulator: SYS_FREQ_MHZ=1000,DDR_RD_LATENCY=0,DDR_BW_BITS=512", true, false);
    args.add_hidden_argument("", "--split_wt_act", "split act and act based on max_wt_size or max_act_size", true, true);
    args.parse(arg_list);
    int retcode = 0;
    if (cmd_action(args, retcode))
    {
        return 0;
    }

    std::map<std::string, aipubt::Any> arg_map;
    for (auto arg : args.get_args())
    {
        if (arg.second->store_true)
        {
            arg_map[arg.first] = true;
        }
        else
        {
            if (arg.first == "dump_tensors" || arg.first == "specify_wt_gm")
            {
                std::string str;
                if (arg.second->value.get_type_id() != GET_ANY_TYPEID<std::string>())
                {
                    str = std::to_string(cast_any<int>(arg.second->value));
                }
                else
                {
                    str = cast_any<std::string>(arg.second->value);
                }
                arg_map[arg.first] = str;
            }
            else
            {
                arg_map[arg.first] = arg.second->value;
            }
        }
    }

    auto graph_file = cast_any<string>(arg_map["graphdef"]);
    if (!utils::file_exists(graph_file))
    {
        ERROR() << "Cannot Open file:" << graph_file;
        return -1;
    }
    auto input_files = utils::split_string(cast_any<string>(arg_map["input"]), ',');
    for (auto f : input_files)
    {
        utils::trim(f);
        if (!utils::file_exists(f))
        {
            ERROR() << "Cannot Open file:" << f;
            return -1;
        }
    }

    GraphPtr g = Graph::parse(graph_file);
    if (!g)
    {
        return -1;
    }
    std::vector<TensorPtr> input_tensors = g->get_input_tensors();
    auto ret = Tensor::fill_tensors_data(input_tensors, input_files);
    if (!ret)
    {
        ERROR() << "Failed to fill input tensors from files:";
        return -1;
    }
    auto bg = aipurun(g, input_tensors, arg_map);
    if (!bg)
    {

        return -1;
    }
    auto hooks = get_aipurun_hooks();
    for (auto h : hooks)
    {
        h(bg);
    }
    if (GCompilerLogger::get_error_count())
    {
        return -2;
    }
    return 0;
}

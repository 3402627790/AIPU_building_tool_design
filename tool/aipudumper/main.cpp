//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2021 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Tom Gao
 * @Date: 2021-07-14 15:37:42
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-06-07 11:13:20
 */
#ifdef MAJOR_VERSION
#undef MAJOR_VERSION
#endif
#ifdef MINOR_VERSION
#undef MINOR_VERSION
#endif
#define MAJOR_VERSION 2
#define MINOR_VERSION 0
#include <algorithm>
#include <builder/tools.h>
#include <fstream>
#include <gcompiler_api.h>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tools/argparser.h>
#include <tools/common_options.h>
#include <utils/string_utils.h>
#include <utils/utils.h>


using namespace aipubt;
DECLARE_VERSION(aipudumper);

#define FORMAT_ADDR(value) "0x" << std::setw(8) << std::setfill('0') << std::hex << value << std::setfill(' ') << std::dec

int main(int argc, const char **argv)
{
    ArgParser args;
    args.add_argument("aipubin", "The aipu excutable binary");
    args.add_argument("-v", "--version", "print the version.", true, true);
    args.add_argument("-h", "--help", "print the help message.", true, true);
    args.add_argument("-i", "--input", "The inputs file for the graph", false, false);
    args.add_argument("-o", "--output", "The output binary file.", string("output.bin"));
    args.add_argument("", "--dump_addr_cfg", "Dump default address config file.", true, true);
    args.add_argument("", "--load_addr_cfg", "Load and apply this address config file.", true, false);
    args.add_argument("", "--disable_gm", "Disable gm config", true, true);
    args.add_argument("", "--ddr_data_gather", "data are gathered in DDR", true, true);
    args.add_argument("", "--multi_core", "set multi core number, default is single core.", true, false);
    args.add_argument("", "--multi_batch", "set multi batch number, default is single batch.", true, false);
    args.add_argument("--simulator", "The simulator executable path.");
    args.add_argument("", "--enable_weight_align", "weight buffer will be aligned to 1:4kbyte 2:64byte", true, false);
    args.add_argument("-p", "--profile", "enable profile counter", true, true);
    args.add_argument("", "--ocm_size",
                      "Setting the SOC SRAM size or On-Chip-Memory size, the allocator will try to use these memory,"
                      " it may cause a slight higher memory consumption, and the ocm or soc sram is no guarrented be used on"
                      " runtime, default is 0. Accept a number in KBytes.",
                      true, false);
    args.add_argument("", "--ocm_addr",
                      "set ocm address", true, false);
    args.add_argument("", "--gm_size",
                      "set ocm address", true, false);
    args.add_argument("", "--put_desc_gm", "put descriptor into gm", true, true);
    args.add_argument("", "--put_ws_gm", "put workspace into gm", true, true);
    args.add_argument("", "--put_weight_gm", "put weight into gm", true, true);
    args.add_argument("", "--input_shape", "Set the input tensor shapes of the graph", true, false);
    args.add_argument("", "--ws_mem_opt", "optimize workspace memory", true, true);
    args.add_argument("", "--reuse_buffer_expand", "use all reuse buffers instead of one buffer", true, true);
    args.add_argument("", "--specify_wt_gm", "specify the weight slot into gm. Can specify multiple slots, for example 3,7,9", true, false);
    args.add_argument("", "--set_asid", "set asid map address, must aligned to 4k, format: asid0=0x86000000,asid1=0x100000000", true, false);
    args.add_argument("", "--set_profile_option", "set profile option for simulator: SYS_FREQ_MHZ=1000,DDR_RD_LATENCY=0,DDR_BW_BITS=512", true, false);
    args.add_argument("", "--max_wt_size", "split wt", true, false);
    args.add_argument("", "--max_act_size", "split act", true, false);
    args.add_argument("", "--split_wt_act", "split act and act based on max_wt_size or max_act_size", true, true);
    args.parse(argc, argv);
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
            if (arg.first == "specify_wt_gm")
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

    auto input_files = utils::split_string(args.get<string>("input"), ',');
    for (auto f : input_files)
    {
        utils::trim(f);
        if (!utils::file_exists(f))
        {
            ERROR() << "Cannot Open file:" << f;
            return 1;
        }
    }
    auto output_file = args.get<string>("output");

    std::vector<TensorPtr> input_tensors;
    for (size_t i = 0; i < input_files.size(); ++i)
    {
        std::ifstream in(input_files[i], std::ifstream::ate | std::ifstream::binary);
        uint32_t size = in.tellg();
        TensorPtr tensor = std::make_shared<Tensor>(input_files[i], size, Tensor::UINT8);
        input_tensors.push_back(tensor);
    }
    auto ret = Tensor::fill_tensors_data(input_tensors, input_files);
    if (!ret)
    {
        ERROR() << "Failed to fill input tensors from files:";
        return 0;
    }

    // INFO() << "aipudumper";

    std::ifstream file;
    file.open(args.get<std::string>("aipubin"), std::ios::binary);
    if (!file)
    {
        ERROR() << "Failed to load file: " << args.get<std::string>("aipubin");
        file.close();
        return 0;
    }
    std::string magic_word;
    char buffer[8];
    file.read(buffer, 8);
    file.close();
    char elf_magic[4] = {0x7f, 'E', 'L', 'F'};
    char aipu_magic[8] = {'A', 'I', 'P', 'U', ' ', 'B', 'I', 'N'};

    if (memcmp(buffer, aipu_magic, 8) == 0)
    {
        INFO() << "load AIPUBIN file";
        aipudumper(input_tensors, arg_map);
    }
    else if (memcmp(buffer, elf_magic, 4) == 0) // x2
    {
        INFO() << "load ELF file";
        elfdumper(input_tensors, arg_map);
    }
    else
    {
        ERROR() << "MAGIC word doesn't match, please check file: " << args.get<std::string>("aipubin");
        return 0;
    }

    return 0;
}

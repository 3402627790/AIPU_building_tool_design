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
 * @Author: Lang Hu
 * @Date: 2024-07-03 11:55:52
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-07-03 11:55:52
 */

#include <builder/tools.h>
#include <tools/argparser.h>
#include <tools/common_options.h>
#include <utils/utils.h>

#ifdef MAJOR_VERSION
#undef MAJOR_VERSION
#endif
#ifdef MINOR_VERSION
#undef MINOR_VERSION
#endif
#define MAJOR_VERSION 1
#define MINOR_VERSION 0

DECLARE_VERSION(aipuestimator);
using namespace aipubt;

int main(int argc, const char **argv)
{
    ArgParser args;
    args.add_argument("graphdef", "The input graph");
    args.add_argument("-v", "--version", "Print the version", true, true);
    args.add_argument("-h", "--help", "Print the help message", true, true);
    args.add_argument("", "--target", "Target for cost model, default is X2_1204", std::string("X2_1204"));
    args.add_argument("", "--perf_json", "Set the graph json path for cost model", true, false);
    args.add_argument("", "--perf_bin", "Set the profile data path for cost model", true, false);
    args.add_argument("", "--svg", "Show the estimated performance data on svg graph", true, false);
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
            arg_map[arg.first] = arg.second->value;
        }
    }

    auto graph_path = cast_any<string>(arg_map["graphdef"]);
    if (!utils::file_exists(graph_path))
    {
        ERROR() << "Cannot open file: " << graph_path;
        return -1;
    }
    GraphPtr g = Graph::parse(graph_path);
    if (!g)
    {
        return -1;
    }

    g = aipuestimator(g, arg_map);
    if (!g)
    {
        return -1;
    }

    return 0;
}

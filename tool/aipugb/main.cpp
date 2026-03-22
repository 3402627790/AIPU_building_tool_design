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
 * @Date: 2020-01-29 14:29:12
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-11-21 10:50:40
 */
#include <algorithm>
#include <builder/tools.h>
#include <fstream>
#include <gcompiler_api.h>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <tools/argparser.h>
#include <tools/common_options.h>
#include <utils/string_utils.h>
#include <utils/utils.h>

using namespace std;
using namespace aipubt;
DECLARE_VERSION(aipugb);

int main(int argc, const char **argv)
{
    ArgParser args;
    get_common_argparser(args);
    args.add_argument("-o", "--output", "The output executable binary file.", string("aipu.bin"));
    // args.add_hidden_argument("", "--revert_format_v4", "Revert graph binary to format v4 (Default is v5).", true, true);

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
    auto graph_file = cast_any<string>(arg_map["graphdef"]);
    if (!utils::file_exists(graph_file))
    {
        ERROR() << "Cannot Open file:" << graph_file;
        return -1;
    }
    GraphPtr g = Graph::parse(graph_file);
    if (!g)
    {
        return -1;
    }
    auto bg = aipugb(g, arg_map);
    if (!bg)
    {
        return -1;
    }
    if (GCompilerLogger::get_error_count())
    {
        return -2;
    }
    return 0;
}

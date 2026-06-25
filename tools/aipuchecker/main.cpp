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
 * @Author: Tom.Gao
 * @Date: 2021-01-28 16:13:00
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-07-19 16:03:20
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>

#include <builder/builder.h>
#include <checker/checker.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <gcompiler_api.h>
#include <tools/argparser.h>
#include <tools/common_options.h>
#include <utils/string_utils.h>
#include <utils/utils.h>

using namespace std;
using namespace aipubt;

int main(int argc, const char **argv)
{
    ArgParser args;
    args.add_argument("graphdef", "The input graph");
    args.add_argument("-v", "--version", "print the version.", true, true);
    args.add_argument("-h", "--help", "print the help message.", true, true);
    args.add_hidden_argument("", "--disable_checker_plugins", "disable all checker plugins.", true, true);

    args.parse(argc, argv);
    if (args.exists("help"))
    {
        cout << args.usage() << "\n";
        return 0;
    }
    if (args.exists("version"))
    {
        cout << argv[0] << " version: " << VERSION << "\n";
        return 0;
    }
    auto graphdef = args.get<string>("graphdef");
    if (!utils::file_exists(graphdef))
    {
        ERROR() << "Cannot Open file:" << graphdef;
        return -1;
    }
    bool plugin_enable = args.exists("disable_checker_plugins") ? false : true;
    aipubt::IRChecker checker(graphdef, nullptr);
    bool valid_IR = checker.check_all(plugin_enable);

    return valid_IR ? 0 : -1;
}

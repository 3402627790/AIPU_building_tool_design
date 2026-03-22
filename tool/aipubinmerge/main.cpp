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
 * @Last Modified time: 2023-04-13 16:01:45
 */
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>

#include <builder/aipu_binfile.h>
#include <builder/builder.h>
#include <builder/mem_allocator.h>
#include <builder/profiler.h>
#include <builder/scheduler.h>
#include <builder/target.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <gcompiler_api.h>
#include <tools/argparser.h>
#include <utils/string_utils.h>
#include <utils/utils.h>

using namespace aipubt;
DECLARE_VERSION(aipubinmerge);

#define FORMAT_ADDR(value) "0x" << std::setw(8) << std::setfill('0') << std::hex << value << std::setfill(' ') << std::dec

int main(int argc, const char **argv)
{
    ArgParser args;
    args.add_argument("aipubins", "The aipu excutable binaries to be merged, split with comma ','.");
    args.add_argument("-o", "--output", "The output aipubin file name.", "output.aipubin");
    args.add_argument("-v", "--version", "print the version.", true, true);
    args.add_argument("-h", "--help", "print the help message.", true, true);

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
    auto aipubins = utils::split_string(args.get<std::string>("aipubins"), ',');
    if (aipubins.size() < 2)
    {
        cout << "you need set at least two aipubin's path for merging.\n";
        cout << args.usage() << "\n";
        return 1;
    }
    AIPUBin out;
    out.load(aipubins[0]);
    for (size_t i = 1; i < aipubins.size(); ++i)
    {
        AIPUBin bin2, temp;
        bin2.load(aipubins[i]);
        AIPUBin::merge(temp, out, bin2);
        out = temp;
    }
    out.write(args.get<std::string>("output"));
    return 0;
}

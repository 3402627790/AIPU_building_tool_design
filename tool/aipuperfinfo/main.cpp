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
 * @Author: Yuchou Gan
 * @Date: 2023-04-10 19:37:42
 * @Last Modified by: Yuchou Gan
 * @Last Modified time: 2023-04-10 19:37:42
 */

#include "json.hpp"
#include <fstream>
#include <iostream>
using namespace nlohmann;

namespace perfdata{
json get_performance_data(std::string graph_json_path, std::string binary_path);
};

int main(int argc, const char **argv)
{
    // This is an example how to use get_performance_data
    // return json structure would be the same with python version
    // get_profiler_data() return dict.
    if (argc < 3)
    {
        std::cerr << "Expect to accept 2 arguments graph_json path and profiler_bin path" << std::endl;
        return -1;
    }
    std::string json_file = argv[1];
    std::string bin_file = argv[2];

    json out = perfdata::get_performance_data(json_file, bin_file);

    if (out.empty())
    {
        return -1;
    }
    std::string out_file;
    if (argc == 4)
    {
        out_file = argv[3];
    }
    else
    {
        out_file = std::string(out["graph_name"]) + ".json";
    }
    std::ofstream ss(out_file);
    ss << out.dump(2);
    return 0;
}

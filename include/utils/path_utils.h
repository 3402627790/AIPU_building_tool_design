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
#ifndef UTILS_PATH_UTILS_H
#define UTILS_PATH_UTILS_H
#include <string>
#include <vector>

namespace utils
{
std::string get_exe_path();
std::string get_exe_dir();
std::string get_temp_dir();
std::string get_home_dir();
std::vector<std::string> list_directory(const std::string &dir);
bool path_exists(const std::string &path);
std::string get_abspath(const std::string &path);
std::string dir_name(const std::string &path);
std::string truncate_path(const std::string &path);
bool mkdir(const std::string &path);
std::string specified_temp_dir(std::string dir = "");
}; // namespace utils

#endif //!UTILS_PATH_UTILS_H
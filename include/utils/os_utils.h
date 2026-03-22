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
 * @Date: 2020-05-13 11:49:49 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-05-13 13:32:37
 */
#ifndef UTILS_OS_UTILS_H
#define UTILS_OS_UTILS_H
#include <functional>
#include <string>
namespace utils
{
void print_backtrace(unsigned int start);

/**
 * @brief Get the env variable
 * 
 * @param name 
 * @return const std::string& empty string if no present
 */
const std::string get_env(const std::string &name);

const std::string get_env(const char *name);

void register_atexit(const std::string &name, std::function<void()> f);
void unregister_atexit(const std::string &name);
void clear_atexit();

}; //namespace utils

#endif //UTILS_OS_UTILS_H
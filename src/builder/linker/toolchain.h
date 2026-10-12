//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang 
 * @Date: 2021-08-09 09:38:52 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-03-22 10:24:10
 */
#include <core/logger.h>
#include <dlfcn.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace aipubt
{
int aipuas(const std::string &s, const std::string &obj);
int aipuld(const std::vector<std::string> &objs, const std::string &out);
int aipuocc(const std::string &c_file, const std::string &out);
int cpsld(const std::vector<std::string> &objs, const std::string &out);
int cpscc(const std::string &c_file, const std::string &out);
int cpshostcc(const std::string &c_file,
              const std::vector<std::string> &objs,
              const std::string &out);
std::string get_devkit_path();
} //namespace aipubt
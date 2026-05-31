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
 * @Date: 2020-04-09 12:53:41
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-08-11 11:39:30
 */
#include "utils/file_utils.h"
#include "utils/path_utils.h"
#include <dirent.h>
#include <fstream>
#include <limits.h>
#include <pwd.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h> //readlink
#include <unistd.h>
namespace utils
{
bool file_exists(const std::string &file)
{
    return path_exists(file);
}

bool remove_file(const std::string &filename)
{
    return (remove(filename.c_str()) != 0);
}
void remove_dir(const std::string &mpath)
{
    std::string cmd = "rm -rf " + mpath;
    system(cmd.c_str());
}

bool copy_file(const std::string &src, const std::string &dest, bool replace)
{
    if (!file_exists(src))
    {
        return false;
    }
    if (file_exists(dest) && !replace)
    {
        return true;
    }
    std::ifstream fsrc(src, std::ios::binary);
    std::ofstream fdest(dest, std::ios::binary);
    fdest << fsrc.rdbuf();
    return true;
}

void stream_copy(std::istream &in, std::ostream &out, std::size_t count)
{
    const std::size_t buffer_size = 4096;
    char buffer[buffer_size];
    while (count > buffer_size)
    {
        in.read(buffer, buffer_size);
        out.write(buffer, buffer_size);
        count -= buffer_size;
    }

    in.read(buffer, count);
    out.write(buffer, count);
}
void stream_fill(std::ostream &out, char b, std::size_t count)
{
    const std::size_t buffer_size = 4096;
    char buffer[buffer_size];
    memset(buffer, b, buffer_size);
    while (count > buffer_size)
    {
        out.write(buffer, buffer_size);
        count -= buffer_size;
    }
    out.write(buffer, count);
}
}; // namespace utils
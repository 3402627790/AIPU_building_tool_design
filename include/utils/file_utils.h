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
 * @Date: 2020-04-09 12:53:12 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-08-11 10:31:35
 */
#include <string>
namespace utils
{

bool file_exists(const std::string &file);
bool remove_file(const std::string &filename);
// using force mode to remove all files in dir
void remove_dir(const std::string &path);
bool copy_file(const std::string &src, const std::string &dest, bool replace = true);
void stream_copy(std::istream &in, std::ostream &out, std::size_t count);
void stream_fill(std::ostream &out, char b, std::size_t count);
}; // namespace utils
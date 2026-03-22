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
 * @Date: 2020-06-04 14:38:47 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-06-04 15:47:19
 */
#ifndef BUILDER_LINKER_ELFREADER
#define BUILDER_LINKER_ELFREADER
#include <cstdio>
#include <cstdlib>
#include <fcntl.h> /* O_RDONLY */
#include <fcntl.h>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h> /* For the size of the file. , fstat */
#include <sys/types.h>
#include <unistd.h>
#include <utility>

namespace aipubt
{
class ELFReader
{
public:
    static bool is_elf_file(const std::string &elfname);
    static bool is_elf_file(const unsigned char *content);
    static const std::string get_kernel_func(const std::string &elfname);
    static std::pair<char *, size_t> get_text_section(const std::string &elfname);
    static std::pair<char *, size_t> get_text_section(const unsigned char *content);
};
} //namespace aipubt
#endif //!BUILDER_LINKER_ELFREADER
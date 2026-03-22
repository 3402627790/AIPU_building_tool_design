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
 * @Date: 2020-06-04 15:03:32 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-06-04 16:12:43
 */

#include "test_builder.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <builder/linker/elfreader.h>
#include <gcompiler_api.h>
#include <utils/string_utils.h>
using namespace aipubt;

void test_elfreader()
{
    std::string name = "test_elf.o";

    if (ELFReader::is_elf_file(name))
    {
        auto ret = ELFReader::get_text_section(name);
        std::ofstream ofs("test.bin", std::ios::binary);
        ofs.write(ret.first, ret.second);
        ofs.close();
    }
    else
    {
        ERROR() << "Not a ELF file: " << name;
    }
}

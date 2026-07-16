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
 * @Author: Kyle Huang
 * @Date: 2026-04-24 14:27:29
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:29
 * */
#ifndef BUILDER_CODE_GENERATROR_H
#define BUILDER_CODE_GENERATROR_H

#include "builder/cps_elf.h"
#include "core/logger.h"
#include <vector>

namespace aipubt
{
class CodeGenerator
{
public:
    void set_kernel_lib(const std::map<std::string, std::string> &kernel_lib);
    void code_gen(HybridELFPtr elf);
    const std::string get_cpshost_header() const;

private:
    void gen_host_code();
    const std::string get_cpshost_code();

    std::map<std::string, std::string> m_kernel_lib;
    std::stringstream m_host_code;
};
} // namespace aipubt
#endif //!BUILDER_CODE_GENERATROR_H
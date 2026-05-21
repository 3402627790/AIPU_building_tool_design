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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Kyle Huang
 * @Date: 2026-05-06 10:44:17
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-05-06 17:15:09
 */
#ifndef BUILDER_LINKER_LINKERCPS_H
#define BUILDER_LINKER_LINKERCPS_H
#include "linker.h"
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

namespace aipubt
{

class LinkerCps : public LinkerImpl
{

public:
    LinkerCps();
    ~LinkerCps();
    // set the generated code
    virtual void set_code(const std::string &code) override
    {
        m_asm = code;
    }

    //return the main offset, now we return the first lib's offset
    virtual void set_codes(std::vector<std::vector<std::string>> &codes) override
    {
        m_codes = codes;
    }
    virtual void set_objects(std::vector<std::string> &objs_filename) override
    {
        m_objs = objs_filename;
    }
    virtual uint32_t link(std::ostream &oss) override;

private:
    std::string m_main;
    std::string m_workdir;
    std::vector<std::vector<std::string>> m_codes;
    std::vector<std::string> m_objs;
    std::string m_asm;
};
} // namespace aipubt

#endif //! BUILDER_LINKER_LINKERCPS_H
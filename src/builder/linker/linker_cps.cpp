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
 * @Author: Neo Wang
 * @Date: 2020-05-27 10:45:52
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-11-15 16:44:54
 */
#include "builder/linker/linker_cps.h"
#include "builder/linker/toolchain.h"
#include "core/config.h"
#include "utils/utils.h"

namespace aipubt
{
LinkerCps::LinkerCps()
{
    if (!m_workdir.size())
    {
        if (BuilderConfig::exist(Constants::global_cwd))
        {
            m_workdir = BuilderConfig::get<std::string>(Constants::global_cwd) + "/ld";
            while (utils::path_exists(m_workdir)) // get a brand new work dir
            {
                m_workdir = m_workdir + "/" + utils::get_uuid();
            }
            utils::mkdir(m_workdir);
        }
        else
        {
            m_workdir = utils::get_temp_dir() + "/" + utils::get_uuid();
            while (utils::path_exists(m_workdir)) // get a brand new work dir
            {
                m_workdir = utils::get_temp_dir() + "/" + utils::get_uuid();
            }
            utils::mkdir(m_workdir);
        }
    }
}

uint32_t LinkerCps::link(std::ostream &oss)
{
    std::string elf = m_workdir + "/" + "main.elf";
    if (cpsld(m_objs, elf) != 0)
    {
        ERROR() << "Build Subgraph error! please double check the kernel function is correct.";
        return 0;
    }
    oss << elf;
    return 0;
}

LinkerCps::~LinkerCps()
{
    if (m_workdir.size())
    {
#if RELEASE_VERSION
        utils::remove_dir(m_workdir);
#endif
    }
}
} // namespace aipubt
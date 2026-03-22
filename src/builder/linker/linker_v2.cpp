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
#include "builder/linker/linker_v2.h"
#include "../text_generator.h"
#include "ZY.h"
#include "builder/ds_code_generator.h"
#include "builder/linker/toolchain.h"
#include "builder/profiler.h"
#include "builder/target.h"
#include "core/config.h"
#include "toolchain.h"
#include "utils/utils.h"
namespace aipubt
{

using namespace AsmZY;
LinkerV2::LinkerV2()
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

LinkerV2::~LinkerV2()
{
    if (m_workdir.size())
    {
#if RELEASE_VERSION
        utils::remove_dir(m_workdir);
#endif
    }
}

uint32_t LinkerV2::link(std::ostream &oss)
{
    std::string s = m_workdir + "/" + "main.s";
    std::string o = m_workdir + "/" + "main.o";
    std::string elf = m_workdir + "/" + "main.elf";
    DEBUG() << "working dir " << m_workdir;
    std::ofstream objfile(s);
    objfile << m_asm;
    objfile.close();
    // system(("cp " + s + " main.s").c_str());
    if (!objfile)
    {
        ERROR() << "Write linker asm to file " << s << " failed.";
        return 0;
    }
    std::ifstream ifile(s);
    // DEBUG() << "gen code:\n"
    //         << ifile.rdbuf();
    if (aipuas(s, o) != 0)
    {
        ERROR() << "Build Subgraph error! please double check the kernel function is correct.";
        return 0;
    }
    auto profiler = AIPUBuilder::get_profiler();
    {
        std::string perf_c = profiler->interface()->get_function("intrusive_implement")(Any());
        std::string perf_c_file = m_workdir + "/" + "intrusive_impl.c";
        std::string perf_o_file = m_workdir + "/" + "intrusive_impl.o";
        std::ofstream objfile(perf_c_file);
        objfile << perf_c;
        objfile.close();
        if (aipuocc(perf_c_file, perf_o_file) != 0)
        {
            ERROR() << "Build Subgraph error! please double check the perf intrusive impelment is correct.";
            return 0;
        }
        m_objs.push_back(perf_o_file);
    }
    if (BuilderConfig::get<bool>(Constants::enable_cl_opt))
    {
        std::string cl_file = m_workdir + "/" + "ds_impl.cl";
        std::string o_file = m_workdir + "/" + "ds_impl.o";
        std::ofstream objfile(cl_file);
        objfile << get_ds_cl_code().str();
        objfile.close();
        if (aipuocc(cl_file, o_file) != 0)
        {
            ERROR() << "Build Subgraph error! please double check the dynamic shape function is correct.";
            return 0;
        }
        m_objs.push_back(o_file);
    }
    static const std::vector<std::string> builtin_lib = {"aiffcllib.a", "tpccllib.a", "aipubt_internal_lib.o", "libcob.a"};
    for (auto inlib : builtin_lib)
    {
        auto fullpath = get_lib_full_filename(inlib);
        if (fullpath != inlib || utils::file_exists(fullpath))
        {
            if (!contains(m_objs, fullpath))
            {
                m_objs.push_back(fullpath);
            }
        }
    }
    std::vector<std::string> objs = {o};
    std::vector<std::string> libs;
    for (uint32_t i = 0; i < m_objs.size(); ++i)
    {
        if (utils::endswith(m_objs[i], ".a"))
        {
            if (contains(objs, m_objs[i]))
            {
                continue;
            }
            if (!contains(libs, m_objs[i]))
            {
                libs.push_back(m_objs[i]);
            }
        }
        else
        {
            objs.push_back(m_objs[i]);
        }
    }
    objs.insert(objs.end(), libs.begin(), libs.end());

    DEBUG() << "linking lib:";
    std::vector<std::string> ld_objs;
    for (auto &obj : objs)
    {
        // reduce continuous duplicated objs
        if (ld_objs.size() && ld_objs.back() == obj)
        {
            continue;
        }
        DEBUG() << obj;
        ld_objs.push_back(obj);
    }
    if (aipuld(ld_objs, elf) != 0)
    {
        ERROR() << "Build Subgraph error! please double check the kernel function is correct.";
        return 0;
    }
    oss << elf;
    // std::ifstream ifs(elf);
    // ifs>>oss;
    return 0;
}
} // namespace aipubt
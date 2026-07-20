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
 * @Date: 2026-04-24 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:33
 */
#include "builder/code_generator.h"
#include "builder/ar/ar_reader.h"
#include "builder/ds_code_generator_cx.h"
#include "builder/linker/linker_cps.h"
#include "builder/linker/toolchain.h"
#include "builder/text_generator.h"
#include "core/config.h"
#include "core/constants.h"
#include "utils/utils.h"

namespace aipubt
{

void CodeGenerator::set_kernel_lib(const std::map<std::string, std::string> &kernel_lib)
{
    m_kernel_lib = kernel_lib;
}

void CodeGenerator::code_gen(HybridELFPtr elf)
{
    LinkerCps linker;
    auto workdir = linker.workdir();

    m_host_code.clear();
    // TODO: gen host code and link host lib

    auto host_code = get_cpshost_code();
    if (host_code.size())
    {
        auto c_file = workdir + "/" + "cpshost.cpp";
        auto o_file = workdir + "/" + "cpshost.o";
        std::ofstream objfile(c_file);
        objfile << host_code;
        objfile.close();
        if (cpshostcc(c_file, o_file) != 0)
        {
            FATAL() << "error occurred while compiling the host code!";
        }
        elf->load(o_file);
    }

    std::set<std::string> libs;

    for (auto &k : m_kernel_lib)
    {
        INFO() << "kernel: " << k.first << " lib: " << k.second;
        bool has_lib = false;
        for (auto lib : libs)
        {
            if (lib == k.second)
            {
                has_lib = true;
                break;
            }

            ARFileReader arf(lib);
            if (arf.is_ar_file() && arf.has_file(k.second))
            {
                has_lib = true;
                break;
            }
        }
        if (has_lib)
        {
            continue;
        }

        if (utils::file_exists(k.second))
        {
            libs.insert(k.second);
        }
        else //find from .a files
        {
            bool found = false;
            for (auto afile : get_all_afiles())
            {
                if (afile.has_file(k.second))
                {
                    libs.insert(afile.name());
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                ERROR() << "cannot find " << k.second;
            }
        }
    }

    std::vector<std::string> objs(libs.begin(), libs.end());

    auto ds_code = get_ds_device_code();
    if (ds_code.size())
    {
        auto c_file = workdir + "/" + "ds_impl.cps";
        auto o_file = workdir + "/" + "ds_impl.o";
        std::ofstream objfile(c_file);
        objfile << ds_code;
        objfile.close();
        if (cpscc(c_file, o_file) != 0)
        {
            FATAL() << "error occurred while compiling the dynamic shape code!";
        }
        objs.push_back(o_file);
    }

    linker.set_objects(objs);
    std::stringstream oss;
    linker.link(oss);
    auto cpself = elf->get_cpself();
    cpself->load(oss.str());
}

const std::string CodeGenerator::get_cpshost_header() const
{
    return R"(
#pragma once
#include "modules/btrt/launcher.h"

extern "C" {

aipubt::rt::Launcher *createLauncher();

}
)";
}

const std::string CodeGenerator::get_cpshost_code()
{
    auto header = get_cpshost_header();
    utils::replace(header, "#pragma once", "");
    auto host_code = m_host_code.str();
    auto ds_host_code = get_ds_host_code();
    std::stringstream oss;
    oss << header
        << "typedef unsigned int uint;\n"
        << "typedef unsigned long long ulonglong;\n\n"
        << host_code;
    if (ds_host_code.size())
    {
        oss << ds_host_code;
    }
    else
    {
        oss << "void eval_output_shape(uint *) {}\n";
    }
    oss << R"(
namespace aipubt::rt {

class HybridLauncher : public Launcher
{
public:
    int run() override
    {
        int ret = Launcher::run();
        if (ret == 0)
        {
            auto pp = get_param_base();
            eval_output_shape((uint *)pp);
        }
        return ret;
    }

    int run_async(AsyncCallback callback,
                  const std::vector<tensor_data_t> &inputs = {},
                  const std::map<std::string, int> &symbols = {}) override
    {
        return Launcher::run_async(callback, inputs, symbols);
    }
};

} // namespace aipubt::rt

#include <dlfcn.h>
#include <stdio.h>

extern "C" {

aipubt::rt::Launcher *createLauncher()
{
    Dl_info info;
    if (dladdr((void *)createLauncher, &info) == 0)
    {
        return NULL;
    }
    if (!info.dli_fname)
    {
        return NULL;
    }
    // printf("Library name: %s\n", info.dli_fname);
    auto launcher = new aipubt::rt::HybridLauncher();
    if (launcher->load_from_file(info.dli_fname) != 0)
    {
        delete launcher;
        return NULL;
    }
    return launcher;
}

}
)";
    return oss.str();
}

} // namespace aipubt
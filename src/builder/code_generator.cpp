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
#include "builder/assembler.h"
#include "builder/ds_code_generator_cx.h"
#include "builder/linker/linker_cps.h"
#include "builder/linker/toolchain.h"
#include "builder/text_generator.h"
#include "core/config.h"
#include "core/constants.h"
#include "utils/utils.h"

namespace aipubt
{

void CodeGenerator::set_graph(GraphPtr g)
{
    m_graph = g;
}

void CodeGenerator::set_device_lib(const std::map<std::string, std::string> &kernel_lib)
{
    m_device_lib = kernel_lib;
}

void CodeGenerator::set_host_lib(const std::map<std::string, std::string> &kernel_lib)
{
    m_host_lib = kernel_lib;
}

void CodeGenerator::gen_device_code(CpsElfPtr elf)
{
    LinkerCps linker;
    auto workdir = linker.workdir();

    std::set<std::string> libs;

    for (auto &k : m_device_lib)
    {
        DEBUG() << "kernel: " << k.first << " lib: " << k.second;
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
        else // find from .a files
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
            FATAL() << "An error occurred while compiling the dynamic shape code!";
        }
        objs.push_back(o_file);
    }

    linker.set_objects(objs);
    std::stringstream oss;
    linker.link(oss);
    elf->load(oss.str());
}

void CodeGenerator::gen_host_code(HybridELFPtr elf)
{
    LinkerCps linker;
    auto workdir = linker.workdir();

    gen_host_funcs();

    std::set<std::string> libs;
    auto lib_paths = get_all_lib_paths();
    for (auto &kv : m_host_lib)
    {
        DEBUG() << "kernel: " << kv.first << " lib: " << kv.second;
        if (utils::file_exists(kv.second))
        {
            libs.insert(utils::get_abspath(kv.second));
        }
        else // find from lib paths
        {
            bool found = false;
            for (auto path : lib_paths)
            {
                auto full_name = path + "/" + kv.second;
                if (utils::file_exists(full_name))
                {
                    libs.insert(utils::get_abspath(full_name));
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                ERROR() << "cannot find " << kv.second;
            }
        }
    }
    std::vector<std::string> objs(libs.begin(), libs.end());

    auto host_code = get_cpshost_code();
    if (host_code.size())
    {
        auto c_file = workdir + "/" + "cpshost.cpp";
        auto o_file = workdir + "/" + "cpshost.o";
        std::ofstream objfile(c_file);
        objfile << host_code;
        objfile.close();
        if (cpshostcc(c_file, objs, o_file) != 0)
        {
            FATAL() << "An error occurred while compiling the host code!";
        }
        elf->load(o_file);
    }
}

void CodeGenerator::gen_host_funcs()
{
    auto param_root = m_graph->global_tensors()["param"];
    auto param_root_object = param_root->memory_offset()->get_base_object();

    auto &oss = m_host_code;
    oss.clear();

    for (auto s : m_graph->nodes())
    {
        auto sgn = SubGraphNode::from_node(s);
        if (sgn)
        {
            auto sg = sgn->subgraph();
            for (auto n : sg->nodes())
            {
                if (!n->attrs().count(Constants::plugin_impl))
                {
                    continue;
                }

                auto kernel_infos = n->get_attr<KernelInfos>(Constants::kernel_infos);
                auto param_bases = n->get_attr<std::vector<PlaceholderPtr>>(Constants::param_bases);
                auto is_host_impl = n->get_attr<bool>(Constants::is_host_impl);
                if (is_host_impl)
                {
                    for (size_t i = 0; i < kernel_infos.size(); ++i)
                    {
                        auto &info = kernel_infos[i];
                        auto func_name = "aipubt_" + utils::verify_name(n->name()) + "_" + std::to_string(i);
                        auto param_base = param_bases[i];
                        auto param_base_object = param_base->memory_offset();
                        auto param_offset = param_base_object->offset() - param_root_object->offset();
                        param_offset += 8 * 4; // real param
                        gen_host_func(oss, func_name, info, param_offset);
                    }
                }
            }
        }
    }
}

void CodeGenerator::gen_host_func(std::stringstream &oss,
                                  const std::string &func_name,
                                  KernelInfo &info,
                                  uint64_t param_offset)
{
    auto tab = std::string(4, ' ');
    size_t num_params = 0;
    oss << "extern \"C\" void " << info.kernel_name << "(";
    for (size_t i = 0; i < info.args.size(); ++i)
    {
        auto &p = info.args[i];
        if (p.surface)
        {
            ERROR() << "unsupported surface parameter!";
        }
        if (p.tensor || p.const_tensor || p.param_list)
        {
            num_params += 2;
            oss << "void *";
        }
        else
        {
            num_params += 1;
            oss << "int";
        }
        if (i != info.args.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << ");\n\n";

    auto pb = "pb";
    auto pp = "pp";
    auto tpp = "tpp";
    oss << "void " << func_name << "(uint64_t " << pb << ")\n{\n"
        << tab << "uint64_t " << pp << " = " << pb << " + " << param_offset << ";\n"
        << tab << "uint32_t *" << tpp << " = new uint32_t[" << num_params << "];\n"
        << tab << "aipubt::rt::Allocator::copy("
        << "(void *)" << tpp << ", " << pp << ", " << num_params * 4 << ");\n";

    size_t pp_offset = 0;
    std::vector<std::string> args;
    std::vector<std::pair<std::string, std::string>> ptrs;
    std::vector<size_t> sizes;
    std::vector<bool> wb;
    for (size_t i = 0; i < info.args.size(); ++i)
    {
        auto &p = info.args[i];
        auto arg = "arg" + std::to_string(i);
        if (p.tensor || p.const_tensor || p.param_list)
        {
            size_t mem_size = 0;
            if (p.tensor)
            {
                mem_size = p.tensor->mem_size();
            }
            else if (p.const_tensor)
            {
                mem_size = p.const_tensor->mem_size();
            }
            else
            {
                mem_size = p.param_list->mem_size();
            }
            auto dp = "dp" + std::to_string(i);
            oss << tab << "char *" << arg << " = new char[" << mem_size << "];\n"
                << tab << "uint64_t " << dp << " = *(uint64_t *)(" << tpp << " + " << pp_offset << ");\n"
                << tab << "aipubt::rt::Allocator::copy("
                << "(void *)" << arg << ", " << dp << ", " << mem_size << ");\n";
            ptrs.push_back({arg, dp});
            sizes.push_back(mem_size);
            wb.push_back(p.tensor ? true : false);
            pp_offset += 2;
            // TODO: support param_list
        }
        else
        {
            oss << tab << "uint32_t " << arg << " = " << tpp << "[" << pp_offset << "];\n";
            pp_offset += 1;
        }
        args.push_back(arg);
    }

    oss << tab << info.kernel_name << "(";
    for (size_t i = 0; i < info.args.size(); ++i)
    {
        auto &p = info.args[i];
        auto &arg = args[i];
        if (p.tensor || p.const_tensor || p.param_list)
        {
            oss << "(void *)" << arg;
        }
        else
        {
            oss << "(int)" << arg;
        }
        if (i != info.args.size() - 1)
        {
            oss << ", ";
        }
    }
    oss << ");\n";

    for (size_t i = 0; i < ptrs.size(); ++i)
    {
        auto &p = ptrs[i];
        if (wb[i])
        {
            oss << tab << "aipubt::rt::Allocator::copy("
                << p.second << ", (void *)" << p.first << ", " << sizes[i] << ");\n";
        }
        oss << tab << "delete[] " << p.first << ";\n";
    }
    oss << tab << "delete[] " << tpp << ";\n"
        << "}\n\n";
}

const std::string CodeGenerator::get_cpshost_header() const
{
    return R"(
#pragma once
#include <modules/btrt/launcher.h>

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
    oss << header;
    if (host_code.size())
    {
        oss << "#include <modules/btrt/allocator.h>\n\n"
            << host_code;
    }
    if (ds_host_code.size())
    {
        oss << "typedef unsigned int uint;\n"
            << "typedef unsigned long long ulonglong;\n\n"
            << ds_host_code;
    }
    oss << R"(
namespace aipubt::rt {

class HybridLauncher : public Launcher
{
public:
    int run() override
    {
        return Launcher::run();
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
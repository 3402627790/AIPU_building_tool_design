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
 * @Date: 2020-01-29 14:23:29
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-01-05 16:47:56
 */

#include "text_generator.h"
#include "builder/ar/ar_reader.h"
#include "builder/linker/elfreader.h"
#include "builder/readonly_param.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "utils/utils.h"
#include <fstream>
#include <mutex>
#include <string>

#define AIPU_LIB_ARCHIVE_FILE "aipulib.a"
namespace aipubt
{
struct TEXTCacher
{
    ~TEXTCacher()
    {
        for (auto kv : text_cache)
        {
            if (kv.second.second > 0 && kv.second.first)
            {
                delete[] kv.second.first;
                kv.second.first = nullptr;
            }
        }
    }
    std::map<std::string, BuilderText> text_cache;
};
static TEXTCacher TEXTCACHER;

std::map<std::string, BuilderText> &get_text_cache_map()
{
    return TEXTCACHER.text_cache;
}

static std::vector<std::string> get_all_lib_pathes()
{
    static std::vector<std::string> ret;
    if (ret.size())
    {
        return ret;
    }
    ret = BuilderConfig::get<std::vector<std::string>>("Lib");
    char *aipulibpath = getenv(Constants::LIB_PATH);
    if (aipulibpath)
    {
        for (auto p : utils::split_string(aipulibpath, ':'))
        {
            ret.push_back(p);
        }
    }
    ret.push_back(".");
    ret.push_back("lib/");
    ret.push_back("../lib/");
    ret.push_back(utils::get_exe_dir() + "/../lib/");
    return ret;
}
static const std::vector<std::string> search_lib(const std::vector<std::string> &available_path, const std::string &suffix = ".a")
{
    std::vector<std::string> ret;
    for (auto path : available_path)
    {
        if (utils::path_exists(path))
        {
            for (auto f : utils::list_directory(path))
            {
                if (utils::endswith(f, suffix))
                {
                    ret.push_back(path + "/" + f);
                }
            }
        }
    }
    return ret;
}
static std::once_flag load_afile_flag;
const std::vector<ARFileReader> &get_all_afiles()
{
    static std::vector<ARFileReader> afiles;
    std::call_once(load_afile_flag, []() {
        std::vector<std::string> lib_path = {
            ".",
            "lib/",
            "../lib/", // cwd
            utils::get_exe_dir() + "/../lib/",
            utils::get_temp_dir()};
        char *aipulibpath = getenv(Constants::LIB_PATH);
        // first options
        for (auto name : search_lib(BuilderConfig::get<std::vector<std::string>>("Lib"), ".a"))
        {
            ARFileReader arf(name);
            if (arf.is_ar_file())
            {
                afiles.push_back(arf);
            }
        }
        if (aipulibpath)
        {
            // second env
            for (auto name : search_lib(utils::split_string(aipulibpath, ':'), ".a"))
            {
                ARFileReader arf(name);
                if (arf.is_ar_file())
                {
                    afiles.push_back(arf);
                }
            }
        }

        // third default
        for (auto name : search_lib(lib_path, ".a"))
        {
            ARFileReader arf(name);
            if (arf.is_ar_file())
            {
                afiles.push_back(arf);
            }
        }
        for (auto arf : afiles)
        {
            DEBUG() << "load *.a: " << arf.name();
        }
    });
    return afiles;
}
inline bool is_in_debug_mode()
{
    return BuilderConfig::get<bool>("DEBUG");
}
inline std::string process_debug_filename(const std::string &filename)
{
    auto ret = filename; // copy
    ret = utils::replace(ret, ".bin", "d.bin");
    return ret;
}
void cache_text_content(const std::string &filename, BuilderText text)
{
    if (text.second & 0xF)
    {
        ERROR() << filename << ": invalid text size, 0x" << std::hex << text.second
                << ". expected 16 bytes aligned";
        return;
    }
    if (TEXTCACHER.text_cache.count(filename))
    {
        return;
    }
    TEXTCACHER.text_cache[filename] = text;
}
std::string complete_text_code_name(const std::string &base_name)
{
    auto target = get_target();
    std::string suffix = "_" + target->name();
    utils::tolower(suffix);
    if (target->isainfo().type == ISA_ZY)
    {
        switch (target->chipinfo().id)
        {
        case 904: // 0904
            suffix = "";
            break;
        case 701:
            suffix = "_zy_sc003";
            break;
        case 1002:
            suffix = "_zy_sc002";
            break;
        default:
            break;
        }
    }
    auto new_name = base_name;
    return utils::replace(new_name, ".bin", suffix + ".bin");
    // return base_name+suffix;
}

const std::string get_lib_full_filename(const std::string &libname)
{
    auto lib_pathes = get_all_lib_pathes();
    //read file
    for (auto path : lib_pathes)
    {
        std::string fullname = path + "/" + libname;
        if (utils::file_exists(fullname))
        {
            return fullname;
        }
    }
    for (auto afile : get_all_afiles())
    {
        if (afile.has_file(libname))
        {
            return afile.name();
        }
    }
    return libname;
}

const std::string get_kernel_func_name(const std::string &libname)
{

    return ELFReader::get_kernel_func(get_lib_full_filename(libname));
}
static BuilderText get_text_content_by_filename_internal(const std::string &filename)
{
    if (TEXTCACHER.text_cache.count(filename))
    {
        return TEXTCACHER.text_cache[filename];
    }
    auto lib_pathes = get_all_lib_pathes();

    //read file
    for (auto path : lib_pathes)
    {
        std::ifstream infile;
        std::string fullname = path + "/" + filename;
        infile.open(fullname, std::ios::binary | std::ios::in);
        if (infile)
        {
            DEBUG() << "load external *.bin: " << fullname;
            if (ELFReader::is_elf_file(fullname))
            {
                return ELFReader::get_text_section(fullname);
            }
            infile.seekg(0, std::ios::end);
            size_t length = infile.tellg();
            infile.seekg(0, std::ios::beg);
            char *buffer = new char[length];
            infile.read(buffer, length);
            infile.close();
            BuilderText ret = {buffer, length};
            // test if is a elf
            cache_text_content(filename, ret);
            return ret;
        }
        infile.close();
    }
    // if there is a elf file
    auto funcname = filename;
    utils::replace(funcname, ".bin", "");
    for (auto afile : get_all_afiles())
    {
        auto lib = afile.get_file_content(filename);
        if (lib.second == 0)
        {
            if (afile.symbol_table().count(funcname))
            {
                lib = afile.get_file_content_by_symbol(funcname);
            }
        }
        if (lib.first && lib.second)
        {
            // test if is a elf
            if (ELFReader::is_elf_file(reinterpret_cast<const unsigned char *>(lib.first)))
            {
                //TODO support all symbol is elf file
                auto ptr = lib.first;
                lib = ELFReader::get_text_section(reinterpret_cast<const unsigned char *>(lib.first));
                delete[] ptr;
            }
            cache_text_content(filename, lib);
            return lib;
        }
    }
    return {};
}

BuilderText get_text_content_by_filename(const std::string &filename)
{
    if (filename.empty())
    {
        return BuilderText();
    }
    if (is_in_debug_mode())
    {
        auto ret = get_text_content_by_filename_internal(process_debug_filename(filename));
        if (ret.first)
        {
            return ret;
        }
    }
    auto ret = get_text_content_by_filename_internal(filename);
    if (!ret.first)
    {
        ERROR() << "Failed to open library:" << filename;
    }
    return ret;
}

text_generator_func get_text_generator(const OpType t)
{
    if (!TEXTGeneratorRegister::get_text_generator_map().count(t))
    {
        return nullptr;
    }
    return TEXTGeneratorRegister::get_text_generator_map()[t];
}

code_generator_func get_code_generator(const OpType t)
{
    if (!TEXTGeneratorRegister::get_code_generator_map().count(t))
    {
        return nullptr;
    }
    return TEXTGeneratorRegister::get_code_generator_map()[t];
}

kernel_generator_func get_kernel_generator(const OpType t)
{
    if (!KernelGeneratorRegister::get_kernel_generator_map().count(t))
    {
        return nullptr;
    }
    return KernelGeneratorRegister::get_kernel_generator_map()[t];
}

TEXTGeneratorRegister::TEXTGeneratorRegister(OpType t, text_generator_func func, code_generator_func code)
{
    (get_text_generator_map())[t] = func;
    (get_code_generator_map())[t] = code;
}

KernelGeneratorRegister::KernelGeneratorRegister(OpType t, kernel_generator_func func)
{
    (get_kernel_generator_map())[t] = func;
}
}; // namespace aipubt

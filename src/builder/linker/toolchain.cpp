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
 * @Date: 2021-08-09 09:38:52
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-03-22 10:52:48
 */
#include "toolchain.h"
#include "builder/target.h"
#include "core/constants.h"
#include "utils/utils.h"
#include <core/logger.h>
#include <dlfcn.h>
#include <link.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

typedef int (*EntryFunc)(int, char **);

namespace aipubt
{
static EntryFunc m_aipuas = nullptr;
static EntryFunc m_aipuld = nullptr;
static EntryFunc m_aipucc = nullptr;

static void load_toolchain();

static const std::string &get_aipuas()
{
    load_toolchain();
    static std::string exe_path;
    if (exe_path.size())
    {
        return exe_path;
    }
    if (m_aipuas != nullptr)
    {
        exe_path = "aipuoas";
        return exe_path;
    }
    //1. try to use env
    auto env = utils::get_env(Constants::AIPUOAS);
    if (!env.empty())
    {
        exe_path = env;
        return exe_path;
    }
    //2. checking path
    auto path = utils::get_env("PATH");
    for (auto d : utils::split_string(path, ':'))
    {
        if (utils::file_exists(d + "/aipuoas"))
        {
            exe_path = "aipuoas";
            return exe_path;
        }
    }
#if RELEASE_VERSION
    WARN() << "Can NOT find aipuoas in PATH. Please make sure you have added it to PATH.";
    exe_path = "aipuoas";
#else
    //3. use default: push the dir to path
    std::string lpath = path + "/project/ai/scratch01/toolchain/llvm/binary/bin/";
    setenv("PATH", lpath.c_str(), 1);
    exe_path = "/project/ai/scratch01/toolchain/llvm/binary/bin/aipuoas";
#endif
    return exe_path;
}

static const std::string &get_aipuld()
{
    load_toolchain();
    static std::string exe_path;
    if (exe_path.size())
    {
        return exe_path;
    }
    if (m_aipuld != nullptr)
    {
        exe_path = "aipuold";
        return exe_path;
    }
    //1. try to use env
    auto env = utils::get_env(Constants::AIPUOLD);
    if (!env.empty())
    {
        exe_path = env;
        return exe_path;
    }
    //2. checking path
    auto path = utils::get_env("PATH");
    for (auto d : utils::split_string(path, ':'))
    {
        if (utils::file_exists(d + "/aipuold"))
        {
            exe_path = "aipuold";
            return exe_path;
        }
    }
#if RELEASE_VERSION
    WARN() << "Can NOT find aipuold in PATH. Please make sure you have added it to PATH.";
    exe_path = "aipuold";
#else
    //3. use default: push the dir to path
    std::string lpath = path + "/project/ai/scratch01/toolchain/llvm/binary/bin/";
    setenv("PATH", lpath.c_str(), 1);
    exe_path = "/project/ai/scratch01/toolchain/llvm/binary/bin/aipuold";
#endif
    return exe_path;
}

static const std::string &get_aipuocc()
{
    load_toolchain();
    static std::string exe_path;
    if (exe_path.size())
    {
        return exe_path;
    }
    if (m_aipucc != nullptr)
    {
        exe_path = "clang";
        return exe_path;
    }
    //1. try to use env
    auto env = utils::get_env(Constants::AIPUOCC);
    if (!env.empty())
    {
        exe_path = env;
        return exe_path;
    }
    //2. checking path
    auto path = utils::get_env("PATH");
    for (auto d : utils::split_string(path, ':'))
    {
        if (utils::file_exists(d + "/aipuocc"))
        {
            exe_path = "aipuocc";
            return exe_path;
        }
    }
#if RELEASE_VERSION
    WARN() << "Can NOT find aipuocc in PATH. Please make sure you have added it to PATH.";
    exe_path = "aipuocc";
#else
    //3. use default: push the dir to path
    std::string lpath = path + "/project/ai/scratch01/toolchain/llvm/binary/bin/";
    setenv("PATH", lpath.c_str(), 1);
    exe_path = "/project/ai/scratch01/toolchain/llvm/binary/bin/aipuocc";
#endif
    return exe_path;
}

int aipuocc(const std::string &c_file, const std::string &out)
{
    auto target = aipubt::get_target();
    std::vector<std::string> args;
    args.push_back(get_aipuocc());
    if (!m_aipucc)
    {
        args.push_back("-target aipu");
        args.push_back("-c");
        args.push_back("-O2");
        // // disable --gc-sections because .bss will be discarded.
        // args.emplace_back("-ffunction-sections");
        args.push_back(c_file);
        args.push_back("-mcpu=" + target->name());
        args.push_back("-o");
        args.push_back(out);
        DEBUG() << "aipuocc args:" << utils::join_string(" ", args);
        auto cmd = utils::join_string(" ", args);
        return system(cmd.c_str());
    }
    else
    {
        std::string s_file = c_file.substr(0, c_file.find_last_of(".")) + ".s";
        args.push_back("-cc1");
        args.emplace_back("-triple=aipux2");
        args.emplace_back("-target-cpu");
        args.emplace_back(target->name());
        args.emplace_back("-x");
        args.emplace_back("cl");
        // // disable --gc-sections because .bss will be discarded.
        // args.emplace_back("-ffunction-sections");
        args.emplace_back("-S");
        args.push_back(c_file);
        args.push_back("-o");
        args.push_back(s_file);
        DEBUG() << "aipuocc args:" << utils::join_string(" ", args);
        std::vector<char *> argv;
        for (auto &&a : args)
        {
            argv.push_back((char *)a.c_str());
        }
        m_aipucc(argv.size(), &argv[0]);

        return aipuas(s_file, out);
    }
}

int aipuas(const std::string &s, const std::string &obj)
{
    auto target = aipubt::get_target();
    std::vector<std::string> args;
    args.push_back(get_aipuas());
    args.push_back(s);
    args.push_back("--arch=aipu");
    if (m_aipuas)
    {
        args.push_back("--mcpu-gb=" + target->name());
    }
    else
    {
        args.push_back("--mcpu=" + target->name());
    }
    // args.push_back("-h");
    args.push_back("-o");
    args.push_back(obj);
    DEBUG() << "aipuas args:" << utils::join_string(" ", args);
    if (!m_aipuas)
    {
        auto cmd = utils::join_string(" ", args);
        return system(cmd.c_str());
    }
    std::vector<char *> argv;
    for (auto &&a : args)
    {
        argv.push_back((char *)a.c_str());
    }

    return m_aipuas(argv.size(), &argv[0]);
}

int aipuld(const std::vector<std::string> &objs, const std::string &out)
{
    std::vector<std::string> args;
    args.push_back(get_aipuld());
    args.insert(args.end(), objs.begin(), objs.end());
    args.push_back("-static");
    // // disable --gc-sections because .bss will be discarded.
    // // if enable --gc-sections, .bss should be kept by KEEP cmd in link script.
    // args.push_back("--gc-sections");
    args.push_back("-o");
    args.push_back(out);
    auto cmd = utils::join_string(" ", args);
    DEBUG() << "linker args:" << utils::join_string(" ", args);
    if (!m_aipuld)
    {
        return system(cmd.c_str());
    }
    std::vector<char *> argv;
    for (auto &&a : args)
    {
        argv.push_back((char *)a.c_str());
    }
    return m_aipuld(argv.size(), &argv[0]);
}

static const std::string &get_cpsld()
{
    static std::string exe_path = "cpsld";
    auto env = utils::get_env(Constants::CPSLD);
    if (!env.empty())
    {
        exe_path = env;
        return exe_path;
    }
    auto path = utils::get_env("PATH");
    for (auto d : utils::split_string(path, ':'))
    {
        if (utils::file_exists(d + "/cpsld"))
        {
            exe_path = "cpsld";
            return exe_path;
        }
    }
#if RELEASE_VERSION
    WARN() << "Can NOT find aipuold in PATH. Please make sure you have added it to PATH.";
    exe_path = "aipuold";
#else
    //3. use default: push the dir to path
    // std::string lpath = path + "/project/ai/scratch01/toolchain/llvm/binary/bin/";
    // setenv("PATH", lpath.c_str(), 1);
    exe_path = "/project/ai/scratch01/compass-toolkit/compiler/latest/bin/cpsld";
#endif
    return exe_path;
}

int cpsld(const std::vector<std::string> &objs, const std::string &out)
{
    std::vector<std::string> args;
    args.push_back(get_cpsld());
    args.insert(args.end(), objs.begin(), objs.end());
    args.push_back("-static");
    // // disable --gc-sections because .bss will be discarded.
    // // if enable --gc-sections, .bss should be kept by KEEP cmd in link script.
    // args.push_back("--gc-sections");
    args.push_back("-o");
    args.push_back(out);
    auto cmd = utils::join_string(" ", args);
    DEBUG() << "linker args:" << utils::join_string(" ", args);

    return system(cmd.c_str());
}

static std::string get_cpscc()
{
    auto env = utils::get_env(Constants::CPSCC);
    if (env.size())
    {
        return env;
    }

    auto path = utils::get_env("PATH");
    auto dirs = utils::split_string(path, ':');
    for (auto &d : dirs)
    {
        if (utils::file_exists(d + "/cpscc"))
        {
            return "cpscc";
        }
    }

#if RELEASE_VERSION
    WARN() << "Can NOT find cpscc in PATH. Please make sure you have added it to PATH.";
    auto exe_path = "cpscc";
#else
    auto exe_path = "/project/ai/scratch01/compass-toolkit/compiler/latest/bin/cpscc";
#endif
    return exe_path;
}

static std::string get_cps_path()
{
    auto env = utils::get_env(Constants::CPS_PATH);
    if (env.size())
    {
        return env;
    }

#if RELEASE_VERSION
    WARN() << "Can NOT find CPS_PATH. Please make sure you have set it up.";
    auto cps_path = "";
#else
    auto cps_path = "/project/ai/scratch01/compass-toolkit/toolkit/latest";
#endif
    return cps_path;
}

int cpscc(const std::string &c_file, const std::string &out)
{
    std::vector<std::string> args;
    args.push_back(get_cpscc());
    auto cps_path = get_cps_path();
    if (cps_path.size())
    {
        args.push_back("--cps-path=" + cps_path);
    }
    args.push_back("-dc");
    args.push_back("-O2");
    args.push_back(c_file);
    args.push_back("-o");
    args.push_back(out);
    auto cmd = utils::join_string(" ", args);
    DEBUG() << "cpscc args:" << cmd;
    return system(cmd.c_str());
}

static std::string get_hostcc()
{
    auto env = utils::get_env(Constants::CPSHOSTCC);
    if (env.size())
    {
        return env;
    }

#if RELEASE_VERSION
    WARN() << "Can NOT find CPSHOSTCC. Use g++ by default.";
#endif
    return "g++";
}

static std::string get_devkit_path()
{
    auto env = utils::get_env(Constants::CPSDEVKIT_PATH);
    if (env.size())
    {
        return env;
    }

#if RELEASE_VERSION
    WARN() << "Can NOT find CPSDEVKIT_PATH. Please make sure you have set it up.";
    auto devkit_path = "";
#else
    auto devkit_path = "/project/ai/scratch01/AIPUBuilder";
#endif
    return devkit_path;
}

int cpshostcc(const std::string &c_file, const std::string &out)
{
    std::vector<std::string> args;
    args.push_back(get_hostcc());
    args.push_back("-fPIC");
    args.push_back("-shared");
    args.push_back("-O2");
    args.push_back(c_file);
    auto devkit_path = get_devkit_path();
    if (devkit_path.size())
    {
        args.push_back("-I" + devkit_path + "/include");
        args.push_back("-L" + devkit_path + "/lib");
    }
    args.push_back("-laipubtrt");
    args.push_back("-ldl");
    args.push_back("-o");
    args.push_back(out);
    auto cmd = utils::join_string(" ", args);
    DEBUG() << "cpshostcc args:" << cmd;
    return system(cmd.c_str());
}

class ToolChainHelper
{
public:
    ToolChainHelper()
    {
        while (dlerror())
            ;
        m_as_handler = dlopen("libaiputoolchain.so", RTLD_LAZY | RTLD_GLOBAL);
        if (m_as_handler)
        {
#if defined(__GNUC__) && !defined(__clang__)
            struct link_map *map;
            dlinfo(m_as_handler, RTLD_DI_LINKMAP, &map);
            if (map)
            {
                char *real = realpath(map->l_name, NULL);
                if (real)
                    DEBUG() << "load toolchain from " << real;
            }
#endif

            m_aipuas = (EntryFunc)dlsym(m_as_handler, "btaipuas");
            auto errorInfo = dlerror();
            if (errorInfo != NULL)
            {
                m_aipuas = nullptr;
                WARN("Dlsym Error:%s.\n", errorInfo);
            }
            m_aipuld = (EntryFunc)dlsym(m_as_handler, "btaipuld");
            errorInfo = dlerror();
            if (errorInfo != NULL)
            {
                m_aipuld = nullptr;
                WARN("Dlsym Error:%s.\n", errorInfo);
            }
            m_aipucc = (EntryFunc)dlsym(m_as_handler, "btaipucc");
            errorInfo = dlerror();
            if (errorInfo != NULL)
            {
                m_aipucc = nullptr;
                WARN("Dlsym Error:%s.\n", errorInfo);
            }
        }
        else
        {
            WARN("Open Error:%s. Unable to use aipuas.\n", dlerror());
            return;
        }
    }
    ~ToolChainHelper()
    {
        if (m_as_handler)
        {
            dlclose(m_as_handler);
        }
    }

private:
    void *m_as_handler = nullptr;
};

static void load_toolchain()
{
#if defined(__ANDROID__) || defined(__USE_TOOLCHAIN_LIB__)
    static ToolChainHelper _tool_chain_helper;
#endif // OS_ANDROID
}
} //namespace aipubt
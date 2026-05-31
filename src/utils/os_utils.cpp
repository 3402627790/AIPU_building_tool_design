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
#include "core/constants.h"
#include "core/logger.h"
#include "utils/utils.h"
#include <array>
#include <functional>
#include <map>
#include <string>

using namespace aipubt;
namespace utils
{
static std::map<std::string, std::function<void()>> *funcs = nullptr;

static std::map<std::string, std::function<void()>> &get_atexit_funcs()

{
    if (!funcs)
    {
        funcs = new std::map<std::string, std::function<void()>>();
    }
    return *funcs;
}

void register_atexit(const std::string &name, std::function<void()> f)
{
    get_atexit_funcs()[name] = f;
}
void unregister_atexit(const std::string &name)
{
    if (get_atexit_funcs().count(name))
    {
        get_atexit_funcs().erase(name);
    }
}
void clear_atexit()
{
    get_atexit_funcs().clear();
}

#if defined(__GNUC__) && !defined(__clang__)
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <gnu/lib-names.h>
#include <map>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

std::string exec(const char *cmd)
{
    std::array<char, 128> buffer;
    std::string result;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
    if (!pipe)
    {
        std::cerr << "Failed to run popen\n";
        return "";
    }
    while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr)
    {
        result += buffer.data();
    }
    return result;
}

const std::map<std::string, size_t> &get_code_memory_map()
{
    static std::map<std::string, size_t> ret;
    if (ret.size())
    {
        return ret;
    }
    FILE *fp;
    char line[2048];
    fp = fopen("/proc/self/maps", "r");
    if (fp == NULL)
    {
        perror("Error opening file");
        return ret;
    }
    while (fgets(line, 2048, fp) != NULL)
    {
        //the is a text section
        if (strstr(line, "r-xp") != NULL)
        {
            std::string s(line);
            std::stringstream ss;
            ss << std::hex << s;
            size_t base = 0;
            ss >> base;
            std::string filename = s.substr(s.rfind(" "), s.size() - s.rfind(" "));
            utils::trim(filename);
            ret[get_abspath(filename)] = base;
            // printf("%s", line);
        }
    }
    return ret;
}

inline void print_bt_symbols(size_t i, char *s, const std::string &exec_file, void *ptr)
{
    size_t idx = 0;
    size_t func_start = 0;
    size_t func_end = 0;
    size_t ptr_start = 0;
    size_t ptr_end = 0;

    char *s_ptr = s;
    while (*s_ptr != '\0')
    {
        if (*s_ptr == '(')
        {
            func_start = idx;
        }
        if (*s_ptr == '[')
        {
            ptr_start = idx;
        }
        if (*s_ptr == '+')
        {
            func_end = idx;
        }
        if (*s_ptr == ']')
        {
            ptr_end = idx;
        }
        if (*s_ptr == ')')
        {
            if (func_end < func_start)
            {
                func_end = idx;
            }
        }
        idx++;
        s_ptr++;
    }
    std::string func(s + func_start + 1, func_end - func_start - 1);
    std::string ptrs(s + ptr_start + 1, ptr_end - ptr_start - 1);
    std::string filename(s, func_start);
    // std::cout<<"filename is :"<<filename<<" func:"<<func<<" ptr:"<<ptrs<<"\n"<<s<<"\n";
    std::stringstream ss;
    ss << std::hex << ptrs;
    size_t ptr_offset = 0;
    ss >> ptr_offset;
    auto &code_mmap = get_code_memory_map();
    char cmd[2048] = {0};
    if (utils::endswith(exec_file, filename))
    {
        filename = exec_file;
        sprintf(cmd, "addr2line %p -e \"%s\" -f -C -p ", ptr, filename.c_str());
    }
    else
    {
        filename = get_abspath(filename);
        if (code_mmap.count(filename))
        {
            if (ptr_offset > code_mmap.at(filename))
            {
                ptr_offset -= code_mmap.at(filename);
            }
        }
        sprintf(cmd, "addr2line 0x%lx -e \"%s\" -f -C -p ", ptr_offset, filename.c_str()); //last parameter is the name of this app
    }

    fprintf(stderr, "%zd %s\t", i, filename.c_str());
    fprintf(stderr, "%s", exec(cmd).c_str());
    // system(cmd);
}

void print_backtrace(unsigned int start)
{
    void *array[64];
    size_t size;
    char **strings;
    size_t i;
    size = backtrace(array, 64);
    strings = backtrace_symbols(array, size);
    char exec_filename[1024] = {0};
    readlink("/proc/self/exe", exec_filename, 1024);
    std::string filename(exec_filename);
    fprintf(stderr, "BackTrace with %zd stack frames. \n", size);
    for (i = start; i < size; i++)
    {
        print_bt_symbols(i - start, strings[i], filename, array[i]);
    }
    free(strings);
    // auto r = get_code_memory_map();
    // for (auto kv : r)
    // {
    //     std::cout << kv.first << " " << kv.second << "\n";
    // }
}

static const std::map<int, std::string> all_signals = {
#ifdef SIGHUP
    {SIGHUP, "SIGHUP"}, /* POSIX.1 */
#endif
#ifdef SIGQUIT
    {SIGQUIT, "SIGQUIT"}, /* POSIX.1 */
#endif
#ifdef SIGTRAP
// {SIGTRAP,"SIGTRAP"}, /* POSIX.1 */
#endif
#ifdef SIGIO
    {SIGIO, "SIGIO"}, /* BSD/Linux */
#endif
    /* C89/C99/C11 standard signals: */
    {SIGABRT, "SIGABRT"},
    {SIGFPE, "SIGFPE"},
    {SIGILL, "SIGILL"},
    // {SIGINT,"SIGINT"},
    {SIGSEGV, "SIGSEGV"},
    /* SIGTERM (C89/C99/C11) is also the terminating signal number */
    {SIGTERM, "SIGTERM"}};
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#define STACK_FRAMES_BUFFERSIZE ((int)128)

static void segment_fault_handler(int sig)
{
    INFO() << "Error: signal: " << sig << "\t" << all_signals.at(sig);
    print_backtrace(3);
    exit(sig);
}

static void atexit_handler()
{
    for (auto kv : get_atexit_funcs())
    {
        kv.second();
    }
    if (funcs)
    {
        delete funcs;
        funcs = nullptr;
    }
}

static int register_all_signal_handler()
{
    auto disable_bt = getenv(Constants::DISABLE_BACKTRACE);
    if (disable_bt)
    {
        std::string disable(disable_bt);
        if (disable == "1" || disable == "True" || disable == "true" || disable == "TRUE")
        {
            return 0;
        }
    }
    for (auto s : all_signals)
    {
        signal(s.first, segment_fault_handler);
    }
    atexit(atexit_handler);
    return 0;
}
static int _nothing = register_all_signal_handler();
#else
void print_backtrace(unsigned int start)
{
}
#endif //__USE_GCC__

}; //namespace utils

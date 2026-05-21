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
 * @Date: 2020-02-09 12:14:57
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-05-18 16:24:18
 */
#ifndef UTILS_LOGGER_H
#define UTILS_LOGGER_H

#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <math.h>
#include <memory>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unistd.h>
namespace utils
{
void print_backtrace(unsigned int start);
};

namespace aipubt
{
namespace GCompilerLogger
{

    size_t get_error_count();
    size_t get_warn_count();
    void increase_error_count();
    void increase_warn_count();
    void clear_error_count();
    void clear_warn_count();
    void set_logger_file(const std::string &filename, const std::string &errlog = "");
    void set_logger_stream(std::shared_ptr<std::ostream> logger, std::shared_ptr<std::ostream> err = nullptr);
    std::shared_ptr<std::ostream> get_logger_file();
    std::shared_ptr<std::ostream> get_logger_err_file();
    std::map<int, std::map<std::string, std::function<void(const std::string &)>>> &get_logger_callbacks(); // logger_callbacks
    int get_logger_level();
    void set_logger_level(int level);

    struct aipubtLoggerOps
    {
        static size_t get_error_count() { return GCompilerLogger::get_error_count(); }
        static size_t get_warn_count() { return GCompilerLogger::get_warn_count(); }
        static void increase_error_count() { GCompilerLogger::increase_error_count(); }
        static void increase_warn_count() { GCompilerLogger::increase_warn_count(); }
        static std::shared_ptr<std::ostream> get_logger_file() { return GCompilerLogger::get_logger_file(); }
        static std::shared_ptr<std::ostream> get_logger_err_file() { return GCompilerLogger::get_logger_err_file(); }
        static std::map<int, std::map<std::string, std::function<void(const std::string &)>>> &get_logger_callbacks()
        {
            return GCompilerLogger::get_logger_callbacks();
        }
        static int get_logger_level() { return GCompilerLogger::get_logger_level(); }
        static void print_backtrace() { utils::print_backtrace(2); }
        static inline bool without_color()
        {
            static bool r = isatty(fileno(stdout)) || get_logger_file();
            return r;
        }
        static inline bool without_color_err()
        {
            static bool r = isatty(fileno(stderr)) || get_logger_err_file();
            return r;
        }
    };
}; // namespace GCompilerLogger
}; // namespace aipubt

#ifdef DEFAULT_LOGGER_OPS
#undef DEFAULT_LOGGER_OPS
#endif
#define DEFAULT_LOGGER_OPS aipubt::GCompilerLogger::aipubtLoggerOps

#include "base_logger.h"

#if __cplusplus >= 201402L
#define DEPRECATED(reason) [[deprecated(reason)]]
#else
#define DEPRECATED(reason) __attribute__((deprecated(reason)))
#endif
#endif //! UTILS_LOGGER_H
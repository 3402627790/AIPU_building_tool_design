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
 * @Last Modified time: 2026-05-19 09:44:21
 */
#ifndef UTILS_BASE_LOGGER_H
#define UTILS_BASE_LOGGER_H

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

#define MSG_FILE_NAME_WIDTH 10
#define _LOGGER_STR_HELPER(x) #x
#define _LOGGER_STR(x) _LOGGER_STR_HELPER(x)

#if defined(__unix__) || defined(__APPLE__)
#define _BEGIN_INFO_ "\033[1;32m"
#define _BEGIN_ERROR_ "\033[1;31m"
#define _BEGIN_DEBUG_ "\033[1;33m"
#define _BEGIN_WARN_ "\033[1;34m"
#define _END_COLOR_ "\033[0m"

#define __FILENAME__ (strrchr(__FILE__, '/') ? (strrchr(__FILE__, '/') + 1) \
                                             : __FILE__)
#else
#define _BEGIN_INFO_ "LOG_COLOR_M"
#define _BEGIN_ERROR_ "LOG_COLOR_R"
#define _BEGIN_DEBUG_ "LOG_COLOR_Y"
#define _END_COLOR_ "LOG_COLOR_W"

#define __FILENAME__ (strrchr(__FILE__, '\\') ? (strrchr(__FILE__, '\\') + 1) \
                                              : __FILE__)
#endif // __unix__, __APPLE__

#define DEFAULT_LOG_LEVEL INT32_MIN
#if RELEASE_VERSION
#undef DEFAULT_LOG_LEVEL
#define DEFAULT_LOG_LEVEL 1
#endif

#define _ERROR_LEVEL (9)
#define _INFO_LEVEL (5)
#define _WARN_LEVEL (2)
#define _DEBUG_LEVEL (1)

#define _FATAL_ERROR_LEVEL (_ERROR_LEVEL + 1)
#ifndef _ASSERTION_LOG_LEVEL
#define _ASSERTION_LOG_LEVEL _FATAL_ERROR_LEVEL
#endif

namespace aipubtLogger
{
struct DefaultLoggerOps
{
    static size_t get_error_count() { return 0; }
    static size_t get_warn_count() { return 0; }
    static void increase_error_count() {}
    static void increase_warn_count() {}
    static std::shared_ptr<std::ostream> get_logger_file() { return nullptr; }
    static std::shared_ptr<std::ostream> get_logger_err_file() { return nullptr; }
    static std::map<int, std::map<std::string, std::function<void(const std::string &)>>> &get_logger_callbacks()
    {
        static std::map<int, std::map<std::string, std::function<void(const std::string &)>>> r;
        return r;
    }
    static int get_logger_level() { return DEFAULT_LOG_LEVEL; }
    static void print_backtrace() {}
};

#ifndef DEFAULT_LOGGER_OPS
#define DEFAULT_LOGGER_OPS aipubtLogger::DefaultLoggerOps
#endif

#if USE_DEFAULT_OPS
#ifdef DEFAULT_LOGGER_OPS
#undef DEFAULT_LOGGER_OPS
#endif
#define DEFAULT_LOGGER_OPS aipubtLogger::DefaultLoggerOps
#endif

template <int loglevel = DEFAULT_LOG_LEVEL, typename LoggerOps = DefaultLoggerOps, bool enable_bt = false>
class LoggerImpl
{
public:
    template <class T>
    LoggerImpl(const T &options)
    {
        init();
        my_cout_ << options;
    }
    ~LoggerImpl()
    {
        auto LOGGER_LEVEL = LoggerOps::get_logger_level();
        if (level() > LOGGER_LEVEL)
        {
            auto m_logger = LoggerOps::get_logger_file();
            auto m_error = LoggerOps::get_logger_err_file();
            if (level() >= _ERROR_LEVEL)
            {
                if (m_error)
                {
                    (*m_error) << my_cout_.str() << std::endl;
                    m_error->flush();
                }

                if (LoggerOps::get_logger_callbacks()[level()].empty())
                {
                    std::cerr << my_cout_.str() << std::endl;
                    // todo make it thread safe
                    LoggerOps::increase_error_count();
                }
                else
                {
                    for (auto ck : LoggerOps::get_logger_callbacks()[level()])
                    {
                        ck.second(my_cout_.str());
                    }
                }
            }
            else
            {
                if (m_logger)
                {
                    (*m_logger) << my_cout_.str() << std::endl;
                    m_logger->flush();
                }
                if (LoggerOps::get_logger_callbacks()[level()].empty())
                {
                    std::cout << my_cout_.str() << std::endl;
                    if (level() == _WARN_LEVEL)
                    {
                        LoggerOps::increase_warn_count();
                    }
                }
                else
                {
                    for (auto ck : LoggerOps::get_logger_callbacks()[level()])
                    {
                        ck.second(my_cout_.str());
                    }
                }
            }
        }
        if (level() >= _FATAL_ERROR_LEVEL)
        {
            LoggerOps::print_backtrace();
            exit(1);
        }
        else if (enable_bt)
        {
            LoggerOps::print_backtrace();
        }
    }
    template <class T>
    inline LoggerImpl &operator<<(const T &x)
    {
        my_cout_ << x;
        return *this;
    }
    LoggerImpl &operator()()
    {
        return *this;
    }

    LoggerImpl &operator()(const char *_str)
    {
        my_cout_ << _str;
        return *this;
    }
    template <typename... Args>
    LoggerImpl &operator()(const std::string &format, Args... args)
    {
        size_t size = snprintf(nullptr, 0, format.c_str(), args...) + 1; // Extra space for '\0'
        if (size <= 0)
        {
            return *this;
        }
        std::unique_ptr<char[]> buf(new char[size]);
        snprintf(buf.get(), size, format.c_str(), args...);
        my_cout_ << buf.get();
        return *this;
    }

    LoggerImpl &_with_end_color()
    {
        if (_without_color() || _without_color_err())
        {
            return *this;
        }
        my_cout_ << _END_COLOR_;
        return *this;
    }
    static inline bool _without_color()
    {
        static bool r = !isatty(fileno(stdout)) || LoggerOps::get_logger_file();
        return r;
    }
    static inline bool _without_color_err()
    {
        static bool r = !isatty(fileno(stderr)) || DEFAULT_LOGGER_OPS::get_logger_err_file();
        return r;
    }

private:
    void init()
    {

        if (loglevel <= _DEBUG_LEVEL)
        {
            if (!_without_color())
            {
                my_cout_ << _BEGIN_DEBUG_;
            }
            my_cout_ << "[D] ";
        }
        else if (loglevel <= _WARN_LEVEL)
        {
            if (!_without_color())
            {
                my_cout_ << _BEGIN_WARN_;
            }
            my_cout_ << "[W] ";
        }
        else if (loglevel <= _INFO_LEVEL)
        {
            if (!_without_color())
            {
                my_cout_ << _BEGIN_INFO_;
            }
            my_cout_ << "[I] ";
        }
        else if (loglevel <= _ERROR_LEVEL)
        {
            if (!_without_color_err())
            {
                my_cout_ << _BEGIN_ERROR_;
            }
            my_cout_ << "[E] ";
        }
        else
        {
            if (!_without_color_err())
            {
                my_cout_ << _BEGIN_ERROR_;
            }
            my_cout_ << "[F] ";
        }
    }

    constexpr int level() const
    {
        return loglevel;
    }
    std::ostringstream my_cout_;
};

#ifndef LOG_MODULE
#define LOG_MODULE ""
#endif

#define LOG_FILE_INFO_STREAM "[" << std::left << std::setw(MSG_FILE_NAME_WIDTH) << __FILENAME__ << ":" \
                                 << std::right << std::setw(4) << __LINE__ << "] "
#ifndef LOG_FILEINFO
#define LOG_FILEINFO LOG_FILE_INFO_STREAM
#endif

}; // namespace aipubtLogger

#endif //! UTILS_BASE_LOGGER_H

#ifdef DEBUG
#undef DEBUG
#undef FATAL
#undef INFO
#undef INFO
#undef TraceERROR
#undef aipu_assert
#endif // DEBUG

#define FATAL(...) ((aipubtLogger::LoggerImpl<_FATAL_ERROR_LEVEL, DEFAULT_LOGGER_OPS>(LOG_MODULE) << LOG_FILEINFO)._with_end_color())(__VA_ARGS__)
#define INFO(...) ((aipubtLogger::LoggerImpl<_INFO_LEVEL, DEFAULT_LOGGER_OPS>(LOG_MODULE) << LOG_FILEINFO)._with_end_color())(__VA_ARGS__)
#define ERROR(...) ((aipubtLogger::LoggerImpl<_ERROR_LEVEL, DEFAULT_LOGGER_OPS>(LOG_MODULE) << LOG_FILEINFO)._with_end_color())(__VA_ARGS__)
#define WARN(...) ((aipubtLogger::LoggerImpl<_WARN_LEVEL, DEFAULT_LOGGER_OPS>(LOG_MODULE) << LOG_FILEINFO)._with_end_color())(__VA_ARGS__)
#define DEBUG(...) ((aipubtLogger::LoggerImpl<_DEBUG_LEVEL, DEFAULT_LOGGER_OPS>(LOG_MODULE) << LOG_FILEINFO)._with_end_color())(__VA_ARGS__)
#define TraceERROR(...) ((aipubtLogger::LoggerImpl<_ERROR_LEVEL, DEFAULT_LOGGER_OPS, true>(LOG_MODULE) << LOG_FILEINFO)._with_end_color())(__VA_ARGS__)
#define aipu_assert(expr)                                                                                                             \
    {                                                                                                                                 \
        if (!(expr))                                                                                                                  \
            (aipubtLogger::LoggerImpl<_ASSERTION_LOG_LEVEL>("[Assertion]") << LOG_FILEINFO)._with_end_color() << (_LOGGER_STR(expr)); \
    }

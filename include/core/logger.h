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
 * @Last Modified time: 2024-08-21 15:58:12
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

#define _FATAL_ERROR_ (aipubt::GCompilerLogger::_begin_error_() + "[F] ")
#define _ERROR_ (aipubt::GCompilerLogger::_begin_error_() + "[E] ")
#define _DEBUG_ (aipubt::GCompilerLogger::_begin_debug_() + "[D] ")
#define _INFO_ (aipubt::GCompilerLogger::_begin_info_() + "[I] ")
#define _WARN_ (aipubt::GCompilerLogger::_begin_warn_() + "[W] ")

#define LOG_LEVEL (aipubt::GCompilerLogger::get_logger_level())
#define _ERROR_LEVEL (9)
#define _INFO_LEVEL (5)
#define _WARN_LEVEL (2)
#define _DEBUG_LEVEL (1)

#define _FATAL_ERROR_LEVEL (_ERROR_LEVEL + 1)
#ifndef _ASSERTION_LOG_LEVEL
#define _ASSERTION_LOG_LEVEL _FATAL_ERROR_LEVEL
#endif

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
    const std::string _end_color();
    const std::string _begin_error_();
    const std::string _begin_debug_();
    const std::string _begin_warn_();
    const std::string _begin_info_();

    template <int loglevel = DEFAULT_LOG_LEVEL>
    class LoggerImpl
    {
    public:
        template <class T>
        LoggerImpl(const T &options)
        {
            my_cout_ << options;
        }
        ~LoggerImpl()
        {
            auto LOGGER_LEVEL = get_logger_level();
            if (level() > LOGGER_LEVEL)
            {
                auto m_logger = get_logger_file();
                auto m_error = get_logger_err_file();
                if (level() >= _ERROR_LEVEL)
                {
                    if (m_error)
                    {
                        (*m_error) << my_cout_.str() << std::endl;
                        m_error->flush();
                    }

                    if (get_logger_callbacks()[level()].empty())
                    {
                        std::cerr << my_cout_.str() << std::endl;
                        // todo make it thread safe
                        increase_error_count();
                    }
                    else
                    {
                        for (auto ck : get_logger_callbacks()[level()])
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
                    if (get_logger_callbacks()[level()].empty())
                    {
                        std::cout << my_cout_.str() << std::endl;
                        if (level() == _WARN_LEVEL)
                        {
                            increase_warn_count();
                        }
                    }
                    else
                    {
                        for (auto ck : get_logger_callbacks()[level()])
                        {
                            ck.second(my_cout_.str());
                        }
                    }
                }
                // if (back_trace && level() < _FATAL_ERROR_LEVEL)
                // {
                //     utils::print_backtrace(2);
                // }
            }
            if (level() >= _FATAL_ERROR_LEVEL)
            {
                utils::print_backtrace(2);
                exit(1);
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

    private:
        constexpr int level() const
        {
            return loglevel;
        }
        std::ostringstream my_cout_;
        bool back_trace = false;
    };

}; // namespace GCompilerLogger
}; // namespace aipubt

#ifndef LOG_MODULE
#define LOG_MODULE ""
#endif

#define LOG_FILE_INFO_STREAM "[" << std::left << std::setw(MSG_FILE_NAME_WIDTH) << __FILENAME__ << ":" \
                                 << std::right << std::setw(4) << __LINE__ << "] "
#ifndef LOG_FILEINFO
#define LOG_FILEINFO LOG_FILE_INFO_STREAM
#endif

#define FATAL(...) ((aipubt::GCompilerLogger::LoggerImpl<_FATAL_ERROR_LEVEL>(_FATAL_ERROR_) << LOG_MODULE << LOG_FILEINFO << (aipubt::GCompilerLogger::_end_color()))(__VA_ARGS__))
#define INFO(...) ((aipubt::GCompilerLogger::LoggerImpl<_INFO_LEVEL>(_INFO_) << LOG_MODULE << LOG_FILEINFO << (aipubt::GCompilerLogger::_end_color()))(__VA_ARGS__))
#define ERROR(...) ((aipubt::GCompilerLogger::LoggerImpl<_ERROR_LEVEL>(_ERROR_) << LOG_MODULE << LOG_FILEINFO << (aipubt::GCompilerLogger::_end_color()))(__VA_ARGS__))
#define WARN(...) ((aipubt::GCompilerLogger::LoggerImpl<_WARN_LEVEL>(_WARN_) << LOG_MODULE << LOG_FILEINFO << (aipubt::GCompilerLogger::_end_color()))(__VA_ARGS__))
#define DEBUG(...) ((aipubt::GCompilerLogger::LoggerImpl<_DEBUG_LEVEL>(_DEBUG_) << LOG_MODULE << LOG_FILEINFO << (aipubt::GCompilerLogger::_end_color()))(__VA_ARGS__))
#define TraceERROR(...) ((aipubt::GCompilerLogger::LoggerImpl<_ERROR_LEVEL>(_ERROR_) << LOG_MODULE << LOG_FILEINFO << (aipubt::GCompilerLogger::_end_color()))(__VA_ARGS__))
#define aipu_assert(expr)                                                                                                                                                                  \
    {                                                                                                                                                                                      \
        if (!(expr))                                                                                                                                                                       \
            (aipubt::GCompilerLogger::LoggerImpl<_ASSERTION_LOG_LEVEL>(_FATAL_ERROR_) << "[Assertion]" << LOG_FILEINFO << (aipubt::GCompilerLogger::_end_color()) << (_LOGGER_STR(expr))); \
    }

#if __cplusplus >= 201402L
#define DEPRECATED(reason) [[deprecated(reason)]]
#else
#define DEPRECATED(reason) __attribute__((deprecated(reason)))
#endif
#endif //! UTILS_LOGGER_H
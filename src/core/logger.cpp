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
 * @Date: 2020-01-29 14:28:02
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-02-28 14:30:22
 */
#include "core.h"
#include "utils/utils.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace aipubt
{
static int LOGGER_LEVEL = DEFAULT_LOG_LEVEL;
static std::atomic<size_t> ERROR_CNT(0);
static std::atomic<size_t> WARN_CNT(0);
static std::shared_ptr<std::ostream> m_logger = nullptr;
static std::shared_ptr<std::ostream> m_error = nullptr;
static std::map<int, std::map<std::string, std::function<void(const std::string &)>>> logger_callbacks;

namespace GCompilerLogger
{

    std::map<int, std::map<std::string, std::function<void(const std::string &)>>> &get_logger_callbacks()
    {
        return logger_callbacks;
    }

    inline bool is_a_tty()
    {
        return isatty(fileno(stdout)) && isatty(fileno(stderr));
    }
    int get_logger_level()
    {
        static const char *logger_level = std::getenv(Constants::LOGLEVEL);
        if (logger_level != NULL && utils::is_number(logger_level))
        {
            LOGGER_LEVEL = std::stoi(logger_level);
            logger_level = nullptr;
        }
        if (LOGGER_LEVEL < -1)
        {
            // try to get env
            const char *logger_level = std::getenv(Constants::LOGLEVEL);
            if (logger_level != NULL && utils::is_number(logger_level))
            {
                LOGGER_LEVEL = std::stoi(logger_level);
            }
            if (LOGGER_LEVEL < -1)
            {
                LOGGER_LEVEL = -1;
            }
        }
        return LOGGER_LEVEL;
    }
    void set_logger_level(int level)
    {
        LOGGER_LEVEL = level;
    }
    const std::string _end_color()
    {
        if (get_logger_err_file() || !is_a_tty())
        {
            return "";
        }
        return _END_COLOR_;
    }
    const std::string _begin_error_()
    {
        if (get_logger_err_file() || !is_a_tty())
        {
            return "";
        }
        return _BEGIN_ERROR_;
    }

    const std::string _begin_debug_()
    {
        if (get_logger_file() || !is_a_tty())
        {
            return "";
        }
        return _BEGIN_DEBUG_;
    }

    const std::string _begin_warn_()
    {
        if (get_logger_file() || !is_a_tty())
        {
            return "";
        }
        return _BEGIN_WARN_;
    }

    const std::string _begin_info_()
    {
        if (get_logger_file() || !is_a_tty())
        {
            return "";
        }
        return _BEGIN_INFO_;
    }

    void set_logger_file(const std::string &filename, const std::string &errlog)
    {
        if (m_logger == nullptr)
        {
            m_logger = std::shared_ptr<std::ostream>(new std::ofstream(filename));
        }
        if (m_error == nullptr)
        {
            if (errlog.size() == 0 || filename == errlog)
            {
                m_error = m_logger;
            }
            else
            {
                m_error = std::shared_ptr<std::ostream>(new std::ofstream(errlog));
            }
        }
    }

    void set_logger_stream(std::shared_ptr<std::ostream> logger, std::shared_ptr<std::ostream> err)
    {
        if (m_logger == nullptr)
        {
            m_logger = std::static_pointer_cast<std::ostream>(logger);
        }
        if (m_error == nullptr)
        {
            if (err == nullptr)
            {
                m_error = m_logger;
            }
            else
            {
                m_error = std::static_pointer_cast<std::ostream>(err);
            }
        }
    }

    std::shared_ptr<std::ostream> get_logger_file()
    {
        return m_logger;
    }
    std::shared_ptr<std::ostream> get_logger_err_file()
    {
        return m_error;
    }

    size_t get_error_count()
    {
        return ERROR_CNT;
    }
    size_t get_warn_count()
    {
        return WARN_CNT;
    }

    void increase_warn_count()
    {
        WARN_CNT++;
    }

    void increase_error_count()
    {
        ERROR_CNT++;
    }

    void clear_error_count()
    {
        ERROR_CNT = 0;
    }
    void clear_warn_count()
    {
        WARN_CNT = 0;
    }
}; // namespace GCompilerLogger

void add_log_callback(const std::string &name, std::function<void(const std::string &)> cb, int level)
{
    logger_callbacks[level][name] = cb;
}
void remove_log_callback(const std::string &name)
{
    for (auto &&cs : logger_callbacks)
    {
        cs.second.erase(name);
    }
}
void add_error_callback(const std::string &name, std::function<void(const std::string &)> cb)
{
    add_log_callback(name, cb, _ERROR_LEVEL);
}
void remove_error_callback(const std::string &name)
{
    remove_log_callback(name);
}
}; // namespace aipubt

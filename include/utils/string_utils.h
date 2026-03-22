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
 * @Date: 2020-01-29 14:28:43 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-11-14 11:00:51
 */
#ifndef UTILS_STRING_UTILS_H
#define UTILS_STRING_UTILS_H

#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace utils
{
struct NullBuf : std::streambuf
{
    char buffer[100];
    int overflow(int c) override
    {
        setp(buffer, buffer + sizeof buffer);
        return c;
    }
    std::streamsize xsputn(const char *, std::streamsize n) override
    {
        return n;
    }
};

std::string &ltrim_digit(std::string &s);
std::string &trim(std::string &s);
std::string verify_name(const std::string &s);
std::string &replace(std::string &s, const char old_c, const char new_c);
std::string &replace(std::string &subject, const std::string &search,
                     const std::string &replace);
bool search(const std::string &s, const std::string &search);
bool startswith(const std::string &s, const std::string &prefix);
bool endswith(const std::string &s, const std::string &suffix);
std::string &tolower(std::string &s);
std::string &toupper(std::string &s);
std::vector<std::string> split_string(const std::string &s, const char &splitter = ',');
std::vector<std::string> split_string(const std::string &s, const std::string &splitter, const int keep_spliter = 0);
std::string join_string(const std::string &join, const std::vector<std::string> &s);
template <typename T>
std::string join_string(const std::string &join, const std::vector<T> &s)
{
    std::ostringstream oss;
    size_t size = s.size();
    if (size)
    {
        oss << s[0];
        for (size_t i = 1; i < size; ++i)
        {
            oss << join << s[i];
        }
    }
    return oss.str();
}
template <typename T>
std::string int_to_hex(T i)
{
    std::stringstream stream;
    stream << "0x" << std::hex << i;
    return stream.str();
}
bool is_number(const std::string &s);
}; // namespace utils

#endif //!UTILS_STRING_UTILS_H
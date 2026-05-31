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
 * @Date: 2020-01-29 14:28:38
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-05-10 17:48:10
 */
#include "utils/string_utils.h"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
namespace utils
{
std::string &ltrim_digit(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isdigit(c); }));
    return s;
}

// trim from start (in place)
static inline void ltrim(std::string &s)
{
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) { return !std::isspace(c); }));
}

// trim from end (in place)
static inline void rtrim(std::string &s)
{
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) { return !std::isspace(c); })
                .base(),
            s.end());
}

std::string &trim(std::string &s)
{
    ltrim(s);
    rtrim(s);
    return s;
}
std::string &replace(std::string &s, const char old_c, const char new_c)
{
    for (unsigned int i = 0; i < s.size(); ++i)
    {
        if (s[i] == old_c)
        {
            s[i] = new_c;
        }
    }
    return s;
}
std::string &replace(std::string &subject, const std::string &search,
                     const std::string &replace)
{
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos)
    {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return subject;
}
std::string &tolower(std::string &s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}
std::string &toupper(std::string &s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}
bool search(const std::string &s, const std::string &search)
{
    return s.rfind(search, 0) != (size_t)-1;
}

bool startswith(const std::string &s, const std::string &prefix)
{
    return s.rfind(prefix, 0) == 0;
}
bool endswith(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() && s.rfind(suffix) == s.size() - suffix.size();
}

std::string verify_name(const std::string &s)
{
    std::string ret = s;
    for (unsigned int i = 0; i < s.size(); ++i)
    {
        if ((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z') || s[i] == '_')
        {
            continue;
        }
        ret[i] = '_';
    }
    return ret;
}

std::vector<std::string> split_string(const std::string &s, const std::string &splitter, const int keep_spliter)
{
    std::vector<std::string> ret;
    int begin_pos = 0;
    auto pos = std::string::npos;
    int size = s.size();
    int split_size = splitter.size();
    while ((pos = s.find(splitter, begin_pos)) != std::string::npos && begin_pos != size)
    {
        ret.push_back(s.substr(begin_pos, pos - begin_pos + keep_spliter));
std::string &replace(std::string &s, const char old_c, const char new_c)
{
    for (unsigned int i = 0; i < s.size(); ++i)
    {
        if (s[i] == old_c)
        {
            s[i] = new_c;
        }
    }
    return s;
}
std::string &replace(std::string &subject, const std::string &search,
                     const std::string &replace)
{
    size_t pos = 0;
    while ((pos = subject.find(search, pos)) != std::string::npos)
    {
        subject.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return subject;
}
std::string &tolower(std::string &s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}
std::string &toupper(std::string &s)
{
    std::transform(s.begin(), s.end(), s.begin(), ::toupper);
    return s;
}
bool search(const std::string &s, const std::string &search)
{
    return s.rfind(search, 0) != (size_t)-1;
}

bool startswith(const std::string &s, const std::string &prefix)
{
    return s.rfind(prefix, 0) == 0;
}
bool endswith(const std::string &s, const std::string &suffix)
{
    return s.size() >= suffix.size() && s.rfind(suffix) == s.size() - suffix.size();
}

std::string verify_name(const std::string &s)
{
    std::string ret = s;
    for (unsigned int i = 0; i < s.size(); ++i)
    {
        if ((s[i] >= '0' && s[i] <= '9') || (s[i] >= 'A' && s[i] <= 'Z') || (s[i] >= 'a' && s[i] <= 'z') || s[i] == '_')
        {
            continue;
        }
        ret[i] = '_';
    }
    return ret;
}

std::vector<std::string> split_string(const std::string &s, const std::string &splitter, const int keep_spliter)
{
    std::vector<std::string> ret;
    int begin_pos = 0;
    auto pos = std::string::npos;
    int size = s.size();
    int split_size = splitter.size();
    while ((pos = s.find(splitter, begin_pos)) != std::string::npos && begin_pos != size)
    {
        ret.push_back(s.substr(begin_pos, pos - begin_pos + keep_spliter));
        // 从最大宽度位置向后倒退查找可以换行的字符
        for (size_t i = end; i > start; --i)
        {
            if (break_set.count(text[i]))
            {
                break_pos = i;
                break;
            }
        }

        if (break_pos != std::string::npos)
        {
            // 情况 1：找到了理想的换行字符
            result += text.substr(start, break_pos - start + 1); // 包含该换行字符
            result += "\n";
            start = break_pos + 1; // 下一行从换行字符的下一个位置开始
        }
        else
        {
            // 情况 2：整行都没有可选的换行字符（比如一长串连续数字或单词）
            // 为了不溢出，只能在最大宽度处强制截断
            result += text.substr(start, max_width);
            result += "\n";
            start = start + max_width;
        }
    }

    return result;
}

}; // namespace utils

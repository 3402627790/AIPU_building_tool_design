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
 * @Date: 2020-04-09 13:01:29
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-08-11 10:58:01
 */

#include "utils/path_utils.h"
#include "utils/string_utils.h"
#ifdef _WIN32
#define PATH_SPLITTER '\\'
#include <windows.h> //GetModuleFileNameW
#else
#define PATH_SPLITTER '/'
#include <limits.h>
#include <unistd.h> //readlink
#endif
#include <cstdint>
#include <dirent.h>
#include <fstream>
#include <pwd.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define UNUSED(x) ((void)(x))
namespace utils
{

std::string get_exe_path()
{
    static std::string ret = "";
    if (ret.size())
    {
        return ret;
    }
#ifdef _WIN32
    wchar_t path[MAX_PATH] = {0};
    GetModuleFileNameW(NULL, path, MAX_PATH);
    return path;
#else
    char result[PATH_MAX];
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);
    ret = std::string(result, (count > 0) ? count : 0);
    return ret;
#endif
}

std::string specified_temp_dir(std::string dir)
{
    static std::string temp_dir = "";
    if (!dir.empty())
        temp_dir = dir;
    return temp_dir;
}

std::string get_exe_dir()
{
    static std::string ret = "";
    if (ret.size())
    {
        return ret;
    }
    auto path = get_exe_path();
    int pos = path.rfind(PATH_SPLITTER);
    ret = path.substr(0, pos + 1);
    return ret;
}
std::vector<std::string> list_directory(const std::string &m_dir)
{
    DIR *dir;
    struct dirent *ent;
    std::vector<std::string> ret;
    if ((dir = opendir(m_dir.c_str())) != NULL)
    {
        while ((ent = readdir(dir)) != NULL)
        {
            ret.push_back(ent->d_name);
        }
        closedir(dir);
    }
    return ret;
}
bool path_exists(const std::string &path)
{
    struct stat info;
    auto bk_errrno = errno;
    auto ret = stat(path.c_str(), &info) == 0;
    errno = bk_errrno;
    return ret;
}

std::string get_temp_dir()
{
    if ((!specified_temp_dir().empty()) &&
        path_exists(specified_temp_dir()))
    {
        return specified_temp_dir();
    }

    char const *folder = getenv("TMPDIR");
    if (folder == 0)
    {
        if (path_exists("/tmp/")) //try use tmp
        {
            folder = "/tmp";
        }
        else
        {
            folder = ".";
        }
    }
    return folder;
}

std::string get_home_dir()
{

    const char *homedir;
    if ((homedir = getenv("HOME")) == NULL)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    return homedir;
}
std::string get_abspath(const std::string &path)
{
    char resolved_path[PATH_MAX];
    auto bk = errno;
    UNUSED(realpath(path.c_str(), resolved_path));
    errno = bk;
    return std::string(resolved_path);
}

std::string truncate_path(const std::string &path)
{
    if (path.size() < 200)
    {
        return path;
    }
    std::string stem = path, suffix = "";
    auto items = split_string(path, ".");
    if (items.size() > 1)
    {
        suffix = "." + items[items.size() - 1];
        stem = replace(stem, suffix, "");
    }
    size_t max_len = 255 - suffix.size();
    if (path.size() > max_len)
    {
        return stem.substr(0, stem.size() - (path.size() - max_len)) + suffix;
    }
    return stem + suffix;
}

bool mkdir(const std::string &path)
{
    const int dir_err = ::mkdir(path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    return dir_err != -1;
}

const std::string get_env(const char *name)
{
    auto r = getenv(name);
    if (r)
    {
        return std::string(r);
    }
    return "";
}

const std::string get_env(const std::string &name)
{
    auto r = getenv(name.c_str());
    if (r)
    {
        return std::string(r);
    }
    return "";
}

std::string dir_name(const std::string &path)
{
    auto pos = path.find_last_of("/\\");
    return path.substr(0, pos);
}

}; // namespace utils

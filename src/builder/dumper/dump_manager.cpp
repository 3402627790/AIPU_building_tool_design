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
 * @Author: Kyle Huang
 * @Date: 2022-06-20 17:00:00
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2022-06-20 17:00:00
 */

#include "dump_manager.h"
#include "core/logger.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
namespace aipubt
{

std::string DumpManager::info_convert_str(const DumpInfo &info)
{
    std::stringstream ss;

    ss << "addr"
       << "=0x" << std::hex << info.addr
       << " # size=0x" << info.size
       << "; align_bytes=0x" << info.align
       << "; mem_type=" << mem_string.at(info.mem_type)
       << "; \n";

    return ss.str();
};

static std::string read_until_delimiters(std::istream &iss, const std::string &delimiters)
{
    std::string line;
    char ch;
    while (iss.get(ch))
    {
        if (delimiters.find(ch) != std::string::npos)
        {
            iss.unget();
            break;
        }
        line.push_back(ch);
    }
    return line;
}

static std::string trim(const std::string &str)
{
    size_t first = str.find_first_not_of(" \t\n\r\f\v");
    if (std::string::npos == first)
    {
        return str;
    }
    size_t last = str.find_last_not_of(" \t\n\r\f\v");
    return str.substr(first, (last - first + 1));
}

DumpInfo DumpManager::str_convert_info(const std::string &str)
{
    DumpInfo info;

    std::istringstream iss(str);
    std::string line;

    while (iss && !iss.eof())
    {
        auto line = read_until_delimiters(iss, ";#");
        size_t pos = line.find('=');
        iss.ignore(1, '\0');
        if (pos == std::string::npos)
        {
            continue;
        }

        std::string key = line.substr(0, pos);
        std::string value = line.substr(pos + 1);

        key = trim(key);
        value = trim(value);

        key.erase(0, key.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "addr")
        {
            info.addr = std::stoul(value, nullptr, 16);
        }
        else if (key == "size")
        {
            info.size = std::stoul(value, nullptr, 16);
        }
        else if (key == "align_bytes")
        {
            info.align = static_cast<uint32_t>(std::stoull(value, nullptr, 16));
        }
        else if (key == "mem_type")
        {
            MemType mem = MemType::ddr;
            for (const auto &pair : mem_string)
            {
                if (pair.second == value)
                {
                    mem = pair.first;
                    break;
                }
            }
            info.mem_type = mem;
        }
    }
    return info;
};

uint64_t DumpManager::str_convert_value(const std::string &str, const std::string &word)
{
    size_t pos = str.find('=');
    if (pos == std::string::npos)
    {
        WARN() << "no " << word;
    }

    std::string key = str.substr(0, pos);
    std::string value = str.substr(pos + 1);

    key.erase(0, key.find_first_not_of(" \t"));
    value.erase(value.find_last_not_of(" \t") + 1);

    uint64_t result = 0;
    if (key.substr(0, 4) == word)
    {
        result = std::stoul(value, nullptr, 16);
    }
    else
    {
        WARN() << "no " << word;
    }
    return result;
}

bool DumpManager::load_addr_cfg(std::string file_name)
{
    std::ifstream file;
    file.open(file_name);
    if (!file)
    {
        ERROR() << "Failed to load file: " << file_name;
        file.close();
        return false;
    }
    std::string sec_name = "";
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line[0] == '#')
        {
            line.erase(0, 1);
        }
        line.erase(0, line.find_first_not_of(" \t"));
        line.erase(line.find_last_not_of(" \t") + 1);
        if (line.substr(0, 1) == "[")
        {
            sec_name = line;
            sec_name = sec_name.replace(sec_name.find("["), 1, "");
            sec_name = sec_name.replace(sec_name.find("]"), 1, "");
        }
        else if (line.empty())
        {
            continue;
        }
        else if (line.substr(0, 7) == "version")
        {
            line.erase(0, 7);
            size_t pos = line.find('.');
            if (pos == std::string::npos)
            {
                WARN() << "no version";
            }
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            key.erase(0, key.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            // INFO() << "key " << key << " value " << value;

            m_major_v = std::stoul(key, nullptr, 10);
            m_minor_v = std::stoul(value, nullptr, 10);

            INFO() << "address config file version: " << m_major_v << "." << m_minor_v;
        }
        else if (line.substr(0, 4) == "asid" && sec_name != "")
        {
            auto asid = str_convert_value(line, "asid");
            uint32_t bss_idx = std::stoull(sec_name.substr(3, 1), nullptr, 10);
            if (m_asid_vec.size() < bss_idx + 1)
            {
                m_asid_vec.resize(bss_idx + 1);
            }
            uint32_t asid_idx = std::stoull(line.substr(4, 1), nullptr, 10);
            if (m_asid_vec[bss_idx].size() < asid_idx + 1)
            {
                m_asid_vec[bss_idx].resize(asid_idx + 1);
            }
            m_asid_vec[bss_idx][asid_idx] = asid;
        }
        else if (line.substr(0, 4) == "addr" && sec_name != "")
        {
            auto info = str_convert_info(line);
            info.name = sec_name;
            m_dump_vec.push_back(info);
        }
    }
    return true;
};

bool DumpManager::dump_addr_cfg(const std::string file_name)
{
    std::ofstream file;
    file.open(file_name, std::ios::out);
    if (!file)
    {
        file.close();
        ERROR() << "Failed to save file: " << file_name;
        return false;
    }

    file << "# version " << to_string(m_major_v) << "." << to_string(m_minor_v) << "\n\n";

    for (size_t i = 0; i < m_asid_vec.size(); ++i)
    {
        auto asid_vec = m_asid_vec[i];
        file << "[bss" << to_string(i) << "]\n";
        for (size_t j = 0; j < asid_vec.size(); ++j)
        {
            file << "asid" << to_string(j) << "=0x" << std::hex << asid_vec[j] << "\n";
        }
        file << "\n";
    }

    for (auto info : m_dump_vec)
    {
        file << "[" << info.name << "]\n";
        file << info_convert_str(info);
        file << "\n";
    }
    file.close();
    return true;
}
bool DumpManager::add_cfg(std::string name, uint32_t addr, const std::vector<MemInfo> &cfg, MemType mem)
{
    DumpInfo info;
    info.name = name;
    info.addr = addr;
    info.size = cfg.back().address + cfg.back().size;
    info.align = cfg.front().align;
    info.mem_type = mem;
    m_dump_vec.push_back(info);

    return true;
}
bool DumpManager::add_buf(std::string name, uint32_t addr, uint32_t size, uint32_t align)
{
    DumpInfo info;
    info.name = name;
    info.addr = addr;
    info.size = size;
    info.align = align;
    info.mem_type = MemType::ddr;
    m_dump_vec.push_back(info);
    return true;
}
std::vector<DumpInfo> DumpManager::get_mem_vec()
{
    return m_dump_vec;
}

bool DumpManager::add_asid(const std::vector<uint64_t> &asid_vec, uint32_t bss_id)
{
    if (bss_id != m_asid_vec.size())
    {
        WARN() << "incorrect order";
        return false;
    }
    m_asid_vec.push_back(asid_vec);
    return true;
}

std::vector<uint64_t> DumpManager::get_asid(uint32_t bss_id)
{
    if (bss_id >= m_asid_vec.size())
    {
        WARN() << "incorrect bss id";
        return std::vector<uint64_t>();
    }
    return m_asid_vec[bss_id];
}

} // namespace aipubt
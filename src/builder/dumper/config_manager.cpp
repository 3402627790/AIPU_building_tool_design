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

#include "config_manager.h"
#include "builder/target.h"
#include "core/logger.h"
#include "utils/string_utils.h"
#include <algorithm>
#include <cstdint>
namespace aipubt
{
/*dump buffer data into binary file*/
static bool dump_buf_to_bin(char *buf, uint32_t size, std::string name)
{
    std::ofstream file;
    file.open(name, std::ios::out | std::ios::binary);
    file.write(buf, size);
    bool ret = !file.fail();
    if (!ret)
    {
        ERROR() << "file open for " << name << " failed! ";
        // cerr << "open failure as expected: " << strerror(errno) << '\n';
    }
    file.close();
    return ret;
}

bool ConfigManager::dump_relative_output()
{
    for (auto output : m_output_vec)
    {
        output.bin = new char[output.size];
        for (auto input : m_input_vec)
        {
            if ((input.addr >= output.addr + output.size) || (output.addr >= input.addr + input.size))
            {
                continue;
            }

            uint64_t output_start = 0;
            uint64_t input_start = 0;
            uint64_t size = 0;
            if (input.addr > output.addr)
            {
                size = input.addr + input.size > output.addr + output.size ? output.addr + output.size - input.addr : input.size;
                if (input.addr + input.size > output.addr + output.size)
                {
                    input_start = 0;
                    output_start = input.addr - output.addr;
                    size = output.addr + output.size - input.addr;
                }
                else
                {
                    input_start = 0;
                    output_start = input.addr - output.addr;
                    size = input.size;
                }
            }
            else
            {
                if (input.addr + input.size > output.addr + output.size)
                {
                    input_start = output.addr - input.addr;
                    output_start = 0;
                    size = output.size;
                }
                else
                {
                    input_start = output.addr - input.addr;
                    output_start = 0;
                    size = input.addr + input.size - output.addr;
                }
            }
            memcpy(output.bin + output_start, input.bin + input_start, size);
        }
        dump_buf_to_bin(output.bin, output.size, output.name);
        delete output.bin;
    }
    return true;
}

std::string ConfigManager::dump_graph_json(std::string prefix, std::string graph_json)
{
    std::string json_file_name = prefix + ".graph.json";
    if (!graph_json.empty())
    {
        std::ofstream json_file(json_file_name, std::ios::out | std::ios::trunc);
        json_file.write(graph_json.c_str(), graph_json.size());
        json_file.close();
    }
    else
    {
        WARN() << "nothing in graph_json! ";
    }
    return json_file_name;
}

bool ConfigManager::gen_config_files(std::string cfg_file_name, std::string graph_json_name, const std::vector<std::string> &profile_option)
{
    std::string calloc_str = m_calloc ? "true" : "false";
    std::string fp_mode_str = m_fp_mode ? "1" : "0";
    std::string log_verbose_str = m_log_verbose ? "true" : "false";
    std::stringstream oss;
    std::string config = m_core > 1 ? m_target + "MP" + std::to_string(m_core) : m_target;
    // if (m_target.find("X3") != std::string::npos)
    // {
    //     set_target(config);
    //     auto split_target = utils::split_string(config, '_');
    //     config.clear();
    //     std::string config_cluster = "K1";
    //     std::string config_core = "C" + std::to_string(get_target()->core_num());
    //     std::string config_aiff = "A" + std::to_string(static_cast<int>(get_target()->aiff_num()));
    //     std::string config_tec = "T" + std::to_string(static_cast<int>(get_target()->tec_num()));
    //     config = split_target[0] + "-" + config_cluster + config_core + config_aiff + config_tec;
    // }
    oss << "# autogenrated by aipurun, do NOT modify!\n\n"
        << "[COMMON]\n"
        << "ENABLE_AVX=false\n"
        << "LOG_FILEPATH= .\n"
        << "LOG_LEVEL="
        << m_log_level
        << "\n"
        << "LOG_VERBOSE="
        << log_verbose_str
        << "\n"
        << "ENABLE_CALLOC="
        << calloc_str
        << "\n"
        << "FP_MODE="
        << fp_mode_str
        << "\n"
        << "GM_SIZE="
        << std::to_string(m_gm_size) + "MiB"
        << "\n"
        << "DDR_SIZE="
        << std::to_string(m_ddr_size) + "GiB"
        << "\n"
        << "CONFIG="
        << config
        << "\n\n";

    if (m_profile)
    {
        oss << "[PROFILE]\n";
        if (m_target.find("X3") != std::string::npos)
        {
            oss << "MODE=0\n";
        }
        else
        {
            oss << "MODE=2\n";
        }
        for (auto option : profile_option)
        {
            oss << option << "\n";
        }
        oss << "print_subg_info=1\n";
        oss << "graph_filename=";
        oss << graph_json_name;
        oss << "\n";
        oss << "\n";
    }

    std::string name;
    int count = 0;
    oss << "[INPUT]\n";

    std::stable_sort(m_input_vec.begin(), m_input_vec.end(), [](const FileInfo &f1, const FileInfo &f2) -> bool {
        return f1.addr < f2.addr;
    });

    for (auto input : m_input_vec)
    {
        if (input.size > 0)
        {
            oss << "FILE" << std::dec << count << "=" << input.name << "\n";
            oss << "BASE" << std::dec << count << "=0x" << std::hex << input.addr << "\n";
            dump_buf_to_bin(input.bin, input.size, input.name);
            count++;
        }
    }
    oss << "COUNT=" << std::dec << count << "\n\n";

    oss << "[HOST]\n"
        << "TCBP_HI=0x" << std::hex << (uint32_t)(m_tcbp >> 32) << "\n"
        << "TCBP_LO=0x" << std::hex << (uint32_t)(m_tcbp & 0xffffffff) << "\n";

    if (m_target.find("X3") != std::string::npos)
    {
        oss << "TCB_NUM=0x" << std::hex << m_tcb_num << "\n";
    }

    oss << "\n";

    count = 0;
    oss << "[OUTPUT]\n";

    std::stable_sort(m_output_vec.begin(), m_output_vec.end(), [](const FileInfo &f1, const FileInfo &f2) -> bool {
        return f1.addr < f2.addr;
    });

    for (auto output : m_output_vec)
    {
        if (output.size > 0)
        {
            oss << "FILE" << std::dec << count << "=" << output.name << "\n";
            oss << "BASE" << std::dec << count << "=0x" << std::hex << output.addr << "\n";
            oss << "SIZE" << std::dec << count << "=0x" << std::hex << output.size << "\n";
            count++;
        }
    }
    oss << "COUNT=" << std::dec << count << "\n";
    std::ofstream ofs;
    ofs.open(cfg_file_name, std::ios::out);
    ofs << oss.str();
    ofs.close();

    return true;
};

bool ConfigManager::set_gm_size(uint32_t size)
{
    m_gm_size = size;
    return true;
}
bool ConfigManager::set_ddr_size(uint32_t size)
{
    m_ddr_size = size;
    return true;
}
bool ConfigManager::set_log_level(uint32_t level)
{
    m_log_level = level;
    return true;
}
bool ConfigManager::set_calloc(bool set)
{
    m_calloc = set;
    return true;
}
bool ConfigManager::set_fp_mode(bool set)
{
    m_fp_mode = set;
    return true;
}
bool ConfigManager::set_log_verbose(bool set)
{
    m_log_verbose = set;
    return true;
}
bool ConfigManager::set_tcbp(uint64_t addr)
{
    m_tcbp = addr;
    return true;
}
bool ConfigManager::set_tcb_num(uint32_t num)
{
    m_tcb_num = num;
    return true;
}
bool ConfigManager::set_profile(bool set)
{
    m_profile = set;
    return true;
}
bool ConfigManager::add_input(std::string name, uint64_t addr, char *bin, uint32_t size)
{
    // m_input[name] = {bin, addr, size};
    m_input_vec.push_back({name, bin, addr, size});
    return true;
}
bool ConfigManager::add_output(std::string name, uint64_t addr, uint32_t size)
{
    // m_output[name] = {nullptr, addr, size};
    m_output_vec.push_back({name, nullptr, addr, size});
    return true;
}
} // namespace aipubt
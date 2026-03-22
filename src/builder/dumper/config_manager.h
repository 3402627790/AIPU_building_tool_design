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
#ifndef BUILDER_CONFIG_MANAGER_H
#define BUILDER_CONFIG_MANAGER_H

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace aipubt
{
class ConfigManager
{
    struct FileInfo
    {
        std::string name;
        char *bin;
        uint64_t addr;
        uint32_t size;
    };

private:
    std::string m_target;
    uint32_t m_core;
    // std::map<std::string, FileInfo> m_input;
    std::vector<FileInfo> m_input_vec;
    // std::map<std::string, FileInfo> m_output;
    std::vector<FileInfo> m_output_vec;
    uint32_t m_gm_size;
    uint32_t m_ddr_size;
    uint32_t m_log_level;
    bool m_calloc;
    bool m_fp_mode;
    bool m_log_verbose;
    uint64_t m_tcbp;
    uint32_t m_tcb_num;
    bool m_profile;

public:
    bool gen_config_files(std::string cfg_file_name, std::string graph_json_name, const std::vector<std::string> &profile_option);
    std::string dump_graph_json(std::string prefix, std::string graph_json);
    bool dump_relative_output();
    bool set_gm_size(uint32_t size);
    bool set_ddr_size(uint32_t size);
    bool set_log_level(uint32_t level);
    bool set_calloc(bool set);
    bool set_fp_mode(bool set);
    bool set_log_verbose(bool set);
    bool set_tcbp(uint64_t addr);
    bool set_tcb_num(uint32_t num);
    bool set_profile(bool set);
    bool add_input(std::string name, uint64_t addr, char *bin, uint32_t size);
    bool add_output(std::string name, uint64_t addr, uint32_t size);
    ConfigManager(std::string target, uint32_t core)
    {
        m_target = target;
        m_core = core;
    };
    ~ConfigManager(){};
};
} // namespace aipubt

#endif
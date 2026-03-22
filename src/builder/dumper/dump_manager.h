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
#ifndef BUILDER_DUMP_MANAGER_H
#define BUILDER_DUMP_MANAGER_H

#include "builder/dumper/dumper_x.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace aipubt
{

struct DumpInfo
{
    std::string name;
    uint32_t addr;
    uint32_t size;
    uint32_t align;
    MemType mem_type; //0: ddr 1: ocm 2: gm
};

class DumpManager
{
private:
    int m_major_v;
    int m_minor_v;
    // std::map<std::string, std::string> m_mem_map;
    // std::map<std::string, std::vector<std::string>> m_reuse_map;
    // std::map<std::string, std::vector<std::string>> m_const_map;
    std::vector<DumpInfo> m_dump_vec;
    std::map<std::string, std::vector<MemInfo>> m_reuse_map;
    std::map<std::string, std::vector<MemInfo>> m_const_map;
    std::vector<std::vector<uint64_t>> m_asid_vec;
    std::string info_convert_str(const DumpInfo &info);
    DumpInfo str_convert_info(const std::string &str);
    uint64_t str_convert_value(const std::string &str, const std::string &word);

public:
    bool load_addr_cfg(std::string file_name);
    bool dump_addr_cfg(std::string file_name);
    bool add_cfg(std::string name, uint32_t addr, const std::vector<MemInfo> &cfg, MemType mem);
    bool add_buf(std::string name, uint32_t addr, uint32_t size, uint32_t align);
    bool add_asid(const std::vector<uint64_t> &asid_vec, uint32_t bss_id);
    std::vector<uint64_t> get_asid(uint32_t bss_id);
    std::vector<DumpInfo> get_mem_vec();
    DumpManager(std::string target, int version)
    {

        m_major_v = ((target == "X3_1304") || (target == "X3_1204") || (target == "X3_1302")) ? 1 : 0;
        m_minor_v = 0;
    };
    ~DumpManager(){};
};
} // namespace aipubt

#endif
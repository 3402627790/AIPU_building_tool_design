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
#ifndef BUILDER_MEM_MANAGER_H
#define BUILDER_MEM_MANAGER_H

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace aipubt
{

#define MAX_BASE_ADDR 0xe0000000 //shouldn't larger than 0xe0000000

class MemManager
{
    enum MemStatus
    {
        TEMP,
        FIXED,
        NONE,
    };

    enum MemType
    {
        REUSE,
        CONST,
        ZEROCPY_CONST,
        OTHER,
    };

    struct MemBlk
    {
        uint32_t addr;
        uint32_t size;
        uint32_t align;
        uint64_t asid_offset;
        MemStatus status;
        MemBlk *next;
        MemBlk *prev;
    };

private:
    std::map<std::string, MemBlk *> m_mem_map;
    std::map<std::string, std::string> m_share_map;
    MemBlk *m_head;
    MemBlk *m_tail;
    void insert_blk(MemBlk *current_blk, MemBlk *new_blk);
    bool assign_mem(MemBlk *blk, MemBlk *start);
    uint64_t m_smmu_addr;
    uint32_t m_smmu_offset;
    bool m_smmu_enable;

public:
    MemManager();
    ~MemManager();
    bool share_mem(std::string name, std::string mem_to_be_shared);
    bool assign_mem(std::string name, uint32_t size, uint64_t asid_addr, uint32_t align, bool fixed = false);
    bool insert_mem(std::string name, uint32_t addr, uint32_t size, uint64_t asid_addr, uint32_t align, bool fixed = false);
    bool fix_mem(std::string name, uint32_t size);
    bool has_mem(std::string name);
    uint64_t get_abs_addr(std::string name);
    uint32_t get_addr(std::string name);
    uint32_t get_size(std::string name);
    uint32_t get_align(std::string name);
    void show();
    bool mem_check();
    uint64_t get_max_addr();
    bool set_smmu(uint64_t asid_addr, uint32_t offset);
};

}; // namespace aipubt
#endif
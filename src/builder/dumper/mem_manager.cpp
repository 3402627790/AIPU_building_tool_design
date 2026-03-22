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

#include "mem_manager.h"
#include "core/logger.h"
#include <algorithm>
#include <cstdint>
namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 4096)
{
    return (s + align - 1) & ~(align - 1);
}

MemManager::MemManager()
{
    m_head = new MemBlk{0, 0, 0, 0, FIXED, nullptr, nullptr};
    m_tail = new MemBlk{0, 0, 0, 0, FIXED, nullptr, nullptr};
    m_head->next = m_tail;
    m_tail->prev = m_head;
    m_smmu_enable = false;
    m_smmu_addr = 0;
    m_smmu_offset = 0;
};

MemManager::~MemManager()
{
    for (auto iter = m_mem_map.begin(); iter != m_mem_map.end(); ++iter)
    {
        delete iter->second;
    }
    delete m_head;
    delete m_tail;
}

void MemManager::insert_blk(MemBlk *current_blk, MemBlk *new_blk)
{
    new_blk->prev = current_blk;
    new_blk->next = current_blk->next;
    current_blk->next->prev = new_blk;
    current_blk->next = new_blk;
}

bool MemManager::assign_mem(MemBlk *blk, MemBlk *start)
{
    bool ret = true;
    MemBlk *current = start;
    while (current != m_tail)
    {
        uint64_t abs_addr = current->addr + current->asid_offset;
        uint64_t blk_addr = blk->asid_offset;
        uint64_t pre_assign_addr = aligned(abs_addr + current->size, blk->align);
        uint64_t new_abs_addr = blk_addr > pre_assign_addr ? blk_addr : pre_assign_addr;
        uint64_t next_addr = current->next->addr + current->next->asid_offset;

        if (((new_abs_addr + blk->size) < next_addr) || (current->next == m_tail))
        {
            if (current->status != FIXED)
            {
                WARN() << "the blk status is not FIXED";
                ret = false;
            }
            else
            {
                blk->addr = new_abs_addr - blk->asid_offset;
                uint64_t total = (uint64_t)blk->addr + blk->size;
                if (m_smmu_enable && blk->asid_offset == m_smmu_addr)
                {
                    if (total > 0x80000000)
                    {
                        ERROR() << "smmu assigned out of bound(>0x8000000)";
                    }
                }
                else
                {
                    if (total > 0xe0000000)
                    {
                        ERROR() << "memory assigned in internal address(>0xe0000000)";
                    }
                }
                insert_blk(current, blk);
                ret = true;
            }
            break;
        }
        current = current->next;
    }
    return ret;
}

bool MemManager::assign_mem(std::string name, uint32_t size, uint64_t asid_addr, uint32_t align, bool fixed)
{
    bool ret = true;
    MemStatus status = fixed ? FIXED : TEMP;
    MemBlk *blk = new MemBlk{0, size, align, asid_addr, status, nullptr, nullptr};
    if (fixed)
    {
        ret = assign_mem(blk, m_head);
    }
    m_mem_map.insert(std::make_pair(name, blk));
    return ret;
}

bool MemManager::share_mem(std::string name, std::string mem_to_be_shared)
{
    bool ret = true;
    if (m_share_map.find(name) != m_share_map.end())
    {
        ERROR() << name << " already exist";
        ret = false;
    }
    if (m_mem_map.find(mem_to_be_shared) == m_mem_map.end())
    {
        ERROR() << "cannot find " << mem_to_be_shared;
        ret = false;
    }
    m_share_map.insert(std::make_pair(name, mem_to_be_shared));
    return ret;
}

bool MemManager::insert_mem(std::string name, uint32_t addr, uint32_t size, uint64_t asid_addr, uint32_t align, bool fixed)
{
    // INFO() << "insert " << name << " " << " addr: 0x" << std::hex << addr;
    bool ret = true;
    MemStatus status = fixed ? FIXED : TEMP;
    MemBlk *blk = new MemBlk{addr, size, align, asid_addr, status, nullptr, nullptr};
    MemBlk *current = m_head;
    MemBlk *last = m_head;
    while (current != m_tail)
    {
        uint64_t abs_current_addr = current->addr + current->asid_offset;
        uint64_t abs_addr = addr + asid_addr;
        if (abs_current_addr > abs_addr)
        {
            insert_blk(last, blk);
            break;
        }
        else if (current->next == m_tail)
        {
            insert_blk(current, blk);
            break;
        }
        last = current;
        current = current->next;
    }
    m_mem_map.insert(std::make_pair(name, blk));
    return ret;
}

bool MemManager::fix_mem(std::string name, uint32_t size)
{
    bool ret = true;
    auto it = m_mem_map.find(name);
    if (it != m_mem_map.end())
    {
        auto blk = it->second;
        if (blk->status == TEMP)
        {
            blk->status = FIXED;
            blk->size = size;
            assign_mem(blk, m_head);
        }
        // else
        // {
        //     WARN() << "duplicate fix mem " << name;
        // }
        ret = true;
    }
    else
    {
        WARN() << "cannot find " << name;
    }
    return ret;
}

bool MemManager::has_mem(std::string name)
{
    bool ret;
    auto it = m_mem_map.find(name);
    if (it != m_mem_map.end())
    {
        ret = true;
    }
    else
    {
        ret = false;
    }
    return ret;
}
uint64_t MemManager::get_abs_addr(std::string name)
{
    uint64_t abs_addr = 0;
    auto it = m_mem_map.find(name);
    if (it != m_mem_map.end())
    {
        auto blk = it->second;
        if (blk->status == TEMP)
        {
            ERROR() << "shouldn't get a unfix address";
        }
        abs_addr = blk->addr + blk->asid_offset;
    }
    else if (m_share_map.find(name) != m_share_map.end())
    {
        auto shared_name = m_share_map[name];
        abs_addr = get_abs_addr(shared_name);
    }
    // else
    // {
    //     DEBUG() << "cannot find " << name;
    // }

    return abs_addr;
}

uint32_t MemManager::get_addr(std::string name)
{
    uint32_t addr = 0;
    auto it = m_mem_map.find(name);
    if (it != m_mem_map.end())
    {
        auto blk = it->second;
        if (blk->status == TEMP)
        {
            ERROR() << "shouldn't get a unfix address";
        }
        if (m_smmu_enable && blk->asid_offset == m_smmu_addr)
        {
            addr = blk->addr + m_smmu_offset;
        }
        else
        {
            addr = blk->addr;
        }
    }
    else if (m_share_map.find(name) != m_share_map.end())
    {
        auto shared_name = m_share_map[name];
        addr = get_addr(shared_name);
    }
    // else
    // {
    //     DEBUG() << "cannot find " << name;
    // }
    return addr;
}
uint32_t MemManager::get_size(std::string name)
{
    uint32_t addr = 0;
    auto it = m_mem_map.find(name);
    if (it != m_mem_map.end())
    {
        auto blk = it->second;
        addr = blk->size;
    }
    else if (m_share_map.find(name) != m_share_map.end())
    {
        auto shared_name = m_share_map[name];
        addr = get_size(shared_name);
    }
    // else
    // {
    //     DEBUG() << "cannot find " << name;
    // }
    return addr;
}

uint32_t MemManager::get_align(std::string name)
{
    uint32_t addr = 0;
    auto it = m_mem_map.find(name);
    if (it != m_mem_map.end())
    {
        auto blk = it->second;
        addr = blk->align;
    }
    else if (m_share_map.find(name) != m_share_map.end())
    {
        auto shared_name = m_share_map[name];
        addr = get_align(shared_name);
    }
    // else
    // {
    //     WARN() << "cannot find " << name;
    // }
    return addr;
}
void MemManager::show()
{
    // for (std::map<std::string, MemBlk *>::iterator it = m_mem_map.begin(); it != m_mem_map.end(); ++it)
    MemBlk *blk = m_head->next;
    std::string name;
    while (blk != m_tail)
    {
        for (std::map<std::string, MemBlk *>::iterator it = m_mem_map.begin(); it != m_mem_map.end(); ++it)
        {
            if (it->second == blk)
            {
                name = it->first;
                break;
            }
        }

        if (name == "")
        {
            WARN() << "cannot find";
            return;
        }

        INFO() << "[MEM]"
               << std::setfill(' ') << std::setw(14) << name << ":"
               << std::setfill(' ') << std::setw(8) << "addr(0x"
               << std::setfill('0') << std::setw(8) << std::hex << blk->addr << ")"
               << std::setfill(' ') << std::setw(8) << "size(0x"
               << std::setfill('0') << std::setw(8) << std::hex << blk->size << ")"
               << std::setfill(' ') << std::setw(15) << "asid_offset(0x"
               << std::setfill('0') << std::setw(16) << std::hex << blk->asid_offset << ")"
               << std::setfill(' ') << std::setw(12) << "abs_addr(0x"
               << std::setfill('0') << std::setw(16) << std::hex << blk->asid_offset + blk->addr << ")";
        blk = blk->next;
    }
    return;
}
bool MemManager::mem_check()
{
    return true;
}
uint64_t MemManager::get_max_addr()
{
    uint64_t max_addr = 0;
    MemBlk *blk = m_head->next;
    while (blk != m_tail)
    {
        if (max_addr < blk->asid_offset + blk->addr + blk->size)
        {
            max_addr = blk->asid_offset + blk->addr + blk->size;
        }
        blk = blk->next;
    }
    return max_addr;
}

bool MemManager::set_smmu(uint64_t asid_addr, uint32_t offset)
{
    m_smmu_enable = true;
    m_smmu_addr = asid_addr;
    m_smmu_offset = offset;
    return true;
}
} // namespace aipubt
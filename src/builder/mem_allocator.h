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
 * @Date: 2020-01-29 14:26:44 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-04-01 17:10:35
 */
#ifndef BUILDER_MEM_ALLOCATOR_H
#define BUILDER_MEM_ALLOCATOR_H
#include <set>
#include <vector>

#include "allocator.h"
#include "core/core.h"
#include "mmu_allocator.h"
#include "utils/utils.h"
namespace aipubt
{
using namespace SegementMMUAllocator;
using MemoryBlk = std::shared_ptr<SegementMMUAllocator::Memory>;
class MemoryAllocator : Allocator
{
public:
    enum SRAMLevel
    {
        //lower is faster for AIPU
        L0 = 0, // internal sram and close to ALU
        L1 = 1, //internal sram but to close
        L2 = 2, //internal sram for sharing data for different unit
        L3 = 3, //DTCM sram
        L4 = 4, //external sram
    };
    MemoryAllocator();
    ~MemoryAllocator();
    MemoryObjectPtr allocate(unsigned int size, MemoryObjectPtr) override;
    void free(MemoryObjectPtr ptr) override;
    std::vector<MemoryObjectPtr> get_allocated();
    virtual std::vector<MemoryObjectPtr> slots()
    {
        return m_slots;
    }
    void enable_soc_sram(bool en)
    {
        m_enable_sram = en;
    }
    virtual void add_soc_sram(uint32_t base, uint32_t size, SRAMLevel l = SRAMLevel::L3);
    virtual void add_soc_sram(MemoryObjectPtr sram, SRAMLevel l = SRAMLevel::L3);
    virtual void remove_soc_sram(uint32_t base);
    virtual void remove_soc_sram(MemoryObjectPtr sram);
    virtual std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> get_liveness();
    virtual uint32_t clock();
    virtual uint32_t get_clock();
    virtual void compress_clock();

    // virtual std::vector<MemoryObjectPtr> get_all_soc_sram();
    virtual void print()
    {
        std::vector<size_t> sizes = {};
        for (auto slot : m_slots)
        {
            sizes.push_back(slot->size >> 10);
        }
        INFO() << utils::join_string("\t", sizes) << "   (KB)";
        INFO() << utils::join_string("\t", m_availability);
        INFO() << "";
    }

protected:
    std::vector<MemoryObjectPtr> m_slots;     //ptr
    std::vector<MemoryObjectPtr> m_allocated; //ptr
    std::vector<bool> m_availability;
    std::vector<MemoryBlk> m_soc_sram;
    std::map<MemoryObjectPtr, std::pair<MemoryBlk, MemoryObjectPtr>> m_allocated_sram; // tensor ptr-> which memory,which block
    bool m_enable_sram;
};

/**
 * @brief New a memory allocator accroding to user's config
 * 
 * @return MemoryAllocator* 
 */
MemoryAllocator *get_mem_allocator();

};     // namespace aipubt
#endif //!BUILDER_MEM_ALLOCATOR_H

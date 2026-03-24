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
 * @Date: 2020-01-29 14:26:48 
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-11-27 17:48:02
 */
#include "mem_allocator.h"
#include "amat_allocator.h"
#include "builder/target.h"
#include "core/config.h"
#include <algorithm>

#define SLOT_ALIGNED_SIZE 512
#define ALIGNED(x) ((x + (SLOT_ALIGNED_SIZE)-1) & ~((SLOT_ALIGNED_SIZE)-1))
namespace aipubt
{
MemoryAllocator::MemoryAllocator()
{
    //add default memory
    //for test
    // m_soc_sram.push_back(std::make_shared<Memory>(0xc2000000, 0x400000));
    m_enable_sram = false;
}

MemoryAllocator::~MemoryAllocator()
{
    for (auto slot : m_slots)
    {
        if (slot)
        {
            // delete slot;
        }
    }
}

void MemoryAllocator::add_soc_sram(uint32_t base, uint32_t size, MemoryAllocator::SRAMLevel l)
{
    for (auto sram : m_soc_sram)
    {
        if (sram->rootblk()->m_base == base)
        {
            WARN() << "The SOC SRAM's base" << std::hex << base << " has be occupied.";
            return;
        }
    }
    auto sram = std::make_shared<MemoryObject>(base);
    sram->size = size;
    add_soc_sram(sram, l);
}
void MemoryAllocator::add_soc_sram(MemoryObjectPtr sram, SRAMLevel l)
{
    if (sram->base_offset != nullptr)
    {
        WARN() << "The SOC SRAM object is a shared object, the object's base must be null.";
        return;
    }
    for (auto s : m_soc_sram)
    {
        if (s->rootblk()->get_base_object() == sram)
        {
            WARN() << "The SOC SRAM's base" << std::hex << sram << " has be occupied.";
            return;
        }
    }
    m_soc_sram.push_back(std::make_shared<Memory>(sram, sram->size));
}

void MemoryAllocator::remove_soc_sram(uint32_t base)
{
    for (uint32_t i = 0; i < m_soc_sram.size(); ++i)
    {
        if (m_soc_sram[i]->rootblk()->m_base == base)
        {
            m_soc_sram.erase(m_soc_sram.begin() + i);
            break;
        }
    }
}

void MemoryAllocator::remove_soc_sram(MemoryObjectPtr base)
{
    for (uint32_t i = 0; i < m_soc_sram.size(); ++i)
    {
        if (m_soc_sram[i]->rootblk()->get_base_object() == base)
        {
            m_soc_sram.erase(m_soc_sram.begin() + i);
            break;
        }
    }
}

std::vector<MemoryObjectPtr> MemoryAllocator::get_allocated()
{
    return m_allocated;
}

static inline size_t get_align(size_t x, size_t align = SLOT_ALIGNED_SIZE)
{
    size_t ret;
    auto target = Target::get_the_target();
    if (target->isainfo().type != ISA_ZY)
    {
        align = std::max(align, size_t(16 * 1024));
    }
    ret = (x + (align)-1) & ~((align)-1);
    return ret;
}

MemoryObjectPtr MemoryAllocator::allocate(unsigned int size, MemoryObjectPtr ptr)
{
    size = get_align(size);
    if (!ptr)
    {
        ptr = MemoryObjectPtr(new MemoryObject(0));
    }
    ptr->size = size;
    // try allocate in soc sram
    if (m_enable_sram)
    {
        for (auto &mem : m_soc_sram)
        {
            auto alloc = mem->allocate(size);
            if (alloc)
            {
                // INFO() << "allocating soc sram:" << size;
                m_allocated_sram[ptr] = {mem, alloc};
                ptr->set_base_offset(alloc->offset());
                m_allocated.push_back(ptr);
                return ptr;
            }
        }
    }

    // INFO() << "request size: " << (size >> 10) << " KB";
    auto it = std::find(m_availability.begin(), m_availability.end(), true);
    int idx = 0;
    int num_slots = m_slots.size();
    if (!num_slots || it == m_availability.end())
    {
        // no available memory
        // MemoryObjectPtr slot = new MemoryObject(m_slots[num_slots - 1].first, m_slots[num_slots - 1].second);
        MemoryObjectPtr slot = MemoryObjectPtr(new MemoryObject(0));
        slot->size = size;
        slot->foot_print = ptr->foot_print;
        m_slots.push_back(slot);
        m_availability.push_back(false);
        idx = num_slots;
    }
    else
    {
        //find the best slot
        int idx_up = -1;
        int idx_down = -1;
        size_t dis_up = INT32_MAX;
        size_t dis_down = INT32_MAX;
        idx = std::distance(m_availability.begin(), it);
        for (auto i = idx; i < num_slots; i++)
        {
            auto slot_size = m_slots[i]->size;
            if (m_availability[i])
            {
                // up
                if (slot_size >= size && (slot_size - size) < dis_up)
                {
                    dis_up = slot_size - size;
                    idx_up = i;
                }
                if (slot_size < size && (size - slot_size) < dis_down)
                {
                    dis_down = size - slot_size;
                    idx_down = i;
                }
            }
        }
        // INFO() << "idx_up: " << idx_up << ", idx_down: " << idx_down << ", dis_up: " << dis_up << ", dis_down: " << dis_down;
        const size_t ratio_up = 95;
        const size_t ratio_down = 95;
        const size_t retio_up_down = 60;
        // up
        if (idx_up >= 0 && (dis_up * 100 / (dis_up + size) <= ratio_up) && (idx_down == -1 || dis_up <= dis_down * retio_up_down))
        {
            idx = idx_up;
            m_slots[idx]->foot_print += ptr->foot_print;
            m_availability[idx] = false;
            // INFO() << "up, " << (dis_up * 100 / (dis_up + size));
        }
        // down
        else if (idx_down >= 0 && (dis_down * 100 / size <= ratio_down) && (idx_up == -1 || dis_up > dis_down * retio_up_down))
        {
            idx = idx_down;
            m_slots[idx]->size = size;
            m_slots[idx]->foot_print += ptr->foot_print;
            m_availability[idx] = false;
            // INFO() << "down," << (dis_down * 100 / size);
        }
        else
        {
            MemoryObjectPtr slot = MemoryObjectPtr(new MemoryObject(0));
            slot->size = size;
            slot->foot_print = ptr->foot_print;
            m_slots.push_back(slot);
            m_availability.push_back(false);
            idx = num_slots;
            // INFO() << "new";
        }
    }
    // INFO() << "idx: " << idx;
    ptr->set_base_offset(m_slots[idx]);
    m_allocated.push_back(ptr);
    return ptr;
}

void MemoryAllocator::free(MemoryObjectPtr ptr)
{
    if (!ptr)
    {
        return;
    }
    if (m_allocated_sram.count(ptr))
    {
        // INFO() << "free soc sram";
        auto p = m_allocated_sram[ptr];
        p.first->free(p.second);
    }
    for (unsigned int i = 0; i < m_availability.size(); ++i)
    {
        if (ptr->base_offset && ptr->base_offset == m_slots[i])
        {
            m_availability[i] = true;
            return;
        }
    }

    // auto mmu = ptr->get_mmu_object();
    // INFO()<<"cannot free ptr , 0x"<<std::hex << ptr->offset();
    // if (mmu)
    // {
    //     INFO() << "base: 0x" << std::hex << mmu->base_offset->offset()
    //            << ", next: 0x" << std::hex << mmu->next_block->offset();
    //     INFO() << "";
    // }
    // m_availability[m_availability.size() - 1] = true;
}

std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> MemoryAllocator::get_liveness()
{
    std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> ret;
    return ret;
}

uint32_t MemoryAllocator::clock()
{
    return 0;
}

uint32_t MemoryAllocator::get_clock()
{
    return 0;
}

void MemoryAllocator::compress_clock()
{
    return;
}

MemoryAllocator *get_mem_allocator()
{
    if (BuilderConfig::get<bool>(Constants::USE_SLOTS))
    {
        return new MemoryAllocator();
    }
    return new AMATAllocator();
}

}; // namespace aipubt

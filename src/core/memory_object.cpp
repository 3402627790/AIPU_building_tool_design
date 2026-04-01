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
 * @Date: 2020-01-29 14:28:07 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-02-10 10:20:29
 */
#include "core.h"
#define MAX(a, b) ((a) > (b) ? (a) : (b))
namespace aipubt
{
MemoryObject::MemoryObject(int64_t base, int64_t off)
{
    base_offset = nullptr;
    m_base = (uint64_t)base;
    relative_offset = (uint64_t)off;
}
MemoryObject::MemoryObject(MemoryObjectPtr base, int off)
{
    m_base = 0;
    base_offset = base;
    relative_offset = off;
};
uint64_t MemoryObject::offset()
{
    if (base_offset)
        return base_offset->offset() + relative_offset;
    return m_base + relative_offset;
}
void MemoryObject::set_base_offset(uint64_t base)
{
    m_base = base;
}
void MemoryObject::set_base_offset(uint32_t base)
{
    m_base = (uint64_t)base;
}
void MemoryObject::set_base_offset(int base)
{
    m_base = (uint64_t)base;
}
void MemoryObject::set_base_offset(MemoryObjectPtr base)
{
    aipu_assert(base.get() != this && "Cannot set base offset as self.");
    // enable_shared_from_this
    if (is_shared_by(base))
    {
        WARN() << "has potential loop for memory object: " << base;
    }
    base_offset = base;
    // base->align = MAX(base->align,align);
}
uint64_t MemoryObject::get_base_memory()
{
    if (base_offset)
    {
        return base_offset->get_base_memory();
    }
    return m_base;
}

uint32_t MemoryObject::get_max_aligned()
{
    uint32_t align_size = this->align;
    if (base_offset && base_offset.get() != this)
    {
        auto size = base_offset->get_max_aligned();
        align_size = align_size > size ? align_size : size;
    }
    return align_size;
}

MemoryObjectPtr MemoryObject::get_base_object()
{
    if (base_offset)
    {
        return base_offset->get_base_object();
    }
    return shared_from_this();
}
bool MemoryObject::is_shared_by(MemoryObjectPtr child)
{
    if (!child)
    {
        return false;
    }
    if (child->base_offset == nullptr)
    {
        return false;
    }
    if (child->base_offset.get() == this)
    {
        return true;
    }
    return is_shared_by(child->base_offset);
}

MemoryObjectPtr MemoryObject::get_mmu_object()
{
    uint32_t relative_offset = 0;
    MemoryObjectPtr ptr = shared_from_this();
    while (ptr)
    {
        relative_offset += ptr->relative_offset;
        if (ptr->base_offset && ptr->next_block)
        {
            // //rewrite ~
            // this->relative_offset = relative_offset;
            // this->base_offset = ptr->base_offset;
            // this->next_block = ptr->next_block;
            return ptr;
        }
        ptr = ptr->base_offset;
    }
    return nullptr;
}
void MemoryObject::reset()
{
    relative_offset = 0;
    m_base = 0;
    align = 128;
    mask = 0;
    size = 0;
    base_offset = nullptr;
    next_block = nullptr;
    mem_flag = 0;
    foot_print = 0;
}

MemoryObjectPtr MemoryObject::merge(const std::vector<MemoryObjectPtr> &to_merge)
{
    auto region = std::make_shared<MemoryObject>(0);
    region->size = 0;
    uint32_t alignment = 32;
    for (uint32_t j = 0; j < to_merge.size(); ++j)
    {
        alignment = MAX(alignment, to_merge[j]->get_max_aligned());
        if (region->size % to_merge[j]->align != 0)
        {
            // align the slot
            region->size += (to_merge[j]->align - region->size % to_merge[j]->align);
        }
        to_merge[j]->relative_offset = region->size;
        to_merge[j]->set_base_offset(region);
        region->size += to_merge[j]->size;
        region->foot_print += to_merge[j]->foot_print;
    }
    region->align = alignment;

    return region;
}

void MemoryObject::set_buf_type(uint32_t type)
{
    mem_flag = mem_flag & ~(0xF << 4);
    auto shifted_type = (type & 0xF) << 4;
    mem_flag = mem_flag | shifted_type;
}

uint32_t MemoryObject::get_buf_type()
{
    uint32_t ret = (mem_flag >> 4) & 0xF;
    return ret;
}

void MemoryObject::set_mem_flag(uint32_t mem)
{
    mem_flag = mem_flag & ~(0xF);
    auto shifted_mem = mem & 0xF;
    mem_flag = mem_flag | shifted_mem;
}

uint32_t MemoryObject::get_mem_flag()
{
    uint32_t ret = mem_flag & 0xF;
    return ret;
}

void MemoryObject::set_attr(uint32_t attr)
{
    mem_flag = mem_flag & ~(0xF << 8);
    auto shifted_attr = (attr & 0xF) << 8;
    mem_flag = mem_flag | shifted_attr;
}

uint32_t MemoryObject::get_attr()
{
    uint32_t ret = (mem_flag >> 8) & 0xF;
    return ret;
}

}; // namespace aipubt

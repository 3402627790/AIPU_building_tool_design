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
 * @Date: 2020-01-29 14:26:57 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-02-21 18:30:47
 */
#include "mmu_allocator.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/logger.h"
#include "utils/utils.h"

#define SRAMBASE 0xf0000000
#define GSRAM0_BASE 0xf8000000
#define GSRAM0_SIZE 0x20000

#define GSRAM1_BASE 0xf8800000
#define GSRAM1_SIZE 0x20000

#define LSRAM_BASE 0xf0000000
#define LSRAM_SIZE 0x8000
#define LSRAM_MAX_SIZE 0x80000
#define LSRAM_RESERVE 0x100000
#define NUM_TEC 4

#define MMU_BASE 0xe0000000
#define MMU_MAX_SIZE 0x4000000

#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define RED_COLOR "\033[1;31m"
#define GREEN_COLOR "\033[1;32m"
#define COLOR_END "\033[0m"
namespace aipubt
{

namespace SegementMMUAllocator
{

    static SRAMFactory sram_factory;

    inline bool is_free(MemoryObjectPtr o)
    {
        return o && o->mask == MemS_FREE;
    }
    inline void MemoryObject_set_status(MemoryObjectPtr o, MemStatus status)
    {
        o->mask = status;
    }
    inline MemStatus mem_status(MemoryObjectPtr o)
    {
        return (MemStatus)o->mask;
    }
    uint32_t free_size(MemoryObjectPtr o)
    {
        uint32_t s = 0;
        aipu_assert(o);
        if (o->next_block)
        {
            s = free_size(o->next_block);
        }
        if (is_free(o))
        {
            s += o->size;
        }
        return s;
    }

    uint32_t max_free_block(MemoryObjectPtr o)
    {
        uint32_t s = 0;
        aipu_assert(o);
        if (o->next_block)
        {
            s = max_free_block(o->next_block);
        }
        if (is_free(o))
        {
            s = MAX(s, o->size);
        }
        return s;
    }

    MemoryObjectPtr split_block(MemoryObjectPtr o, uint32_t size)
    {
        if (!o)
        {
            return o;
        }
        auto right = MemoryObjectPtr(new MemoryObject(o, size));
        right->size = o->size - size;
        if (o->next_block)
        {
            auto next = o->next_block;
            next->relative_offset = right->size;
            next->base_offset = right;
            right->next_block = next;
        }
        right->base_offset = o;
        o->next_block = right;
        o->size = size;
        return o;
    }
    void merge_block(MemoryObjectPtr blk)
    {
        if (!blk)
            return;
        if (blk->base_offset && (blk->base_offset->next_block == blk) // check if the base is prev link
            && is_free(blk->base_offset) && is_free(blk))
        {
            auto prev = blk->base_offset;
            prev->next_block = blk->next_block;
            prev->size += blk->size;
            if (prev->next_block)
            {
                prev->next_block->base_offset = prev;
                prev->next_block->relative_offset = prev->size;
            }
            // delete blk;
            blk->reset(); //for free avoid cycle ref

            blk = prev;
        }
        if (is_free(blk) && blk->next_block && is_free(blk->next_block))
        {
            blk->size += blk->next_block->size;
            auto next = blk->next_block;
            blk->next_block = next->next_block;
            if (blk->next_block)
            {
                blk->next_block->base_offset = blk;
                blk->next_block->relative_offset = blk->size;
            }
            // delete next;
            next->reset();
        }
    }

    MemoryObjectPtr find_best_block(MemoryObjectPtr o, uint32_t size)
    {
        MemoryObjectPtr ret = nullptr;
        aipu_assert(o);
        if (is_free(o) && o->size >= size)
        {

            ret = o;
        }
        else if (o->next_block)
        {
            ret = find_best_block(o->next_block, size);
        }
        return ret;
    }

    Memory::Memory(uint32_t base, uint32_t size)
    {
        root = MemoryObjectPtr(new MemoryObject(base, 0));
        root->size = size;
    }

    Memory::Memory(MemoryObjectPtr base, uint32_t size)
    {
        root = MemoryObjectPtr(new MemoryObject(base, 0));
        root->size = size;
    }

    Memory::~Memory()
    {
        auto next = root;
        while (next)
        {
            auto nnext = next->next_block;
            next->reset();
            next = nnext;
        }
        root = nullptr;
    }

    MemoryObjectPtr Memory::allocate(uint32_t size)
    {
        auto blk = find_best_block(root, size);
        auto alloc = split_block(blk, size);
        if (alloc)
        {
            MemoryObject_set_status(alloc, MemS_USED);
        }
        return alloc;
    }

    void Memory::merge_free_blks()
    {
        auto blk = root;
        while (blk)
        {
            if (is_free(blk) && blk->next_block && is_free(blk->next_block))
            {
                blk->size += blk->next_block->size;
                auto next = blk->next_block;
                blk->next_block = next->next_block;
                if (blk->next_block)
                {
                    blk->next_block->base_offset = blk;
                    blk->next_block->relative_offset = blk->size;
                }
                // delete next;
                next->reset();
            }
            else
            {
                blk = blk->next_block;
            }
        }
    }

    //change the block status
    // mark the root->offset()+offset as Status with size=size
    void change_blk_status(MemoryObjectPtr root, uint32_t off, uint32_t size, MemStatus status)
    {
        uint32_t offset = root->offset() + off;
        auto blk = root;
        while (blk && blk->offset() < offset && blk->next_block)
        {
            blk = blk->next_block;
        }
        if (blk->offset() != offset)
        {
            split_block(blk, offset - blk->offset());
            blk = blk->next_block;
        }
        if (blk->size < size)
        {
            auto next = blk->next_block;
            while (next && blk->size < size)
            {
                blk->size += next->size;
                blk->next_block = next->next_block;
                if (blk->next_block)
                {
                    blk->next_block->base_offset = blk;
                    blk->next_block->relative_offset = blk->size;
                }
                // delete next;
                next->reset();
                next = blk->next_block;
            }
        }
        if (blk->size > size)
        {
            blk = split_block(blk, size);
        }
        MemoryObject_set_status(blk, status);
    }

    void Memory::free(MemoryObjectPtr ptr)
    {
        MemoryObject_set_status(ptr, MemS_FREE);
        merge_block(ptr);
    }

    SegMMU::SegMMU(uint32_t base, std::vector<Memory *> mems_) : Memory(base, 0), mems(mems_)
    {
        uint32_t size = 0;
        for (auto mem : mems)
        {
            m_interval.push_back({size, size + mem->root->size, mem});
            size += mem->root->size;
        }
        root->size = size;
        sync();
    }

    SegMMU::~SegMMU()
    {

        auto next = root;
        while (next)
        {
            auto nnext = next->next_block;
            next->reset();
            next = nnext;
        }
        root = nullptr;
    }
    void SegMMU::sync()
    {
        //sync the shadow memory and phy memory
        for (auto inter : m_interval)
        {
            auto blk = inter.mem->root;
            uint32_t base = blk->offset();
            while (blk)
            {
                change_blk_status(root, blk->offset() - base + inter.start, blk->size, mem_status(blk));
                blk = blk->next_block;
            }
        }
        merge_free_blks();
    }
    MemoryObjectPtr SegMMU::allocate(uint32_t size)
    {
        sync();
        auto blk = Memory::allocate(size);
        uint32_t begin = blk->offset() - root->offset();
        uint32_t end = begin + blk->size;
        bool not_end = false;
        for (auto inter : m_interval)
        {
            if (begin >= inter.start && begin < inter.end)
            {
                if (end < inter.end)
                {
                    change_blk_status(inter.mem->root, begin - inter.start, size, MemS_USED);
                    break;
                }
                else
                {
                    not_end = true;
                    change_blk_status(inter.mem->root, begin - inter.start, inter.end - begin + inter.start, MemS_USED);
                    continue;
                }
            }
            if (not_end)
            {
                if (end > inter.end)
                {
                    change_blk_status(inter.mem->root, 0, inter.end - inter.start, MemS_USED);
                }
                else
                {
                    not_end = false;
                    change_blk_status(inter.mem->root, 0, end - inter.start, MemS_USED);
                }
            }
        }
        return blk;
    }

    uint32_t SegMMU::get_free_size(uint32_t mem_idx)
    {
        if (mem_idx >= mems.size())
        {
            return 0;
        }
        auto mem = mems[mem_idx];
        uint32_t s = 0;
        uint32_t max_size = 0;
        bool last_blk_is_free = false;
        auto blk = mem->rootblk();
        if (is_free(blk))
        {
            max_size = blk->size;
            s = blk->size;
            last_blk_is_free = true;
        }
        while (blk->next_block)
        {
            blk = blk->next_block;
            if (is_free(blk))
            {
                if (last_blk_is_free)
                {
                    s += blk->size;
                }
                else
                {
                    s = blk->size;
                }

                last_blk_is_free = true;
            }
            else
            {
                last_blk_is_free = false;
                s = 0;
            }
            if (s > max_size)
            {
                max_size = s;
            }
        }
        return max_size;
    }

    uint32_t SegMMU::get_mem_size(uint32_t mem_idx)
    {
        if (mem_idx >= mems.size())
        {
            return 0;
        }
        auto mem = mems[mem_idx];
        uint32_t s = 0;
        auto blk = mem->rootblk();
        s = blk->size;
        while (blk->next_block)
        {
            blk = blk->next_block;
            s += blk->size;
        }
        return s;
    }
    void SegMMU::free(MemoryObjectPtr ptr)
    {
        sync();
        MemoryObject_set_status(ptr, MemS_FREE);
        uint32_t begin = ptr->offset() - root->offset();
        uint32_t end = begin + ptr->size;
        bool not_end = false;
        for (auto inter : m_interval)
        {
            if (begin >= inter.start && begin < inter.end)
            {
                if (end < inter.end)
                {
                    change_blk_status(inter.mem->root, begin - inter.start, ptr->size, MemS_FREE);
                    break;
                }
                else
                {
                    not_end = true;
                    change_blk_status(inter.mem->root, begin - inter.start, inter.end - begin + inter.start, MemS_FREE);
                    continue;
                }
            }
            if (not_end)
            {
                if (end > inter.end)
                {
                    change_blk_status(inter.mem->root, 0, inter.end - inter.start, MemS_FREE);
                }
                else
                {
                    not_end = false;
                    change_blk_status(inter.mem->root, 0, end - inter.start, MemS_FREE);
                }
            }
            inter.mem->merge_free_blks();
        }
        sync();
    }

    std::ostream &operator<<(std::ostream &os, const Memory &mem)
    {
        auto blk = mem.rootblk();
        os << "[" << std::hex;
        while (blk)
        {
            if (is_free(blk))
            {
                os << GREEN_COLOR;
            }
            else
            {
                os << RED_COLOR;
            }

            os << "0x" << blk->size << COLOR_END << "|";
            blk = blk->next_block;
        }
        os << "]";
        return os;
    }

    SRAMFactory::~SRAMFactory()
    {
        // for (auto kv : this->m_sram)
        // {
        //     if (kv.second)
        //     {
        //         // delete kv.second;
        //     }
        // }
        m_sram.clear();
    }
    MemoryObjectPtr SRAMFactory::get_sram_by_addr(size_t addr)
    {
        if (m_sram.count(addr))
        {
            return m_sram[addr];
        }
        //check the address is  legal
        if (addr >= SRAMBASE && !(addr & 0x7fff))
        {
            auto target = get_target();
            size_t sram_size = 0;
            if (addr & 0x8000000) //gsram
            {
                sram_size = target->gsram_size();
            }
            else
            {
                sram_size = target->lsram_size();
            }
            if (sram_size > 0)
            {
                auto obj = MemoryObjectPtr(new MemoryObject((uint32_t)addr, 0));
                obj->size = sram_size;
                m_sram[addr] = obj;
                return obj;
            }
        }
        return nullptr;
    }
    MemoryObjectPtr SRAMFactory::generate_memory(size_t addr, size_t sram_size)
    {
        if (sram_size > 0)
        {
            auto obj = MemoryObjectPtr(new MemoryObject((uint32_t)addr, 0));
            obj->size = sram_size;
            m_sram[addr] = obj;
            return obj;
        }
        return nullptr;
    }
}; //namespace SegementMMUAllocator

using namespace SegementMMUAllocator;
MMUAllocator::MMUAllocator(TensorPtr workspace, size_t ddr_size) : m_workspace(workspace)
{
    init_memory(ddr_size);
}
MMUAllocator::~MMUAllocator()
{
    for (auto mem : m_mem)
    {
        if (mem)
        {
            delete mem;
        }
    }
    for (auto ddr : m_ddr)
    {
        if (ddr)
        {
            delete ddr;
        }
    }
    for (auto seg : m_seg)
    {
        if (seg)
        {
            delete seg;
        }
    }
    for (auto seg : m_ddr_seg)
    {
        if (seg)
        {
            delete seg;
        }
    }
}
MemoryObjectPtr MMUAllocator::allocate(unsigned int size, MemoryObjectPtr ptr)
{
    return allocate(size, ptr, {});
}
MemoryObjectPtr MMUAllocator::allocate(unsigned int size, MemoryObjectPtr ptr, const std::set<MemoryObjectPtr> &slow_objects)
{
    if (!ptr)
    {
        return nullptr;
    }
    auto mem = find_best_memory(size, ptr->align, slow_objects);
    if (mem)
    {
        auto blk = mem->allocate(size);
        uint32_t mod = blk->offset() % ptr->align;
        uint32_t extra_for_align = mod == 0 ? 0 : (ptr->align - mod);
        if (extra_for_align)
        {
            mem->free(blk);
            size += extra_for_align;
            blk = mem->allocate(size);
        }
        if (contains(m_mem, mem))
        {
            ptr->m_base = blk->offset();
            ptr->relative_offset = extra_for_align;
        }
        else
        {
            ptr->relative_offset = blk->offset() - mem->rootblk()->offset() + extra_for_align;
            auto segmem = static_cast<SegMMU *>(mem);
            ptr->base_offset = segmem->memories()[0]->rootblk()->base_offset;
            ptr->next_block = segmem->memories()[1]->rootblk()->base_offset;
        }
        m_allocated[ptr] = {blk, mem, size};
        // DEBUG() << "allocator size: 0x" << std::hex << size << " \toffset:0x" << blk->offset()
        //         << " ptr: " << ptr << std::dec;
        // INFO() << *(m_mem[0]) << "\t" << *(m_mem[1]);
        // INFO() << *(m_seg[0]);
    }
    else
    {
        // DEBUG() << std::hex << "!!!cannot allocate  size: 0x" << size;
        return nullptr;
    }
    return ptr;
}
void MMUAllocator::free(MemoryObjectPtr ptr)
{
    if (ptr)
    {
        if (m_allocated.count(ptr))
        {
            auto pointer = m_allocated[ptr];
            // the memory splits to two blocks
            if (ptr->next_block && pointer.size > pointer.blk->size && pointer.blk->next_block)
            {
                pointer.mem->free(pointer.blk->next_block);
            }
            pointer.mem->free(pointer.blk);
            m_allocated.erase(ptr);
            // DEBUG() << std::hex << "free size: 0x" << pointer.size << " ptr: " << ptr << " offset: 0x" << pointer.blk->offset();
            // INFO() << *(m_mem[0]) << "\t" << *(m_mem[1]);
            // INFO() << *(m_seg[0]);
            // INFO() << "free:" << *pointer.mem;
        }
        else
        {
            // DEBUG() << "cannot free pointer:" << ptr;
        }
    }
}

static std::set<MemoryObjectPtr> get_all_objects(MemoryObjectPtr ptr)
{
    std::set<MemoryObjectPtr> all_objects;
    if (ptr->base_offset)
    {
        auto objects = get_all_objects(ptr->base_offset);
        for (auto obj : objects)
        {
            all_objects.insert(obj);
        }
    }
    if (ptr->next_block)
    {
        auto objects = get_all_objects(ptr->next_block);
        for (auto obj : objects)
        {
            all_objects.insert(obj);
        }
    }
    all_objects.insert(ptr);
    return all_objects;
}

Memory *MMUAllocator::find_best_memory(uint32_t size, uint32_t align, const std::set<MemoryObjectPtr> &slow_objects, bool sram_only)
{
    using block_pair = std::pair<MemoryObjectPtr, Memory *>;
    std::vector<block_pair> available_block;
    // try sram
    for (auto mem : m_mem)
    {
        auto block = find_best_block(mem->rootblk(), size);
        if (block)
        {
            available_block.push_back({block, mem});
        }
    }
    if (!sram_only)
    {
        //try sram mmu
        if (available_block.empty())
        {
            for (auto mem : m_seg)
            {
                mem->sync();
                auto block = find_best_block(mem->rootblk(), size);
                if (block)
                {
                    available_block.push_back({block, mem});
                }
            }
        }
        //try ddr mmu
        if (available_block.empty())
        {
            std::pair<MemoryObjectPtr, Memory *> best_block = {0, 0};
            int best_sram_size = 0;
            for (auto mem : m_ddr_seg)
            {
                mem->sync();
                auto block = find_best_block(mem->rootblk(), size);
                // if the relative_offset!=0, this is a ddr region
                if (block)
                {
                    int sram_size = mem->get_mem_size(0) - (block->offset() - mem->rootblk()->offset());
                    if (sram_size > 0 && best_block.first == 0)
                    {
                        best_block = {block, mem};
                    }
                    if (sram_size > best_sram_size)
                    {
                        best_block = {block, mem};
                        best_sram_size = sram_size;
                    }
                }
            }
            if (best_block.first)
            {
                return best_block.second;
            }
        }
    }

    if (available_block.empty())
    {
        return nullptr;
    }
    std::vector<block_pair> fast_blocks;

    std::set<MemoryObjectPtr> all_slow_ojbects;
    for (auto ptr : slow_objects)
    {
        auto all_objects = get_all_objects(ptr);
        for (auto obj : all_objects)
        {
            //INFO() << obj;
            all_slow_ojbects.insert(obj);
        }
    }
    auto mem_to_srams = [](Memory *mem, MMUAllocator *allocator) -> std::set<Memory *> {
        std::set<Memory *> srams;
        if (allocator->is_sram_mem(mem))
        {
            srams.insert(mem);
        }
        else
        {
            auto seg = static_cast<SegMMU *>(mem);
            for (size_t i = 0; i < 2; i++)
            {
                if (allocator->is_sram_mem(seg->memories()[i]))
                {
                    srams.insert(seg->memories()[i]);
                }
            }
        }
        return srams;
    };
    std::set<Memory *> slow_srams;
    for (auto ptr : all_slow_ojbects)
    {
        if (ptr && m_allocated.count(ptr))
        {
            auto srams = mem_to_srams(m_allocated[ptr].mem, this);
            for (auto sram : srams)
            {
                slow_srams.insert(sram);
            }
        }
    }
    for (auto block : available_block)
    {
        bool is_fast_block = true;
        for (auto sram : mem_to_srams(block.second, this))
        {
            if (slow_srams.count(sram))
            {
                is_fast_block = false;
                break;
            }
        }
        if (is_fast_block)
        {
            fast_blocks.push_back(block);
        }
    }
    auto find_smallest_block = [size](std::vector<block_pair> block, uint32_t align) -> block_pair {
        block_pair ret = {0, 0};
        size_t actual_size = size;
        for (unsigned int i = 0; i < block.size(); ++i)
        {
            auto mod = block[i].first->offset() % align;
            auto need_size = size + (mod == 0 ? 0 : align - mod);
            if (block[i].first->size >= need_size)
            {
                if (ret.first == nullptr)
                {

                    ret = block[i];
                    actual_size = need_size;
                }
                // try to use small size
                else if (actual_size > need_size)
                {
                    ret = block[i];
                    actual_size = need_size;
                }
                // try to use smaller block
                else if ((ret.first->size > block[i].first->size))
                {
                    ret = block[i];
                    actual_size = need_size;
                }
            }
        }
        return ret;
    };
    block_pair best_block = find_smallest_block(fast_blocks.size() > 0 ? fast_blocks : available_block, align);
    return best_block.second;
}

bool MMUAllocator::is_sram_address(size_t addr)
{
    for (auto m : m_mem)
    {
        auto root = m->rootblk();
        if (root->offset() <= addr && addr < root->offset() + root->size)
        {
            return true;
        }
    }
    return false;
}

bool MMUAllocator::is_sram_mem(Memory *mem)
{
    bool ret = contains(m_mem, mem);
    return ret;
}

void MMUAllocator::init_memory(size_t ddr_size)
{
    auto target = get_target();
    if (target->isainfo().type >= ISA_ZYv5)
    {
        auto gsram = new Memory(sram_factory.get_sram_by_addr(GSRAM0_BASE), target->gsram_size());
        m_mem.push_back(gsram);
        return;
    }
    MemoryObjectPtr ws = nullptr;
    if (ddr_size != 0)
    {
        //16K align
        ddr_size = ((ddr_size + 16 * 1024 - 1) >> 14) << 14;

        //workspace must be 1MB align
        ws = m_workspace->memory_offset();
        //we make the align as the ddr_size
        size_t alignment = 1024 * 1024; //default 1M
        if ((target->isainfo().type == ISA_ZYv2) || (target->isainfo().type == ISA_ZYv3))
        {
            alignment = 16 * 1024; //16K
        }
        else if (target->isainfo().type == ISA_ZY)
        {
            alignment = 16 * 1024;
            while (alignment < ddr_size)
            {
                alignment = alignment << 1;
            }
            //max is 1M
            alignment = MAX(alignment, 1024 * 1024);
        }
        ws->align = MAX(alignment, ws->align);
        int num_ddr = 2;
        if (target->gsram_size() >= 32 * 1024)
        {
            num_ddr = 4;
        }
        m_workspace->shape()[0] = MAX(m_workspace->shape().size(), (int)(num_ddr * ddr_size));
        ddr_size = m_workspace->shape()[0] / num_ddr;
    }

    // apply 1MB, GSRAM0/1: 1MB, LSRAM0/1: 1MB

    auto lsram0ddr = std::make_shared<MemoryObject>(0, 0);
    MemoryObjectPtr lsram1ddr = std::make_shared<MemoryObject>(0, ddr_size);
    lsram0ddr->relative_offset = 0;
    lsram0ddr->size = ddr_size;
    lsram0ddr->base_offset = ws;
    lsram1ddr->relative_offset = ddr_size;
    lsram1ddr->size = ddr_size;
    lsram1ddr->base_offset = ws;
    auto ddr0 = new Memory(MemoryObjectPtr(lsram0ddr), ddr_size);
    auto ddr1 = new Memory(MemoryObjectPtr(lsram1ddr), ddr_size);

    m_ddr.push_back(ddr0);
    m_ddr.push_back(ddr1);
    Memory *gsram0 = nullptr;
    Memory *gsram1 = nullptr;
    MemoryObjectPtr gsram0ddr = std::make_shared<MemoryObject>(0, 2 * ddr_size);
    MemoryObjectPtr gsram1ddr = std::make_shared<MemoryObject>(0, 3 * ddr_size);
    if (target->gsram_size() >= 32 * 1024)
    {

        gsram0ddr->relative_offset = 2 * ddr_size;
        gsram0ddr->size = ddr_size;
        gsram0ddr->base_offset = ws;
        gsram1ddr->relative_offset = 3 * ddr_size;
        gsram1ddr->size = ddr_size;
        gsram1ddr->base_offset = ws;

        auto ddr2 = new Memory(MemoryObjectPtr(gsram0ddr), ddr_size);
        auto ddr3 = new Memory(MemoryObjectPtr(gsram1ddr), ddr_size);
        gsram0 = new Memory(sram_factory.get_sram_by_addr(GSRAM0_BASE), target->gsram_size());
        gsram1 = new Memory(sram_factory.get_sram_by_addr(GSRAM1_BASE), target->gsram_size());

        m_ddr.push_back(ddr2);
        m_ddr.push_back(ddr3);
        m_ddr_seg.push_back(new SegMMU(MMU_BASE, {gsram0, ddr2}));
        m_ddr_seg.push_back(new SegMMU(MMU_BASE, {gsram1, ddr3}));
        m_mem.push_back(gsram0);
        m_mem.push_back(gsram1);
        // the mmu is use two physical memory
        auto mmu0 = new SegMMU(MMU_BASE, {gsram0, gsram1});
        m_seg.push_back(mmu0);
    }
    for (unsigned int i = 0; i < target->tec_num(); ++i)
    {
        auto lsram0 = new Memory(sram_factory.get_sram_by_addr(LSRAM_BASE + i * LSRAM_RESERVE), target->lsram_size());
        auto lsram1 = new Memory(sram_factory.get_sram_by_addr(LSRAM_BASE + i * LSRAM_RESERVE + LSRAM_MAX_SIZE), target->lsram_size());

        m_mem.push_back(lsram0);
        m_mem.push_back(lsram1);
        m_seg.push_back(new SegMMU(MMU_BASE, {lsram0, lsram1}));
        if (i == 0)
        {
            m_ddr_seg.push_back(new SegMMU(MMU_BASE, {lsram0, ddr0}));
            m_ddr_seg.push_back(new SegMMU(MMU_BASE, {lsram1, ddr1}));
        }
    }
}

}; // namespace aipubt

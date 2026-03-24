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
 * @Date: 2020-01-29 14:26:52 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-02-21 18:03:05
 */
#ifndef BUILDER_MMU_ALLOCATOR
#define BUILDER_MMU_ALLOCATOR

#include <map>
#include <ostream>
#include <stdint.h>
#include <vector>

#include "allocator.h"
#include "core/memory_object.h"
namespace aipubt
{
class MMUAllocator;
namespace SegementMMUAllocator
{
    enum MemStatus : uint32_t
    {
        MemS_FREE = 0,
        MemS_USED = 1,
    };

    // using MemoryObject as Memory block
    // the mask is the memblk status
    // the mask default is 0 i.e. FREE
    using Memblk = MemoryObjectPtr;
    class Memory
    {
    public:
        friend std::ostream &operator<<(std::ostream &os, const Memory &mem);
        Memory(uint32_t base, uint32_t size);
        Memory(MemoryObjectPtr base, uint32_t size);
        virtual ~Memory();
        virtual MemoryObjectPtr allocate(uint32_t size);
        virtual void free(MemoryObjectPtr ptr);
        MemoryObjectPtr rootblk() const
        {
            return root;
        }

    protected:
        void merge_free_blks();
        MemoryObjectPtr root;
        friend class SegMMU;
    };
    class SegMMU : public Memory
    {
    public:
        // friend std::ostream &operator<<(std::ostream &os, const SegMMU *mem);
        SegMMU(uint32_t base, std::vector<Memory *> mems_);
        ~SegMMU();
        MemoryObjectPtr allocate(uint32_t size);
        void free(MemoryObjectPtr ptr);
        void sync();
        std::vector<Memory *> &memories()
        {
            return mems;
        }
        uint32_t get_free_size(uint32_t mem_idx);
        uint32_t get_mem_size(uint32_t mem_idx);

    protected:
        struct MemInterval
        {
            uint64_t start;
            uint64_t end;
            Memory *mem;
        };
        std::vector<Memory *> mems;
        std::vector<MemInterval> m_interval;
    };

    struct Pointer
    {
        MemoryObjectPtr blk;
        Memory *mem;
        size_t size;
    };

    class SRAMFactory
    {
    public:
        SRAMFactory() {}
        ~SRAMFactory();
        MemoryObjectPtr get_sram_by_addr(size_t addr);
        MemoryObjectPtr generate_memory(size_t addr, size_t sram_size);

    private:
        std::map<size_t, MemoryObjectPtr> m_sram;
    };
} // namespace SegementMMUAllocator

using namespace SegementMMUAllocator;
class MMUAllocator : public Allocator
{
public:
    MMUAllocator(TensorPtr workspace, size_t ddr_size = 1024 * 1024);
    ~MMUAllocator();
    MemoryObjectPtr allocate(unsigned int size, MemoryObjectPtr ptr);
    MemoryObjectPtr allocate(unsigned int size, MemoryObjectPtr ptr, const std::set<MemoryObjectPtr> &slow_objects);
    void free(MemoryObjectPtr ptr);
    bool is_sram_address(size_t addr);

private:
    Memory *find_best_memory(uint32_t size, uint32_t align, const std::set<MemoryObjectPtr> &slow_objects, bool sram_only = false);
    void init_memory(size_t ddr_size);
    bool is_sram_mem(Memory *mem);
    std::map<MemoryObjectPtr, Pointer> m_allocated;
    std::vector<Memory *> m_mem;
    std::vector<Memory *> m_ddr;
    std::vector<SegMMU *> m_seg;
    std::vector<SegMMU *> m_ddr_seg;
    TensorPtr m_workspace = nullptr;
};

}; // namespace aipubt

#endif //!BUILDER_MMU_ALLOCATOR
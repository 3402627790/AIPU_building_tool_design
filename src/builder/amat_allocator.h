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
 * @Date: 2022-06-13 10:18:20 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-12 16:50:00
 */

#ifndef BUILDER_AMAT_ALLOCATOR_H
#define BUILDER_AMAT_ALLOCATOR_H
#include <vector>

#include "core/core.h"
#include "mem_allocator.h"
#include "utils/utils.h"
namespace aipubt
{
class AMATAllocator : public MemoryAllocator
{
    // implementation of AMAT(AutoMemoryAllocationTool from Chunying)
public:
    AMATAllocator();
    ~AMATAllocator();
    MemoryObjectPtr allocate(unsigned int size, MemoryObjectPtr) override;
    void free(MemoryObjectPtr ptr) override;
    std::vector<MemoryObjectPtr> slots() override;
    void add_soc_sram(uint32_t base, uint32_t size, MemoryAllocator::SRAMLevel l = MemoryAllocator::SRAMLevel::L3) override;
    void add_soc_sram(MemoryObjectPtr sram, MemoryAllocator::SRAMLevel l = MemoryAllocator::SRAMLevel::L3) override;
    void remove_soc_sram(uint32_t base) override;
    void remove_soc_sram(MemoryObjectPtr base) override;
    void reset();
    void print() override;
    std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> get_liveness() override;
    uint32_t clock() override;
    uint32_t get_clock() override;
    void compress_clock() override;

protected:
    struct AllocationObj
    {
        uint64_t size;
        uint64_t offset;
        uint32_t life_start;
        uint32_t life_end;
    };
    struct DTCM
    {
        MemoryObjectPtr sram;
        MemoryAllocator::SRAMLevel level;
    };
    void apply_dtcm_opt();
    std::vector<std::pair<MemoryObjectPtr, AllocationObj>> allocate_ptrs(std::vector<std::pair<MemoryObjectPtr, uint32_t>> &ptrs, uint64_t max_size = UINT64_MAX);
    std::vector<AllocationObj> liveness_overlap(const std::vector<std::pair<MemoryObjectPtr, AllocationObj>> &allocated, std::pair<uint32_t, uint32_t> &liveness);
    uint32_t find_available_gap(std::vector<AllocationObj> &overlap, uint32_t size, bool &found, uint32_t align = 1);
    uint64_t count_memory_footprint(const std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> &allocated);
    std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> cump0(std::vector<std::pair<MemoryObjectPtr, uint32_t>> &ptrs, uint32_t sram);
    std::vector<std::pair<MemoryObjectPtr, AMATAllocator::AllocationObj>> cump1(std::vector<std::pair<MemoryObjectPtr, uint32_t>> &ptrs, uint32_t sram);
    std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> m_liveness;
    std::map<MemoryObjectPtr, uint32_t> m_ptr_size;
    std::vector<std::pair<MemoryObjectPtr, uint32_t>> m_tensors;
    std::vector<DTCM> m_dtcm;
    std::set<MemoryObjectPtr> m_use_ddr;
    std::set<MemoryObjectPtr> m_in_tcm;
    uint32_t m_clock;
    uint32_t m_start_clock = 65535;
    bool m_refresh;
};
};     // namespace aipubt
#endif //!BUILDER_AMAT_ALLOCATOR_H

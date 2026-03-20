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
 * @Date: 2020-02-09 12:14:52 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-02-10 10:20:09
 */
#ifndef BUILDER_MEMORY_OBJECT_H
#define BUILDER_MEMORY_OBJECT_H
#include <cstdint>
#include <memory>
#include <vector>
namespace aipubt
{
struct MemoryObject;
using MemoryObjectPtr = std::shared_ptr<MemoryObject>;
struct MemoryObject : public std::enable_shared_from_this<MemoryObject>
{
    enum flag
    {
        normal = 0,
        tiled_internal = 1,
        zero_copy = 2,
        prior_in_gm = 3,
    };

    enum type
    {
        reuse = 0,
        constant = 1,
    };

    enum attr
    {
        none = 0,
        loop_scan = 1,
    };

    uint64_t relative_offset = 0;
    uint64_t m_base = 0;
    uint32_t align = 32; //Byte
    uint64_t mask = 0;
    uint64_t size = 0;
    uint64_t foot_print = 0;

    MemoryObjectPtr base_offset = nullptr;
    // if the memory object is a virtual memory buffer, contains with several physical memory,
    // the we use a linked list for the physical memory
    MemoryObjectPtr next_block = nullptr;

    // 8:11 attr flag
    // 4:7 buf flag
    // 0:3 mem flag
    uint32_t mem_flag = 0;
    int32_t buffer_id = -1; // the bss section index, this value should write by build_bss
    int32_t bss_id = -1;    // the bss index, this value should write by build_bss
    MemoryObject(int64_t base, int64_t off = 0);
    MemoryObject(MemoryObjectPtr base, int off = 0);
    uint64_t offset();
    void set_base_offset(uint64_t base);
    void set_base_offset(uint32_t base);
    void set_base_offset(int base);
    void set_base_offset(MemoryObjectPtr base);
    bool is_shared_by(MemoryObjectPtr child);
    MemoryObjectPtr get_base_object();
    MemoryObjectPtr get_mmu_object();
    uint64_t get_base_memory();
    uint32_t get_max_aligned();
    void reset();
    static MemoryObjectPtr merge(const std::vector<MemoryObjectPtr> &to_merge);
    void set_buf_type(uint32_t type);
    uint32_t get_buf_type();
    void set_mem_flag(uint32_t mem);
    uint32_t get_mem_flag();
    void set_attr(uint32_t attr);
    uint32_t get_attr();
};
}; // namespace aipubt

#endif // !BUILDER_MEMORY_OBJECT_H

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
#include "test_builder.h"

#include <iostream>
#include <string>
#include <algorithm>
#include <set>
#include <vector>

#include <gcompiler_api.h>
#include <utils/string_utils.h>
#include <builder/mem_allocator.h>
#include <builder/mmu_allocator.h>
#include <builder/scheduler.h>
#include <builder/builder.h>
using namespace aipubt;

void test_mmu_allocator()
{
    MMUAllocator m_allocate(0);
    MemoryObjectPtr ptr = MemoryObjectPtr(new MemoryObject(0));
    MemoryObjectPtr mem0 = m_allocate.allocate(0x100, ptr);
    MemoryObjectPtr mem1 = m_allocate.allocate(0x2000, ptr);
    MemoryObjectPtr mem2 = m_allocate.allocate(0x1000, ptr);
    m_allocate.free(mem0);
    MemoryObjectPtr mem3 = m_allocate.allocate(0x800, ptr);
    m_allocate.free(mem2);
    MemoryObjectPtr mem4 = m_allocate.allocate(0x200, ptr);
    m_allocate.free(mem1);
    MemoryObjectPtr mem5 = m_allocate.allocate(0x10000, ptr);
    MemoryObjectPtr mem6 = m_allocate.allocate(0x22000, ptr);
    m_allocate.free(mem3);
    m_allocate.free(mem4);
    MemoryObjectPtr mem7 = m_allocate.allocate(0x8000, ptr);
    m_allocate.free(mem6);
    m_allocate.free(mem5);
    MemoryObjectPtr mem9 = m_allocate.allocate(0x23000, ptr);
    m_allocate.free(mem7);
    MemoryObjectPtr mem10 = m_allocate.allocate(0x8000, ptr);
    m_allocate.free(mem9);
    m_allocate.free(mem10);
}

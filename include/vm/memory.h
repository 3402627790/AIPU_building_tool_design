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
 * @Date: 2024-12-16 11:41:19 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-16 14:53:35
 */
#pragma once
#include <algorithm>
#include <array>
#include <asm/asm.h>
#include <core/fp16.h>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace aipubt
{

class Memory
{
#define MEM_DEFAULT_PAGE ((uint64_t)(64 * 1024))
    mutable std::unordered_map<uint64_t, char *> pages;
    mutable std::mutex m_mtx;

    char *get_page(uint64_t offset) const
    {
        offset = offset & (~((uint64_t)(MEM_DEFAULT_PAGE - 1)));
        std::lock_guard<std::mutex> lock(m_mtx);
        if (pages.count(offset))
        {
            return pages.at(offset);
        }
        //page fault
        auto p = new char[MEM_DEFAULT_PAGE];
        pages[offset] = p;
        return p;
    }

public:
    ~Memory()
    {
        for (auto kv : pages)
        {
            if (kv.second)
            {
                delete[] kv.second;
            }
        }
        pages.clear();
    }
    template <typename T>
    T read(uint64_t offset) const
    {
        T ret;
        read(offset, (char *)(&ret), sizeof(T));
        return ret;
    }
    void read(uint64_t offset, char *data, size_t size) const
    {
        uint64_t base = offset & (~((uint64_t)(MEM_DEFAULT_PAGE - 1)));
        offset = offset & (MEM_DEFAULT_PAGE - 1);
        while (offset + size > MEM_DEFAULT_PAGE)
        {
            auto s = MEM_DEFAULT_PAGE - offset;
            memcpy(data, get_page(base) + offset, s);
            data += s;
            base += MEM_DEFAULT_PAGE;
            size -= s;
            offset = 0;
        }
        memcpy(data, get_page(base) + offset, size);
    }

    void write(uint64_t offset, char *data, size_t size)
    {
        uint64_t base = offset & (~((uint64_t)(MEM_DEFAULT_PAGE - 1)));
        offset = offset & (MEM_DEFAULT_PAGE - 1);
        while (offset + size > MEM_DEFAULT_PAGE)
        {
            auto s = MEM_DEFAULT_PAGE - offset;
            memcpy(get_page(base) + offset, data, s);
            data += s;
            base += MEM_DEFAULT_PAGE;
            size -= s;
            offset = 0;
        }
        memcpy(get_page(base) + offset, data, size);
    }

    template <typename T>
    void write(uint64_t offset, const T &v)
    {
        write(offset, (char *)(&v), sizeof(T));
    }
};

using MemoryPtr = std::shared_ptr<Memory>;
} // namespace aipubt
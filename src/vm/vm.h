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
 * @Date: 2024-12-13 10:05:59 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-16 14:34:53
 */
#pragma once
#include "asmtranscode.h"
#include "core.h"
#include "utils/threadpool.h"

namespace aipubt
{

class VM
{
public:
    VM(uint32_t core_num = 2, uint32_t tec_num = 4)
    {
        _init();
    }

    void run(uint32_t offset)
    {
        std::vector<std::future<void>> res;
        for (auto i = 0u; i < m_core.size(); ++i)
        {
            res.push_back(thread_pool()->enqueue([this, i, offset]() { m_core[i].run(offset); }));
        }
        for (auto &&r : res)
        {
            r.wait();
        }
    }

    std::shared_ptr<Memory> get_memory() { return m_mem; }
    void set_mem(std::shared_ptr<Memory> mem) { m_mem = mem; }

protected:
    void _init()
    {
    }
    static utils::ThreadPool *thread_pool(uint32_t num = DEFAULT_THREAD_NUM)
    {
        static utils::ThreadPool *tp = nullptr;
        if (tp == nullptr)
        {
            tp = new utils::ThreadPool(num);
        }
        return tp;
    }
    std::vector<Core> m_core;
    std::shared_ptr<Memory> m_mem;
};
} // namespace aipubt

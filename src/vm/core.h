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
 * @Last Modified time: 2024-12-16 14:21:13
 */
#pragma once
#include "asmtranscode.h"
#include "tec.h"
#include "utils/threadpool.h"
#include <mutex>

namespace aipubt
{
constexpr int32_t SMSIZE = 128 * 1024;

class VM;
class TransCode;
class Core
{
    std::vector<TecStatus> m_tec;
    std::vector<TransCode> m_dynarec;
    std::shared_ptr<Memory> m_mem = 0;
    std::shared_ptr<utils::Barrier> m_barrier = 0;
    uint8_t m_sm[SMSIZE] = {0};

    void _init()
    {
        for (auto i = 0u; i < m_tec.size(); ++i)
        {
            m_dynarec[i] = m_dynarec[0];
            m_tec[i].tec = &m_dynarec[i];
            m_tec[i].core = this;
            m_tec[i].set_mem(m_mem);
        }
        m_barrier = std::make_shared<utils::Barrier>(m_tec.size());
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

public:
    Core(uint32_t tec_num = 4) : m_tec(tec_num), m_dynarec(tec_num) {}
    void set_code(const char *data, uint32_t size, uint32_t base_offset = 0)
    {
        if (m_tec.size() == 0)
            return;
        m_dynarec[0] = TransCode(data, size, base_offset);
    }
    void set_mem(std::shared_ptr<Memory> m)
    {
        m_mem = m;
        _init();
    }
    std::vector<TecStatus> &get_tec() { return m_tec; }
    auto get_memory() { return m_mem; }
    auto get_sm() { return m_sm; }
    auto run_async(uint32_t offset)
    {
        std::vector<std::future<void>> res;
        for (auto i = 0u; i < m_tec.size(); ++i)
        {
            res.push_back(thread_pool()->enqueue([this, i, offset]() { m_tec[i].run(offset); }));
        }
        return res;
    }
    void run(uint32_t offset)
    {
        for (auto &&r : run_async(offset))
        {
            r.wait();
        }
    }
    void tec_sync()
    {
        std::cout << "waiting for tec_sync\n";
        m_barrier->wait();
        std::cout << "tec_sync done \n";
    }
};

} // namespace aipubt

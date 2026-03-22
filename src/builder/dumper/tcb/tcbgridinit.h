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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Kyle Huang
 * @Date: 2022-06-20 17:00:00
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2022-06-20 17:00:00
 */
#ifndef BUILDER_TCB_GRID_INIT_H
#define BUILDER_TCB_GRID_INIT_H

#include "tcb.h"

namespace aipubt
{
class TCBGridInit : virtual public TCB
{
protected:
    uint64_t m_gm_addr = 0;
    uint32_t m_gm_size = 0;
    uint32_t m_gm_sync_offset = 0;
    uint32_t m_gm_sync_size = 0;
    uint32_t m_core = 0;
    uint32_t m_group_num = 0;
    uint32_t m_l2_flush = 0;

public:
    TCBGridInit()
    {
        m_gm_addr = 0;
        m_gm_size = 0;
        m_gm_sync_offset = 0;
        m_gm_sync_size = 0;
        m_core = 0;
        m_group_num = 0;
        m_is_task = false;
    }
    bool set_gm(uint64_t addr, uint32_t size, uint32_t sync_offset, uint32_t sync_size);
    bool set_core(uint32_t core);
    bool set_group_num(uint32_t grou_num);
    bool set_l2_flush();
    virtual ~TCBGridInit(){};
};

}; // namespace aipubt
#endif
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
#include "tcbgridinit.h"
#include <memory.h>

namespace aipubt
{

bool TCBGridInit::set_gm(uint64_t addr, uint32_t size, uint32_t sync_offset, uint32_t sync_size)
{
    m_gm_addr = addr;
    m_gm_size = size;
    m_gm_sync_offset = sync_offset;
    m_gm_sync_size = sync_size;
    return true;
}

bool TCBGridInit::set_core(uint32_t core)
{
    m_core = core;
    return true;
}

bool TCBGridInit::set_group_num(uint32_t group_num)
{
    m_group_num = group_num;
    return true;
}

bool TCBGridInit::set_l2_flush()
{
    m_l2_flush = 1;
    return true;
}

} // namespace aipubt
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
#include "tcbgridinitx3.h"
#include "core/logger.h"
#include <memory.h>

namespace aipubt
{

bool TCBGridInitX3::gen_tcb(char *ptr, uint32_t current, uint32_t next) const
{
    uint32_t dep_type = 2;
    uint32_t flag = ((m_l2_flush & 0x1) << 22) | ((dep_type & 0x3) << 4) | 0; //grid init
    uint32_t remap_size = (m_gm_size >> 18) - 1;
    uint32_t remap_mode = 0;               //time priority
    uint32_t remap_en = m_gm_size ? 1 : 0; // disable gm
    uint32_t gm_remap_ctrl = (remap_size & 0xff) << 8 | (remap_mode & 0x1) << 1 | (remap_en & 0x1);
    uint32_t sync_mode = m_gm_sync_size ? 1 : 0;
    uint32_t sync_size = ceil((float)m_gm_sync_size / (256 * 1024));
    uint32_t gm_sync_ctrl = ((sync_mode & 0x3) << 30) | (sync_size & 0xfff);
    uint32_t gm_addr_h = m_gm_addr >> 32;
    uint32_t gm_addr_l = m_gm_addr & 0xffffffff;

    memset(ptr, 0x0, TCB_SIZE);
    set_tcb_value(ptr, 0, 0, flag, 32);
    set_tcb_value(ptr, 1, 0, next, 32);
    set_tcb_value(ptr, 2, 0, m_group_num, 32);    //group num
    set_tcb_value(ptr, 4, 0, 1, 16);              //Grid interrupt
    set_tcb_value(ptr, 5, 16, m_grid_id, 16);     //gridID
    set_tcb_value(ptr, 5, 0, 0, 16);              //groupID
    set_tcb_value(ptr, 12, 0, gm_remap_ctrl, 32); //gm remap control
    set_tcb_value(ptr, 13, 0, gm_sync_ctrl, 32);  //gm sync control
    set_tcb_value(ptr, 14, 0, gm_addr_l, 32);     //gm remap addr low
    set_tcb_value(ptr, 15, 0, gm_addr_h, 32);     //gm remap addr high
    set_tcb_value(ptr, 30, 0, 0, 16);             //group dep0
    set_tcb_value(ptr, 30, 16, 0, 16);            //group dep1
    set_tcb_value(ptr, 31, 0, 0, 16);             //group dep2
    set_tcb_value(ptr, 31, 16, 0, 16);            //group dep3

    return true;
}

uint32_t TCBGridInitX3::get_size() const
{
    return TCB_SIZE;
}

} // namespace aipubt
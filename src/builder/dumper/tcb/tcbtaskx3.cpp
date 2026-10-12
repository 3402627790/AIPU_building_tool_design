//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------

#include "tcbtaskx3.h"
#include <memory.h>

namespace aipubt
{

bool TCBTaskX3::set_warmup(uint32_t len)
{
    warm_len = len;
    return true;
}

bool TCBTaskX3::gen_tcb(char *ptr, uint32_t current, uint32_t next) const
{
    uint32_t broadcast_start = m_broadcast ? 1 : 0;
    memset(ptr, 0x0, TCB_SIZE);
    uint32_t flag = (2 & 0xf) | ((m_dptype & 0x3) << 4) | ((m_groupend & 0x7) << 6) | ((m_broadcast & 0x7) << 16) | ((broadcast_start & 0x1) << 20);
    uint32_t core_id = 0;
    uint32_t tec_intr = 1 << 2;
    set_tcb_value(ptr, 0, 0, flag, 32);
    set_tcb_value(ptr, 1, 0, next, 32);
    set_tcb_value(ptr, 3, 0, m_spc, 32);
    set_tcb_value(ptr, 4, 0, tec_intr, 32); //interrupt
    set_tcb_value(ptr, 5, 0, m_group_id, 16);
    set_tcb_value(ptr, 5, 16, m_grid_id, 16);
    set_tcb_value(ptr, 6, 0, m_task_id, 16);
    set_tcb_value(ptr, 6, 16, warm_len, 8);
    set_tcb_value(ptr, 7, 0, 1, 32);           //gridDim.x
    set_tcb_value(ptr, 7, 16, 1, 32);          //gridDim.y
    set_tcb_value(ptr, 8, 0, 1, 16);           //gridDim.z
    set_tcb_value(ptr, 8, 16, m_tec_num, 16);  //groupDim.x
    set_tcb_value(ptr, 9, 0, 1, 16);           //groupDim.y
    set_tcb_value(ptr, 9, 16, 1, 16);          //groupDim.z
    set_tcb_value(ptr, 10, 0, 1, 16);          //groupid.x
    set_tcb_value(ptr, 10, 16, 0, 16);         //groupid.y
    set_tcb_value(ptr, 11, 16, m_task_id, 16); //taskid.x
    set_tcb_value(ptr, 13, 0, m_sp, 32);
    set_tcb_value(ptr, 14, 0, m_pp, 32);
    set_tcb_value(ptr, 15, 0, m_dp, 32);
    set_tcb_value(ptr, 16, 0, m_cp, 32);
    set_tcb_value(ptr, 17, 0, m_print, 32);
    set_tcb_value(ptr, 18, 0, m_prof, 32);
    set_tcb_value(ptr, 19, 0, core_id, 16);
    set_tcb_value(ptr, 27, 0, current, 32); //current tcbp
    set_tcb_value(ptr, 28, 0, m_gpp, 32);   //global parameter pointer
    set_tcb_value(ptr, 29, 0, 0, 32);
    set_tcb_value(ptr, 30, 0, m_dep0, 16);  //group dep0
    set_tcb_value(ptr, 30, 16, m_dep1, 16); //group dep1
    set_tcb_value(ptr, 31, 0, m_dep2, 16);  //group dep2
    set_tcb_value(ptr, 31, 16, m_dep3, 16); //group dep3
    return true;
}

uint32_t TCBTaskX3::get_size() const
{
    return TCB_SIZE;
}

} // namespace aipubt
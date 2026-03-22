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
#include "tcbsync.h"
#include "builder/target.h"
#include <memory.h>
#include <stdlib.h>
namespace aipubt
{

bool TCBSync::set_gm(uint32_t gm_ctl0, uint32_t gm_ctl1)
{
    m_gm_ctl0 = gm_ctl0;
    m_gm_ctl1 = gm_ctl1;
    return true;
}
/*tcbsync is copied from tcbinit, so only need to update gm_ctrl related info*/
bool TCBSync::gen_tcb(char *ptr, uint32_t current, uint32_t next) const
{
    auto flag = 0x1a0;
    set_tcb_value(ptr, 0, 0, flag, 16);
    set_tcb_value(ptr, 1, 0, next, 32);
    set_tcb_value(ptr, 4, 0, 0xf, 32);       //gm_ctl
    set_tcb_value(ptr, 6, 0, m_gm_ctl0, 32); //gm_region0_ctl
    set_tcb_value(ptr, 7, 0, m_gm_ctl1, 32); //gm_region1_ctl
    return true;
}

}; // namespace aipubt

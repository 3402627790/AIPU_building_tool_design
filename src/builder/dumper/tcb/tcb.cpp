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
#include "tcb.h"
#include "builder/target.h"
#include <memory.h>
#include <stdlib.h>
namespace aipubt
{
bool TCB::set_tcb_value(char *ptr, uint32_t line, uint32_t bitAddr, uint32_t value, uint32_t bitLength) const
{
    //char = 1 byte = 8 bit
    char *new_ptr = ptr + line * 4 + bitAddr / 8;
    switch (bitLength)
    {
    case 8:
        *(uint8_t *)new_ptr = value;
        break;
    case 16:
        *(uint16_t *)new_ptr = value;
        break;
    case 32:
        *(uint32_t *)new_ptr = value;
        break;
    default:
        break;
    }

    return true;
}

bool TCB::set_grid_id(uint32_t id)
{
    m_grid_id = id;
    return true;
}

bool TCB::set_group_id(uint32_t id)
{
    m_group_id = id;
    return true;
}

bool TCB::set_dptype(uint32_t type)
{
    m_dptype = type;
    return true;
}
bool TCB::set_end(uint32_t end)
{
    m_groupend = end;
    return true;
}

bool TCB::set_group_dep(uint32_t dep, uint32_t group)
{
    group = (0x1 << 15) | group;
    switch (dep)
    {
    case 0:
        m_dep0 = group;
        break;
    case 1:
        m_dep1 = group;
        break;
    case 2:
        m_dep2 = group;
        break;
    case 3:
        m_dep3 = group;
        break;
    default:
        break;
    }
    return true;
}

bool TCB::is_task()
{
    return m_is_task;
}

}; // namespace aipubt

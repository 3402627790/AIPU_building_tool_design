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
#include "tcbgroupinit.h"
#include <memory.h>

namespace aipubt
{

bool TCBGroupInit::set_asid_addr(uint32_t asid, uint64_t addr)
{
    bool ret = true;
    switch (asid)
    {
    case 0:
        m_asid0_addr = addr;
        break;
    case 1:
        m_asid1_addr = addr;
        break;
    case 2:
        m_asid2_addr = addr;
        break;
    case 3:
        m_asid3_addr = addr;
        break;
    default:
        ret = false;
        break;
    }
    return ret;
}

bool TCBGroupInit::set_smmu(uint32_t addr, uint32_t size, uint32_t asid)
{
    m_smmu0_addr = addr;
    if (m_smmu0_addr != (m_smmu0_addr >> 14 << 14))
    {
        return false;
    }
    m_smmu0_size = size;
    m_smmu0_asid = asid;
    return true;
}

} // namespace aipubt
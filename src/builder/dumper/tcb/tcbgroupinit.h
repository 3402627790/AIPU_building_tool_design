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
#ifndef BUILDER_TCB_GROUP_INIT_H
#define BUILDER_TCB_GROUP_INIT_H

#include "tcb.h"

namespace aipubt
{
class TCBGroupInit : virtual public TCB
{
protected:
    uint32_t m_smmu0_addr;
    uint32_t m_smmu0_size;
    uint32_t m_smmu0_asid;
    uint64_t m_asid0_addr;
    uint64_t m_asid1_addr;
    uint64_t m_asid2_addr;
    uint64_t m_asid3_addr;

public:
    TCBGroupInit()
    {
        m_smmu0_addr = 0;
        m_smmu0_size = 0;
        m_smmu0_asid = 0;
        m_asid0_addr = 0;
        m_asid1_addr = 0;
        m_asid2_addr = 0;
        m_asid3_addr = 0;
        m_is_task = false;
    }
    bool set_smmu(uint32_t addr, uint32_t size, uint32_t asid);
    bool set_asid_addr(uint32_t asid, uint64_t addr);
    virtual ~TCBGroupInit(){};
};

}; // namespace aipubt
#endif
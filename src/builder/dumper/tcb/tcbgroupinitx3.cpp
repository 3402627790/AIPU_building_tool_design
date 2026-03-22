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
#include "tcbgroupinitx3.h"
#include "core/logger.h"
#include <memory.h>

namespace aipubt
{

bool TCBGroupInitX3::gen_tcb(char *ptr, uint32_t current, uint32_t next) const
{
    uint16_t flag = (1 & 0xf) | ((m_dptype & 0x3) << 4) | ((m_groupend & 0x7) << 6);
    uint32_t rd_en = 1;
    uint32_t wr_en = 1;
    uint32_t size = 0; //4GB
    uint32_t asid0_l = ((m_asid0_addr & 0xfffff000) | ((rd_en & 0x1) << 6) | ((wr_en & 0x1) << 5) | (size & 0x1f));
    uint32_t asid0_h = m_asid0_addr >> 32;
    uint32_t asid1_l = ((m_asid1_addr & 0xfffff000) | ((rd_en & 0x1) << 6) | ((wr_en & 0x1) << 5) | (size & 0x1f));
    uint32_t asid1_h = m_asid1_addr >> 32;
    uint32_t asid2_l = ((m_asid2_addr & 0xfffff000) | ((rd_en & 0x1) << 6) | ((wr_en & 0x1) << 5) | (size & 0x1f));
    uint32_t asid2_h = m_asid2_addr >> 32;
    uint32_t asid3_l = ((m_asid3_addr & 0xfffff000) | ((rd_en & 0x1) << 6) | ((wr_en & 0x1) << 5) | (size & 0x1f));
    uint32_t asid3_h = m_asid3_addr >> 32;

    uint32_t core_smmu_ctrl = m_smmu0_size > 0 ? (0x1 << 5) | (0x1 << 4) | 0x1 : 0;
    // uint32_t core_smmu_remap_ctrl0 = ((m_smmu0_asid & 0x3) << 18) | ((0 & 0x3) << 16) | (0xf << 12) | (0xf << 8);
    uint32_t core_smmu_remap_ctrl0 = ((m_smmu0_asid & 0x3) << 30) | ((m_smmu0_asid & 0x3) << 26) | ((m_smmu0_asid & 0x3) << 22) | ((m_smmu0_asid & 0x3) << 18) |
                                     (0xf << 12) | (0xf << 8);
    uint32_t core_smmu_remap_ctrl1 = ((m_smmu0_asid & 0x3) << 30) | ((m_smmu0_asid & 0x3) << 26) | ((m_smmu0_asid & 0x3) << 22) | ((m_smmu0_asid & 0x3) << 18) |
                                     (0xf << 12) | (0xf << 8);
    uint32_t smmu0_ctrl0 = m_smmu0_addr & 0xfffffc000; //no limit
    uint32_t smmu0_ctrl1 = 0;
    uint32_t smmu1_ctrl0 = m_smmu0_addr + 0x1000000; //no limit
    uint32_t smmu1_ctrl1 = 0;
    uint32_t smmu2_ctrl0 = m_smmu0_addr + 0x2000000; //no limit
    uint32_t smmu2_ctrl1 = 0;
    uint32_t smmu3_ctrl0 = m_smmu0_addr + 0x3000000; //no limit
    uint32_t smmu3_ctrl1 = 0;
    uint32_t smmu4_ctrl0 = m_smmu0_addr + 0x4000000; //no limit
    uint32_t smmu4_ctrl1 = 0;
    uint32_t smmu5_ctrl0 = m_smmu0_addr + 0x5000000; //no limit
    uint32_t smmu5_ctrl1 = 0;
    uint32_t smmu6_ctrl0 = m_smmu0_addr + 0x6000000; //no limit
    uint32_t smmu6_ctrl1 = 0;
    uint32_t smmu7_ctrl0 = m_smmu0_addr + 0x7000000; //no limit
    uint32_t smmu7_ctrl1 = 0;

    memset(ptr, 0x0, TCB_SIZE);
    set_tcb_value(ptr, 0, 0, flag, 32);                  //flag
    set_tcb_value(ptr, 1, 0, core_smmu_ctrl, 32);        //core mmu seg ctrl
    set_tcb_value(ptr, 2, 0, core_smmu_remap_ctrl0, 32); //core remap ctrl0
    set_tcb_value(ptr, 3, 0, core_smmu_remap_ctrl1, 32); //core remap ctrl1
    set_tcb_value(ptr, 4, 0, 0, 16);                     //group interrupt
    set_tcb_value(ptr, 5, 16, m_grid_id, 16);            //gridID
    set_tcb_value(ptr, 5, 0, m_group_id, 16);            //groupID
    set_tcb_value(ptr, 6, 0, smmu0_ctrl0, 32);           //seg0_ctl0
    set_tcb_value(ptr, 7, 0, smmu0_ctrl1, 32);           //seg0_ctl1
    set_tcb_value(ptr, 8, 0, smmu1_ctrl0, 32);           //seg1_ctl0
    set_tcb_value(ptr, 9, 0, smmu1_ctrl1, 32);           //seg1_ctl1
    set_tcb_value(ptr, 10, 0, smmu2_ctrl0, 32);          //seg2_ctl0
    set_tcb_value(ptr, 11, 0, smmu2_ctrl1, 32);          //seg2_ctl1
    set_tcb_value(ptr, 12, 0, smmu3_ctrl0, 32);          //seg3_ctl0
    set_tcb_value(ptr, 13, 0, smmu3_ctrl1, 32);          //seg3_ctl1
    set_tcb_value(ptr, 14, 0, smmu4_ctrl0, 32);          //seg4_ctl0
    set_tcb_value(ptr, 15, 0, smmu4_ctrl1, 32);          //seg4_ctl1
    set_tcb_value(ptr, 16, 0, smmu5_ctrl0, 32);          //seg5_ctl0
    set_tcb_value(ptr, 17, 0, smmu5_ctrl1, 32);          //seg5_ctl1
    set_tcb_value(ptr, 18, 0, smmu6_ctrl0, 32);          //seg6_ctl0
    set_tcb_value(ptr, 19, 0, smmu6_ctrl1, 32);          //seg6_ctl1
    set_tcb_value(ptr, 20, 0, smmu7_ctrl0, 32);          //seg7_ctl0
    set_tcb_value(ptr, 21, 0, smmu7_ctrl1, 32);          //seg7_ctl1
    set_tcb_value(ptr, 22, 0, asid0_l, 32);              //ASID0_L/ASID0_ctl0
    set_tcb_value(ptr, 23, 0, asid0_h, 32);              //ASID0_H
    set_tcb_value(ptr, 24, 0, asid1_l, 32);              //ASID1_L/ASID0_ctl0
    set_tcb_value(ptr, 25, 0, asid1_h, 32);              //ASID1_H
    set_tcb_value(ptr, 26, 0, asid2_l, 32);              //ASID2_L/ASID0_ctl0
    set_tcb_value(ptr, 27, 0, asid2_h, 32);              //ASID2_H
    set_tcb_value(ptr, 28, 0, asid3_l, 32);              //ASID3_L/ASID0_ctl0
    set_tcb_value(ptr, 29, 0, asid3_h, 32);              //ASID3_H
    set_tcb_value(ptr, 30, 0, m_dep0, 16);               //group dep0
    set_tcb_value(ptr, 30, 16, m_dep1, 16);              //group dep1
    set_tcb_value(ptr, 31, 0, m_dep2, 16);               //group dep0
    set_tcb_value(ptr, 31, 16, m_dep3, 16);              //group dep1

    return true;
}

uint32_t TCBGroupInitX3::get_size() const
{
    return TCB_SIZE;
}

} // namespace aipubt
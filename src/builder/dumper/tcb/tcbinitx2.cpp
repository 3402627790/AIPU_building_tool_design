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
#include "tcbinitx2.h"
#include "core/logger.h"
#include <map>
#include <memory.h>
namespace aipubt
{

bool TCBInitX2::gen_tcb(char *ptr, uint32_t current, uint32_t next) const
{
    uint32_t dtcm_en_0 = 0;
    uint32_t rd_en_0 = 1;
    uint32_t wr_en_0 = 1;
    uint32_t size_0 = 0; //4GB
    uint32_t as_lo_0 = ((m_asid0_addr & 0xfffffc00) | (dtcm_en_0 & 0x1) << 7) | ((rd_en_0 & 0x1) << 6) | ((wr_en_0 & 0x1) << 5) | (size_0 & 0x1f);
    uint32_t as_hi_0 = m_asid0_addr >> 32;
    uint32_t as_lo_1 = ((m_asid1_addr & 0xfffffc00) | (dtcm_en_0 & 0x1) << 7) | ((rd_en_0 & 0x1) << 6) | ((wr_en_0 & 0x1) << 5) | (size_0 & 0x1f);
    uint32_t as_hi_1 = m_asid1_addr >> 32;
    uint32_t as_lo_2 = ((m_asid2_addr & 0xfffffc00) | (dtcm_en_0 & 0x1) << 7) | ((rd_en_0 & 0x1) << 6) | ((wr_en_0 & 0x1) << 5) | (size_0 & 0x1f);
    uint32_t as_hi_2 = m_asid2_addr >> 32;
    uint32_t as_lo_3 = ((m_asid3_addr & 0xfffffc00) | (dtcm_en_0 & 0x1) << 7) | ((rd_en_0 & 0x1) << 6) | ((wr_en_0 & 0x1) << 5) | (size_0 & 0x1f);
    uint32_t as_hi_3 = m_asid3_addr >> 32;

    // uint32_t smmu_cfg = m_smmu_en ? Target::get_the_target()->core_num() : 0;
    uint32_t smmu_cfg = m_core;
    uint32_t flag = (smmu_cfg & 0x1f) << 16;

    uint32_t gm0_lo = m_gm_addr & 0xffffffff;
    uint32_t gm0_hi = m_gm_addr >> 32;

    uint32_t gm1_lo = 0;
    uint32_t gm1_hi = 0;

    std::map<std::uint32_t, std::uint32_t> remap_size_map =
        {
            {512, 0},
            {1024, 1},
            {2048, 2},
            {4096, 3},
            {8192, 4},
            {16384, 5},
            {32768, 6},
        };

    uint32_t remap_size = 0;
    if (m_gm_size)
    {
        auto it = remap_size_map.find((m_gm_size >> 10));
        if (it != remap_size_map.end())
        {
            remap_size = it->second;
        }
        else
        {
            ERROR() << "gm size error";
        }
    }

    uint32_t remap_en = m_gm_size ? 1 : 0; // disable gm
    uint32_t gm_ctl = ((remap_size & 0xf) << 16) | (remap_en & 0x1);
    uint32_t data_sync = m_gm_sync_size ? 1 : 0;
    uint32_t sync_size = ceil((float)m_gm_sync_size / 0x1000);
    uint32_t gm_ctl0 = ((data_sync & 0x1) << 30) | (m_gm_sync_offset & 0xffff000) | sync_size;
    uint32_t gm_ctl1 = 0;

    memset(ptr, 0x0, TCB_SIZE);
    set_tcb_value(ptr, 0, 0, flag, 32);
    set_tcb_value(ptr, 1, 0, next, 32);
    set_tcb_value(ptr, 4, 0, gm_ctl, 32); //gm_ctl
    set_tcb_value(ptr, 5, 16, m_grid_id, 16);
    set_tcb_value(ptr, 6, 0, gm_ctl0, 32);  //gm_region0_ctl
    set_tcb_value(ptr, 7, 0, gm_ctl1, 32);  //gm_region1_ctl
    set_tcb_value(ptr, 8, 0, gm0_lo, 32);   //gm_region0_addr_l
    set_tcb_value(ptr, 9, 0, gm0_hi, 32);   //gm_region0_addr_h
    set_tcb_value(ptr, 10, 0, gm1_lo, 32);  //gm_region1_addr_l
    set_tcb_value(ptr, 11, 0, gm1_hi, 32);  //gm_region1_addr_h
    set_tcb_value(ptr, 12, 0, as_lo_0, 32); //as0_l
    set_tcb_value(ptr, 13, 0, as_hi_0, 32); //as0_h
    set_tcb_value(ptr, 14, 0, as_lo_1, 32); //as1_l
    set_tcb_value(ptr, 15, 0, as_hi_1, 32); //as1_h
    set_tcb_value(ptr, 16, 0, as_lo_2, 32); //as2_l
    set_tcb_value(ptr, 17, 0, as_hi_2, 32); //as2_h
    set_tcb_value(ptr, 18, 0, as_lo_3, 32); //as3_l
    set_tcb_value(ptr, 19, 0, as_hi_3, 32); //as3_h

    char *tcb_ptr = ptr;
    uint32_t seg_remap_share_en = 1;
    uint32_t seg_remap_en = 1;
    uint32_t mem_ctrl_en = 1;
    uint32_t mem_seg_ctrl = (seg_remap_share_en & 0x1) << 5 | (seg_remap_en & 0x1) << 4 | (mem_ctrl_en & 0x1);

    uint32_t core_seg_remap_ctrl = 0;

    int core_num = m_core;
    // int core_num = 3;
    uint32_t base_0_addr = m_smmu0_addr >> 14;
    uint32_t size_0_limit = 0; //no limit = 32M
    uint32_t core_seg0_ctrl0 = (base_0_addr & 0x3ffff) << 14 | (size_0_limit & 0x7f);

    for (int i = 0; i < core_num / 2 + core_num % 2; ++i)
    {
        tcb_ptr += TCB_SIZE;
        memset(tcb_ptr, 0x0, TCB_SIZE);
        set_tcb_value(tcb_ptr, 6, 0, mem_seg_ctrl, 32);        //mem segment control
        set_tcb_value(tcb_ptr, 7, 0, core_seg_remap_ctrl, 32); //segment remap control
        set_tcb_value(tcb_ptr, 8, 0, core_seg0_ctrl0, 32);     //seg 0 ctrl 0
        set_tcb_value(tcb_ptr, 9, 0, 0, 32);                   //seg 0 ctrl 1
        set_tcb_value(tcb_ptr, 10, 0, 0, 32);                  //seg 1 ctrl 0
        set_tcb_value(tcb_ptr, 11, 0, 0, 32);                  //seg 1 ctrl 1
        set_tcb_value(tcb_ptr, 12, 0, 0, 32);                  //seg 2 ctrl 0
        set_tcb_value(tcb_ptr, 13, 0, 0, 32);                  //seg 2 ctrl 1
        set_tcb_value(tcb_ptr, 14, 0, 0, 32);                  //seg 3 ctrl 0
        set_tcb_value(tcb_ptr, 15, 0, 0, 32);                  //seg 3 ctrl 1

        if ((i == core_num / 2) && (core_num % 2 != 0))
            break;

        set_tcb_value(tcb_ptr, 22, 0, mem_seg_ctrl, 32);        //mem segment control
        set_tcb_value(tcb_ptr, 23, 0, core_seg_remap_ctrl, 32); //segment remap control
        set_tcb_value(tcb_ptr, 24, 0, core_seg0_ctrl0, 32);     //seg 0 ctrl 0
        set_tcb_value(tcb_ptr, 25, 0, 0, 32);                   //seg 0 ctrl 1
        set_tcb_value(tcb_ptr, 26, 0, 0, 32);                   //seg 1 ctrl 0
        set_tcb_value(tcb_ptr, 27, 0, 0, 32);                   //seg 1 ctrl 1
        set_tcb_value(tcb_ptr, 28, 0, 0, 32);                   //seg 2 ctrl 0
        set_tcb_value(tcb_ptr, 29, 0, 0, 32);                   //seg 2 ctrl 1
        set_tcb_value(tcb_ptr, 30, 0, 0, 32);                   //seg 3 ctrl 0
        set_tcb_value(tcb_ptr, 31, 0, 0, 32);                   //seg 3 ctrl 1
    }

    return true;
}

uint32_t TCBInitX2::get_size() const
{
    return TCB_SIZE + (m_core / 2 + m_core % 2) * TCB_SIZE;
}

} // namespace aipubt
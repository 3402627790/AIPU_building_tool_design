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

#include "tcb_manager.h"
#include "core/logger.h"
#include <algorithm>
#include <core/config.h>
#include <core/constants.h>
#include <cstdint>
#define MIN(a, b) ((a) < (b) ? (a) : (b))
namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

bool TcbManager::set_stack_size(uint32_t size)
{
    m_stack_size = aligned(size, 0x1000);
    return true;
}

bool TcbManager::set_text_size(uint32_t size)
{
    m_text_size = size;
    return true;
}

bool TcbManager::set_data_size(uint32_t size)
{
    m_data_size = size;
    return true;
}

bool TcbManager::set_print_size(uint32_t size)
{
    m_print_size = size;
    return true;
}

bool TcbManager::set_l2_flush()
{
    m_l2_flush = true;
    return true;
}

void TcbManager::add_grid_init_tcb(uint64_t gm_remap_addr, uint32_t gm_size, uint32_t gm_sync_in_size, uint32_t group_num)
{
    TCBGridInit *tcb_grid_init;
    m_grid_id++;
    if (m_target == HWTARGET::X2)
    {
        auto tcb_init_x2 = new TCBInitX2;
        tcb_init_x2->set_asid_addr(0, m_asid[0]);
        tcb_init_x2->set_asid_addr(1, m_asid[1]);
        tcb_init_x2->set_asid_addr(2, m_asid[2]);
        tcb_init_x2->set_asid_addr(3, m_asid[3]);
        if (m_smmu_size > 0)
        {
            tcb_init_x2->set_smmu(m_smmu_addr, m_smmu_size, m_smmu_asid);
        }
        tcb_grid_init = tcb_init_x2;
    }
    else
    {
        tcb_grid_init = new TCBGridInitX3;
        if (m_l2_flush)
        {
            tcb_grid_init->set_l2_flush();
        }
    }
    tcb_grid_init->set_gm(gm_remap_addr, gm_size, 0, gm_sync_in_size);
    tcb_grid_init->set_group_id(m_group_id);
    tcb_grid_init->set_core(m_core_num);
    tcb_grid_init->set_group_num(group_num);
    tcb_grid_init->set_grid_id(m_grid_id);
    m_tcb_vec.push_back(tcb_grid_init);
}

void TcbManager::add_group_init_tcb(uint32_t sg_id,
                                    bool depd_all,
                                    const std::vector<uint32_t> &depd_id)
{
    m_group_id++;
    if (m_target == HWTARGET::X2)
    {
        return;
    }
    m_id_map[sg_id] = m_group_id;
    TCBGroupInitX3 *tcb_group_init = new TCBGroupInitX3;
    tcb_group_init->set_asid_addr(0, m_asid[0]);
    tcb_group_init->set_asid_addr(1, m_asid[1]);
    tcb_group_init->set_asid_addr(2, m_asid[2]);
    tcb_group_init->set_asid_addr(3, m_asid[3]);
    tcb_group_init->set_group_id(m_group_id);
    tcb_group_init->set_grid_id(m_grid_id);
    uint32_t dptype = m_group_id == 1 ? 0 : depd_all ? 2 : 1;
    tcb_group_init->set_dptype(dptype);
    if (m_smmu_size > 0)
    {
        tcb_group_init->set_smmu(m_smmu_addr, m_smmu_size, m_smmu_asid);
    }
    for (size_t k = 0; k < depd_id.size(); ++k)
    {
        tcb_group_init->set_group_dep(k, m_id_map[depd_id[k]]);
    }
    m_tcb_vec.push_back(tcb_group_init);
}

void TcbManager::add_task_tcb(uint32_t sg_id,
                              bool sg_depd_all,
                              std::vector<uint32_t> sg_depd_id,
                              uint32_t text_addr,
                              uint32_t rodata_addr,
                              uint32_t global_param_addr,
                              uint32_t prof_addr,
                              uint32_t ro_addr,
                              uint32_t warmup_len)
{
    TCBTask *tcb;
    for (size_t i = 0; i < (size_t)m_tec_num; ++i)
    {
        uint32_t sp_offset = (sg_depd_all && (i == 0)) ? 0 : m_private_buf_offset;
        uint32_t dp_offset = sp_offset + m_stack_size;
        uint32_t print_offset = dp_offset + m_data_size;
        uint32_t dptype;
        if (m_target == HWTARGET::X2)
        {
            dptype = (sg_depd_all && i == 0) ? 2 : 0;
            tcb = new TCBTaskX2;
        }
        else
        {
            dptype = 0; //depend relation is set in group tcb
            auto tcb_x3 = new TCBTaskX3;
            tcb_x3->set_warmup(warmup_len);
            tcb = tcb_x3;
        }
        tcb->set_id(m_group_id, i, 0);
        tcb->set_spc(text_addr);
        tcb->set_dptype(dptype);
        tcb->set_grid_id(m_grid_id);
        tcb->set_tec_num(m_tec_num);
        for (size_t k = 0; k < sg_depd_id.size(); ++k)
        {
            tcb->set_group_dep(k, m_id_map[sg_depd_id[k]]);
        }
        if (i == m_tec_num - 1)
        {
            tcb->set_end(1); // tec end
        }
        uint32_t sp = sp_offset;
        uint32_t pp = ro_addr;
        uint32_t dp = dp_offset;
        uint32_t cp = rodata_addr;
        uint32_t print = print_offset;
        uint32_t prof = prof_addr;
        uint32_t gpp = global_param_addr;
        tcb->set_pointer(sp, pp, dp, cp, print, prof, gpp);

        m_tcb_vec.push_back(tcb);
        m_private_buf_offset = print_offset + m_print_size;
    }
    m_private_buf_size = std::max(m_private_buf_offset, m_private_buf_size);
    m_ws_buf_size = std::max(m_ws_buf_size, m_ws_buf_size);
}

void TcbManager::gen_tcb(char *ptr, uint32_t addr, uint32_t stack_offset)
{
    char *tcb_ptr = ptr;
    uint32_t current = addr, next = addr;
    for (size_t i = 0; i < m_tcb_vec.size(); ++i)
    {
        if (m_tcb_vec[i]->is_task())
        {
            ((TCBTask *)m_tcb_vec[i])->set_stack_offset(stack_offset);
            if (i != m_tcb_vec.size() - 1)
            {
                ((TCBTask *)m_tcb_vec[i])->disable_done_irq();
            }
        }
        uint32_t size = m_tcb_vec[i]->get_size();
        if (i == m_tcb_vec.size() - 1)
        {
            next = 0;
            m_tcb_vec[i]->set_end(7);
        }
        else
        {
            next = next + size;
        }
        uint32_t next_addr = (next == 0) ? 0 : (m_target == HWTARGET::X2) ? (m_asid[0] + next) & 0xffffffff : next;
        m_tcb_vec[i]->gen_tcb(tcb_ptr, current, next_addr);
        tcb_ptr = tcb_ptr + size;
        // INFO() << "gen tcb " << i << " current 0x" << std::hex << current << " next 0x" << next;
        current = next;
    }
}

uint32_t TcbManager::get_private_buf_size()
{
    return m_private_buf_size;
}
uint32_t TcbManager::get_tcb_size()
{
    uint32_t total_size = 0;
    for (size_t i = 0; i < m_tcb_vec.size(); ++i)
    {
        total_size += m_tcb_vec[i]->get_size();
    }

    return total_size;
}
uint32_t TcbManager::get_tcb_num()
{
    return m_tcb_vec.size();
}

void TcbManager::set_asid(const std::vector<uint64_t> &asid)
{
    m_asid = asid;
}

void TcbManager::set_smmu(uint32_t addr, uint32_t size, uint32_t asid)
{
    m_smmu_addr = addr;
    m_smmu_size = size;
    m_smmu_asid = asid;
}

} // namespace aipubt
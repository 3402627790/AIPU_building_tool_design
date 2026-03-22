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
#ifndef BUILDER_TCB_MANAGER_H
#define BUILDER_TCB_MANAGER_H

#include "builder/aipu_elf.h"
#include "core/logger.h"
#include "tcb/tcbgridinitx3.h"
#include "tcb/tcbgroupinitx3.h"
#include "tcb/tcbinitx2.h"
#include "tcb/tcbtask.h"
#include "tcb/tcbtaskx2.h"
#include "tcb/tcbtaskx3.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace aipubt
{

class TcbManager
{
    enum HWTARGET
    {
        X2,
        X3,
        OTHERS,
    };

private:
    std::vector<TCB *> m_tcb_vec;
    int m_group_id = 0;
    int m_grid_id = 0;
    uint32_t m_text_size;
    uint32_t m_stack_size;
    uint32_t m_data_size;
    uint32_t m_print_size;
    uint32_t m_core_num;
    uint32_t m_tec_num;
    HWTARGET m_target;
    uint32_t m_private_buf_offset;
    uint32_t m_private_buf_size;
    uint32_t m_ws_buf_size;
    std::vector<uint32_t> m_ws_buf_offset;
    std::map<uint32_t, uint32_t> m_id_map;
    uint32_t m_broadcast;
    std::vector<uint64_t> m_asid;
    uint32_t m_smmu_addr;
    uint32_t m_smmu_size;
    uint32_t m_smmu_asid;
    bool m_l2_flush;

public:
    TcbManager(uint32_t core_num, uint32_t tec_num, const std::string &target)
    {
        m_ws_buf_size = 0;
        m_private_buf_offset = 0;
        m_private_buf_size = 0;
        m_text_size = 0;
        m_stack_size = 0;
        m_data_size = 0;
        m_print_size = 0;
        m_core_num = core_num;
        m_tec_num = tec_num;
        m_broadcast = 0;
        m_smmu_addr = 0;
        m_smmu_size = 0;
        m_smmu_asid = 0;
        m_l2_flush = false;

        if (target.find("X2") != std::string::npos)
        {
            m_target = X2;
        }
        else if (target.find("X3") != std::string::npos)
        {
            m_target = X3;
        }
        else
        {
            WARN() << "target not found";
            m_target = OTHERS;
        }
    };
    ~TcbManager()
    {
        for (auto tcb : m_tcb_vec)
        {
            delete tcb;
        }
    };

    bool set_text_size(uint32_t size);
    bool set_stack_size(uint32_t size);
    bool set_data_size(uint32_t size);
    bool set_print_size(uint32_t size);

    void setup_init_tcb(uint64_t gm_remap_addr,
                        uint32_t gm_size,
                        bool gm_sync_in);

    void add_task_tcb(uint32_t sg_id,
                      bool sg_depd_all,
                      std::vector<uint32_t> sg_depd_id,
                      uint32_t text_addr,
                      uint32_t rodata_addr,
                      uint32_t global_param_addr,
                      uint32_t prof_addr,
                      uint32_t ro_addr,
                      uint32_t warmup_len = 0);
    void add_grid_init_tcb(uint64_t gm_remap_addr,
                           uint32_t gm_size,
                           uint32_t gm_sync_in_size,
                           uint32_t group_num);
    void add_group_init_tcb(uint32_t sg_id,
                            bool depd_all,
                            const std::vector<uint32_t> &depd_id);
    uint32_t get_private_buf_size();
    uint32_t get_tcb_size();
    uint32_t get_tcb_num();
    void gen_tcb(char *ptr, uint32_t addr, uint32_t stack_offset);
    void set_asid(const std::vector<uint64_t> &asid);
    void set_smmu(uint32_t addr, uint32_t size, uint32_t asid);
    bool set_l2_flush();
};

}; // namespace aipubt
#endif
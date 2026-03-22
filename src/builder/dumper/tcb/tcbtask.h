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
#ifndef BUILDER_TCB_TASK_H
#define BUILDER_TCB_TASK_H

#include "tcb.h"

namespace aipubt
{
class TCBTask : public TCB
{
protected:
    uint32_t m_spc;     // text address
    uint32_t m_task_id; // tec#
    uint32_t m_sp, m_pp, m_dp, m_cp, m_print, m_prof, m_gpp;
    uint32_t m_delay, m_gm_dma, m_sm_dma;
    uint32_t m_tec_num;

    /**
     * +------+------------------+---------------+--------+--------------------------------+
     * | Bits |       Name       | Default value | Access |          Description           |
     * +------+------------------+---------------+--------+--------------------------------+
     * | 11   | pool_irq_en      | 0x0           | RW     | Command pool interrupt enable. |
     * | 10   | cluster_irq_en   | 0x0           | RW     | Cluster interrupt enable.      |
     * | 9    | core_irq_en      | 0x0           | RW     | AIPU core interrupt enable.    |
     * | 8    | tec_irq_en       | 0x0           | RW     | TEC interrupt enable.          |
     * | 7:6  | Reserved         | 0x0           | RO     |                                |
     * | 5    | signal_irq_en    | 0x0           | RW     | IRQ Signal interrupt enable.   |
     * | 4    | error_irq_en     | 0x0           | RW     | Pool error interrupt enable.   |
     * | 3    | fault_irq_en     | 0x0           | RW     | Fault interrupt enable.        |
     * | 2    | exception_irq_en | 0x0           | RW     | Exception interrupt enable.    |
     * | 1    | wait_irq_en      | 0x0           | RW     | Wait interrupt enable.         |
     * | 0    | done_irq_en      | 0x0           | RW     | Done interrupt enable.         |
     * +------+------------------+---------------+--------+--------------------------------+
     */
    uint32_t m_interrupt = 0xc2f;

public:
    TCBTask()
    {
        m_delay = 0;
        m_gm_dma = 0;
        m_sm_dma = 0;
        m_is_task = true;
    }
    virtual bool gen_tcb(char *ptr, uint32_t current, uint32_t next) const = 0;
    virtual uint32_t get_size() const = 0;
    virtual ~TCBTask(){};
    bool set_id(uint32_t groupid, uint32_t taskid, uint32_t gridid);
    bool set_spc(uint32_t spc);
    bool set_pointer(uint32_t sp, uint32_t pp, uint32_t dp, uint32_t cp, uint32_t print, uint32_t prof, uint32_t gpp);
    bool set_delay(uint32_t delay);
    bool set_dma(uint32_t gm_dma, uint32_t sm_dma);
    bool set_stack_offset(uint32_t offset);
    bool set_tec_num(uint32_t tec_num);
    bool disable_done_irq();
};

};     // namespace aipubt
#endif //!BUILDER_TCB_TASK_H
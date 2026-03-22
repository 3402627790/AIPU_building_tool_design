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
#ifndef BUILDER_TCB_GENERATOR_H
#define BUILDER_TCB_GENERATOR_H

#include <cstdint>
#include <vector>

namespace aipubt
{
#define TCB_SIZE 128 // 32(bit) x 32(line)
class TCB
{
public:
    bool set_grid_id(uint32_t id);
    bool set_group_id(uint32_t id);
    bool set_group_dep(uint32_t dep, uint32_t group);
    bool set_dptype(uint32_t type);
    bool set_end(uint32_t end);
    bool is_task();
    virtual bool gen_tcb(char *ptr, uint32_t current, uint32_t next) const = 0;
    virtual uint32_t get_size() const = 0;
    virtual ~TCB(){};

protected:
    uint32_t m_dep0 = 0;
    uint32_t m_dep1 = 0;
    uint32_t m_dep2 = 0;
    uint32_t m_dep3 = 0;
    uint32_t m_dptype = 0;
    uint32_t m_groupend = 0;
    uint32_t m_broadcast = 0;
    uint32_t m_grid_id = 0;
    uint32_t m_group_id = 0;
    bool m_is_task = false;
    bool set_tcb_value(char *ptr, uint32_t line, uint32_t bitAddr, uint32_t value, uint32_t bitLength) const;
};

};     // namespace aipubt
#endif //!BUILDER_TCB_GENERATOR_H
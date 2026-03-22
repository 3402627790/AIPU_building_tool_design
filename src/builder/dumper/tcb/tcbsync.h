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
#ifndef BUILDER_TCB_SYNC_H
#define BUILDER_TCB_SYNC_H

#include "tcb.h"
#include <cstdint>
#include <vector>

namespace aipubt
{
#define TCB_SIZE 128 // 32(bit) x 32(line)

class TCBSync : public TCB
{
private:
    uint32_t m_gm_ctl0;
    uint32_t m_gm_ctl1;

public:
    TCBSync()
    {
        m_gm_ctl0 = 0;
        m_gm_ctl1 = 0;
    }
    bool set_gm(uint32_t gm_ctl0, uint32_t gm_ctl1);
    virtual bool gen_tcb(char *ptr, uint32_t current, uint32_t next) const;
    virtual ~TCBSync(){};
};

};     // namespace aipubt
#endif //!BUILDER_TCB_GENERATOR_H
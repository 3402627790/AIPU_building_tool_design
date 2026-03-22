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
#ifndef BUILDER_TCB_GRID_INIT_X3_H
#define BUILDER_TCB_GRID_INIT_X3_H

#include "tcbgridinit.h"

namespace aipubt
{
class TCBGridInitX3 : public TCBGridInit
{

public:
    bool gen_tcb(char *ptr, uint32_t current, uint32_t next) const;
    uint32_t get_size() const;
    TCBGridInitX3() {}
    ~TCBGridInitX3(){};
};

}; // namespace aipubt
#endif
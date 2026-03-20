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
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Alex Zheng
 * @Date: 2025-05-08 14:57:41
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2025-05-08 20:21:24
 */
#ifndef DATALAYOUT_SCHEDULER_H
#define DATALAYOUT_SCHEDULER_H

#include "core/core.h"
#include "core/pass_manager.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <stdint.h>
#include <vector>

namespace aipubt
{

class DataLayoutSchedule : public PassManager
{
public:
    DataLayoutSchedule();
    ~DataLayoutSchedule() {}
    static PassManagerPtr create()
    {
        auto ptr = new DataLayoutSchedule();
        return PassManagerPtr(ptr);
    }
};

} // namespace aipubt
#endif // DATALAYOUT_SCHEDULER_H
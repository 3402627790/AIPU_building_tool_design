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
 * @Author: Neo Wang 
 * @Date: 2022-05-13 14:12:11 
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-08-04 01:13:15
 */
#include "statistic_pass.h"
#include "core/basepass.h"
#include "core/core.h"
#include <algorithm>
#include <set>

/**
 * @brief This file is all passes apply for graph analysis, which would not change the graph at all!
 *        all data is store to graph's attribute, with key: `_analysis`
 *
 */
namespace aipubt
{
const std::vector<uint64_t> get_big_buffer_thresholds()
{
    std::vector<uint64_t> BIG_BUFFER_THRESHOLDS = {
        0,
        512 * 1024,
        1   * 1024 * 1024,
        2   * 1024 * 1024,
        4   * 1024 * 1024,
        8   * 1024 * 1024,
        16  * 1024 * 1024,
        32  * 1024 * 1024,
        64  * 1024 * 1024,
        128 * 1024 * 1024,
    };
    return BIG_BUFFER_THRESHOLDS;
}


}; //namespace aipubt
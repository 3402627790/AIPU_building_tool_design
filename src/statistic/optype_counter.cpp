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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-05-13 14:30:46
 */
#include "optype_counter.h"
#include "core/core.h"

namespace aipubt
{

bool OpTypeCounter::operator()(GraphPtr g) const
{
    if (g)
    {
        for (auto n : g->nodes())
        {
            m_res[n->type()] = m_res[n->type()] + 1;
        }
    }

    return false;
}
}; // namespace aipubt
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2021 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Tom.Gao
 * @Date: 2021-01-25 17:06:21
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-04-28 14:02:28
 */
#ifndef CORE_GSIM_GSIM_H
#define CORE_GSIM_GSIM_H

#include "core/basepass.h"
#include "core/graph.h"
#include "core/pass_manager.h"

namespace aipubt
{
class GSim : public PassManager
{
public:
    GSim();
    static bool is_float_ir(GraphPtr g);
};
}; // namespace aipubt

#endif //!CORE_GSIM_GSIM_H
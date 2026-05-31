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
 * @Date: 2024-06-21 16:38:20 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-16 17:15:24
 */
#pragma once
#include "common.h"

namespace aipubt
{

void fp_vadd(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx);
void fp16_vadd(Xbyak::CodeGenerator &cg, RegMap &reg_map, const Instruction &ins, const TransContext &ctx);

}; // namespace aipubt
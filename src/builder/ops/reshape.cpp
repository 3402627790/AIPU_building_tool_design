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
 * @Author: Samule Han
 * @Date: 2020-03-13 17:15:02
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-10-13 15:26:11
 */
#include "builder/mem_remapping.h"
#include "builder/optimizer.h"
#include "builder/param_check.h"
#include "builder/plugin.h"
#include "builder/ro_generator.h"
#include "builder/target.h"
#include "builder/text_generator.h"
#include "core/constants.h"
#include "core/logger.h"

namespace aipubt
{

RO_GENERATOR(OpType_Reshape)
{
}
RO_GENERATOR_END(OpType_Reshape);

TEXT_GENERATOR(OpType_Reshape)
{
    return "";
}
TEXT_GENERATOR_END(OpType_Reshape);

MEMORY_REMAPPING_OP(OpType_Reshape)
{
    auto &out = node->outputs()[0];
    auto &inp = node->inputs()[0];
    out->memory_offset()->relative_offset = 0;
    out->memory_offset()->set_base_offset(inp->memory_offset());
}
MEMORY_REMAPPING_OP_END(OpType_Reshape);
} // namespace aipubt

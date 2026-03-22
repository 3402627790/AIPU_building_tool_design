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
#include <vector>
namespace aipubt
{

RO_GENERATOR(OpType_InternalSpliter)
{
}
RO_GENERATOR_END(OpType_InternalSpliter);

TEXT_GENERATOR(OpType_InternalSpliter)
{
    return "";
}
TEXT_GENERATOR_END(OpType_InternalSpliter);

MEMORY_REMAPPING_OP(OpType_InternalSpliter)
{
    auto inp = node->inputs()[0];
    uint32_t relative_offset=0;
    for (uint32_t i = 0; i < node->outputs().size(); ++i)
    {
        auto t = node->outputs()[i];
        t->memory_offset()->set_base_offset(inp->memory_offset());
        t->memory_offset()->relative_offset = relative_offset;
        relative_offset += t->mem_size();
    }
}
MEMORY_REMAPPING_OP_END(OpType_InternalSpliter);
}; // namespace aipubt

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
 * @Author: Lang Hu
 * @Date: 2024-10-28 11:52:59
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-10-28 11:52:59
 */
#include "builder/builder.h"
#include "builder/lowering/lowering.h"
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
MEMORY_REMAPPING_OP(OpType_CacheUpdate)
{
    int index = get_real_node(node)->params()["block_index"];
    auto inp0 = node->inputs()[0];
    size_t block_size = inp0->mem_size() / inp0->shape()[0];
    if (inp0->shape().dim() == 3)
    {
        block_size = inp0->mem_size();
    }

    auto out0 = node->outputs()[0];
    out0->memory_offset()->relative_offset = index * block_size;
    out0->memory_offset()->set_base_offset(inp0->memory_offset());

    auto out1 = node->outputs()[1];
    out1->memory_offset()->relative_offset = 0;
    out1->memory_offset()->set_base_offset(inp0->memory_offset());
}
MEMORY_REMAPPING_OP_END(OpType_CacheUpdate);
} // namespace aipubt

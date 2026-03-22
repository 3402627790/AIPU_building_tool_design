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
 * @Date: 2025-05-13 10:33:19
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-05-13 10:33:19
 */
#include "builder/builder.h"
#include "builder/mem_remapping.h"
#include "builder/ops/ops.h"
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
RO_GENERATOR(OpType_IfEnder)
{
    ro.push_back(node->inputs()[0]); // call_back
    ro.push_back(node->inputs()[1]); // cond
    ro.push_back(node->get_attr<TensorPtr>(Constants::descriptorbase));
}
RO_GENERATOR_END(OpType_IfEnder);

DESC_GENERATOR(OpType_IfEnder)
{
    std::vector<std::pair<TensorPtr, TensorPtr>> copy_then, copy_else;
    std::vector<Expr> copy_then_dim, copy_else_dim;
    size_t output_num = node->outputs().size();
    for (size_t i = 2; i < 2 + output_num; ++i)
    {
        auto then_inp = node->inputs()[i];
        auto else_inp = node->inputs()[i + output_num];
        auto out = node->outputs()[i - 2];
        copy_then.push_back({then_inp, out});
        copy_else.push_back({else_inp, out});
        for (int d = 0; d < then_inp->shape().dim(); ++d)
        {
            copy_then_dim.push_back(then_inp->shape().symbol()[d]);
        }
        for (int d = 0; d < else_inp->shape().dim(); ++d)
        {
            copy_else_dim.push_back(else_inp->shape().symbol()[d]);
        }
    }
    descriptor.push_back((uint32_t)copy_then.size());
    descriptor.push_back((uint32_t)copy_else.size());
    for (auto &&pairs : {copy_then, copy_else})
    {
        for (auto &&p : pairs)
        {
            auto dtype_size = Tensor::dtype_size(p.first->type());
            Expr data_size = dtype_size * p.first->shape().symbol().size();
            uint32_t src_base = get_memory_base(p.first);
            uint32_t dst_base = get_memory_base(p.second);
            descriptor.push_back(p.first);
            descriptor.push_back(p.second);
            descriptor.push_back(data_size);
            descriptor.push_back((src_base << 16) | dst_base);
        }
    }
    aipu_assert(copy_then_dim.size() == copy_else_dim.size());
    descriptor.push_back((uint32_t)copy_then_dim.size());
    for (auto &&dims : {copy_then_dim, copy_else_dim})
    {
        for (auto &&d : dims)
        {
            descriptor.push_back(d);
        }
    }
}
DESC_GENERATOR_END(OpType_IfEnder);

TEXT_GENERATOR(OpType_IfEnder)
{
    return "";
}
TEXT_GENERATOR_END(OpType_IfEnder);

KERNEL_GENERATOR(OpType_IfEnder)
{
    return "";
}
KERNEL_GENERATOR_END(OpType_IfEnder);

MEMORY_REMAPPING_OP(OpType_IfEnder)
{
}
MEMORY_REMAPPING_OP_END(OpType_IfEnder);
} // namespace aipubt

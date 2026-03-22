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
 * @Date: 2024-08-27 10:52:15
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-27 10:52:15
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
RO_GENERATOR(OpType_ForRangeEnder)
{
    int depend_input_num = get_control_param(node, "dependency_inputs_num");
    int scan_output_num = get_control_param(node, "scan_outputs_num");
    ro.push_back(node->inputs()[0]);                    // call_back
    ro.push_back(node->inputs()[1]);                    // frame_ptr
    ro.push_back(node->inputs()[2]);                    // scan_len
    ro.push_back(node->inputs()[3]);                    // max_count
    ro.push_back(node->inputs()[4]);                    // index
    ro.push_back(node->inputs()[5]);                    // cond_in
    ro.push_back(node->inputs()[6 + depend_input_num]); // cond_out
    ro.push_back(depend_input_num);
    ro.push_back(scan_output_num);
    ro.push_back(node->get_attr<TensorPtr>(Constants::descriptorbase));
}
RO_GENERATOR_END(OpType_ForRangeEnder);

DESC_GENERATOR(OpType_ForRangeEnder)
{
    int depend_input_num = get_control_param(node, "dependency_inputs_num");
    for (int i = 0; i < depend_input_num; ++i)
    {
        auto out = node->outputs()[i];
        auto inp = node->inputs()[i + 6];
        auto dtype_size = Tensor::dtype_size(inp->type());
        // cannot use output symbol shape until ender done
        Expr data_size = dtype_size * inp->shape().symbol().size();
        uint32_t src_base = get_memory_base(out);
        uint32_t dst_base = get_memory_base(inp);
        descriptor.push_back(out);
        descriptor.push_back(inp);
        descriptor.push_back(data_size);
        descriptor.push_back((src_base << 16) | dst_base);
    }
    for (size_t i = depend_input_num; i < node->outputs().size(); ++i)
    {
        auto out = node->outputs()[i];
        auto inp = node->inputs()[i + 7 + depend_input_num];
        int start_dim = 0;
        Expr batch_num = 1;
        if (out->shape().dim() == inp->shape().dim())
        {
            start_dim = 1;
            batch_num = inp->shape().symbol()[0];
        }
        Expr batch_size = Tensor::dtype_size(inp->type());
        for (int j = start_dim; j < inp->shape().dim(); ++j)
        {
            batch_size *= inp->shape().symbol()[j];
        }
        uint32_t src_base = get_memory_base(inp);
        uint32_t dst_base = get_memory_base(out);
        descriptor.push_back(inp);
        descriptor.push_back(out);
        descriptor.push_back(batch_num);
        descriptor.push_back(batch_size);
        descriptor.push_back((src_base << 16) | dst_base);
    }
}
DESC_GENERATOR_END(OpType_ForRangeEnder);

TEXT_GENERATOR(OpType_ForRangeEnder)
{
    return "";
}
TEXT_GENERATOR_END(OpType_ForRangeEnder);

KERNEL_GENERATOR(OpType_ForRangeEnder)
{
    return "";
}
KERNEL_GENERATOR_END(OpType_ForRangeEnder);

MEMORY_REMAPPING_OP(OpType_ForRangeEnder)
{
    int depend_input_num = get_control_param(node, "dependency_inputs_num");
    for (int i = 0; i < depend_input_num; ++i)
    {
        auto out = node->outputs()[i];
        auto inp = node->inputs()[i + 7 + depend_input_num];
        out->memory_offset()->relative_offset = 0;
        out->memory_offset()->set_base_offset(inp->memory_offset());
    }
}
MEMORY_REMAPPING_OP_END(OpType_ForRangeEnder);
} // namespace aipubt

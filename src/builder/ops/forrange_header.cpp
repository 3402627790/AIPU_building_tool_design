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
#include "builder/optimizer.h"
#include "builder/param_check.h"
#include "builder/plugin.h"
#include "builder/ro_generator.h"
#include "builder/target.h"
#include "builder/text_generator.h"
#include "core/constants.h"
#include "core/logger.h"
#include "core/subgraph.h"

namespace aipubt
{
NodeParamValue get_control_param(NodePtr n, const std::string &param)
{
    auto sg = std::dynamic_pointer_cast<SubGraph>(n->graph());
    auto sgn = sg->subgraphnode();
    if (sgn->params().count(param))
    {
        return sgn->params()[param];
    }
    return NodeParamValue();
}

uint32_t get_memory_base(TensorPtr t)
{
    uint32_t src_base = 0; // ddr
    if (t->memory_offset()->get_base_memory() >= 0xf8000000)
    {
        src_base = 8; // sram
    }
    return src_base;
}

RO_GENERATOR(OpType_ForRangeHeader)
{
    int depend_input_num = get_control_param(node, "dependency_inputs_num");
    int scan_output_num = get_control_param(node, "scan_outputs_num");
    ro.push_back(node->outputs()[0]); // call_back
    ro.push_back(node->outputs()[1]); // frame_ptr
    ro.push_back(node->outputs()[2]); // scan_len
    ro.push_back(node->outputs()[3]); // max_count
    ro.push_back(node->outputs()[4]); // index
    ro.push_back(node->outputs()[5]); // cond_in
    ro.push_back(depend_input_num);
    ro.push_back(scan_output_num);
    ro.push_back(node->get_attr<TensorPtr>(Constants::descriptorbase));
}
RO_GENERATOR_END(OpType_ForRangeHeader);

DESC_GENERATOR(OpType_ForRangeHeader)
{
    int depend_input_num = get_control_param(node, "dependency_inputs_num");
    for (int i = 2; i < 2 + depend_input_num; ++i)
    {
        auto inp = node->inputs()[i];
        auto out = node->outputs()[i + 4];
        auto dtype_size = Tensor::dtype_size(inp->type());
        // TODO: support dynamic shape
        // Expr data_size = dtype_size * inp->shape().symbol().size();
        uint32_t data_size = dtype_size * inp->shape().size();
        uint32_t src_base = get_memory_base(inp);
        uint32_t dst_base = get_memory_base(out);
        descriptor.push_back(inp);
        descriptor.push_back(out);
        descriptor.push_back(data_size);
        descriptor.push_back((src_base << 16) | dst_base);
    }
}
DESC_GENERATOR_END(OpType_ForRangeHeader);

TEXT_GENERATOR(OpType_ForRangeHeader)
{
    return "";
}
TEXT_GENERATOR_END(OpType_ForRangeHeader);

KERNEL_GENERATOR(OpType_ForRangeHeader)
{
    return "";
}
KERNEL_GENERATOR_END(OpType_ForRangeHeader);

MEMORY_REMAPPING_OP(OpType_ForRangeHeader)
{
    auto inp = node->inputs()[0];
    auto out = node->outputs()[3];
    out->memory_offset()->relative_offset = 0;
    out->memory_offset()->set_base_offset(inp->memory_offset());

    inp = node->inputs()[1];
    out = node->outputs()[5];
    out->memory_offset()->relative_offset = 0;
    out->memory_offset()->set_base_offset(inp->memory_offset());

    int depend_input_num = get_control_param(node, "dependency_inputs_num");
    for (size_t i = 2 + depend_input_num; i < node->inputs().size(); ++i)
    {
        auto inp = node->inputs()[i];
        auto out = node->outputs()[i + 4];
        out->memory_offset()->relative_offset = 0;
        out->memory_offset()->set_base_offset(inp->memory_offset());
    }
}
MEMORY_REMAPPING_OP_END(OpType_ForRangeHeader);
} // namespace aipubt

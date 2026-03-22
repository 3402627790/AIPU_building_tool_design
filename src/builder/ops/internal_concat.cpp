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
 * @Last Modified time: 2021-11-03 14:39:47
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
#include <set>
#include <vector>
namespace aipubt
{

RO_GENERATOR(OpType_InternalConcatenated)
{
}
RO_GENERATOR_END(OpType_InternalConcatenated);

TEXT_GENERATOR(OpType_InternalConcatenated)
{
    return "";
}
TEXT_GENERATOR_END(OpType_InternalConcatenated);

MEMORY_REMAPPING_OP(OpType_InternalConcatenated)
{
    auto axis_remap = [](TensorShape &shape, DataLayout output_layout,
                         DataLayout input_layout, int axis)
    {
        std::set<DataLayout> nchwc_support_layout = {
            DataLayout_NCHWC4,
            DataLayout_NCHWC8,
            DataLayout_NCHWC16,
            DataLayout_NCHWC32,
            DataLayout_NCHWC16T8,
            DataLayout_NCHWC16T1,
            DataLayout_NCHWC32T8,
            DataLayout_NCHWC32T1,
        };

        std::set<DataLayout> ndchwc_support_layout = {
            DataLayout_NDCHWC16,
            DataLayout_NDCHWC32,
        };

        std::set<DataLayout> ncdhwc_support_layout = {
            DataLayout_NCDHWC16,
            DataLayout_NCDHWC32,
        };

        int nchwc_remap[4] = {0, 2, 3, 1};
        int ndchwc_remap[5] = {0, 1, 3, 4, 2};
        int ncdhwc_remap[5] = {0, 2, 3, 4, 1};

        auto layout = output_layout;
        if (layout == DataLayout_NHWC)
        {
            layout = input_layout;
        }
        if (nchwc_support_layout.count(layout))
        {
            if (axis < 0)
                axis = shape.dim() + axis - 1;
            if (axis < 4)
                return nchwc_remap[axis];
        }
        else if (ndchwc_support_layout.count(layout))
        {
            if (axis < 0)
                axis = shape.dim() + axis - 1;
            if (axis < 5)
                return ndchwc_remap[axis];
        }
        else if (ncdhwc_support_layout.count(layout))
        {
            if (axis < 0)
                axis = shape.dim() + axis - 1;
            if (axis < 5)
                return ncdhwc_remap[axis];
        }
        else if (axis < 0)
        {
            axis = shape.dim() + axis;
        }
        return axis;
    };
    auto forward_reshape_memory = [](TensorPtr input, MemoryObjectPtr base, uint32_t offset)
    {
        NodePtr p = input->node();
        while (p->type() == OpType_Reshape)
        {
            auto input = p->inputs()[0];
            auto input_ptr = input->memory_offset()->get_base_object();
            if (input_ptr != base)
            {
                input_ptr->set_base_offset(base);
                input_ptr->relative_offset = offset;
            }
            input->node()->set_attr("fixed_mem", true);
            auto output = p->outputs()[0];
            auto output_ptr = output->memory_offset()->get_base_object();
            if (output_ptr != base)
            {
                output_ptr->set_base_offset(base);
                output_ptr->relative_offset = offset;
            }
            p = p->parents()[0];
        }
        return;
    };

    auto refine_output_stride = [](TensorShape &out_shape, std::vector<uint32_t> &strides)
    {
        std::set<DataLayout> nchwc16_tail_layout = {
            DataLayout_NCHWC16T8,
            DataLayout_NCHWC16T1,
        };
        std::set<DataLayout> nchwc32_tail_layout = {
            DataLayout_NCHWC32T8,
            DataLayout_NCHWC32T1,
        };
        std::map<DataLayout, std::set<DataLayout>> prime_to_tail = {
            {DataLayout_NCHWC16, nchwc16_tail_layout},
            {DataLayout_NCHWC32, nchwc32_tail_layout},
        };

        if (nchwc16_tail_layout.count(out_shape.layout()) ||
            nchwc32_tail_layout.count(out_shape.layout()))
        {
            TensorShape temp_out = out_shape;
            temp_out.c() = temp_out.c_total() - temp_out.c_tail();
            for (auto &p2t : prime_to_tail)
            {
                if (p2t.second.count(out_shape.layout()))
                {
                    temp_out.change_layout(p2t.first);
                }
            }
            temp_out.reset_strides();
            strides = temp_out.strides();
        }
        return strides;
    };

    auto &out = node->outputs()[0];

    for (auto inp : node->inputs())
    {
        if (inp->memory_offset()->get_base_object() == out->memory_offset())
        {
            return; // already remaped
        }
    }

    auto input_layout = node->inputs()[0]->shape().layout();
    int axis = 0, remaped_axis = 0;
    ssize_t dsize = Tensor::dtype_size(out->type());

    if (node->params().count("axis"))
    {
        axis = node->params()["axis"];
        remaped_axis = axis_remap(node->inputs()[0]->shape(), out->shape().layout(),
                                  input_layout, axis);
    }
    if (out->shape().layout() != DataLayout_NHWC)
    {
        axis = remaped_axis;
    }

    // TODO!! the share memory of concat is up_broadcast...
    bool can_share = true;
    uint32_t offset = 0;
    auto strides = out->shape().strides();
    refine_output_stride(out->shape(), strides);
    // INFO() << node->name();
    for (auto it : node->inputs())
    {
        // INFO() << "\t parent: " << it->node()->name();
        auto ptr = it->memory_offset()->get_base_object();
        if (ptr->offset() != it->memory_offset()->offset())
        {
            can_share = false;
        }
        if (ptr->get_base_object() != out->memory_offset() && ptr->get_base_object() != out->memory_offset()->get_base_object())
        {
            ptr->set_base_offset(out->memory_offset());
        }
        ptr->relative_offset = offset;
        forward_reshape_memory(it, out->memory_offset(), offset);
        offset += (it->shape()[remaped_axis] * strides[axis] * dsize);
        it->node()->set_attr("fixed_mem", true);
    }
    if (!can_share)
    {
        // TODO!!
    }
}
MEMORY_REMAPPING_OP_END(OpType_InternalConcatenated);
}; // namespace aipubt

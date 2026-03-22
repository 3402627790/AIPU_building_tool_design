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
 * @Author: Alex.Zheng
 * @Date: 2022-11-11 17:05:05
 * @Last Modified by: Alex.Zheng
 * @Last Modified time: 2022-11-11 17:19:33
 */
#include "builder/lowering/lowering.h"
#include "builder/ops/ops.h"
#include "core/core.h"
#include <algorithm>
#include <vector>

namespace aipubt
{

static int axis_remap(DataLayout layout, int axis)
{
    std::set<DataLayout> nchwc_support_layout = {
        DataLayout_NCHWC4,
        DataLayout_NCHWC8,
        DataLayout_NCHWC16,
        DataLayout_NCHWC32,
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

    if (nchwc_support_layout.count(layout) && axis < 4)
    {
        return nchwc_remap[axis];
    }
    if (ndchwc_support_layout.count(layout) && axis < 5)
    {
        return ndchwc_remap[axis];
    }
    if (ncdhwc_support_layout.count(layout) && axis < 5)
    {
        return ncdhwc_remap[axis];
    }
    return axis;
}

// Check support layout
static bool check_layout_support(NodePtr n)
{
    const std::vector<TensorPtr> &inputs = n->inputs();
    const std::vector<TensorPtr> &outputs = n->outputs();
    // 1. Check unsupport layout
    std::set<DataLayout> not_Support_layout = {
        DataLayout_NCHWC16T8,
        DataLayout_NCHWC32T8,
        DataLayout_NCHWC16T1,
        DataLayout_NCHWC32T1,
        DataLayout_NCHWC16T8Sparse,
        DataLayout_NCHWC32T8Sparse,
        DataLayout_NCHWC16T1Sparse,
        DataLayout_NCHWC32T1Sparse,
        DataLayout_NDCHWC16T8,
        DataLayout_NDCHWC32T8,
        DataLayout_NDCHWC16T1,
        DataLayout_NDCHWC32T1,
    };
    auto exception = [](TensorPtr input, TensorPtr output) {
        if (((output->shape().layout() == DataLayout_NHWC) && (input->shape().layout() == DataLayout_NCHWC32) && (output->shape().c() == 32)) || ((output->shape().layout() == DataLayout_NHWC) && (input->shape().layout() == DataLayout_NCHWC16) && (output->shape().c() == 16)))
            return true;
        else
            return false;
    };
    if ((outputs.size() == 0) && (inputs.size() == 0))
        return false;
    for (auto output : outputs)
    {
        if ((output->shape().layout() != inputs[0]->shape().layout()) && (!exception(output, inputs[0])))
        {
            return false;
        }
        if (!output->shape().is_continuous())
        {
            return false;
        }
    }
    if (not_Support_layout.count(inputs[0]->shape().layout()))
    {
        return false;
    }
    if (not_Support_layout.count(outputs[0]->shape().layout()))
    {
        return false;
    }
    return true;
}

static bool check_quantization_support(NodePtr n)
{
    auto input = n->inputs()[0];
    auto params = get_real_node(n)->params();
    if (params.count("scale_value"))
    {
        auto scales = params["scale_value"];
        std::vector<int> shift = params["shift_value"];
        for (uint32_t i = 0; i < scales.arr.size(); i++)
        {
            if (((int)scales[0] != (int)scales[i]) || (shift[0] != shift[i]))
            {
                return false;
            }
        }
    }
    else
    {
        for (auto output : n->outputs())
        {
            if ((input->quantization().offset() != output->quantization().offset()) || (input->quantization().scale() != output->quantization().scale()))
                return false;
        }
    }
    return true;
}

// Check support axis
static bool check_axis_support(NodePtr n)
{
    auto &params = get_real_node(n)->params();
    int axis = params["axis"];
    auto inputs = n->inputs();
    auto in_shape = inputs[0]->shape();
    int in_dim_size = 1;
    if (axis < 0)
        axis = in_shape.dim() + axis;
    axis = axis_remap(in_shape.layout(), axis);
    for (auto dim = in_shape.begin(); (dim != in_shape.end()) && ((dim - in_shape.begin()) < axis); dim++)
    {
        in_dim_size *= (*dim);
    }
    return (in_dim_size == 1);
}

static bool check_bytes_align(NodePtr n)
{
    auto &params = get_real_node(n)->params();
    int axis = params["axis"];

    if (axis < 0)
    {
        axis = n->outputs()[0]->shape().dim() + axis;
    }

    uint32_t relative_addr = 0;

    for (uint32_t i = 0; i < n->outputs().size(); ++i)
    {
        auto t = n->outputs()[i];
        uint32_t offset = 1;
        for (int i = axis; i < t->shape().dim(); ++i)
        {
            offset = offset * t->shape()[i];
        }
        offset = offset * t->dtype_size(t->type());

        if ((relative_addr % t->memory_offset()->align) != 0)
        {
            // INFO() << "relative addr: " << relative_addr << " align " << t->memory_offset()->align;
            return false;
        }
        relative_addr += offset;
    }

    return true;
}

void remove_split(GraphPtr g)
{
    auto creteria = [](NodePtr n) {
        if (get_real_op_type(n) != OpType_Split)
        {
            return false;
        }

        if (n->is_dynamic())
        {
            return false;
        }

        static std::set<OpType> nonsupport_output_type =
            {
                OpType_InternalConcatenated,
            };
        for (auto output : n->outputs())
        {
            for (auto consumer : output->consumers())
            {
                if (consumer->type() == OpType_LayoutConvertor && is_a_remapping_convertor(consumer, false))
                {
                    for (auto cc : consumer->children())
                    {
                        if (nonsupport_output_type.count(cc->type()))
                        {
                            return false;
                        }
                    }
                }
                if (nonsupport_output_type.count(consumer->type()))
                {
                    return false;
                }
            }
        }

        if (!check_layout_support(n))
        {
            return false;
        }

        if (!check_axis_support(n))
        {
            return false;
        }
        if (!check_quantization_support(n))
        {
            return false;
        }
        if (!check_bytes_align(n))
        {
            return false;
        }

        return true;
    };

    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!creteria(n))
        {
            continue;
        }

        std::string name;
        auto sgn = SubGraphNode::from_node(n);
        if (sgn)
        {
            name = sgn->subgraph()->nodes()[0]->name();
        }
        else
        {
            name = n->name();
        }

        auto split = Node::create(name, OpType_InternalSpliter);
        split->params() = std::move(get_real_node(n)->params());
        split->constants() = std::move(get_real_node(n)->constants());
        split->attrs() = std::move(get_real_node(n)->attrs());
        ADD_LOCATION_BY_NODE(split, n);
        g->replace_node_by(n, split);
    }
}

} // namespace aipubt
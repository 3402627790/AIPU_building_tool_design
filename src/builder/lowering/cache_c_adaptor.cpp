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
 * @Date: 2025-09-08 09:51:58
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-09-08 09:51:58
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
static int get_ds_alignment(TensorPtr t)
{
    auto dtype = t->type();
    if (Tensor::dtype_size(dtype) == 2)
    {
        return 16;
    }
    else if (Tensor::dtype_size(dtype) == 1)
    {
        return 32;
    }
    return 8;
}

static int criteria(NodePtr n)
{
    if (get_target()->isainfo().type < ISA_ZYv6)
    {
        return 1;
    }

    if (n->type() != OpType_CacheUpdate)
    {
        return 1;
    }

    int axis = 2;
    if (n->params().count("axis"))
    {
        axis = n->params()["axis"];
    }
    if (axis != 3)
    {
        return 1;
    }

    auto inp = n->inputs()[2];
    if (inp->shape().dim() != 4)
    {
        return 1;
    }

    int alignment = get_ds_alignment(inp);
    if (inp->shape().c_total() % alignment == 0)
    {
        return 1;
    }

    return alignment;
}

void cache_c_adaptor(GraphPtr g)
{
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        int alignment = criteria(n);
        if (alignment < 2)
        {
            continue;
        }

        auto inp = n->inputs()[2];
        auto out = n->outputs()[0];
        auto ndim = inp->shape().dim();

        // create pad node
        std::vector<std::pair<uint32_t, uint32_t>> pads(ndim, {0, 0});
        pads.back().second = (alignment - inp->shape()[3] % alignment) % alignment;
        int32_t constant_value = 0 - inp->quantization().offset();
        auto pad_out = ops::pad(inp, pads, ops::PadMode::Constant, constant_value);
        auto pad = pad_out->node();
        if (enable_ds)
        {
            std::vector<std::vector<Expr>> ds_pads(ndim, {0, 0});
            ds_pads.back()[1] = (alignment - inp->shape().symbol()[3] % alignment) % alignment;
            pad->params()["ds_pads"] = ds_pads;
            pad->attrs()[Constants::c_align] = alignment;
        }
        pad->infer_shape(true);
        g->add_node(pad);
        ADD_LOCATION_BY_NODE(pad, n);

        // update node
        n->add_input(pad_out, n->remove_input(inp));
        std::vector<int> shape_list = out->shape();
        shape_list[3] = (shape_list[3] + alignment - 1) / alignment * alignment;
        out->shape() = TensorShape(shape_list, out->shape().layout());
        if (enable_ds)
        {
            auto shapes = n->get_attr<std::vector<SymbolShape>>(Constants::output_symbol_shape);
            std::vector<Expr> shape = shapes[0];
            shape[3] = (shape[3] + alignment - 1).floordiv(alignment) * alignment;
            shapes[0] = shape;
            n->set_attr(Constants::output_symbol_shape, shapes);
        }
        n->infer_shape(true);
    }
}
} // namespace aipubt

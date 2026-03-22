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
 * @Date: 2025-02-24 16:16:18
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2025-02-24 16:16:18
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
    if (get_target()->isainfo().type < ISA_ZYv5)
    {
        return 1;
    }

    if (n->type() != OpType_Softmax)
    {
        return 1;
    }

    if (!n->is_dynamic())
    {
        return 1;
    }

    auto inp = n->inputs()[0];
    auto out = n->outputs()[0];

    if ((inp->type() != Tensor::FP16 || out->type() != Tensor::FP16) &&
        (inp->type() != Tensor::BFP16 || out->type() != Tensor::BFP16) &&
        (inp->type() != Tensor::FP8E4M3FN || out->type() != Tensor::FP8E4M3FN) &&
        (inp->type() != Tensor::FP8E5M2 || out->type() != Tensor::FP8E5M2))
    {
        return 1;
    }

    if (inp->shape().dim() != 4)
    {
        return 1;
    }

    int alignment = get_ds_alignment(inp);

    if (inp->shape().symbol().c()->is_constant() &&
        (inp->shape().c() % alignment == 0 || inp->shape().c() >= 4096))
    {
        return 1;
    }

    return alignment;
}

void hwa_softmax_c_adaptor(GraphPtr g)
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

        auto inp = n->inputs()[0];
        auto out = n->outputs()[0];
        auto ndim = inp->shape().dim();

        // create pad node
        std::vector<std::pair<uint32_t, uint32_t>> pads(ndim, {0, 0});
        pads.back().second = (alignment - inp->shape()[-1] % alignment) % alignment;
        int32_t constant_value = -1024; // -inf
        auto pad_out = ops::pad(inp, pads, ops::PadMode::Constant, constant_value);
        auto pad = pad_out->node();
        if (enable_ds)
        {
            std::vector<std::vector<Expr>> ds_pads(ndim, {0, 0});
            ds_pads.back()[1] = (alignment - inp->shape().symbol()[-1] % alignment) % alignment;
            pad->params()["ds_pads"] = ds_pads;
            pad->attrs()[Constants::c_align] = alignment;
        }
        pad->infer_shape(true);
        g->add_node(pad);
        ADD_LOCATION_BY_NODE(pad, n);

        // update node
        n->add_input(pad_out, n->remove_input(inp));
        auto new_shape = out->shape();
        new_shape[-1] = (out->shape()[-1] + alignment - 1) / alignment * alignment;
        auto new_name = g->get_valid_tensor_name(out->name());
        TensorPtr new_out = std::make_shared<Tensor>(new_name, new_shape, out->type());
        new_out->quantization() = out->quantization();
        n->add_output(new_out, n->remove_output(out));
        n->infer_shape(true);

        // create crop node
        std::vector<std::pair<int32_t, int32_t>> crops;
        for (int i = 0; i < ndim; ++i)
        {
            crops.push_back({0, out->shape()[i]});
        }
        auto crop_out = ops::crop(new_out, crops);
        auto crop = crop_out->node();
        crop->add_output(out, crop->remove_output(crop_out));
        if (enable_ds)
        {
            std::vector<std::vector<Expr>> ds_crops;
            for (int i = 0; i < ndim; ++i)
            {
                ds_crops.push_back({0, out->shape().symbol()[i]});
            }
            crop->params()["ds_crops"] = ds_crops;
            crop->attrs()[Constants::c_align] = alignment;
        }
        crop->infer_shape(true);
        g->add_node(crop);
        ADD_LOCATION_BY_NODE(crop, n);
    }
}
} // namespace aipubt

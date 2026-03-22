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
 * @Date: 2024-05-03 13:59:06
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-05-03 13:59:06
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{

static bool criteria(NodePtr n, bool &pad_in, bool &crop_out)
{
    if (n->type() != OpType_FullyConnected)
    {
        return false;
    }

    if (n->inputs()[0]->type() == Tensor::FP16 ||
        n->inputs()[0]->type() == Tensor::BFP16 ||
        n->outputs()[0]->type() == Tensor::FP16 ||
        n->outputs()[0]->type() == Tensor::BFP16)
    {
        return false;
    }

    if (!n->inputs()[0]->shape().symbol().n()->is_constant())
    {
        pad_in = true;
    }

    if (!n->outputs()[0]->shape().symbol().n()->is_constant())
    {
        crop_out = true;
    }

    return pad_in || crop_out;
}

static NodePtr pad_input(GraphPtr g, NodePtr n)
{
    auto inp = n->inputs()[0];
    auto in_shape = inp->shape();
    std::vector<std::pair<uint32_t, uint32_t>> pads(in_shape.dim(), {0, 0});
    int32_t constant_value = 0 - inp->quantization().offset();

    auto consumers = inp->consumers();
    auto pad_out = ops::pad(inp, pads, ops::PadMode::Constant, constant_value);
    auto pad_node = pad_out->node();
    for (auto c : consumers)
    {
        if (c->type() == OpType_FullyConnected)
        {
            c->add_input(pad_out, c->remove_input(inp));
        }
    }

    std::vector<std::vector<Expr>> ds_pads(in_shape.dim(), {0, 0});
    ds_pads[0][1] = in_shape.n() - in_shape.symbol().n();
    pad_node->params()["ds_pads"] = ds_pads;
    pad_node->infer_shape(true);
    g->add_node(pad_node);

    return pad_node;
}

static NodePtr crop_output(GraphPtr g, NodePtr n)
{
    auto out = n->outputs()[0];
    auto consumers = out->consumers();
    std::vector<int> indexs;
    for (auto c : consumers)
    {
        indexs.push_back(c->remove_input(out));
    }
    auto crop_in = std::make_shared<Tensor>(*out);
    crop_in->set_io_tensor(false);
    crop_in->set_name(g->get_valid_tensor_name(out->name()));
    n->add_output(crop_in, n->remove_output(out));
    n->infer_shape(true);

    auto out_shape = out->shape();
    std::vector<std::pair<int32_t, int32_t>> crops;
    for (int i = 0; i < out_shape.dim(); ++i)
    {
        crops.push_back({0, out_shape[i]});
    }

    auto crop_out = ops::crop(crop_in, crops);
    auto crop_node = crop_out->node();
    crop_node->add_output(out, crop_node->remove_output(crop_out));
    for (uint32_t i = 0; i < consumers.size(); ++i)
    {
        consumers[i]->add_input(out, indexs[i]);
    }

    std::vector<std::vector<Expr>> ds_crops;
    for (int i = 0; i < out_shape.dim(); ++i)
    {
        ds_crops.push_back({0, out_shape.symbol()[i]});
    }
    crop_node->params()["ds_crops"] = ds_crops;
    crop_node->infer_shape(true);
    g->add_node(crop_node);

    return crop_node;
}

void eliminate_dynamic_fc(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        bool pad_in = false, crop_out = false;

        if (!criteria(n, pad_in, crop_out))
        {
            continue;
        }

        if (pad_in)
        {
            auto pad_node = pad_input(g, n);
            ADD_LOCATION_BY_NODE(pad_node, n);
        }

        if (crop_out)
        {
            auto crop_node = crop_output(g, n);
            ADD_LOCATION_BY_NODE(crop_node, n);
        }
    }
}

} // namespace aipubt

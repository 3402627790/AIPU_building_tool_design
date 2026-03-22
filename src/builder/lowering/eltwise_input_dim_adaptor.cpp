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
 * @Date: 2024-04-30 10:06:39
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-04-30 10:06:39
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{

static bool criteria(NodePtr n)
{
    if (n->type() != OpType_Eltwise)
    {
        return false;
    }
    if (n->inputs().size() != 2)
    {
        return false;
    }
    auto &in0_shape = n->inputs()[0]->shape();
    auto &in1_shape = n->inputs()[1]->shape();
    if (in0_shape.dim() == in1_shape.dim() && in0_shape.dim() < 4)
    {
        return true;
    }
    return false;
}

static NodePtr get_input_reshape_node(GraphPtr g, NodePtr n, TensorPtr inp)
{
    std::vector<int> shape = inp->shape();
    std::vector<Expr> ds_shape = inp->shape().symbol();
    while (shape.size() < 4)
    {
        shape.insert(shape.begin(), 1);
        ds_shape.insert(ds_shape.begin(), 1);
    }
    auto reshape_out = ops::reshape(inp, TensorShape(shape, DataLayout_NHWC));
    auto reshape_node = reshape_out->node();
    reshape_node->params()["ds_shape"] = ds_shape;
    reshape_node->infer_shape(true);
    n->add_input(reshape_out, n->remove_input(inp));
    g->add_node(reshape_node);
    return reshape_node;
}

static NodePtr get_output_reshape_node(GraphPtr g, NodePtr n)
{
    auto out = n->outputs()[0];
    auto consumers = out->consumers();
    std::vector<int> indexs;
    for (auto c : consumers)
    {
        indexs.push_back(c->remove_input(out));
    }
    auto new_out = std::make_shared<Tensor>(*out);
    new_out->set_io_tensor(false);
    new_out->set_name(g->get_valid_tensor_name(out->name()));
    new_out->shape() = n->inputs()[0]->shape();
    n->add_output(new_out, n->remove_output(out));
    n->infer_shape(true);

    auto reshape_out = ops::reshape(new_out, out->shape());
    auto reshape_node = reshape_out->node();
    reshape_node->add_output(out, reshape_node->remove_output(reshape_out));
    std::vector<Expr> ds_shape = out->shape().symbol();
    reshape_node->params()["ds_shape"] = ds_shape;
    reshape_node->infer_shape(true);
    for (uint32_t i = 0; i < consumers.size(); ++i)
    {
        consumers[i]->add_input(out, indexs[i]);
    }
    g->add_node(reshape_node);
    return reshape_node;
}

void eltwise_input_dim_adaptor(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }

        auto reshape0_node = get_input_reshape_node(g, n, n->inputs()[0]);
        ADD_LOCATION_BY_NODE(reshape0_node, n);

        auto reshape1_node = get_input_reshape_node(g, n, n->inputs()[1]);
        ADD_LOCATION_BY_NODE(reshape1_node, n);

        auto reshape_out_node = get_output_reshape_node(g, n);
        ADD_LOCATION_BY_NODE(reshape_out_node, n);
    }
}

} // namespace aipubt

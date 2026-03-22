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
 * @Date: 2024-11-18 10:44:26
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-11-18 10:44:26
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
using check_fn = std::function<bool(NodePtr)>;
using lower_fn = std::function<void(NodePtr, NodePtr)>;

static bool check_eltwise(NodePtr n)
{
    if (n->type() != OpType_Eltwise)
    {
        return false;
    }
    if (n->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    return true;
}

static void lower_eltwise(NodePtr elt, NodePtr n)
{
    auto elt_in0 = elt->inputs()[0];
    auto elt_in1 = elt->inputs()[1];
    n->clear_inputs();
    n->add_input(elt_in0);
    std::vector<int> shape_list = {
        elt_in0->shape().n() * elt_in0->shape().h(),
        1,
        elt_in0->shape().w(),
        elt_in0->shape().c(),
    };
    auto shape = TensorShape(shape_list, elt_in0->shape().layout());
    n->params()["shape"] = shape;
    n->outputs()[0]->shape() = shape;
    std::vector<Expr> ds_shape = {
        elt_in0->shape().symbol().n() * elt_in0->shape().symbol().h(),
        1,
        elt_in0->shape().symbol().w(),
        elt_in0->shape().symbol().c(),
    };
    n->params()["ds_shape"] = ds_shape;
    n->infer_shape(true);
    MODIFY_NODE(n);

    auto reshape1_out = ops::reshape(elt_in1, shape);
    auto reshape1_node = reshape1_out->node();
    reshape1_node->params()["ds_shape"] = ds_shape;
    reshape1_node->infer_shape(true);
    elt->graph()->add_node(reshape1_node);
    ADD_LOCATION_BY_NODE(reshape1_node, elt);

    auto reshape0_out = n->outputs()[0];
    auto consumers = reshape0_out->consumers();
    elt->clear_inputs();
    elt->add_input(reshape0_out);
    elt->add_input(reshape1_out);
    auto elt_out = elt->outputs()[0];
    elt_out->shape() = shape;
    elt->infer_shape(true);
    MODIFY_NODE(elt);

    for (auto c : consumers)
    {
        c->add_input(elt_out, c->remove_input(reshape0_out));
        c->infer_shape(true);
        MODIFY_NODE(c);
    }
}

static bool check_pad(NodePtr n)
{
    if (n->type() != OpType_Pad)
    {
        return false;
    }
    if (n->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (!n->attrs().count(Constants::w_align) &&
        !n->attrs().count(Constants::c_align))
    {
        return false;
    }
    return true;
}

static void lower_pad(NodePtr pad, NodePtr n)
{
    auto pad_in = pad->inputs()[0];
    n->clear_inputs();
    n->add_input(pad_in);
    std::vector<int> shape_list = {
        pad_in->shape().n() * pad_in->shape().h(),
        1,
        pad_in->shape().w(),
        pad_in->shape().c(),
    };
    auto shape = TensorShape(shape_list, pad_in->shape().layout());
    n->params()["shape"] = shape;
    auto reshape_out = n->outputs()[0];
    reshape_out->shape() = shape;
    std::vector<Expr> ds_shape = {
        pad_in->shape().symbol().n() * pad_in->shape().symbol().h(),
        1,
        pad_in->shape().symbol().w(),
        pad_in->shape().symbol().c(),
    };
    n->params()["ds_shape"] = ds_shape;
    n->infer_shape(true);
    MODIFY_NODE(n);

    auto consumers = reshape_out->consumers();
    pad->add_input(reshape_out, pad->remove_input(pad_in));
    auto pad_out = pad->outputs()[0];
    shape_list = {
        pad_out->shape().n() * pad_out->shape().h(),
        1,
        pad_out->shape().w(),
        pad_out->shape().c(),
    };
    pad_out->shape() = TensorShape(shape_list, pad_out->shape().layout());
    pad->infer_shape(true);
    MODIFY_NODE(pad);

    for (auto c : consumers)
    {
        c->add_input(pad_out, c->remove_input(reshape_out));
        c->infer_shape(true);
        MODIFY_NODE(c);
    }
}

static std::map<OpType, std::pair<check_fn, lower_fn>> supported_ops = {
    {OpType_Eltwise, {check_eltwise, lower_eltwise}},
    {OpType_Pad, {check_pad, lower_pad}},
};

static bool criteria(NodePtr n)
{
    if (n->type() != OpType_Reshape)
    {
        return false;
    }
    if (n->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    auto in_shape = n->inputs()[0]->shape();
    auto out_shape = n->outputs()[0]->shape();
    if (in_shape.dim() != 4 || out_shape.dim() != 4)
    {
        return false;
    }
    if (in_shape[0] * in_shape[1] != out_shape[0] ||
        out_shape[1] != 1 ||
        in_shape[2] != out_shape[2] ||
        in_shape[3] != out_shape[3])
    {
        return false;
    }
    auto p = n->parents()[0];
    if (p->children().size() > 1)
    {
        return false;
    }
    if (!supported_ops.count(p->type()))
    {
        return false;
    }
    if (!supported_ops[p->type()].first(p))
    {
        return false;
    }
    return true;
}

void lift_reshape(GraphPtr g)
{
    while (true)
    {
        bool is_lifted = false;
        auto nodes = g->nodes();
        for (auto n : nodes)
        {
            if (!criteria(n))
            {
                continue;
            }

            auto p = n->parents()[0];
            supported_ops[p->type()].second(p, n);

            is_lifted = true;
        }
        if (!is_lifted)
        {
            break;
        }
    }
}
} // namespace aipubt

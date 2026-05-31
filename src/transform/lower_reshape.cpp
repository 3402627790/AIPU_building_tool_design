//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-03-30 13:38:32
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-07-05 10:40:45
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"
#include <map>
#include <vector>

namespace aipubt
{
static inline bool same_channel(NodePtr n0, NodePtr n1)
{
    // n0 output is n1 inputs
    if (n0->inputs()[0]->shape().dim() == 0)
    {
        return false;
    }
    if (n0->outputs()[0] == n1->inputs()[0] &&
        n0->inputs()[0]->shape()[-1] == n0->outputs()[0]->shape()[-1] &&
        n0->outputs()[0]->shape()[-1] == n1->outputs()[0]->shape()[-1] &&
        n1->outputs()[0]->shape()[-1] == n0->inputs()[0]->shape()[-1])
    {
        return true;
    }
    return false;
}

static bool checking_bn(NodePtr node, NodePtr child, bool change_param)
{
    if (child->type() == OpType_BatchNorm)
    {
        auto reshape_input_dim = node->inputs()[0]->shape().dim();
        if (reshape_input_dim < 2 || reshape_input_dim > 6)
        {
            return false;
        }
        int axis = child->params()["axis"];
        if (axis != child->outputs()[0]->shape().dim() - 1)
        {
            return false;
        }
        // pre-channel ax+b, same shape[-1]
        if (!same_channel(node, child))
        {
            return false;
        }

        // =============================================================================
        // change param
        if (change_param)
        {
            child->params()["axis"] = node->inputs()[0]->shape().dim() - 1;
        }
    }
    return true;
}

static bool checking_ln(NodePtr node, NodePtr child, bool change_param)
{
    if (child->type() == OpType_LayerNorm)
    {
        std::vector<int> axis = child->params()["axis"];
        if (axis.size() != 1 || axis[0] != child->outputs()[0]->shape().dim() - 1)
        {
            return false;
        }
        TensorShape tmp = node->outputs()[0]->shape();
        if (node->inputs()[0]->shape().dim() > node->outputs()[0]->shape().dim())
        {
            tmp = node->inputs()[0]->shape();
        }
        if (tmp.dim() > 4 || tmp.dim() < 2)
        {
            // tpc ln requirement
            return false;
        }
        // pre-channel ax+b, same shape[-1]
        if (!same_channel(node, child))
        {
            return false;
        }

        // =============================================================================
        // change param
        if (change_param)
        {
            child->params()["axis"] = {node->inputs()[0]->shape().dim() - 1};
        }
    }
    return true;
}

static bool checking_act(NodePtr node, NodePtr child, bool change_param)
{
    if (child->type() == OpType_Activation)
    {
        std::string method = child->params()["method"];
        if (method == "CRELU")
        {
            return false;
        }
        if (method == Constants::act_prelu &&
            (!same_channel(node, child) || child->constants()["negative_slope"]->shape().dim() > 1))
        {
            // pre-channel negative slope, same shape[-1]
            return false;
        }
    }
    return true;
}

static bool checking_softmax(NodePtr node, NodePtr child, bool change_param)
{
    if (child->type() == OpType_Softmax)
    {
        int axis = child->params()["axis"];
        if (axis != child->outputs()[0]->shape().dim() - 1)
        {
            return false;
        }
        // pre-channel ax+b, same shape[-1]
        if (!same_channel(node, child))
        {
            return false;
        }

        // =============================================================================
        // change param
        if (change_param)
        {
            child->params()["axis"] = node->inputs()[0]->shape().dim() - 1;
        }
    }
    return true;
}

static bool checking_dequantize(NodePtr node, NodePtr child, bool change_param)
{
    if (child->type() == OpType_DeQuantize)
    {
        if (!same_channel(node, child))
        {
            // pre-channel
            return false;
        }
    }
    return true;
}

static bool checking_quantize(NodePtr node, NodePtr child, bool change_param)
{
    if (child->type() == OpType_Quantize)
    {
        if (!same_channel(node, child))
        {
            // pre-channel
            return false;
        }
    }
    return true;
}
static bool checking_cast(NodePtr node, NodePtr child, bool change_param)
{
    if (child->type() == OpType_Cast)
    {
        if (!same_channel(node, child))
        {
            // pre-channel
            return false;
        }
    }
    return true;
}

static std::map<OpType, std::function<bool(NodePtr, NodePtr, bool)>> lower_supported_op = {
    {OpType_BatchNorm, checking_bn},
    {OpType_LayerNorm, checking_ln},
    {OpType_Activation, checking_act},
    {OpType_Erf, checking_act},
    {OpType_Softmax, checking_softmax},
    {OpType_DeQuantize, checking_dequantize},
    {OpType_Quantize, checking_quantize},
    {OpType_Cast, checking_cast},
};

static inline bool lower_creteria(NodePtr node)
{
    if (node->type() != OpType_Reshape)
    {
        return false;
    }
    if (node->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (node->children().size() != 1 || !lower_supported_op.count(node->children()[0]->type()))
    {
        return false;
    }
    if (node->children()[0]->outputs().size() != 1)
    {
        return false;
    }
    auto child = node->children()[0];
    auto _f = lower_supported_op[child->type()];
    if (!_f(node, child, true))
    {
        return false;
    }

    return true;
}

bool lower_reshape(GraphPtr g)
{
    auto m_nodes = g->nodes();
    bool is_changed = false;
    for (auto &&n : m_nodes)
    {
        if (!lower_creteria(n))
        {
            continue;
        }

        auto child = n->children()[0];
        auto inp = n->inputs()[0];
        auto inter = n->outputs()[0];
        auto out = child->outputs()[0];

        n->clear_inputs();
        n->clear_outputs();
        child->clear_inputs();
        child->clear_outputs();

        inter->shape() = inp->shape();
        inter->quantization() = out->quantization();
        inter->set_type(out->type());

        child->add_input(inp);
        child->add_output(inter);
        n->add_input(inter);
        n->add_output(out);

        is_changed = true;
    }
    return is_changed;
}

bool duplicate_reshape(GraphPtr g)
{
    auto m_nodes = g->nodes();
    bool is_changed = false;
    for (auto &&n : m_nodes)
    {
        if (n->type() != OpType_Reshape || n->children().size() <= 1)
        {
            continue;
        }
        // it will make death loop with CSE_subnode
        // we only duplicate_reshape when at least one child can lower reshape
        bool having_one_at_least = false;
        for (auto &c : n->children())
        {
            if (lower_supported_op.count(c->type()) &&
                lower_supported_op[c->type()](n, c, false))
            {
                having_one_at_least = true;
                break;
            }
        }
        if (!having_one_at_least)
        {
            continue;
        }

        auto n_children = n->children();
        // Only create new reshape node and output tensor for all the children except first one, and keep the original
        // reshape node(so output tensor is also kept, and no need to update io tensor if it happens to be io tensor).
        for (size_t i = 1; i < n_children.size(); i++)
        {
            auto reshape_node_name = g->get_valid_node_name(n->name() + "_reshape" + std::to_string(i));
            NodePtr reshape = Node::create(reshape_node_name, OpType_Reshape);

            ADD_LOCATION_BY_NODE(reshape, n);
            g->add_node(reshape);
            for (auto &&param : n->params())
            {
                reshape->params()[param.first] = param.second;
            }
            reshape->add_input(n->inputs());

            TensorPtr _out = std::make_shared<Tensor>(reshape->name(), n->outputs()[0]->shape(), n->outputs()[0]->type());
            _out->quantization() = n->outputs()[0]->quantization();
            reshape->add_output(_out);
            auto idx = n_children[i]->remove_input(n->outputs()[0]);
            n_children[i]->add_input(_out, idx);
        }

        is_changed = true;
    }
    return is_changed;
}

bool lower_reshape_combine(GraphPtr g)
{
    bool is_changed = false;
    bool is_continue = false;
    while (1)
    {
        is_continue = false;
        for (auto _f : {lower_reshape, duplicate_reshape, lower_reshape_reshape_eltwise, lower_reshape_eltwise, lower_reshape_reshape_add, lower_reshape_add, duplicate_reshape})
        {
            is_continue |= _f(g);
            is_changed |= is_continue;
            is_changed |= fuse_reshape(g);
            is_changed |= eliminate_reshape(g);
        }
        if (!is_continue)
        {
            break;
        }
    }
    is_changed |= lower_reshape(g);
    is_changed |= CSE_subnode(g);

    return is_changed;
}
}; // namespace aipubt

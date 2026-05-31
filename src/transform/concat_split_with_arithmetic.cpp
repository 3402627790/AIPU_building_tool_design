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
 * @Date: 2022-05-09 09:52:47
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-05-12 11:14:03
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"

namespace aipubt
{
static std::vector<OpType> _unary_ops = {
    OpType_Abs,
    OpType_Acos,
    OpType_Acosh,
    OpType_Asin,
    OpType_Asinh,
    OpType_Tan,
    OpType_Ceil,
    OpType_Cosine,
    OpType_Cosh,
    OpType_Exp,
    OpType_Floor,
    OpType_Log,
    OpType_Negative,
    OpType_Reciprocal,
    OpType_Rsqrt,
    OpType_Sqrt,
    OpType_Square,
    OpType_Round,
    OpType_Sine,
    OpType_Sinh,
    OpType_Elu,
    OpType_RELU,
    OpType_RELU1,
    OpType_RELU6,
    OpType_Silu,

    OpType_Activation,
    // OpType_Selu,
    // OpType_Sigmoid,
    // OpType_Tanh,
    // OpType_HardSwish,
    // OpType_HardSigmoid,
};

static inline bool criteria(NodePtr n)
{
    auto _check_node = [](NodePtrList nodes) -> bool {
        std::map<std::string, int32_t> _optype_map;
        for (auto &n : nodes)
        {
            auto n_type = n->type();
            auto key = optype_to_string(n_type);
            if (contains(_unary_ops, n_type))
            {
                if (n_type == OpType_Activation)
                {
                    key = key + n->params()["method"];
                    // TODO: checking prelu negative_slope data is the same with other branch
                }

                int32_t v = _optype_map.count(key) ? _optype_map[key] : 0;
                _optype_map.insert(std::pair<std::string, int32_t>(key, v + 1));
            }
            else
            {
                return false;
            }
        }
        return _optype_map.size() == 1 ? true : false;
    };

    auto _check_tensor = [](const std::vector<TensorPtr> &ts) -> bool {
        std::vector<Tensor::Dtype> _FLOAT_DTYPE = {Tensor::BFP16, Tensor::FP16, Tensor::FP32, Tensor::FP64};
        for (auto tensor : ts)
        {
            // now only support Float CompassIR
            if (!contains(_FLOAT_DTYPE, tensor->type()))
            {
                return false;
            }
            if (tensor->is_io_tensor())
            {
                return false;
            }
        }
        return true;
    };

    if (n->type() == OpType_Concat)
    {
        auto ret = _check_tensor(n->inputs());
        if (!ret)
        {
            return ret;
        }
        auto parents = n->parents();
        for (auto pn : n->parents())
        {
            if (pn->children().size() > 1)
            {
                return false;
            }
        }
        return _check_node(parents);
    }
    else if (n->type() == OpType_Split)
    {
        auto ret = _check_tensor(n->outputs());
        if (!ret)
        {
            return ret;
        }
        auto children = n->children();
        for (auto cn : n->children())
        {
            if (cn->parents().size() > 1)
            {
                return false;
            }
        }
        return _check_node(children);
    }
    return false;
}

bool concat_split_with_arithmetic(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        if (n->type() == OpType_Concat)
        {
            auto axis = n->params()["axis"];
            auto unary_node = std::move(n->parents()[0]);

            std::vector<TensorPtr> new_its;
            for (auto &it : n->inputs())
            {
                auto it_n = it->node();
                new_its.push_back(it_n->inputs()[0]);
            }
            auto concat_out = ops::concat(new_its, axis);

            unary_node->clear_inputs();
            unary_node->clear_outputs();
            unary_node->add_input(concat_out);
            unary_node->add_output(n->outputs()[0]);

            auto new_concat = concat_out->node();
            g->add_node(new_concat);
            ADD_LOCATION_BY_NODE(new_concat, n);

            for (auto pn : n->parents())
            {
                pn->clear_outputs();
                pn->clear_inputs();
                g->eliminate_node(pn);
            }
            n->clear_outputs();
            n->clear_inputs();
            is_changed |= g->eliminate_node(n);
        }
        else if (n->type() == OpType_Split)
        {
            auto axis = n->params()["axis"];
            auto splits = n->params()["splits"];
            std::vector<TensorPtr> new_ots;
            for (auto cn : n->children())
            {
                new_ots.push_back(cn->outputs()[0]);
            }

            auto inp_t = n->inputs()[0];
            auto unary_node = std::move(n->children()[0]);
            auto unary_ot = std::make_shared<Tensor>(g->get_valid_tensor_name(unary_node->name()), inp_t->shape(), inp_t->type());
            unary_node->clear_inputs();
            unary_node->clear_outputs();
            unary_node->add_input(inp_t);
            unary_node->add_output(unary_ot);

            auto split_ot = ops::split(unary_ot, new_ots.size(), axis);
            auto split_node = split_ot[0]->node();
            split_node->clear_outputs();
            split_node->add_output(new_ots);
            split_node->params()["splits"] = splits;
            g->add_node(split_node);
            ADD_LOCATION_BY_NODE(split_node, n);

            for (auto cn : n->children())
            {
                cn->clear_inputs();
                cn->clear_outputs();
                g->eliminate_node(cn);
            }
            n->clear_inputs();
            n->clear_outputs();
            is_changed |= g->eliminate_node(n);
        }
    }
    return is_changed;
}

}; // namespace aipubt
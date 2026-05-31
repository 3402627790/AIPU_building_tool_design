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
 * @Author: Xiaolong Zhang
 * @Date: 2023-12-06 13:38:32
 * @Last Modified by: Xiaolong Zhang
 * @Last Modified time: 2023-12-07 10:40:45
 *       |                                              |
 *    reshape/activation/cast                       crop(channel)
 *       |                          ====>               |
 *      crop(channel)                               reshape/act/cast
 *       |                                              |
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"
#include <map>
#include <vector>

namespace aipubt
{

static bool checking_act(NodePtr act_node)
{
    if (act_node->type() != OpType_Activation)
    {
        return false;
    }

    std::string method = act_node->params()["method"];
    if (method == "CRELU")
    {
        return false;
    }

    if (method == Constants::act_prelu &&
        act_node->constants()["negative_slope"]->shape().dim() > 1)
    {
        // pre-channel negative slope, same shape[-1]
        return false;
    }

    if (!(act_node->parents()[0]->type() == OpType_FullyConnected ||
          act_node->parents()[0]->type() == OpType_Cast))
    {
        return false;
    }

    return true;
}

static bool checking_cast(NodePtr cast_node)
{
    if (cast_node->type() != OpType_Cast)
    {
        return false;
    }

    std::vector<OpType> pattern = {OpType_Activation, OpType_Cast, OpType_FullyConnected};
    auto cur_node = cast_node;
    for (auto &t : pattern)
    {
        if (cur_node->children().size() > 1)
        {
            return false;
        }

        if (cur_node->parents()[0]->type() == t)
        {
            cur_node = cur_node->parents()[0];
        }
    }

    return true;
}

static bool checking_reshape(NodePtr reshape)
{
    if (reshape->type() != OpType_Reshape)
    {
        return false;
    }

    auto inp = reshape->inputs()[0];
    auto outp = reshape->outputs()[0];
    if (inp->shape()[-1] != outp->shape()[-1])
    {
        return false;
    }

    return true;
}

static void lift_crop_node(GraphPtr g, NodePtr cast_node)
{
    auto crop = cast_node->children()[0];
    auto inp_tensor = cast_node->inputs()[0];
    auto inter_tensor = cast_node->outputs()[0];
    auto out_tensor = crop->outputs()[0];
    inter_tensor->shape() = out_tensor->shape();
    inter_tensor->set_type(inp_tensor->type());
    inter_tensor->quantization() = inp_tensor->quantization();
    crop->clear_inputs();
    crop->clear_outputs();
    crop->add_input(inp_tensor);
    crop->add_output(inter_tensor);
    MODIFY_NODE(crop);
    cast_node->clear_inputs();
    cast_node->clear_outputs();
    cast_node->add_input(inter_tensor);
    cast_node->add_output(out_tensor);
    MODIFY_NODE(crop);
}

static void lift_crop_reshape(GraphPtr g, NodePtr reshape)
{
    auto crop = reshape->children()[0];

    auto inp = reshape->inputs()[0];
    auto inter = reshape->outputs()[0];
    auto out = crop->outputs()[0];

    auto origin_crop_param = crop->params()["crops"];
    NodeParamValue m_crop;
    for (int i = 0; i < inp->shape().dim(); ++i)
    {
        if (i == inp->shape().dim() - 1)
        {
            m_crop.push_back(origin_crop_param[inter->shape().dim() - 1]);
        }
        else
        {
            m_crop.push_back({0, inp->shape()[i]});
        }
    }
    crop->params()["crops"] = m_crop;
    if (crop->params().count("ds_crops"))
    {
        std::vector<std::vector<tongue::Expr>> origin_ds_crops = crop->params()["ds_crops"];
        std::vector<std::vector<tongue::Expr>> ds_crops;
        for (int i = 0; i < inp->shape().dim(); ++i)
        {
            if (i == inp->shape().dim() - 1)
            {
                ds_crops.push_back(origin_ds_crops[inter->shape().dim() - 1]);
            }
            else
            {
                ds_crops.push_back({0, inp->shape().symbol()[i]});
            }
        }
        crop->params()["ds_crops"] = ds_crops;
    }

    auto s = inp->shape();
    s[-1] = out->shape()[-1];
    inter->shape() = s;

    crop->clear_inputs();
    crop->clear_outputs();
    crop->add_input(inp);
    crop->add_output(inter);
    crop->infer_shape(true);
    MODIFY_NODE(crop);
    reshape->clear_inputs();
    reshape->clear_outputs();
    reshape->add_input(inter);
    reshape->add_output(out);
    reshape->params()["shape"] = out->shape();
    if (reshape->params().count("ds_output_shape"))
    {
        reshape->params()["ds_output_shape"] =
            std::vector<std::vector<Expr>>{out->shape().symbol()};
    }
    if (reshape->params().count("ds_shape"))
    {
        reshape->params()["ds_shape"] = std::vector<Expr>{out->shape().symbol()};
    }
    return;
}

static std::map<OpType, std::function<bool(NodePtr)>> lift_crop_check = {
    {OpType_Activation, checking_act},
    {OpType_Cast, checking_cast},
    {OpType_Reshape, checking_reshape},
};

static std::map<OpType, std::function<void(GraphPtr, NodePtr)>> lift_executor = {
    {OpType_Activation, lift_crop_node},
    {OpType_Cast, lift_crop_node},
    {OpType_Reshape, lift_crop_reshape},
};

static inline bool lift_crop_creteria(NodePtr node)
{
    if (node->type() != OpType_Crop)
    {
        return false;
    }

    auto crop_parent = node->parents()[0];
    for (auto &inp : crop_parent->inputs())
    {
        if (inp->is_io_tensor())
        {
            return false;
        }
    }

    if (node->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    if (node->children().size() > 0 && node->children()[0]->type() == OpType_Convolution)
    {
        return false;
    }

    if (node->parents()[0]->children().size() != 1 ||
        !lift_crop_check.count(node->parents()[0]->type()))
    {
        return false;
    }

    int32_t axis = 0;
    int32_t change_axis_count = 0;
    auto crop_params = node->params()["crops"];
    for (int i = 0; i < node->inputs()[0]->shape().dim(); ++i)
    {
        int32_t dim_length = static_cast<int32_t>(crop_params[i][1]) - static_cast<int32_t>(crop_params[i][0]);
        int32_t s = node->inputs()[0]->shape()[i];
        if (s != dim_length)
        {
            change_axis_count++;
            axis = i;
        }
    }

    if (change_axis_count != 1) // only support one dim croped for now.
    {
        return false;
    }

    // only support 1-dim channel crop;
    int32_t parent_dim = node->inputs()[0]->shape().dim();
    if (axis != parent_dim - 1)
    {
        return false;
    }

    auto _f = lift_crop_check[node->parents()[0]->type()];
    if (!_f(node->parents()[0]))
    {
        return false;
    }

    return true;
}

bool lift_crop(GraphPtr g)
{
    bool is_changed = false;
    bool is_loop_search = false;
    while (1)
    {
        is_loop_search = false;
        for (auto &&n : g->nodes())
        {
            if (!lift_crop_creteria(n))
            {
                continue;
            }

            auto p = n->parents()[0];
            auto executor = lift_executor[p->type()];

            executor(g, p);

            is_changed = true;
            is_loop_search |= is_changed;
        }

        if (!is_loop_search)
        {
            break;
        }
    }
    return is_changed;
}
}; // namespace aipubt

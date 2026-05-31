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
 * @Author: Samuel Han
 * @Date: 2024-08-29 20:30:22
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-05 11:47:49
 */

#include "quantization_transform/passes/passes.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace qtlibn
{

inline static bool criteria(aipubt::TensorPtr t)
{
    auto cur_node = t->node();
    std::string trigger_mode_attr_name = "insert_cast_in_io_tensor";
    if (cur_node->attrs().count(trigger_mode_attr_name))
    {
        std::string trigger_mode = cur_node->get_attr<aipubt::NodeParamValue>(trigger_mode_attr_name);
        std::transform(trigger_mode.begin(), trigger_mode.end(), trigger_mode.begin(), ::toupper);
        // cur_node->set_attr(Constants::trigger_float_op, trigger_mode);
        if (trigger_mode == "DISABLE")
        {
            return false;
        }
    }

    return aipubt::Tensor::is_float(t->type()) && aipubt::Tensor::dtype_bits(t->type()) > 16;
}

inline static bool criteria_in(aipubt::NodePtr n)
{
    return n->type() == aipubt::OpType_Quantize ? false : true;
}
inline static bool criteria_out(aipubt::NodePtr n)
{
    return n->type() == aipubt::OpType_DeQuantize ? false : true;
}

void insert_cast_in_io_tensor(aipubt::GraphPtr g)
{
    auto _insert_cast_at_in_edge = [](aipubt::GraphPtr g, aipubt::TensorPtr t, aipubt::NodePtr cur_node) {
        auto cast_t = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(t->name()), t->shape(), t->type());
        auto cast_node = aipubt::Node::create(g->get_valid_node_name(cur_node->name()), aipubt::OpType_Cast);
        // cast_node->params()["to_dtype"] = setted_dtype;
        cast_node->params()["to_dtype"] = t->type();
        cast_node->params()["ignore_scale_zp"] = true;
        cast_node->params()["clip_mode"] = (std::string) "SATURATION";
        cast_node->add_input(t);
        cast_node->add_output(cast_t);
        cast_t->quantization() = t->quantization();
        auto idx = cur_node->remove_input(t);
        cur_node->add_input(cast_t, idx);
        cast_node->copy_attrs(cur_node);
        // cur_node->infer_shape();
        // cast_node->infer_shape();
        if (cur_node->params().count("ds_output_shape"))
        {
            cast_node->params()["ds_output_shape"] = t->node()->params()["ds_output_shape"];
        }
        g->add_node(cast_node);
    };
    auto _insert_cast_at_out_edge = [](aipubt::GraphPtr g, aipubt::TensorPtr t, aipubt::NodePtr cur_node) {
        auto cast_t = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(t->name()), t->shape(), t->type());
        auto cast_node = aipubt::Node::create(g->get_valid_node_name(cur_node->name()), aipubt::OpType_Cast);
        cast_t->set_io_tensor(false);
        cast_node->params()["to_dtype"] = t->type();
        cast_node->params()["ignore_scale_zp"] = true;
        cast_node->params()["clip_mode"] = (std::string) "SATURATION";
        cast_node->add_input(cast_t);
        if (t->consumers().size())
        {
            for (auto &consumer : t->consumers())
            {
                auto idx = consumer->remove_input(t);
                consumer->add_input(cast_t, idx);
            }
        }
        cast_node->add_output(t);
        cast_t->quantization() = t->quantization();
        auto removed_idx = cur_node->remove_output(t);
        cur_node->add_output(cast_t, removed_idx);
        cast_node->copy_attrs(cur_node);
        // cur_node->infer_shape();
        // cast_node->infer_shape();
        if (cur_node->params().count("ds_output_shape"))
        {
            cast_node->params()["ds_output_shape"] = cur_node->params()["ds_output_shape"];
        }
        g->add_node(cast_node);
    };

    auto g_intensors = g->get_input_tensors();
    auto g_outtensors = g->get_output_tensors();
    for (auto &in_t : g_intensors)
    {
        if (!criteria(in_t))
        {
            continue;
        }

        auto consumers = in_t->consumers();
        for (auto &cons : consumers)
        {
            if (criteria_in(cons))
            {
                _insert_cast_at_in_edge(g, in_t, cons);
            }
        }
    }

    for (auto &out_t : g_outtensors)
    {

        if (!criteria(out_t))
        {
            continue;
        }
        auto cur_node = out_t->node();
        if (criteria_out(cur_node))
        {
            _insert_cast_at_out_edge(g, out_t, cur_node);
        }
    }
}

} // namespace qtlibn

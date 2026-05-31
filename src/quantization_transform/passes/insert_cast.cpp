//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2023 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2023-04-08 22:27:33
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-11 17:54:40
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

static bool criteria(aipubt::NodePtr n)
{
    if (n->type() != aipubt::OpType_DeQuantize && n->type() != aipubt::OpType_Quantize)
    {
        return false;
    }

    if (n->attrs().count("trigger_float_op"))
    {
        std::string trigger_fp_mode = n->get_attr<aipubt::NodeParamValue>("trigger_float_op");
        std::transform(trigger_fp_mode.begin(), trigger_fp_mode.end(), trigger_fp_mode.begin(), ::toupper);
        if (trigger_fp_mode == "DISABLE")
        {
            return false;
        }
    }
    else
    {
        return false;
    }
    return true;
}

static void _insert_cast_for_dequantize(aipubt::NodePtr node, const std::string &n_name, const std::string &t_name)
{
    auto t = node->inputs()[0];
    auto cast_op = aipubt::Node::create(n_name, aipubt::OpType_Cast);
    auto cast_o_t = std::make_shared<aipubt::Tensor>(t_name, t->shape(), aipubt::Tensor::INT16);
    cast_o_t->quantization() = t->quantization();
    cast_op->add_input(t);
    cast_op->add_output(cast_o_t);
    cast_op->params()["to_dtype"] = aipubt::Tensor::INT16;

    node->clear_inputs();
    node->add_input(cast_o_t);

    node->graph()->add_node(cast_op);
}
static void _insert_cast_for_quantize(aipubt::NodePtr node, const std::string &n_name, const std::string &t_name)
{
    auto t = node->outputs()[0];
    auto cast_op = aipubt::Node::create(n_name, aipubt::OpType_Cast);
    auto cast_o_t = std::make_shared<aipubt::Tensor>(t_name, t->shape(), aipubt::Tensor::INT16);
    cast_o_t->quantization() = t->quantization();
    cast_op->add_input(cast_o_t);
    cast_op->add_output(t);
    cast_op->params()["to_dtype"] = aipubt::Tensor::INT16;

    node->clear_outputs();
    node->add_output(cast_o_t);
    node->graph()->add_node(cast_op);
}

void insert_cast_for_dequantize_and_quantize(aipubt::GraphPtr g)
{
    return;
    // in order to pass the modelxy_tflite model, we use a simple method to insert cast.
    // and we will implement the insert cast like opt has done, which uses the ops dtype map from lib.
    auto nodes = g->nodes();
    for (auto &n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto n_name = g->get_valid_node_name("cast");
        auto t_name = g->get_valid_tensor_name(n_name);

        if (n->type() == aipubt::OpType_DeQuantize)
        {
            _insert_cast_for_dequantize(n, n_name, t_name);
        }
        else
        {
            _insert_cast_for_quantize(n, n_name, t_name);
        }
    }
}

} // namespace qtlibn

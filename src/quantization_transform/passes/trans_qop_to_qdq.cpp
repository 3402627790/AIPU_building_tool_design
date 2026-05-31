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
 * @Date: 2025-04-16 16:06:25
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-04-16 16:06:47
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
static inline bool criteria(aipubt::NodePtr node)
{
    if (node->type() != aipubt::OpType_Pow)
    {
        return false;
    }

    for (auto &inp : node->inputs())
    {
        if (aipubt::Tensor::is_float(inp->type()))
        {
            return false;
        }
        if (inp->quantization().empty())
        {
            return false;
        }
    }

    for (auto &out : node->outputs())
    {
        if (aipubt::Tensor::is_float(out->type()))
        {
            return false;
        }
        if (out->quantization().empty())
        {
            return false;
        }
    }

    return true;
}

void trans_op_to_qdq(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        if (!criteria(node))
        {
            continue;
        }

        for (uint32_t i = 0; i < node->inputs().size(); i++)
        {
            auto inp = node->inputs()[i];
            node->remove_input(inp);

            aipubt::NodePtr new_dq_node = aipubt::Node::create(g->get_valid_node_name(node->name() + "dequantize"), aipubt::OpType_DeQuantize);
            auto new_dq_out = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(inp->name()), inp->shape(), aipubt::Tensor::FP32);
            new_dq_node->add_input(inp);
            new_dq_node->add_output(new_dq_out);
            new_dq_node->attrs() = node->attrs();

            node->add_input(new_dq_out, i);

            g->add_node(new_dq_node);
        }

        for (uint32_t i = 0; i < node->outputs().size(); i++)
        {
            auto out = node->outputs()[i];

            auto new_q_in = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(out->name()), out->shape(), aipubt::Tensor::FP32);
            node->remove_output(out);
            node->add_output(new_q_in, i);

            auto new_quantize_node = aipubt::Node::create(g->get_valid_node_name(node->name() + "quantize"), aipubt::OpType_Quantize);
            new_quantize_node->add_input(new_q_in);
            new_quantize_node->add_output(out);
            new_quantize_node->attrs() = node->attrs();

            g->add_node(new_quantize_node);
        }
    }
}

} // namespace qtlibn

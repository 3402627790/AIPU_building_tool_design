//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Alex Zheng
 * @Date: 2026-08-28 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time:  2026-08-28 17:14:20
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt::ops
{
#define TensorCheck(t, str)                                                  \
    if (t == nullptr)                                                        \
    {                                                                        \
        ERROR() << "OP 'grouped_experts' input '" << str << "' is nullptr."; \
        return {};                                                           \
    }

static std::string update_rule_to_string(UpdateRule update_rule)
{
    std::map<UpdateRule, std::string> m_table =
        {
            {UpdateRule::GatedDelta, "gated_delta"},
            {UpdateRule::Linear, "linear"},
            {UpdateRule::Gated, "gated"},
            {UpdateRule::Delta, "delta"},
        };

    if (m_table.count(update_rule))
    {
        return m_table[update_rule];
    }
    return {};
}

TensorPtrList linear_attention(TensorPtrList inputs,
                               int q_num_heads,
                               int k_num_heads,
                               int v_num_heads,
                               float scale,
                               UpdateRule update_rule,
                               bool has_present_state,
                               int chunk_size)
{

    if (inputs.size() < 5)
    {
        ERROR() << "OP linear_attention at least the following inputs are required:\n"
                << "query,key,value,decay,beta";
        return {};
    }

    auto g = inputs[0]->node()->graph();

    TensorPtrList outputs;
    auto node = Node::create(g->get_valid_node_name("linear_attention"), OpType_LinearAttention);
    for (auto inp : inputs)
    {
        if (inp != nullptr)
        {
            node->add_input(inp);
        }
    }
    auto value = inputs[2];
    int Hq = q_num_heads;
    int Hk = k_num_heads;
    int Hv = v_num_heads;
    int output_v = 0;
    int dv = value->shape()[-1] / Hv;
    int B = value->shape()[0];
    int T = value->shape()[1];
    if (Hk == Hv)
    {
        output_v = Hq * dv;
    }
    else if (Hq == Hk && Hv > Hk)
    {
        output_v = Hv * dv;
    }

    TensorShape output_shape({B, T, output_v});
    auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("linear_attention/output"), output_shape);
    add_tensor_info(value, output);
    node->add_output(output);
    outputs.push_back(output);

    if (has_present_state)
    {
        auto query = inputs[0];
        int dk = query->shape()[-1] / Hq;
        TensorShape present_state_shape({B, Hv, dk, dv});
        auto present_state = std::make_shared<Tensor>(g->get_valid_tensor_name("linear_attention/present_state"), present_state_shape);
        present_state->set_type(Tensor::FP32);
        node->add_output(present_state);
        outputs.push_back(present_state);
    }
    auto &params = node->params();
    params["q_num_heads"] = q_num_heads;
    params["k_num_heads"] = k_num_heads;
    params["v_num_heads"] = v_num_heads;
    params["scale"] = v_num_heads;
    params["update_rule"] = update_rule_to_string(update_rule);
    params["chunk_size"] = chunk_size;
    params["scale"] = scale;

    g->add_node(node);
    node->infer_shape(true);
    quantize(node);
    return outputs;
}

} // namespace aipubt::ops

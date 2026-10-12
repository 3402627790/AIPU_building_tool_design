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
 * @Date: 2026-08-20 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time:  2026-08-20 17:14:20
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
    }

static std::string layout_to_string(GroupedExpertsLayout l)
{
    std::map<GroupedExpertsLayout, std::string> m_table =
        {
            {GroupedExpertsLayout::E_O_I, "E_O_I"},
            {GroupedExpertsLayout::E_I_O, "E_I_O"},
        };

    if (m_table.count(l))
    {
        return m_table[l];
    }
    return {};
}
static std::string gate_up_to_string(GroupedExpertsGL l)
{
    TensorCheck(ffn_gate_weight, "ffn_gate_weight");
    TensorCheck(ffn_up_weight, "ffn_up_weight");
    TensorCheck(ffn_down_weight, "ffn_down_weight");

    auto g = input->node()->graph();

    TensorShape output_shape = input->shape();
    auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("shared_expert"), output_shape);
    add_tensor_info(input, output);

    auto node = Node::create(g->get_valid_node_name("shared_expert"), OpType_SharedExpert);
    node->add_input(input);
    node->add_output(output);
    auto &params = node->params();

    params["activation"] = activation.type;
    if (activation.params_f.size() > 0)
    {
        params["clamp_limit"] = activation.params_f[0];
    }
    params["activation"] = activation.type;
    params["hidden_size"] = input->shape()[1];
    params["intermediate_size"] = ffn_down_weight->shape()[1];

    node->constants()["ffn_gate_weight"] = ffn_gate_weight;
    node->constants()["ffn_up_weight"] = ffn_up_weight;
    node->constants()["ffn_down_weight"] = ffn_down_weight;
    if (ffn_gate_bias != nullptr)
        node->constants()["ffn_gate_bias"] = ffn_gate_bias;
    if (ffn_up_bias != nullptr)
        node->constants()["ffn_up_bias"] = ffn_up_bias;
    if (ffn_down_bias != nullptr)
        node->constants()["ffn_down_bias"] = ffn_down_bias;
    g->add_node(node);
    node->infer_shape(true);
    quantize(node);
    return output;
}

} // namespace aipubt::ops
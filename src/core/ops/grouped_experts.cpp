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
    std::map<GroupedExpertsGL, std::string> m_table =
        {
            {GroupedExpertsGL::Concatenated, "concatenated"},
            {GroupedExpertsGL::Interleaved, "interleaved"},
        };

    if (m_table.count(l))
    {
        return m_table[l];
    }
    return {};
}

TensorPtr grouped_experts(TensorPtr permuted_states,
                          TensorPtr expert_offsets,
                          TensorPtr expert_counts,
                          TensorPtr gate_up_weights,
                          TensorPtr down_weights,
                          const GroupedExpertsLayout gate_up_weight_layout,
                          const GroupedExpertsLayout down_weight_layout,
                          const GroupedExpertsGL gate_up_layout,
                          const Activation &activation,
                          TensorPtr gate_up_bias,
                          TensorPtr down_bias)
{
    TensorCheck(permuted_states, "permuted_states");
    TensorCheck(expert_offsets, "expert_offsets");
    TensorCheck(expert_counts, "expert_counts");
    TensorCheck(gate_up_weights, "gate_up_weights");
    TensorCheck(down_weights, "down_weights");

    auto g = permuted_states->node()->graph();

    int32_t R = permuted_states->shape()[0];
    TensorShape output_shape = permuted_states->shape();
    auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("grouped_experts"), output_shape);
    add_tensor_info(permuted_states, output);
    auto node = Node::create(g->get_valid_node_name("grouped_experts"), OpType_GroupedExperts);
    node->add_input(permuted_states);
    node->add_input(expert_offsets);
    node->add_input(expert_counts);
    node->add_output(output);

    auto &params = node->params();
    params["gate_up_weight_layout"] = layout_to_string(gate_up_weight_layout);
    params["down_weight_layout"] = layout_to_string(down_weight_layout);
    params["gate_up_layout"] = gate_up_to_string(gate_up_layout);

    params["num_experts"] = expert_counts->shape()[0];
    params["hidden_size"] = R;
    params["intermediate_size"] = down_weights->shape()[2];

    params["activation"] = activation.type;
    if (activation.params_f.size() > 0)
    {
        params["clamp_limit"] = activation.params_f[0];
    }
    if (activation.params_f.size() > 1)
    {
        params["activation_alpha"] = activation.params_f[1];
    }

    node->constants()["gate_up_weights"] = gate_up_weights;
    node->constants()["down_weights"] = down_weights;
    if (gate_up_bias != nullptr)
        node->constants()["gate_up_bias"] = gate_up_bias;
    if (down_bias != nullptr)
        node->constants()["down_bias"] = down_bias;
    g->add_node(node);
    node->infer_shape(true);
    quantize(node);
    return output;
}

} // namespace aipubt::ops
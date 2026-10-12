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

namespace aipubt
{
namespace ops
{
    TensorPtrList moe_permute(TensorPtr input,
                              TensorPtr top_k_ids,
                              int32_t num_experts)
    {

        auto g = input->node()->graph();
        int32_t T = input->shape()[0], H = input->shape()[1];
        int32_t K = top_k_ids->shape()[1];
        int32_t R = T * K;
        TensorShape permuted_states_shape({R, H});
        TensorShape route_pos_shape({T, K});
        TensorShape expert_counts_shape({num_experts});
        TensorShape expert_offsets_shape({num_experts + 1});
        auto permuted_states = std::make_shared<Tensor>(g->get_valid_tensor_name("permuted_states"),
                                                        permuted_states_shape);
        add_tensor_info(input, permuted_states);
        auto route_pos = std::make_shared<Tensor>(g->get_valid_tensor_name("route_pos"),
                                                  route_pos_shape);
        route_pos->set_type(Tensor::INT64);
        auto expert_counts = std::make_shared<Tensor>(g->get_valid_tensor_name("expert_counts"),
                                                      expert_counts_shape);
        expert_counts->set_type(Tensor::INT32);
        auto expert_offsets = std::make_shared<Tensor>(g->get_valid_tensor_name("expert_offsets"),
                                                       expert_offsets_shape);
        expert_offsets->set_type(Tensor::INT32);

        auto node = Node::create(g->get_valid_node_name("moe_permute"), OpType_MoEPermute);
        node->add_input(input);
        node->add_input(top_k_ids);
        node->add_output(permuted_states);
        node->add_output(route_pos);
        node->add_output(expert_counts);
        node->add_output(expert_offsets);
        auto &params = node->params();
        params["num_experts"] = num_experts;
        params["top_k"] = K;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return {permuted_states, route_pos, expert_counts, expert_offsets};
    }
} // namespace ops
} // namespace aipubt
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
#define PARAM_CHECK(p, value, name)                                              \
    if (p == value)                                                              \
    {                                                                            \
        ERROR() << "The parameter " << name << " must be set to a valid value, " \
                << "now is " << std::to_string(int(p));                          \
        return {};                                                               \
    }

static std::string method_to_string(MoeRoutingMethod m)
{
    std::map<MoeRoutingMethod, std::string> m_table =
        {
            {MoeRoutingMethod::None, "none"},
            {MoeRoutingMethod::Renormalize, "renormalize"},
            {MoeRoutingMethod::RenormalizeNaive, "renormalize_naive"},
            {MoeRoutingMethod::Deepseekv3, "deepseekv3"},
        };

    if (m_table.count(m))
    {
        return m_table[m];
    }
    return {};
}
static std::string select_to_string(MoeRoutingSelect m)
{
    std::map<MoeRoutingSelect, std::string> m_table =
        {
            {MoeRoutingSelect::None, "none"},
            {MoeRoutingSelect::Topk, "top_k"},
            {MoeRoutingSelect::Hash, "hash"},
        };

    if (m_table.count(m))
    {
        return m_table[m];
    }
    return {};
}
static std::string scoring_func_to_string(MoeRouterScoringFunc m)
{
    std::map<MoeRouterScoringFunc, std::string> m_table =
        {
            {MoeRouterScoringFunc::Softmax, "softmax"},
            {MoeRouterScoringFunc::Raw, "raw"},
            {MoeRouterScoringFunc::Sqrtsoftplus, "sqrtsoftplus"},
        };

    if (m_table.count(m))
    {
        return m_table[m];
    }
    return {};
}

TensorPtrList moe_router(
    TensorPtrList inputs,
    TensorPtr router_weight,
    MoeRoutingParam param)
{
    PARAM_CHECK(param.routing_method, MoeRoutingMethod::None, "routing_method");
    PARAM_CHECK(param.num_experts, -1, "num_experts");
    PARAM_CHECK(param.top_k, -1, "top_k");
    PARAM_CHECK(param.hidden_size, -1, "hidden_size");
    PARAM_CHECK(param.router_select, MoeRoutingSelect::None, "router_select");

    if ((param.routing_method == MoeRoutingMethod::Deepseekv3 &&
         param.router_select != MoeRoutingSelect::Hash) ||
        param.routing_method == MoeRoutingMethod::Renormalize)
    {
        if (param.bias == nullptr)
        {
            ERROR() << "Deepseek or GPT need bias";
            return {};
        }
    }
    if (param.routing_method == MoeRoutingMethod::Deepseekv3 &&
        param.router_select == MoeRoutingSelect::Hash)
    {
        if (inputs.size() != 2)
        {
            ERROR() << "Deepseek hash mode need input [hidden_states,input_ids].";
            return {};
        }
        if (param.expert_id_table == nullptr)
        {
            ERROR() << "Deepseek hash mode need expert_id_table tensor.";
            return {};
        }
    }
    TensorPtr input = inputs[0];
    int32_t T = input->shape()[0];
    auto g = input->node()->graph();
    TensorShape top_k_shape({T, param.top_k});
    Tensor::Dtype top_k_dtype = Tensor::BFP16;
    Tensor::Dtype top_k_ids_dtype = Tensor::INT64;
    if (param.top_k_weights_dtype != Tensor::VOID)
    {
        top_k_dtype = param.top_k_weights_dtype;
    }
    auto top_k_weights = std::make_shared<Tensor>(g->get_valid_tensor_name("top_k_weights"), top_k_shape, top_k_dtype);
    auto top_k_ids = std::make_shared<Tensor>(g->get_valid_tensor_name("top_k_ids"), top_k_shape, top_k_ids_dtype);

    TensorPtrList outputs;
    auto node = Node::create(g->get_valid_node_name("moe_router"), OpType_MoERouter);

    node->add_input(input);
    node->add_output(top_k_weights);
    node->add_output(top_k_ids);
    outputs.push_back(top_k_weights);
    outputs.push_back(top_k_ids);

    auto &n_params = node->params();
    n_params["routing_method"] = method_to_string(param.routing_method);
    n_params["num_experts"] = param.num_experts;
    n_params["top_k"] = param.top_k;
    n_params["hidden_size"] = param.hidden_size;
    n_params["router_select"] = select_to_string(param.router_select);
    n_params["scoring_func"] = scoring_func_to_string(param.scoring_func);

    if (param.routing_method == MoeRoutingMethod::Deepseekv3)
    {
        if (param.route_scale != -1)
            n_params["route_scale"] = param.route_scale;
        if (param.norm_eps != -1)
            n_params["norm_eps"] = param.norm_eps;
    }

    node->constants()["router_weight"] = router_weight;

    if (param.bias != nullptr)
    {
        if (param.routing_method == MoeRoutingMethod::Deepseekv3)
        {
            node->constants()["correction_bias"] = param.bias;
        }
        else if (param.routing_method == MoeRoutingMethod::Renormalize)
        {
            node->constants()["router_bias"] = param.bias;
        }
    }
    if (param.routing_method == MoeRoutingMethod::Deepseekv3 &&
        param.router_select == MoeRoutingSelect::Hash)
    {
        node->add_input(inputs[1]);
        node->constants()["expert_id_table"] = param.expert_id_table;
    }
    g->add_node(node);
    node->infer_shape(true);
    quantize(node);
    return outputs;
}

} // namespace aipubt::ops
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
 * @Date: 2022-10-28 15:59:56
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-21 14:28:25
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "core/location.h"
#include "passes.h"
#include "quantization_transform/common.h"

namespace qtlibn
{
using namespace aipubt;
/*
dequantize_in, dequantize_out, exp_out, mul_out, quantize_out
[[119, 135], [-1.2705879, 0.9882355], [0.28066656, 2.6864898], [23.014658, 217.60568], [255, 255]]
[[122, 139], [-0.84705925, 1.5529404], [0.4286737, 4.725344], [9.859495, 108.682915], [198, 255]]
[[122, 136], [-0.84705925, 1.1294117], [0.4286737, 3.0938358], [15.860928, 179.44247], [240, 255]]
[[121, 137], [-0.9882355, 1.2705879], [0.3722329, 3.5629468], [30.523098, 288.5987], [255, 255]]
[[121, 135], [-0.9882355, 0.9882355], [0.3722329, 2.6864898], [50.251442, 362.67612], [255, 255]]
[[121, 130], [-0.9882355, 0.28235245], [0.3722329, 1.326246], [118.7423, 423.07248], [255, 255]]
*/
static std::string criteria(std::vector<aipubt::NodePtr> &ns)
{
    auto node_num = ns.size();
    auto dequantize_n0 = ns[0];
    auto quantize_n4 = ns[node_num - 1];
    if (dequantize_n0->inputs()[0]->quantization().empty() || quantize_n4->outputs()[0]->quantization().empty())
    {
        return (std::string) "unmatched";
    }

    if ((dequantize_n0->attrs().count("unmerged") && dequantize_n0->get_attr<bool>("unmerged")) ||
        (dequantize_n0->attrs().count("unquantifiable") && (bool)dequantize_n0->get_attr("unquantifiable")))
    {
        return (std::string) "unmerged";
    }
    else
    {
        for (auto &n : ns)
        {
            if (n->outputs()[0]->is_io_tensor())
            {
                return (std::string) "unmatched";
            }
        }
        return (std::string) "merged";
    }

    // return (std::string)"unmerged";
}
static void _fuse_to_activation(aipubt::GraphPtr g, std::vector<aipubt::NodePtr> &m)
{
    // DEBUG() << "now do fuse float ops [dequantize, exp, mul, quantize] to [split, actiavtion, concat]";
    int32_t node_num = m.size();
    auto dequantize_n = m[0];
    auto constant_n = m[2];
    auto quantize_n = m[node_num - 1];
    auto deq_n_inp = dequantize_n->inputs()[0];
    auto axis = deq_n_inp->shape().dim() - 1;
    auto split_t = aipubt::ops::split(deq_n_inp, deq_n_inp->shape()[-1], axis);
    auto const_vals = constant_n->constants()["weights"]->data<float>();
    auto dtype = quantize_n->outputs()[0]->type();

    std::string method = "UNKNOWN";
    std::string act_name = "activation/" + method;

    auto split_size = split_t.size();
    std::vector<aipubt::TensorPtr> activation_outputs;
    for (uint32_t ss = 0; ss < split_size; ss++)
    {
        auto activation_n0 = aipubt::Node::create(g->get_valid_node_name(act_name), aipubt::OpType_Activation);
        auto activation_output0 = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(act_name), split_t[ss]->shape(), dtype);
        activation_output0->quantization() = quantize_n->outputs()[0]->quantization();
        activation_n0->add_input(split_t[ss]);
        activation_n0->add_output(activation_output0);
        activation_n0->params()["method"] = method;
        activation_n0->params()["constant_value"] = aipubt::NodeParamValue(const_vals[ss]);
        g->add_node(activation_n0);
        ADD_LOCATION_BY_NODE(activation_n0, m[1]);
        activation_outputs.push_back(activation_output0);
    }

    auto concat_t = aipubt::ops::concat(activation_outputs, axis, quantize_n->outputs()[0]->quantization());
    auto concat_node = concat_t->node();
    concat_node->clear_outputs();
    concat_node->add_output(quantize_n->outputs()[0]);
    g->add_node(split_t[0]->node());
    g->add_node(concat_node);
    ADD_LOCATION_BY_NODE(concat_node, m[1]);
    ADD_LOCATION_BY_NODE(split_t[0]->node(), m[1]);
    dequantize_n->clear_inputs();
    dequantize_n->clear_outputs();
    for (int32_t i = 1; i < node_num - 1; i++)
    {
        m[i]->clear_inputs();
        m[i]->clear_outputs();
    }
    quantize_n->clear_inputs();
    quantize_n->clear_outputs();
    for (int32_t i = 0; i < node_num; i++)
    {
        g->remove_node(m[i]);
    }
}

static void _convert_to_aipu_fp(std::vector<aipubt::NodePtr> &m)
{
    // auto node_num = m.size();
    // for (uint32_t i = 0; i < node_num - 1; i++)
    // {
    //     auto node = m[i];
    //     for (auto &ot : node->outputs())
    //     {
    //         ot->set_type(aipubt::Tensor::FP16);
    //         ot->quantization().set_unquantifiable(true);
    //     }

    //     for (auto &ct : node->constants())
    //     {
    //         ct.second->set_type(aipubt::Tensor::FP16);
    //         ct.second->quantization().set_unquantifiable(true);
    //     }
    //     m[i]->set_attr("unquantifiable", true);
    // }
}

void fuse_float_op_between_dequantize_and_quantize(aipubt::GraphPtr g)
{
    std::vector<aipubt::GraphMatcherPattern> patterns = {
        {{{"0", aipubt::OpType_DeQuantize}, {"1", aipubt::OpType_Exp}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Eltwise}, {"4", aipubt::OpType_Quantize}},
         {{"0", "1"}, {"1", "3"}, {"2", "3"}, {"3", "4"}}},
        {{{"0", aipubt::OpType_DeQuantize}, {"1", aipubt::OpType_Exp}, {"2", aipubt::OpType_Constant}, {"3", aipubt::OpType_Mul}, {"4", aipubt::OpType_Quantize}},
         {{"0", "1"}, {"1", "3"}, {"2", "3"}, {"3", "4"}}},
        {{{"0", aipubt::OpType_DeQuantize}, {"1", aipubt::OpType_Exp}, {"2", aipubt::OpType_BatchNorm}, {"3", aipubt::OpType_Quantize}},
         {{"0", "1"}, {"1", "2"}, {"2", "3"}}},
    };

    // bool is_changed = false;
    for (auto &pattern : patterns)
    {
        aipubt::GraphMatcher matcher(pattern);
        matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                auto crit = criteria(m);
                if (crit == "unmatched")
                {
                    continue;
                }
                if (crit == "merged")
                {
                    _fuse_to_activation(g, m);
                    is_merged = true;
                }
                else if (crit == "unmerged")
                {
                    _convert_to_aipu_fp(m);
                }
                else
                {
                    ERROR() << "unsuppport critera return value.";
                }
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
}

} // namespace qtlibn

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
 * @Author: ALex Zheng
 * @Date: 2021-07-22 14:29:19
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:32:25
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
namespace
{
    static bool is_support_act(NodePtr n, Activation act)
    {
        std::vector<std::string> activation_support =
        {
            "RELU",
            "TANH",
            "SIGMOID",
            "AFFINE",
            "LEAKYRELU",
            "THRESHOLDEDRELU",
            "HARDSIGMOID",
            "ELU",
            "SOFTSIGN",
            "SOFTPLUS",
            "CLIP",
        };
        if (std::find(activation_support.begin(), activation_support.end(), act.type) != activation_support.end())
        {
            bool ret=true;
            if(act.type == "CLIP")
            {
                if((act.params_quantized) && (act.params_i.size() !=2))
                {
                    WARN() << "Node: " << n->name() <<"activation " << act.type << " expect 2 params.";
                    ret = false;
                }
                else if((!act.params_quantized) && (act.params_f.size() !=2))
                {
                    WARN() << "Node: " << n->name() <<"activation " << act.type << " expect 2 params.";
                    ret = false;
                }
            }
            return ret;
        }
        else
        {
            WARN() << "Node: " << n->name() << " may not support activation " << act.type;
            return false;
        }
    }

    static bool append_activations(NodePtr n, std::vector<Activation> activations)
    {
        std::vector<Activation> activation_default = {
            Activation("SIGMOID"),
            Activation("TANH"),
        };
        auto &params = n->params();
        NodeParamValue activations_params;
        for(size_t i=0; i<activation_default.size(); ++i)
        {
            Activation act = activation_default[i];
            if((activations.size()>i) && (is_support_act(n, activations[i])))
            {
                act = activations[i];
            }
            std::string act_name = act.type;
            params["activations"].push_back(NodeParamValue(act_name));
            NodeParamValue activation_params;
            append_activation_params(n, activation_params, act);
            activations_params.push_back(activation_params);
        }
        params["activations_params"] = activations_params;
        return true;
    }

    static std::vector<TensorPtr> gru_common(OpType optype,
                                             TensorPtr input,
                                             TensorPtr h0,
                                             TensorPtr weights,
                                             TensorPtr biases,
                                             const GruOutSequence &out_sequence,
                                             const std::vector<Activation> &activations,
                                             const GruDirection &direction,
                                             const Quantization &quant)
    {
        if (input->shape().dim() != 3 || h0->shape().dim() != 2)
        {
            ERROR() << "Dimensions of input,h0 must be 3,2!";
            return {};
        }

        std::string gru_name = "gru_v3";
        if (optype == OpType_GRUv1)
            gru_name = "gru_v1";
        std::map<GruOutSequence, std::vector<std::string>> out_sequence_maps =
        {
            {GruOutSequence::H, {"H"}},
            {GruOutSequence::Hn, {"Hn"}},
            {GruOutSequence::H_Hn, {"H", "Hn"}},
        };

        std::map<GruDirection, std::string> direction_maps =
        {
            {GruDirection::Forward, "forward"},
            {GruDirection::Reverse, "reverse"},
        };
        auto out_sequence_str = out_sequence_maps[out_sequence];
        int32_t batch_size = input->shape()[0];
        int32_t time_steps = input->shape()[1];
        int32_t input_size = input->shape()[2];
        int32_t cell_size = h0->shape()[1];

        if (weights->shape().dim() != 2)
        {
            ERROR() << "Dimensions of weights must be 2!";
            return {};
        }

        if ((weights->shape()[0] != 3 * cell_size) || (weights->shape()[1] != (input_size + cell_size)))
        {
            ERROR() << "W_weight dimensions must be [3 * cell size, (input size+cell_size)]!";
            return {};
        }

        std::vector<TensorPtr> outputs;
        auto g = input->node()->graph();
        for (auto sequence : out_sequence_str)
        {
            if (sequence == "Hn")
            {
                TensorPtr hn = std::make_shared<Tensor>(g->get_valid_tensor_name(gru_name + "/hn"), h0->shape());
                hn->set_type(input->type());
                hn->quantization() = input->quantization();
                hn->shape().set_layout(input->shape().layout());
                outputs.push_back(hn);
                break;
            }
            if (sequence == "H")
            {
                TensorPtr h = std::make_shared<Tensor>(g->get_valid_tensor_name(gru_name + "/h"),
                                                       TensorShape(batch_size, time_steps, cell_size));
                h->set_type(input->type());
                h->quantization() = input->quantization();
                h->shape().set_layout(input->shape().layout());
                outputs.push_back(h);
            }
        }

        auto node = Node::create(g->get_valid_node_name(gru_name), optype);
        node->add_input(input);
        node->add_input(h0);
        for (auto output : outputs)
        {
            add_tensor_info(input, output, quant);
            node->add_output(output);
        }

        node->constants()["weights"] = weights;
        node->constants()["biases"] = biases;

        g->add_node(node);

        auto &params = node->params();
        params["time_steps"] = time_steps;
        params["input_size"] = input_size;
        params["cell_size"] = cell_size;
        params["direction"] = direction_maps[direction];
        for (auto sequence : out_sequence_str)
        {
            params["out_sequence"].push_back(sequence);
        }
        append_activations(node, activations);
        quantize(node);
        return outputs;
    }
}

    std::vector<TensorPtr> gru_v3(TensorPtr input,
                                  TensorPtr h0,
                                  TensorPtr weights,
                                  TensorPtr biases,
                                  const GruOutSequence &out_sequence,
                                  const std::vector<Activation> &activations,
                                  const GruDirection &direction,
                                  const Quantization &quant)
    {
        return gru_common(OpType_GRUv3, input, h0, weights, biases, out_sequence,
                          activations, direction, quant);
    }

    std::vector<TensorPtr> gru(TensorPtr input,
                               TensorPtr h0,
                               TensorPtr weights,
                               TensorPtr biases,
                               const GruOutSequence &out_sequence,
                               const std::vector<Activation> &activations,
                               const GruDirection &direction,
                               const Quantization &quant)
    {
        return gru_v3(input, h0, weights, biases, out_sequence,
                      activations, direction, quant);
    }

    std::vector<TensorPtr> gru_v1(TensorPtr input,
                                  TensorPtr h0,
                                  TensorPtr weights,
                                  TensorPtr biases,
                                  const GruOutSequence &out_sequence,
                                  const std::vector<Activation> &activations,
                                  const GruDirection &direction,
                                  const Quantization &quant)
    {
        return gru_common(OpType_GRUv1, input, h0, weights, biases, out_sequence,
                          activations, direction, quant);
    }
}; //namespace ops
}; // namespace aipubt

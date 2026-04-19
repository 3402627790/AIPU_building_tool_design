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
 * @Author: Neo Wang
 * @Date: 2021-04-14 11:04:33
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-05-22 14:37:00
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    // RELU: max(0, x)
    // TANH: (1 - e^{-2x}) / (1 + e^{-2x})
    // SIGMOID: 1/(1 + e^{-x})
    // LEAKYRELU: x if x >= 0 else alpha * x
    // THRESHOLDEDRELU: x if x >= alpha else 0
    // HARDSIGHMOID: min(max (alpha * x + beta, 0), 1)
    // ELU: x if x >= 0 else alpha * (e^x -1)
    // SOFTSIGN: x / (1 + |x|)
    // SOFTPLUS: log(1 + e^x)
    static bool is_support_act(NodePtr n, Activation act)
    {
        std::vector<std::string> activation_support = {
            "RELU",
            "TANH",
            "SIGMOID",
            "LEAKYRELU",
            "THRESHOLDEDRELU",
            "HARDSIGHMOID",
            "ELU",
            "SOFTSIGN",
            "SOFTPLUS",
        };
        if (std::find(activation_support.begin(), activation_support.end(), act.type) != activation_support.end())
        {
            return true;
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
            params["activations"].push_back(NodeParamValue(act.type));
            NodeParamValue activation_params;
            append_activation_params(n, activation_params, act);
            activations_params.push_back(activation_params);
        }
        params["activations_params"] = activations_params;
        return true;
    }

    std::vector<TensorPtr> basic_lstm(TensorPtr input,
                                      TensorPtr hidden_state,
                                      TensorPtr cell_gate,
                                      TensorPtr weights,
                                      TensorPtr biases,
                                      std::vector<Activation> activations,
                                      const Quantization &cell_quant,
                                      const Quantization &out_quant,
                                      const LSTMOutSequence &out_sequence)
    {
        TensorPtr new_input = input;
        if (input->shape().dim() == 2)
        {
            // If the input is a 2D tensor, then reshape it to 3D(batch_size, time_step, input_size)
            // by setting time_step to 1.
            TensorShape r_shape(input->shape());
            r_shape.reshape(input->shape()[0], 1, input->shape()[1]);
            new_input = reshape(input, r_shape);
        }

        if (new_input->shape().dim() != 3 ||
            hidden_state->shape().dim() != 2 ||
            cell_gate->shape().dim() != 2)
        {
            ERROR() << "rank of input,hidden_state,cell_gate should be 3,2,2!";
            return {};
        }

        int32_t output_size = cell_gate->shape()[1];
        if (output_size != cell_gate->shape()[1])
        {
            ERROR() << "dimension size of cell_state_in[1],output_state_in[1] should be equal!";
            return {};
        }

        if (4 * output_size != weights->shape()[0] ||
            weights->shape()[0] != biases->shape()[0])
        {
            ERROR() << "dimension size of input_weights[0],bias[0] should be equal!";
            return {};
        }

        int32_t batch_size = new_input->shape()[0];
        int32_t time_step = new_input->shape()[1];
        int32_t input_size = new_input->shape()[2];

        if (batch_size != hidden_state->shape()[0] ||
            batch_size != cell_gate->shape()[0])
        {
            ERROR() << "dimension of input[-2],hidden_state[0],cell_gate[0] should be equal!";
            return {};
        }

        std::map<LSTMOutSequence, std::vector<std::string>> out_sequence_maps =
            {
                {LSTMOutSequence::Y, {"Y"}},
                {LSTMOutSequence::H, {"H"}},
                {LSTMOutSequence::C, {"C"}},
                {LSTMOutSequence::Y_H, {"Y", "H"}},
                {LSTMOutSequence::Y_C, {"Y", "C"}},
                {LSTMOutSequence::H_C, {"H", "C"}},
                {LSTMOutSequence::Y_H_C, {"Y", "H", "C"}},
            };

        auto g = new_input->node()->graph();

        auto node = Node::create(g->get_valid_node_name("lstm"), OpType_BasicLSTM);
        node->add_input(new_input);
        node->add_input(hidden_state);
        node->add_input(cell_gate);

        auto &params = node->params();
        auto out_sequence_str = out_sequence_maps[out_sequence];
        std::vector<TensorPtr> outputs;
        for (auto sequence : out_sequence_str)
        {
            params["out_sequence"].push_back(sequence);
            if (sequence == "Y")
            {
                TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("lstm/output"), TensorShape(batch_size, time_step, output_size));
                add_tensor_info(new_input, output, out_quant);
                outputs.push_back(output);
                node->add_output(output);
            }
            else if (sequence == "H")
            {
                TensorPtr output_hidden = std::make_shared<Tensor>(g->get_valid_tensor_name("lstm/hidden"), TensorShape(batch_size, output_size));
                add_tensor_info(hidden_state, output_hidden, out_quant);
                outputs.push_back(output_hidden);
                node->add_output(output_hidden);
            }
            else if (sequence == "C")
            {
                TensorPtr output_cell = std::make_shared<Tensor>(g->get_valid_tensor_name("lstm/cell"), TensorShape(batch_size, output_size));
                add_tensor_info(cell_gate, output_cell, cell_quant);
                outputs.push_back(output_cell);
                node->add_output(output_cell);
            }
        }

        node->constants()["weights"] = weights;
        node->constants()["biases"] = biases;
        params["time_steps"] = time_step;
        params["input_size"] = input_size;
        params["cell_size"] = output_size;
        params["direction"] = "forward";
        append_activations(node, activations);
        g->add_node(node);
        quantize(node);

        return outputs;
    }

}; //namespace ops
}; // namespace aipubt

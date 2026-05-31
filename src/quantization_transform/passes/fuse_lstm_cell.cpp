//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Yuxian Chen
 * @Date: 2023-06-21 11:32:32
 * @Last Modified by: Yuxian Chen
 * @Last Modified time: 2023-06-21 11:32:32
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
static inline bool criteria_for_fc(aipubt::NodePtr fc)
{
    /* Check FullyConnected op:
       1) with_activation should be NONE;
       2) biases should be 0. */
    if (fc->params().count("with_activation") > 0 && fc->params()["with_activation"] != "NONE")
    {
        return false;
    }

    auto fc_biases = fc->constants()["biases"];
    auto fc_biases_data = fc_biases->data<int32_t>();
    bool fc_biases_are_zero = std::all_of(fc_biases_data, (fc_biases_data + fc_biases->size()), [](int32_t i) { return i == 0; });
    if (!fc_biases_are_zero || (fc_biases->quantization().scale() != 1.0) || (fc_biases->quantization().offset() != 0))
        return false;
    return true;
}

static inline bool criteria_for_weights_biases(aipubt::NodePtr fc1, aipubt::NodePtr fc2, aipubt::NodePtr bias,
                                               uint32_t &input_size, uint32_t &cell_size)
{
    /* Get input_size and cell_size; and check weights and biases for LSTM:
       1) shape of weights from FullyConnected;
       2) shape of biases of Eltwise(bias) node. */
    if (!criteria_for_fc(fc1) || !criteria_for_fc(fc2))
    {
        return false;
    }
    auto fc1_weights_shape = fc1->constants()["weights"]->shape();
    auto fc2_weights_shape = fc2->constants()["weights"]->shape();
    auto bias_shape = bias->constants()["weights"]->shape();
    if (fc1_weights_shape.dim() != 2 || fc2_weights_shape.dim() != 2 || (bias_shape.dim() != 2 && bias_shape.dim() != 1))
    {
        return false;
    }
    int bias_size = bias_shape.size();
    if (fc1_weights_shape[0] != fc2_weights_shape[0] || fc1_weights_shape[0] != bias_size)
    {
        return false;
    }
    input_size = fc1_weights_shape[1];
    cell_size = fc2_weights_shape[1];
    if (bias_size != int(4 * cell_size))
    {
        return false;
    }
    return true;
}

static inline bool criteria_for_activations(aipubt::NodePtr act_i, aipubt::NodePtr act_c,
                                            aipubt::NodePtr act_f, aipubt::NodePtr act_o,
                                            aipubt::NodePtr act_cout)
{
    /* Check the method of activation nodes. */
    if (act_f->params()["method"] != "SIGMOID" || act_o->params()["method"] != "SIGMOID")
    {
        return false;
    }

    if (act_i->params()["method"] != "SIGMOID" || act_c->params()["method"] != "TANH")
    {
        return false;
    }

    if (act_cout->params()["method"] != "TANH")
    {
        return false;
    }
    return true;
}

static inline bool criteria_for_eltwise(aipubt::NodePtr add_fc, aipubt::NodePtr add_bias,
                                        aipubt::NodePtr mul_f_cpre, aipubt::NodePtr mul_i_c,
                                        aipubt::NodePtr cout, aipubt::NodePtr hout)
{
    /* Check method and with_activation of Eltwise nodes. */
    std::vector<aipubt::NodePtr> add_nodes = {add_fc, add_bias, cout};
    for (auto add_node : add_nodes)
    {
        if ((add_node->type() == aipubt::OpType_Eltwise && add_node->params()["method"] != "ADD") ||
            (add_node->params().count("with_activation") && add_node->params()["with_activation"] != "NONE"))
        {
            return false;
        }
    }

    std::vector<aipubt::NodePtr> mul_nodes = {mul_f_cpre, mul_i_c, hout};
    for (auto mul_node : mul_nodes)
    {
        if ((mul_node->type() == aipubt::OpType_Eltwise && mul_node->params()["method"] != "MUL") ||
            (mul_node->params().count("with_activation") && mul_node->params()["with_activation"] != "NONE"))
        {
            return false;
        }
    }
    return true;
}

static inline bool criteria_for_split_outs(std::map<std::string, aipubt::NodePtr> &nodes_map,
                                           std::vector<aipubt::NodePtr> act_nodes)
{
    /* Check whether the outputs of Split are activation nodes and whether there is
       only one consumer for each output. Also add activation nodes to nodes_map. */
    std::vector<std::string> act_names_in_sequence{"act_i", "act_f", "act_c", "act_o"};
    std::size_t act_name_index = 0;

    auto split_outputs = nodes_map["split"]->outputs();
    for (auto out : split_outputs)
    {
        if (out->consumers().size() == 1 && contains(act_nodes, out->consumers()[0]))
        {
            auto act_key_name = act_names_in_sequence[act_name_index++];
            nodes_map[act_key_name] = out->consumers()[0];
        }
        else
        {
            return false;
        }
    }
    return true;
}

static void get_top_scale_zp(aipubt::NodePtr node, float_t &scale, int32_t &zp)
{
    /* Get scale and offset of node's first output. */
    if (!node)
    {
        ERROR() << "Meets invalid node in get_top_scale_zp!";
        return;
    }
    auto outs = node->outputs();
    if (outs.size() < 1)
    {
        ERROR() << "Cannot get scale/zp for node " << node->name();
        return;
    }
    scale = outs[0]->quantization().scale();
    zp = outs[0]->quantization().offset();
    return;
}

static aipubt::ComputeTensor convert_ifco_to_icfo(aipubt::NodePtr node, int32_t split_axis = 0)
{
    /* Convert data from ifco format to icfo format. */
    aipubt::ComputeTensor weights = node->constants()["weights"];
    auto weights_list = weights.split(4, split_axis);
    auto new_weights = aipubt::ComputeTensor::concat({weights_list[0], weights_list[2], weights_list[1], weights_list[3]}, split_axis);
    return new_weights;
}

static void do_fuse_lstm_cell(aipubt::GraphPtr g,
                              aipubt::GraphMatcherPattern& begin_pattern,
                              aipubt::GraphMatcherPattern& end_pattern)
{
    aipubt::GraphMatcher begin_matcher(begin_pattern);
    begin_matcher.set_mode(aipubt::GraphMatcher::MatchMode::RelaxedMode);
    auto begin_matches = begin_matcher.match(g);
    if (!begin_matches.size())
    {
        return;
    }
    aipubt::GraphMatcher end_matcher(end_pattern);
    end_matcher.set_mode(aipubt::GraphMatcher::MatchMode::RelaxedMode);
    auto end_matches = end_matcher.match(g);
    if (!end_matches.size())
    {
        return;
    }

    std::set<aipubt::NodePtr> matched;
    for (auto &&begin_m : begin_matches)
    {
        if (matched.count(begin_m[0]))
        {
            continue;
        }

        // Save all the matched nodes in begin_m to nodes_map, except activation nodes.
        std::map<std::string, aipubt::NodePtr> nodes_map{};
        auto act_index = begin_pattern.first.size() - 4;
        for (std::size_t index = 0; index < act_index; ++index)
        {
            auto node_name = begin_pattern.first[index].first;
            nodes_map[node_name] = begin_m[index];
        }

        // Save activations nodes to nodes_map after checking their connections with Split node.
        std::vector<aipubt::NodePtr> act_nodes(begin_m.begin() + act_index, begin_m.end());
        if (!criteria_for_split_outs(nodes_map, act_nodes))
        {
            continue;
        }

        // Now all the nodes in begin_m should be saved to nodes_map.
        if (nodes_map.size() != begin_m.size())
        {
            continue;
        }

        /* Two FullyConnected nodes before Add node could be in random sequence. Check whether the parent node is Reshape
           with specified input shape. If yes, then it's the fc node(fc1) that connects with input sequences; otherwise
           it's the fc node(fc2) that connects with previous hout(initial h). Swap fc nodes in nodes_map if needed. */
        aipubt::NodePtr reshape_node;
        std::vector<std::string> fc_nodes = {"fc1", "fc2"};
        for (auto fc : fc_nodes)
        {
            if (nodes_map[fc]->parents().size() == 1)
            {
                auto reshape = nodes_map[fc]->parents()[0];
                if (reshape->type() != aipubt::OpType_Reshape || reshape->inputs().size() < 1 || reshape->inputs()[0]->shape().dim() != 3)
                {
                    continue;
                }
                reshape_node = reshape;
                if (fc != "fc1")
                {
                    std::swap(nodes_map["fc1"], nodes_map["fc2"]);
                    break;
                }
            }
        }
        if (!reshape_node)
        {
            continue;
        }

        /* Check whether activation i and c have the same child(mul_i_c). */
        if (nodes_map["act_i"]->children().size() != 1 || nodes_map["act_c"]->children().size() != 1 || nodes_map["act_i"]->children()[0] != nodes_map["act_c"]->children()[0])
        {
            continue;
        }
        nodes_map["mul_i_c"] = nodes_map["act_c"]->children()[0];

        /* Get Mul node(mul_f_cpre). */
        if (nodes_map["act_f"]->children().size() != 1)
        {
            continue;
        }
        nodes_map["mul_f_cpre"] = nodes_map["act_f"]->children()[0];

        /* Get one of the input tensor(initial_c). */
        aipubt::TensorPtr initial_c = nullptr;
        for (auto input : nodes_map["mul_f_cpre"]->inputs())
        {
            if (input != nodes_map["act_f"]->outputs()[0])
            {
                initial_c = input;
                break;
            }
        }
        if (!initial_c)
        {
            continue;
        }

        /* Get one of the output node(hout). */
        if (nodes_map["act_o"]->children().size() != 1)
        {
            continue;
        }
        nodes_map["hout"] = nodes_map["act_o"]->children()[0];

        /* Check whether the Eltwise nodes that are already found belongs to end matches.
           If yes, add other nodes to nodes_map as well. */
        bool end_matched = false;
        for (auto end_m : end_matches)
        {
            if (end_m[4] != nodes_map["hout"])
            {
                continue;
            }
            if (end_m[0] == nodes_map["mul_i_c"])
            {
                if (end_m[1] != nodes_map["mul_f_cpre"])
                {
                    continue;
                }
            }
            else if (end_m[1] == nodes_map["mul_i_c"])
            {
                if (end_m[0] != nodes_map["mul_f_cpre"])
                {
                    continue;
                }
            }
            nodes_map["cout"] = end_m[2];
            nodes_map["act_cout"] = end_m[3];
            end_matched = true;
            break;
        }
        if (!end_matched)
        {
            continue;
        }

        /* Check Activation nodes. */
        if (!criteria_for_activations(nodes_map["act_i"], nodes_map["act_c"],
                                      nodes_map["act_f"], nodes_map["act_o"],
                                      nodes_map["act_cout"]))
        {
            continue;
        }

        /* Check Eltwise nodes. */
        if (!criteria_for_eltwise(nodes_map["add_fc"], nodes_map["add_bias"],
                                  nodes_map["mul_f_cpre"], nodes_map["mul_i_c"],
                                  nodes_map["cout"], nodes_map["hout"]))
        {
            continue;
        }

        /* Check weights and biases, and get input_size and cell_size from them. */
        uint32_t input_size = 0, cell_size = 0;
        if (!criteria_for_weights_biases(nodes_map["fc1"], nodes_map["fc2"], nodes_map["bias"], input_size, cell_size) || input_size == 0 || cell_size == 0)
        {
            continue;
        }

        /* Check Split node. */
        int32_t axis = nodes_map["split"]->params()["axis"];
        std::vector<int32_t> splits = nodes_map["split"]->params()["splits"];
        if (axis != 1 || !std::all_of(splits.begin(), splits.end(), [=](int32_t i) { return i == (int32_t)cell_size; }))
        {
            continue;
        }

        /* Finish checking. */
        matched.insert(begin_m[0]);

        /* Create a new node whose type is BasicLSTM. */
        auto node = aipubt::Node::create(g->get_valid_node_name(nodes_map["hout"]->name() + "_lstm"), aipubt::OpType_BasicLSTM);

        /* Add weights to lstm node. */
        auto new_weights = convert_ifco_to_icfo(nodes_map["fc1"]);
        auto new_recurrent_weights = convert_ifco_to_icfo(nodes_map["fc2"]);
        node->constants()["weights"] = aipubt::ComputeTensor::concat({new_weights, new_recurrent_weights}, 1);
        // Make node weights have same quantization info as fc1.
        node->constants()["weights"]->quantization() = nodes_map["fc1"]->constants()["weights"]->quantization();

        float fc1_scale = nodes_map["fc1"]->constants()["weights"]->quantization().scale();
        int32_t fc1_zp = nodes_map["fc1"]->constants()["weights"]->quantization().offset();
        float fc2_scale = nodes_map["fc2"]->constants()["weights"]->quantization().scale();
        int32_t fc2_zp = nodes_map["fc2"]->constants()["weights"]->quantization().offset();
        node->constants()["weights"]->quantization().set_scales({fc1_scale, fc2_scale});
        node->constants()["weights"]->quantization().set_offsets({fc1_zp, fc2_zp});

        /* Add biases to lstm node. */
        aipubt::TensorShape biases_shape = {int(4 * cell_size)};
        int32_t split_axis = nodes_map["bias"]->constants()["weights"]->shape().dim() ==2 ? 1 : 0;
        auto new_biases = convert_ifco_to_icfo(nodes_map["bias"], split_axis).reshape(biases_shape);
        node->constants()["biases"] = new_biases;
        node->constants()["biases"]->quantization() = nodes_map["bias"]->constants()["weights"]->quantization();

        /* Add params activations_scale and activations_zp to lstm node. */
        std::vector<float_t> act_scale_list;
        std::vector<int32_t> act_zp_list;
        // There is no forget_bias for this pattern.
        auto act_keys = {"add_bias", "act_i", "act_c", "act_f", "act_o",
                         "mul_i_c", "mul_f_cpre", "act_cout", "cout", "hout"};
        for (auto key : act_keys)
        {
            float_t scale;
            int32_t zp;
            get_top_scale_zp(nodes_map[key], scale, zp);
            act_scale_list.push_back(scale);
            act_zp_list.push_back(zp);
        }
        node->params()["activations_scale"] = act_scale_list;
        node->params()["activations_zp"] = act_zp_list;

        /* Add attrs and params to lstm node. */
        node->set_attr("set_default_placeholder_info", true);
        node->params()["activations"] = {"SIGMOID", "TANH", "TANH"};
        node->params()["cell_size"] = cell_size;
        node->params()["direction"] = "forward";
        node->params()["input_size"] = input_size;
        node->params()["time_steps"] = 1;
        node->params()["out_sequence"] = {"H", "C"};

        /* Set inputs for lstm node. */
        auto input = reshape_node->inputs()[0];
        auto initial_h = nodes_map["fc2"]->inputs()[0];
        node->add_input({input, initial_h, initial_c});

        /* Set 2 outputs for lstm node. The first output is hout. */
        auto hout_tensor = nodes_map["hout"]->outputs()[0];
        aipubt::TensorPtr output_hidden = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(node->name() + "_hidden"),
                                                                           hout_tensor->shape(),
                                                                           hout_tensor->type());
        node->add_output(output_hidden, 0);
        output_hidden->quantization() = hout_tensor->quantization();
        auto hout_children = nodes_map["hout"]->children();
        for (auto hout_child : hout_children)
        {
            hout_child->replace_input_by(hout_tensor, output_hidden);
        }
        if (hout_tensor->is_io_tensor())
        {
            hout_tensor->set_io_tensor(false);
            output_hidden->set_io_tensor(true);
        }

        /* Set the second output(cout) for lstm node. */
        auto cout_tensor = nodes_map["cout"]->outputs()[0];
        aipubt::TensorPtr output_cell = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(node->name() + "_cell"),
                                                                         cout_tensor->shape(),
                                                                         cout_tensor->type());
        node->add_output(output_cell, 1);
        output_cell->quantization() = cout_tensor->quantization();
        auto cout_children = nodes_map["cout"]->children();
        for (auto cout_child : cout_children)
        {
            if (cout_child != nodes_map["act_cout"])
            {
                cout_child->replace_input_by(cout_tensor, output_cell);
            }
        }
        if (cout_tensor->is_io_tensor())
        {
            cout_tensor->set_io_tensor(false);
            output_cell->set_io_tensor(true);
        }

        /* Add the lstm node to graph. */
        g->add_node(node);

        /* After merging, the nodes in nodes_map could be removed.
           Check whether output is io tensor before removing. */
        std::vector<aipubt::NodePtr> nodes_to_remove;
        for (auto &map : nodes_map)
        {
            nodes_to_remove.push_back(map.second);
        }
        if(reshape_node->outputs()[0]->consumers().size() == 1)
            nodes_to_remove.push_back(reshape_node);
        auto sorted_nodes = g->topological_sort(nodes_to_remove);
        std::reverse(sorted_nodes.begin(), sorted_nodes.end());
        for (auto sorted_node : sorted_nodes)
        {
            bool can_remove = true;
            for (auto out : sorted_node->outputs())
            {
                if (out->is_io_tensor())
                {
                    can_remove = false;
                }
            }
            if (can_remove)
            {
                g->remove_node(sorted_node, true);
            }
        }
    }

    return;
}
void fuse_lstm_cell(aipubt::GraphPtr g)
{
 /* Merge one lstm cell into BasicLSTM with time step=1. */
    // begin_pattern, end_pattern
    static std::vector<std::pair<aipubt::GraphMatcherPattern,aipubt::GraphMatcherPattern>> patterns =
    {
        {
            {//begin_pattern
                {
                    // {"reshape", aipubt::OpType_Reshape},
                    {"fc1", aipubt::OpType_FullyConnected},
                    {"fc2", aipubt::OpType_FullyConnected},
                    {"add_fc", aipubt::OpType_Eltwise},
                    {"bias", aipubt::OpType_Constant},
                    {"add_bias", aipubt::OpType_Eltwise},
                    {"split", aipubt::OpType_Split},
                    {"act_i", aipubt::OpType_Activation},
                    {"act_f", aipubt::OpType_Activation},
                    {"act_c", aipubt::OpType_Activation},
                    {"act_o", aipubt::OpType_Activation},
                },
                {
                    // {"reshape", "fc1"},
                    {"fc1", "add_fc"},
                    {"fc2", "add_fc"},
                    {"add_fc", "add_bias"},
                    {"bias", "add_bias"},
                    {"add_bias", "split"},
                    {"split", "act_i"},
                    {"split", "act_f"},
                    {"split", "act_c"},
                    {"split", "act_o"},
                }
            },
            {//end_pattern
                {
                    {"mul_f_cpre", aipubt::OpType_Eltwise},
                    {"mul_i_c", aipubt::OpType_Eltwise},
                    {"cout", aipubt::OpType_Eltwise},
                    {"act_cout", aipubt::OpType_Activation},
                    {"hout", aipubt::OpType_Eltwise},
                },
                {
                    {"mul_f_cpre", "cout"},
                    {"mul_i_c", "cout"},
                    {"cout", "act_cout"},
                    {"act_cout", "hout"},
                }
            },
        },

        {
            {//begin_pattern
                {
                    // {"reshape", aipubt::OpType_Reshape},
                    {"fc1", aipubt::OpType_FullyConnected},
                    {"fc2", aipubt::OpType_FullyConnected},
                    {"add_fc", aipubt::OpType_Add},
                    {"bias", aipubt::OpType_Constant},
                    {"add_bias", aipubt::OpType_Add},
                    {"split", aipubt::OpType_Split},
                    {"act_i", aipubt::OpType_Activation},
                    {"act_f", aipubt::OpType_Activation},
                    {"act_c", aipubt::OpType_Activation},
                    {"act_o", aipubt::OpType_Activation},
                },
                {
                    // {"reshape", "fc1"},
                    {"fc1", "add_fc"},
                    {"fc2", "add_fc"},
                    {"add_fc", "add_bias"},
                    {"bias", "add_bias"},
                    {"add_bias", "split"},
                    {"split", "act_i"},
                    {"split", "act_f"},
                    {"split", "act_c"},
                    {"split", "act_o"},
                }
            },
            {//end_pattern
                {
                    {"mul_f_cpre", aipubt::OpType_Mul},
                    {"mul_i_c", aipubt::OpType_Mul},
                    {"cout", aipubt::OpType_Add},
                    {"act_cout", aipubt::OpType_Activation},
                    {"hout", aipubt::OpType_Mul},
                },
                {
                    {"mul_f_cpre", "cout"},
                    {"mul_i_c", "cout"},
                    {"cout", "act_cout"},
                    {"act_cout", "hout"},
                }
            },
        },
    };

    for(auto& pattern: patterns)
    {
        do_fuse_lstm_cell(g, pattern.first, pattern.second);
    }
}

}; // namespace qtlibn
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
 * =====================================================================================
 *
 *       Filename:  split_matmul_for_large_outc.cpp
 *
 *    Description:  split matmul if output channel is large
 *
 *        Version:  1.0
 *        Created:  11/11/21 14:31:36
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */

 #include "builder/lowering/lowering.h"
 #include "builder/target.h"
 #include "core/core.h"
#include <algorithm>
#include <functional>
#include <map>

namespace aipubt
{

void split_matmul_for_large_outc(GraphPtr graph)
{
    auto criterion = [](NodePtr node) -> bool {
        return (node->type() == OpType_MatMul) && (node->inputs().size() == 2) && (node->outputs().size() == 1) &&
               (node->inputs()[0]->shape().dim() == 4) && (node->inputs()[1]->shape().dim() == 4) &&
               (node->outputs()[0]->shape().dim() == 4);
    };

    auto need_split_matmul = [](GraphPtr graph, NodePtr node, int max_outc) -> bool {
        auto arch = aipubt::get_target()->isainfo().type;
        if ((arch >= aipubt::ISA_ZYv6) && !node->params()["trans_b"])
        {
            return false;
        }
        auto output_shape = node->outputs()[0]->shape();
        auto outc = output_shape[3];
        return outc > max_outc;
    };

    auto split_matmul = [need_split_matmul](GraphPtr graph, NodePtr node, int max_outc) {
        if (!need_split_matmul(graph, node, max_outc))
        {
            return;
        }
        auto output_shape = node->outputs()[0]->shape();
        auto outc = output_shape[3];
        int split_num = (outc + max_outc - 1) / max_outc;
        if (split_num < 2)
        {
            return;
        }

        // split weight node
        auto weight_input = node->inputs()[1];
        NodePtr split_node = Node::create(weight_input->name() + "/spliter", OpType_Split);

        graph->add_node(split_node);
        ADD_LOCATION_BY_NODE(split_node, node);
        int outc_axis = 3;
        if (node->params()["trans_b"])
        {
            outc_axis = 2;
        }
        split_node->params()["axis"] = outc_axis;
        std::vector<int> splitted_channels;
        std::vector<TensorPtr> splitted_weight_tensors;
        for (auto i = 0; i < split_num; i++)
        {
            int channel = max_outc;
            if (i == split_num - 1)
            {
                channel = outc - max_outc * (split_num - 1);
            }
            splitted_channels.push_back(channel);
            split_node->params()["splits"].push_back(channel);

            auto split_out_shape = weight_input->shape();
            split_out_shape[outc_axis] = channel;
            auto split_output = std::make_shared<Tensor>(split_node->name() + "/output/" + std::to_string(i),
                                                         split_out_shape, weight_input->type());
            split_output->quantization() = weight_input->quantization();
            splitted_weight_tensors.push_back(split_output);
            split_node->add_output(split_output);
        }
        split_node->add_input(weight_input);

        // concat matmul output
        auto concat_node = Node::create(node->name() + "_concat", OpType_Concat);

        graph->add_node(concat_node);
        ADD_LOCATION_BY_NODE(concat_node, node);
        concat_node->params()["axis"] = 3;
        for (auto i = 0; i < split_num; i++)
        {
            concat_node->params()["scale_value"].push_back(NodeParamValue(1));
            concat_node->params()["scale_type"].push_back(Tensor::Dtype::UINT8);
            concat_node->params()["shift_value"].push_back(NodeParamValue(0));
            concat_node->params()["shift_type"].push_back(Tensor::Dtype::UINT8);
        }

        // split matmul node
        for (auto i = 0; i < split_num; i++)
        {
            auto matmul_node = Node::create(node->name() + "/splitted/" + std::to_string(i), OpType_MatMul);

            graph->add_node(matmul_node);
            ADD_LOCATION_BY_NODE(matmul_node, node);
            matmul_node->params().insert(node->params().begin(), node->params().end());
            matmul_node->add_input(node->inputs()[0]);
            matmul_node->add_input(splitted_weight_tensors[i]);
            auto matmul_out_shape = node->outputs()[0]->shape();
            matmul_out_shape.c() = splitted_channels[i];
            auto matmul_output =
                std::make_shared<Tensor>(matmul_node->name() + "/output", matmul_out_shape, node->outputs()[0]->type());
            matmul_output->quantization() = node->outputs()[0]->quantization();
            matmul_node->add_output(matmul_output);
            concat_node->add_input(matmul_output);
        }
        for (auto &output : node->outputs())
        {
            concat_node->add_output(output);
        }

        // remove original matmul node
        graph->remove_node(node);
    };

    auto max_outc = get_aiff_max_out_c();
    if (max_outc <= 1)
    {
        WARN() << "split_matmul_for_large_outc doesn't support target" << aipubt::get_target()->name();
    }

    auto nodes = graph->nodes();
    for (auto node : nodes)
    {
        if (!criterion(node))
        {
            continue;
        }

        split_matmul(graph, node, max_outc);
    }
}
} // namespace aipubt

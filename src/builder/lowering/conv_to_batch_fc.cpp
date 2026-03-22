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
 *       Filename:  conv_to_batch_fc.cpp
 *
 *    Description:  convert Conv (kernel=stride && large kernel) to FC for precision purposes.
 *    e.g. Conv in vit: [1,384,384,3] -> [1,12,12,768]
 *    - kernel_x=kernel_y=32
 *    - stride_x=stride_y=32
 *    The split_conv_kernel pass will split it into multiple conv+eltwise, which reduces the precision.
 *    The conv_to_batch_fc pass will convert this Conv node into the following nodes to solve the problem:
 *    - Reshape: [1,384,384,3] -> [1,12,32,12,32,3]
 *    - Transpose: [1,12,32,12,32,3] -> [1,12,12,32,32,3]
 *    - Reshape: [1,12,12,32,32,3] -> [12x12,32x32x3]=[144,3072]
 *    - FC: [144,3072] -> [144,768]
 *    - Reshape: [144,768] -> [1,12,12,768]
 *    Note that although the FC node has a larger batch (more AIFF descriptor load/execution), the batch_FC_to_conv
 *    pass will optimize it to an equivalent 1x1 Conv node so that we only need one AIFF descriptor load/execution.
 *
 *        Version:  1.0
 *        Created:  11/10/22 17:40:06
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
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <map>
#include <set>

namespace aipubt
{

void conv_to_batch_fc(GraphPtr graph)
{
    auto criterion = [](NodePtr node) -> bool {
        std::map<aipubt::ISAType, int> max_kernel_size_map = {
            {aipubt::ISA_ZY, 6},
            {aipubt::ISA_ZYv2, 6},
            {aipubt::ISA_ZYv3, 7},
            {aipubt::ISA_ZYv5, 7},
            {aipubt::ISA_ZYv6, 7},
        };

        if (node->type() != OpType_Convolution)
        {
            return false;
        }

        auto arch = aipubt::get_target()->isainfo().type;
        int max_kernel_size = 1;
        if (max_kernel_size_map.find(arch) == max_kernel_size_map.end())
        {
            WARN() << "conv_to_batch_fc doesn't support target" << aipubt::get_target()->name();
            return false;
        }
        max_kernel_size = max_kernel_size_map[arch];

        auto input_shape = node->inputs()[0]->shape();
        auto output_shape = node->outputs()[0]->shape();
        int kernel_x = node->params()["kernel_x"];
        int kernel_y = node->params()["kernel_y"];
        int stride_x = node->params()["stride_x"];
        int stride_y = node->params()["stride_y"];
        return (node->type() == OpType_Convolution) &&
               (kernel_x == stride_x) &&
               (kernel_y == stride_y) &&
               ((kernel_x > max_kernel_size) || (kernel_y > max_kernel_size)) &&
               (input_shape.w() == output_shape.w() * kernel_x) &&
               (input_shape.h() == output_shape.h() * kernel_y) &&
               (node->params()["dilation_x"] == 1) &&
               (node->params()["dilation_y"] == 1) &&
               (node->params()["group"] == 1) &&
               (node->params()["pad_left"] == 0) &&
               (node->params()["pad_right"] == 0) &&
               (node->params()["pad_top"] == 0) &&
               (node->params()["pad_bottom"] == 0);
    };

    auto nodes = graph->nodes();
    for (auto node : nodes)
    {
        if (!criterion(node))
        {
            continue;
        }

        auto conv_input = node->inputs()[0];
        auto conv_output = node->outputs()[0];
        auto input_shape = conv_input->shape();
        auto output_shape = conv_output->shape();
        auto kernel_x = node->params()["kernel_x"];
        auto kernel_y = node->params()["kernel_y"];

        TensorShape transpose_input_shape(input_shape.n(), output_shape.h(), kernel_y, output_shape.w(), kernel_x, input_shape.c_total());
        transpose_input_shape.set_layout(DataLayout_Flat);
        auto transpose_reshape_node = ops::reshape(conv_input, transpose_input_shape)->node();

        graph->add_node(transpose_reshape_node);
        ADD_LOCATION_BY_NODE(transpose_reshape_node, node);

        std::vector<uint32_t> perm = {0, 1, 3, 2, 4, 5};
        auto transpose_node = ops::transpose(transpose_reshape_node->outputs()[0], perm)->node();

        graph->add_node(transpose_node);
        ADD_LOCATION_BY_NODE(transpose_node, node);

        auto transpose_output = transpose_node->outputs()[0];
        auto transpose_output_shape = transpose_output->shape();
        TensorShape fc_input_shape(transpose_output_shape[0] * transpose_output_shape[1] * transpose_output_shape[2],
                                   transpose_output_shape[3] * transpose_output_shape[4] * transpose_output_shape[5]);
        fc_input_shape.set_layout(DataLayout_Flat);
        auto fc_in_reshape_node = ops::reshape(transpose_output, fc_input_shape)->node();

        graph->add_node(fc_in_reshape_node);
        ADD_LOCATION_BY_NODE(fc_in_reshape_node, node);

        auto node_name = graph->get_valid_node_name(node->name() + "_fc");
        NodePtr fc_node = Node::create(node_name, OpType_FullyConnected);
        fc_node->add_input(fc_in_reshape_node->outputs()[0]);
        TensorShape fc_output_shape = fc_input_shape;
        fc_output_shape[1] = output_shape.c_total();
        fc_output_shape.set_layout(DataLayout_Flat);
        auto tensor_name = graph->get_valid_tensor_name(fc_node->name());
        TensorPtr fc_output = std::make_shared<Tensor>(tensor_name, fc_output_shape, conv_output->type());
        fc_output->quantization() = conv_output->quantization();
        fc_node->add_output(fc_output);

        graph->add_node(fc_node);
        ADD_LOCATION_BY_NODE(fc_node, node);

        auto conv_weight = node->constants()["weights"];
        auto fc_weight = ComputeTensor::zeros(conv_weight->shape(), conv_weight->type());
        fc_weight->shape().set_layout(DataLayout_Flat);
        fc_weight[{{0, -1}, {0, -1}, {0, -1}, {0, -1}}] = conv_weight;
        fc_weight = fc_weight.reshape(fc_output_shape[1], fc_input_shape[1]);
        fc_weight->make_memory_continuous();
        fc_node->constants()["weights"] = fc_weight;
        for (auto &kv : node->constants())
        {
            if (kv.first != "weights")
            {
                fc_node->constants()[kv.first] = node->constants()[kv.first];
            }
        }

        std::set<std::string> excluded_conv_params = {
            "kernel_x",
            "kernel_y",
            "dilation_x",
            "dilation_y",
            "stride_x",
            "stride_y",
            "group",
            "pad_left",
            "pad_right",
            "pad_top",
            "pad_bottom",
        };
        for (auto &kv : node->params())
        {
            if (excluded_conv_params.find(kv.first) == excluded_conv_params.end())
            {
                fc_node->params()[kv.first] = node->params()[kv.first];
            }
        }
        fc_node->set_attr("per_channel_quant", node->get_attr("per_channel_quant"));

        auto fc_out_reshape_node = ops::reshape(fc_node->outputs()[0], output_shape)->node();

        graph->add_node(fc_out_reshape_node);
        ADD_LOCATION_BY_NODE(fc_out_reshape_node, node);
        fc_out_reshape_node->remove_output(0);
        fc_out_reshape_node->add_output(conv_output, 0);

        graph->remove_node(node);
    }
}

} // namespace aipubt

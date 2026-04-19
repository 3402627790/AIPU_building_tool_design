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
 * @Date: 2021-07-13 10:55:59
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:34:59
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    // layerlib doesn't support non-4D input
    std::vector<TensorPtr> channel_shuffle(TensorPtr input,
                                           int32_t groups,
                                           int32_t split)
    {
        if (split < 1 || input->shape()[input->shape().dim() - 1] % split != 0)
        {
            ERROR() << "Please make sure split is positive integer and divisible by input channel.";
            return {};
        }

        if (input->shape().dim() > 4)
        {
            ERROR() << "Doesn't support > 4D input.";
            return {};
        }

        TensorShape refer_outshape = input->shape();
        refer_outshape[refer_outshape.dim() - 1] = refer_outshape[refer_outshape.dim() - 1] / split;
        TensorPtr new_input = input;
        if (input->shape().dim() != 4)
        {
            new_input = reshape(input, reshape_to_4d(TensorShape(input->shape())));
        }

        GraphPtr g = new_input->node()->graph();
        auto node = Node::create(g->get_valid_node_name("channel_shuffle"), OpType_ChannelShuffle);
        auto &params = node->params();
        params["group"] = groups;
        params["splits"] = split;
        node->add_input(new_input);
        g->add_node(node);

        TensorShape output_shape = new_input->shape();
        output_shape[output_shape.dim() - 1] = output_shape[output_shape.dim() - 1] / split;

        std::vector<TensorPtr> outputs;
        for (int32_t i = 0; i < split; ++i)
        {
            auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("channel_shuffle_" + std::to_string(i)), output_shape);
            add_tensor_info(new_input, output);
            node->add_output(output);
            TensorPtr new_output = output;
            if (output->shape().dim() != input->shape().dim())
            {
                new_output = reshape(output, refer_outshape);
            }
            outputs.push_back(new_output);
        }
        node->infer_shape(true);
        quantize(node);

        return outputs;
    }

}; // namespace ops
}; // namespace aipubt

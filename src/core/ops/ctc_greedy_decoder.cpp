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
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:33:23
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr ctc_greedy_decoder(TensorPtr input,
                                 TensorPtr seq_len,
                                 bool merge_repeated,
                                 const TensorShape &output_shape)
    {
        if (input->shape().dim() != 3)
        {
            ERROR() << "input rank should be 3!";
            return nullptr;
        }
        if (seq_len->shape().dim() != 1 || seq_len->shape()[0] != input->shape()[0])
        {
            ERROR() << "seq_len rank should be 1, and seq_len length should be equal to input[0]!";
            return nullptr;
        }

        auto g = input->node()->graph();
        // new_shape.set_layout(input->shape().layout());
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("ctc_greedy_decoder"), output_shape);
        if (input->quantization().empty())
        {
            output->set_type(Tensor::INT32);
        }
        else
        {
            output->set_type(Tensor::UINT16);
            Quantization quant(1.0, 0, 16, input->quantization().scheme());
            output->quantization() = quant;
        }

        auto node = Node::create(g->get_valid_node_name("ctc_greedy_decoder"), OpType_CTCGreedyDecoder);
        node->add_input(input);
        node->add_input(seq_len);
        node->add_output(output);
        node->params()["merge_repeated"] = merge_repeated ? "true" : "false";

        g->add_node(node);
        quantize(node);
        return output;
    }
}; //namespace ops
}; // namespace aipubt
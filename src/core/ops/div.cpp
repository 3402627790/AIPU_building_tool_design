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
 * @Last Modified time: 2022-08-13 11:33:03
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    // layerlib div has already supported multibroadcast
    TensorPtr div(TensorPtr input1, TensorPtr input2, const Quantization &quant)
    {
        auto new_inputs = do_multibroadcast({input1, input2});
        if (new_inputs.empty())
        {
            ERROR() << "inputs don't support multibroadcast!";
            return nullptr;
        }
        if (Tensor::dtype_bits(input1->type()) != Tensor::dtype_bits(input2->type()))
        {
            ERROR() << "inputs should have same bitwidth!";
            return nullptr;
        }

        auto g = input1->node()->graph();
        TensorShape output_shape(new_inputs[0]->shape());
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("div"), output_shape);
        add_tensor_info(new_inputs[0], output, quant);

        auto node = Node::create(g->get_valid_node_name("div"), OpType_Div);
        node->add_input(new_inputs[0]);
        node->add_input(new_inputs[1]);
        node->add_output(output);
        g->add_node(node);
        quantize(node);
        return output;
    }

}; //namespace ops
}; // namespace aipubt
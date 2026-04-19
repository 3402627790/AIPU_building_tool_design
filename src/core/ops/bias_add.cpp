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
 * @Last Modified time: 2022-08-13 23:26:26
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr bias_add(TensorPtr input,
                       TensorPtr bias,
                       const Quantization &quant)
    {
        GraphPtr g = input->node()->graph();
        TensorShape output_shape = input->shape();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("bias_add"), output_shape);
        add_tensor_info(input, output, quant);

        auto node = Node::create(g->get_valid_node_name("bias_add"), OpType_BiasAdd);
        node->add_input(input);
        node->add_output(output);
        node->constants()["biases"] = bias;

        g->add_node(node);
        quantize(node);
        return output;
    }
}; //namespace ops
}; // namespace aipubt
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
 * @Last Modified time: 2022-08-13 11:32:07
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr instance_normalization(TensorPtr input,
                                     TensorPtr gamma,
                                     TensorPtr beta,
                                     float epsilon,
                                     const Quantization &quant)
    {
        TensorPtr new_input = input;
        if (input->shape().dim() != 4)
        {
            auto new_shape = reshape_to_4d(input->shape());
            new_input = reshape(input, new_shape);
        }
        auto in_dtype_size = Tensor::dtype_size(input->type());
        auto w_dtype_size = Tensor::dtype_size(gamma->type());
        if (in_dtype_size != w_dtype_size)
        {
            ERROR() << "input data type width should be same with gamma data type width!";
            return nullptr;
        }
        if (input->shape().c() != gamma->shape()[0] ||
            input->shape().c() != beta->shape()[0])
        {
            ERROR() << "gamma/beta dimension size should be equal to input[c]!";
            return nullptr;
        }

        auto g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("instance_norm"), new_input->shape());
        add_tensor_info(new_input, output, quant);

        auto node = Node::create(g->get_valid_node_name("instance_norm"), OpType_InstanceNorm);
        node->add_input(new_input);
        node->add_output(output);
        node->params()["epsilon"] = NodeParamValue(epsilon);
        node->constants()["weights"] = gamma;
        node->constants()["biases"] = beta;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

}; //namespace ops
}; // namespace aipubt
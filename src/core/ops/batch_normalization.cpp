
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
 * @Date: 2021-07-07 16:26:26
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:35:23
 */
#include <cmath>
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr batch_normalization(TensorPtr input,
                                  TensorPtr weights,
                                  TensorPtr biases,
                                  int32_t axis,
                                  float epsilon,
                                  const Quantization &quant)
    {
        if (axis < 0)
        {
            axis = axis + input->shape().dim();
        }

        if (axis <= 0 || axis >= input->shape().dim()) // axis cannot be 0
        {
            ERROR() << "axis should be in [-rank, rank) of input!";
            return nullptr;
        }
        if (input->shape().dim() < 2 || input->shape().dim() > 4)
        {
            ERROR() << "invalid input rank, should be in [2,4]";
            return nullptr;
        }

        auto g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("batch_norm"), input->shape());
        add_tensor_info(input, output, quant);

        auto node = Node::create(g->get_valid_node_name("batch_norm"), OpType_BatchNorm);
        node->add_input(input);
        node->add_output(output);
        node->params()["axis"] = axis;
        node->params()["epsilon"] = NodeParamValue(epsilon);
        node->constants()["weights"] = weights;
        node->constants()["biases"] = biases;

        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

}; // namespace ops
}; // namespace aipubt

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
 * @Author: Alex Zheng
 * @Date: 2021-04-14 11:04:33
 * @Last Modified by: Alex Zheng
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
    TensorPtr group_normalization(TensorPtr input,
                                  TensorPtr gamma,
                                  TensorPtr beta,
                                  int32_t axis,
                                  int32_t group,
                                  float epsilon,
                                  const Quantization &quant)
    {
        TensorPtr new_input = input;
        if (input->shape().dim() < 3 || input->shape().dim() > 4)
        {
            ERROR() << "Input rank should be 3 or 4!";
            return nullptr;
        }

        if ((input->shape()[input->shape().dim() - 1]) % group != 0)
        {
            ERROR() << "Input channels must be divisible by group!";
            return nullptr;
        }

        if (gamma->shape().dim() != 1 || beta->shape().dim() != 1)
        {
            ERROR() << "Gamma & beta rank should be 1!";
            return nullptr;
        }

        axis = axis < 0 ? axis + input->shape().dim() : axis;
        if (axis < 0 || axis >= input->shape().dim())
        {
            ERROR() << "Axis is invalid, it should be in [-rank, rank)!";
            return nullptr;
        }

        if (input->shape().dim() == 3)
        {
            auto new_shape = reshape_to_4d(input->shape());
            new_input = reshape(input, new_shape);
            axis += 1;
        }

        if (gamma->shape()[0] != new_input->shape()[axis] ||
            beta->shape()[0] != new_input->shape()[axis])
        {
            ERROR() << "gamma or beta shape cannot match input! \n"
                    << "input[" << axis << "]:" << new_input->shape()[axis]
                    << "gamma[" << 0 << "]:" << gamma->shape()[0]
                    << "beta[" << 0 << "]:" << beta->shape()[0];
            return nullptr;
        }

        auto g = new_input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("group_norm"), new_input->shape());
        add_tensor_info(new_input, output, quant);

        auto node = Node::create(g->get_valid_node_name("group_norm"), OpType_GroupNorm);
        node->add_input(new_input);
        node->add_output(output);
        auto &params = node->params();
        params["axis"] = axis;
        params["group"] = group;
        params["epsilon"] = NodeParamValue(epsilon);
        node->constants()["weights"] = gamma;
        node->constants()["biases"] = beta;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

}; //namespace ops
}; // namespace aipubt
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
 * @Last Modified time: 2022-08-13 11:31:27
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr matmul(TensorPtr a,
                     TensorPtr b,
                     bool adj_a,
                     bool adj_b,
                     float fused_multiplier,
                     const Quantization &quant)
    {
        GraphPtr g = a->node()->graph();
        auto a_shape = a->shape(), b_shape = b->shape();
        // Check the shape of two input tensors
        int32_t a_dim = a_shape.dim();
        int32_t b_dim = b_shape.dim();
        if (a_dim == 0 || b_dim == 0)
        {
            ERROR() << "Multiplication by scalar is unsupported!";
            return nullptr;
        }
        if (a_dim < 2 && b_dim < 2)
        {
            ERROR() << "Multiplication of two 1-D arrays is unsupported!";
            return nullptr;
        }
        // Valid inputs:
        // 1) the shapes except the last two dims are same, like (2,3,5,6) and (2,3,6,5)
        // 2) the shapes except the last two dims are different, but can be broadcast,
        //    like (2,3,5,6) and (3,6,5), or (2,3,5,6) and (1,3,6,5).
        int32_t min_dim = (a_dim < b_dim) ? a_dim : b_dim;
        for (int32_t i = 0; i < min_dim - 2; i++)
        {
            int32_t a_index = a_dim - 3 - i;
            int32_t b_index = b_dim - 3 - i;
            if (a_shape[a_index] != b_shape[b_index] && a_shape[a_index] != 1 && b_shape[b_index] != 1)
            {
                ERROR() << "operands could not be broadcast. shape[" << a_index
                        << "] of the first input should be equal to shape[" << b_index
                        << "] of the second input, or one of them is 1";
                return nullptr;
            }
        }

        int32_t max_dim = (a_dim > b_dim) ? a_dim : b_dim;
        std::vector<int32_t> output_dim;
        int32_t a_start_index = 0, b_strat_index = 0, delta_dim = 0;
        if (a_dim > b_dim)
        {
            a_start_index = a_dim - b_dim;
            delta_dim = a_start_index;
            for (int32_t i = 0; i < delta_dim; i++)
            {
                output_dim.push_back(a_shape[i]);
            }
        }
        else if (a_dim < b_dim)
        {
            b_strat_index = b_dim - a_dim;
            delta_dim = b_strat_index;
            for (int32_t i = 0; i < delta_dim; i++)
            {
                output_dim.push_back(b_shape[i]);
            }
        }
        for (int32_t i = 0; i < max_dim - delta_dim - 2; i++)
        {
            int32_t a_i_shape = a_shape[a_start_index + i], b_i_shape = b_shape[b_strat_index + i];
            output_dim.push_back(((a_i_shape > b_i_shape) ? a_i_shape : b_i_shape));
        }

        // Change the shape of inputs if transpose is needed before multiplication
        if (adj_a)
        {
            int32_t temp_shape = a_shape[-1];
            a_shape[-1] = a_shape[-2];
            a_shape[-2] = temp_shape;
        }
        if (adj_b)
        {
            int32_t temp_shape = b_shape[-1];
            b_shape[-1] = b_shape[-2];
            b_shape[-2] = temp_shape;
        }

        if (a_shape[-1] == b_shape[-2])
        {
            output_dim.push_back(a_shape[-2]);
            output_dim.push_back(b_shape[-1]);
        }
        else
        {
            ERROR() << "Shape does not match!";
            return nullptr;
        }

        TensorShape output_shape(output_dim);
        output_shape.set_layout(a->shape().layout());
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("matmul"), output_shape);
        auto node = Node::create(g->get_valid_node_name("matmul"), OpType_MatMul);
        add_tensor_info(a, output, quant);

        if (!quant.empty() && output->quantization().aipu_dtype() != Tensor::VOID)
        {
            output->set_type(output->quantization().aipu_dtype());
        }

        node->add_input(a);
        node->add_input(b);
        node->add_output(output);
        auto &params = node->params();
        params["trans_a"] = adj_a;
        params["trans_b"] = adj_b;
        params["fused_multiplier"] = fused_multiplier;

        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }
}; // namespace ops
}; // namespace aipubt
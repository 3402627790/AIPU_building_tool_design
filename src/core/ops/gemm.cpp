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
 * @Date: 2022-08-18 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2022-08-18 11:04:33
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr gemm(TensorPtr A,
                   TensorPtr B,
                   TensorPtr C,
                   const float alpha,
                   const bool trans_a,
                   const bool trans_b,
                   const float beta,
                   const Quantization &quant)
    {

        if ((A->shape().dim() != 2) || (B->shape().dim() != 2))
        {
            ERROR() << "Only Support 2-d shape."
                    << " A:" << A->shape().dim()
                    << ",B:" << B->shape().dim();
            return nullptr;
        }
        int32_t dim_M = A->shape()[0];
        int32_t dim_K_A = A->shape()[1];
        int32_t dim_K_B = B->shape()[0];
        int32_t dim_N = B->shape()[1];
        if (trans_a)
        {
            dim_M = A->shape()[1];
            dim_K_A = A->shape()[0];
        }

        if (trans_b)
        {
            dim_K_B = B->shape()[1];
            dim_N = B->shape()[0];
        }

        if (dim_K_A != dim_K_B)
        {
            ERROR() << "The dimensions used to multiply must be the same.";
            return nullptr;
        }
        if (C != nullptr)
        {
            if ((C->shape().dim() != 2) ||
                (C->shape()[0] != dim_M && C->shape()[0] != 1) ||
                (C->shape()[1] != dim_N))
            {
                ERROR() << "The C tensor shape is not right,should be(" << dim_M << ", " << dim_N << ") or (1, " << dim_N << ").";
                return nullptr;
            }
        }

        auto g = A->node()->graph();
        TensorShape output_shape(dim_M, dim_N);
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("gemm"), output_shape);
        auto node = Node::create(g->get_valid_node_name("gemm"), OpType_Gemm);
        add_tensor_info(A, output, quant);

        node->add_input(A);
        node->add_input(B);
        if (C != nullptr)
            node->add_input(C);
        node->add_output(output);
        auto &params = node->params();
        params["alpha"] = alpha;
        params["beta"] = beta;
        params["trans_a"] = trans_a;
        params["trans_b"] = trans_b;
        g->add_node(node);
        quantize(node);
        return output;
    }
}; // namespace ops
}; // namespace aipubt
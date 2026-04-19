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
    TensorPtr in_topk(TensorPtr input1, TensorPtr input2, int32_t k)
    {
        if (Tensor::is_float(input2->type()))
        {
            ERROR() << "input 2 not support Float type.";
            return nullptr;
        }

        GraphPtr g = input1->node()->graph();
        u_int32_t batch_size = input1->shape().n();

        TensorShape output_shape(batch_size);
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("in_topk"), output_shape);
        auto node = Node::create(g->get_valid_node_name("in_topk"), OpType_InTopK);
        output->set_type(Tensor::UINT8);
        Quantization quant(1.0, 0, 8);
        output->quantization() = quant;

        node->add_input(input1);
        node->add_input(input2);
        node->add_output(output);
        auto &params = node->params();
        params["k"] = k;
        g->add_node(node);
        quantize(node);
        return output;
    }
}; // namespace ops
}; // namespace aipubt
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
 * @Last Modified time: 2022-04-29 11:30:37
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtrList nonzero(TensorPtr input)
    {
        auto g = input->node()->graph();
        TensorShape Y1_shape(input->shape().dim(), input->shape().size());
        TensorShape Y2_shape(1);
        Tensor::Dtype Y1_dtype = Tensor::UINT8;
        if (input->shape().size() < 256)
        {
            Y1_dtype = Tensor::UINT8;
        }
        else if (input->shape().size() < 65536)
        {
            Y1_dtype = Tensor::UINT16;
        }
        else
        {
            Y1_dtype = Tensor::INT32;
        }
        auto Y1 = std::make_shared<Tensor>(g->get_valid_tensor_name("nonzero/Y1"), Y1_shape, Y1_dtype);
        auto Y2 = std::make_shared<Tensor>(g->get_valid_tensor_name("nonzero/Y2"), Y2_shape, Tensor::INT32);
        auto node = Node::create(g->get_valid_node_name("nonzero"), OpType_NonZero);
        node->add_input(input);
        node->add_output(Y1);
        node->add_output(Y2);
        g->add_node(node);
        return {Y1, Y2};
    }
} // namespace ops
} // namespace aipubt
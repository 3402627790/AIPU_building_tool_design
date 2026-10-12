//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
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
    TensorPtr nonzero(TensorPtr input)
    {
        auto g = input->node()->graph();
        TensorShape out_shape(input->shape().dim(), input->shape().size());
        Tensor::Dtype out_dtype = Tensor::UINT8;
        if (input->shape().size() < 256)
        {
            out_dtype = Tensor::UINT8;
        }
        else if (input->shape().size() < 65536)
        {
            out_dtype = Tensor::UINT16;
        }
        else
        {
            out_dtype = Tensor::UINT32;
        }
        auto out = std::make_shared<Tensor>(g->get_valid_tensor_name("nonzero/Y1"), out_shape, out_dtype);
        auto node = Node::create(g->get_valid_node_name("nonzero"), OpType_NonZero);
        node->add_input(input);
        node->add_output(out);
        g->add_node(node);
        return out;
    }
} // namespace ops
} // namespace aipubt
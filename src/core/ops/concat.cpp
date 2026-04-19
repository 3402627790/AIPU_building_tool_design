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
 * @Last Modified time: 2022-08-13 11:34:48
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr concat(const std::vector<TensorPtr> &tensors,
                     int32_t axis_p,
                     const Quantization &quant)
    {
        // TODO check the shape~
        auto g = tensors[0]->node()->graph();
        auto node = Node::create(g->get_valid_node_name("concat"), OpType_Concat);
        auto shape = tensors[0]->shape();
        int32_t axis = axis_p;
        if (axis < 0)
        {
            axis += shape.dim();
        }
        auto t0 = tensors[0];
        for (auto t : tensors)
        {
            if (t->shape().dim() != t0->shape().dim())
            {
                ERROR() << "Shape mismatch. t0 shape:" << t0->shape()
                        << ",t shape:" << t->shape();
                return nullptr;
            }
            for (int i = 0; i < t->shape().dim(); ++i)
            {
                if (i == axis)
                    continue;
                if ((!t0->shape().symbol()[i]->is_constant() || !t->shape().symbol()[i]->is_constant()) &&
                    !t0->shape().symbol()[i]->equal(t->shape().symbol()[i]))
                {
                    ERROR() << "Shape symbol mismatch. t0 shape:" << t0->shape().symbol().to_string()
                            << ", t shape:" << t->shape().symbol().to_string();
                    return nullptr;
                }
            }
        }
        shape[axis] = 0;
        for (auto t : tensors)
        {
            shape[axis] += t->shape()[axis];
            node->add_input(t);
        }
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("concat"), shape);
        add_tensor_info(tensors[0], output, quant);

        node->add_output(output);
        node->params()["axis"] = axis;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

}; // namespace ops
}; // namespace aipubt
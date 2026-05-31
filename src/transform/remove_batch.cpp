//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-03-30 13:55:50
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-30 16:35:20
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{

bool remove_batch(GraphPtr g0)
{
    auto g = g0->clone();
    int batch = 0;
    auto input_tensors = g->get_input_tensors();
    for (uint32_t i = 0; i < input_tensors.size(); ++i)
    {
        if (i == 0)
        {
            batch = input_tensors[i]->shape()[0];
        }
        if (batch != input_tensors[i]->shape()[0])
        {
            batch = 0;
            break;
        }
    }
    if (batch == 0)
    {
        DEBUG() << "Can not remove batch, bottom nodes' shape batch are different.";

        return false;
    }
    auto bottom = g->get_bottom_nodes();
    // initialize bottome nodes' batch
    for (auto n : bottom)
    {
        if (n->type() == OpType_Input)
        {
            n->outputs()[0]->shape()[0] = 1;
        }
        else if (n->type() == OpType_Constant)
        {
            ComputeTensor w = n->constants()["weights"];
            w = w[{{0}}];
            w->make_memory_continuous();
            n->constants()["weights"] = w;
            n->outputs()[0]->shape()[0] = 1;
        }
        else
        {
            // only support input/const as bottom nodes
            DEBUG() << "Can not remove batch, only support Input and Constant as bottom node, but got type " << n->type() << "\n"
                    << n->to_string();

            return false;
        }
    }
    g->topological_sort();
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (n->type() == OpType_Input || n->type() == OpType_Constant)
        {
            continue;
        }
        if (n->type() == OpType_Reshape)
        {
            TensorShape shape = n->params()["shape"];
            if (shape[0] != batch) // reshape batch mismatch
            {
                DEBUG() << "Can not remove batch, reshape's batch is not match graph batch: " << shape << " global batch=" << batch;

                return false;
            }
            shape[0] = 1;
            n->params()["shape"] = shape;
            n->outputs()[0]->shape() = shape;
            continue;
        }
        auto o_shapes = n->infer_shape();
        if (o_shapes.size() == 0)
        {
            DEBUG() << "Can not remove batch, node " << n->name() << " infer shape failed.\n"
                    << n->to_string();
            return false;
        }
        for (uint32_t i = 0; i < o_shapes.size(); ++i)
        {
            n->outputs()[i]->shape() = o_shapes[i];
        }
    }
    g->clone(g0);
    return true;
}

}; // namespace aipubt
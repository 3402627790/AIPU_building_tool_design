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
 * @Author: Alex Zheng
 * @Date: 2026-08-20 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time:  2026-08-20 17:14:20
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt::ops
{

TensorPtr moe_finalize(TensorPtrList inputs,
                       const TensorShape output_shape)
{
    auto g = inputs[0]->node()->graph();
    // check output shape
    if (output_shape.dim() == 3)
    {
        if (output_shape[0] * output_shape[1] != inputs[0]->shape()[0] || output_shape[2] != inputs[0]->shape()[1])
        {
            ERROR() << "output shape error";
            return nullptr;
        }
    }
    else
    {
        ERROR() << "output shape error";
        return nullptr;
    }

    auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("moe_finalize"), output_shape);
    auto node = Node::create(g->get_valid_node_name("moe_finalize"), OpType_MoEFinalize);
    add_tensor_info(inputs[0], output);
    for (auto input : inputs)
    {
        node->add_input(input);
    }
    node->add_output(output);

    g->add_node(node);
    node->infer_shape(true);
    quantize(node);
    return output;
}

} // namespace aipubt::ops
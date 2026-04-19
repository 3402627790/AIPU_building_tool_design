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
 * @Date: 2025-07-11 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2025-07-11 11:04:33
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtrList dummy(TensorPtrList inputs)
    {
        auto g = inputs[0]->node()->graph();
        auto node = Node::create(g->get_valid_node_name("dummpy"), OpType_Dummy);
        TensorPtrList outputs;
        int i = 0;
        for (auto inp : inputs)
        {
            node->add_input(inp);
            auto output = std::make_shared<Tensor>(
                g->get_valid_tensor_name("dummpy/" + inp->name() + "/output" + std::to_string(i)), inp->shape());
            add_tensor_info(inp, output);
            node->add_output(output);
            outputs.push_back(output);
            i++;
        }
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return outputs;
    }
}; // namespace ops
}; // namespace aipubt
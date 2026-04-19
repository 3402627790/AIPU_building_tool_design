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
 * @Date: 2023-10-21 13:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2023-10-21 11:35:09
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr bounding_box(TensorPtr input1,
                            TensorPtr input2,
                            const Quantization &quant)
    {
        if (input1->shape() != input2->shape())
        {
            ERROR() << "The boxes,delta tensor rank must be the same!";
            return nullptr;
        }
        auto g = input1->node()->graph();
        TensorShape output_shape(input1->shape());
        output_shape.set_layout(input1->shape().layout());
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("bounding_box"), output_shape);
        add_tensor_info(input1, output, quant);

        auto node = Node::create(g->get_valid_node_name("bounding_box"), OpType_BoundingBox);
        node->add_input(input1);
        node->add_input(input2);
        node->add_output(output);
        g->add_node(node);
        quantize(node);

        return output;
    }
}; //namespace ops
}; // namespace aipubt
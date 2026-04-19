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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-13 18:36:16
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
namespace aipubt
{
namespace ops
{
    TensorPtr convert_layout(TensorPtr input, DataLayout convert_to)
    {
        auto g = input->node()->graph();

        // if (input->shape().check_layout())
        // {
        //     ERROR() << "the input layout is:" << input->shape().layout() << " which is mismatch with shape:" << input->shape();
        //     return nullptr;
        // }
        auto outshape = input->shape();
        if (!outshape.change_layout(convert_to))
        {
            ERROR() << "the input layout is:" << input->shape().layout() << " which can not change to layout:" << convert_to;
            return nullptr;
        }

        auto base_name = input->name() + "/layout/" + datalayout_to_string(convert_to);
        auto node = Node::create(g->get_valid_node_name(base_name), OpType_LayoutConvertor);
        g->add_node(node);

        node->params()["from"] = datalayout_to_string(input->shape().layout());
        node->params()["to"] = datalayout_to_string(convert_to);
        node->add_input(input);
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name(base_name), outshape, input->type());
        output->quantization() = input->quantization();

        node->add_output(output);
        return output;
    }

}; //namespace ops
}; // namespace aipubt
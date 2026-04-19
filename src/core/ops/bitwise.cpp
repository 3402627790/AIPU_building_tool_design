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
 * @Author: ALex Zheng
 * @Date: 2021-07-13 10:55:59
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 23:26:26
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>


namespace aipubt
{
namespace ops
{
TensorPtr bitwise(std::vector<TensorPtr> inputs,
                  BitwiseMethod method,
                  const Quantization &quant)
{
    std::map<BitwiseMethod, std::string> table = {
        {BitwiseMethod::AND, "AND"},
        {BitwiseMethod::NOT, "NOT"},
        {BitwiseMethod::OR, "OR"},
        {BitwiseMethod::XOR, "XOR"},
    };
    if (!table.count(method))
    {
        ERROR() << "Doesn't support method!";
        return nullptr;
    }
    std::string method_str = table[method];
    auto input1 = inputs[0];
    auto g = input1->node()->graph();
    auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("bitwise/" + method_str), input1->shape());
    add_tensor_info(input1, output, quant);

    auto node = Node::create(g->get_valid_node_name("bitwise/" + method_str), OpType_Bitwise);
    node->add_input(input1);
    if(method != BitwiseMethod::NOT)
    {
        auto input2 = inputs[1];
        node->add_input(input2);
    }
    node->add_output(output);
    g->add_node(node);
    auto &params = node->params();
    params["method"] = method_str;
    quantize(node);
    return output;
}

}; //namespace ops
}; // namespace aipubt
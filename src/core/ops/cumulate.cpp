
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
 * @Date: 2021-12-03 16:26:26
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2022-12-03 11:33:19
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
TensorPtr cumulate(TensorPtr input,
                    CumulateMethod method,
                    int32_t axis,
                    bool exclusive,
                    bool reverse)
{
    std::map<CumulateMethod, std::string> table = {
        {CumulateMethod::SUM, "SUM"},
        {CumulateMethod::PROD, "PROD"},
    };
    if (!table.count(method))
    {
        ERROR() << "Doesn't support method!";
        return nullptr;
    }
    std::string method_str = table[method];
    auto g = input->node()->graph();
    auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("cumulate/" + method_str), input->shape());
    add_tensor_info(input, output);

    auto node = Node::create(g->get_valid_node_name("cumulate/" + method_str), OpType_Cumulate);
    node->add_input(input);
    node->add_output(output);
    g->add_node(node);
    auto &params = node->params();
    params["method"] = method_str;
    params["axis"] = axis;
    params["exclusive"] = exclusive;
    params["reverse"] = reverse;

    quantize(node);
    return output;
}
}; //namespace ops
}; // namespace aipubt

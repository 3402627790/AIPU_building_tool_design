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
 * @Date: 2022-08-18 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2022-08-18 11:04:33
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> filter(std::vector<TensorPtr> inputs)
    {
        std::vector<TensorPtr> outputs;
        int32_t input_num = inputs.size() - 1;
        if ((input_num > 8) || (input_num < 1))
        {
            ERROR() << "Supports input range [1, 8].";
            return {};
        }

        auto g = inputs[0]->node()->graph();
        auto node = Node::create(g->get_valid_node_name("filter"), OpType_Filter);
        for (auto i = 0; i < input_num; i++)
        {
            node->add_input(inputs[i]);
            auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("filter_" + std::to_string(i)), inputs[i]->shape());
            add_tensor_info(inputs[i], output);
            node->add_output(output);
            outputs.push_back(output);
        }

        node->add_input(inputs[input_num]);
        TensorShape length_shape(1);
        auto length = std::make_shared<Tensor>(g->get_valid_tensor_name("filter/length"), length_shape);
        Quantization quant(1.0, 0, 16);
        length->set_type(Tensor::UINT16);
        length->quantization() = quant;
        node->add_output(length);
        outputs.push_back(length);

        auto &params = node->params();
        params["num"] = input_num;
        params["axis"] = 0; //  lib not support ,but IR check need.
        g->add_node(node);
        quantize(node);
        return outputs;
    }
}; //namespace ops
}; // namespace aipubt
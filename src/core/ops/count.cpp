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
 * @Date: 2022-08-16 17:05:05
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-10 21:04:10
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>

namespace aipubt
{
namespace ops
{
    TensorPtr count(TensorPtr input, NodeParamValue min, NodeParamValue max, uint32_t nbins, bool discrete)
    {
        auto g = input->node()->graph();

        TensorShape count_shape(input->shape().n(), nbins);
        count_shape.set_layout(input->shape().layout());
        TensorPtr count_output = std::make_shared<Tensor>(g->get_valid_tensor_name("count"), count_shape);
        count_output->set_type(Tensor::Dtype::UINT16);
        count_output->quantization() = input->quantization();

        auto node = Node::create(g->get_valid_node_name("count"), OpType_Count);
        node->add_input(input);
        node->add_output(count_output);

        bool quantized = false;
        if (min.t != NodeParamValue::NodeParamType_float &&
            max.t != NodeParamValue::NodeParamType_float)
        {
            DEBUG() << "count quantized";
            quantized = true;
        }
        node->params()["min"] = min;
        node->params()["max"] = max;
        node->set_attr("params_quantized", NodeParamValue(quantized));
        node->params()["nbins"] = nbins;
        node->params()["discrete"] = NodeParamValue(discrete);
        g->add_node(node);

        return count_output;
    }

}; // namespace ops
}; // namespace aipubt
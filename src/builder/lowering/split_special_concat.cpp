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
 * @Author: Yuchou.Gan
 * @Date: 2020-02-25 17:58:05
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2021-11-04 15:08:34
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include <set>
namespace aipubt
{
bool criterion(NodePtr node)
{
    aipubt::ISAType arch = aipubt::get_target()->isainfo().type;
    // only from z3 has nchwc32t8
    if ((arch < aipubt::ISA_ZYv3))
    {
        return false;
    }
    if (node->type() != OpType_Concat)
    {
        return false;
    }

    auto inputs = node->inputs();
    auto output = node->outputs()[0];
    auto params = node->params();
    // inputs number == 3
    if (inputs.size() != 3)
    {
        return false;
    }
    // axis == 3
    int axis = params["axis"];
    if (axis != 3)
    {
        return false;
    }
    // both inputs are 8bit c16, same zp in input/output
    if (Tensor::dtype_size(output->type()) != 1)
    {
        return false;
    }
    if (output->shape().layout() != DataLayout_NHWC)
    {
        return false;
    }
    for (auto &in : inputs)
    {
        if (output->quantization().offset() != in->quantization().offset())
        {
            return false;
        }
        if (in->type() != output->type())
        {
            return false;
        }
        if (in->shape().layout() != output->shape().layout())
        {
            return false;
        }
        if (in->shape().c_total() != 16)
        {
            return false;
        }
    }
    // no scale/shift
    std::set<std::string> quant_var = {"scale_type", "scale_value", "shift_type", "shift_value"};
    for (auto &qv : quant_var)
    {
        if (params.count(qv))
        {
            return false;
        }
    }

    return true;
};

// for ai-benchmark punet 62.025ms to 36.022ms when rd200 wr40 1core
void split_special_concat(GraphPtr g)
{
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }

        auto inputs = n->inputs();
        // create concat0 node
        NodePtr concat0 = Node::create(n->name(), OpType_Concat);
        auto node_name = g->get_valid_node_name(n->name() + ".AIPU_split");
        n->set_name(node_name);
        concat0->params()["axis"] = n->params()["axis"];

        // create outputs
        TensorShape concat0_shape = n->outputs()[0]->shape();
        // must be DataLayout_NHWC
        concat0_shape[3] = inputs[0]->shape()[3] + inputs[1]->shape()[3];
        auto tensor_name = g->get_valid_tensor_name(concat0->name());
        TensorPtr concat0_output = std::make_shared<Tensor>(tensor_name, concat0_shape, n->outputs()[0]->type());
        concat0_output->quantization() = n->outputs()[0]->quantization();
        concat0->add_output(concat0_output);

        concat0->add_input(inputs[0]);
        concat0->add_input(inputs[1]);
        n->clear_inputs();
        n->add_input(concat0_output);
        n->add_input(inputs[2]);

        g->add_node(concat0);
        ADD_LOCATION_BY_NODE(concat0, n);
    }
}
}; // namespace aipubt

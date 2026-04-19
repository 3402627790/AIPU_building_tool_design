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
 * @Date: 2025-05-30 15:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2025-05-30 15:04:33
 */
#include <core/core.h>
#include <core/ifnode.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtrList If(TensorPtrList inputs, GraphPtr then_branch, GraphPtr else_branch)
    {
        auto logical_input = inputs[0];
        if (logical_input->type() != Tensor::UINT8 && logical_input->type() != Tensor::BOOL)
        {
            ERROR() << "The logical input should have type UINT8/BOOL, but now its dtype is " << logical_input->type();
            return {};
        }
        if (then_branch == nullptr || else_branch == nullptr)
        {
            ERROR() << "The then_branch or else_branch is nullptr.";
            return {};
        }

        if (then_branch->get_output_tensors().size() != else_branch->get_output_tensors().size())
        {
            ERROR() << "The then_branch and else_branch should have same outputs.";
            return {};
        }
        for (size_t i = 0; i < then_branch->get_output_tensors().size(); ++i)
        {
            if (then_branch->get_output_tensors()[i]->shape().dim() !=
                else_branch->get_output_tensors()[i]->shape().dim())
            {
                ERROR() << "The then_branch and else_branch output " << i
                        << " should have same dimension.";
                return {};
            }
        }
        if (inputs.size() != 1 + then_branch->get_input_tensors().size() + else_branch->get_input_tensors().size())
        {
            ERROR() << "The input size not right";
            return {};
        }
        std::vector<std::vector<int>> outputs_dim;
        for (size_t i = 0; i < then_branch->get_output_tensors().size(); ++i)
        {
            std::vector<int> output_dim;
            for (auto j = 0; j < then_branch->get_output_tensors()[i]->shape().dim(); ++j)
            {
                if (then_branch->get_output_tensors()[i]->shape()[j] > else_branch->get_output_tensors()[i]->shape()[j])
                    output_dim.push_back(then_branch->get_output_tensors()[i]->shape()[j]);
                else
                    output_dim.push_back(else_branch->get_output_tensors()[i]->shape()[j]);
            }
            outputs_dim.push_back(output_dim);
        }

        auto g = logical_input->node()->graph();
        auto node = Node::create(g->get_valid_node_name("If"), OpType_If);
        for (auto inp : inputs)
        {
            node->add_input(inp);
        }
        std::vector<TensorPtr> outputs;
        for (size_t i = 0; i < then_branch->get_output_tensors().size(); ++i)
        {
            TensorShape output_shape(outputs_dim[i], then_branch->get_output_tensors()[i]->shape().layout());
            auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("If"), output_shape);
            output->set_type(then_branch->get_output_tensors()[i]->type());
            node->add_output(output);
            outputs.push_back(output);
        }
        g->add_node(node);
        auto &params = node->params();
        params["then_branch"] = then_branch->name();
        params["else_branch"] = else_branch->name();
        int then_input_num = then_branch->get_input_tensors().size();
        int else_input_num = else_branch->get_input_tensors().size();
        params["then_branch_inputs_num"] = then_input_num;
        params["else_branch_inputs_num"] = else_input_num;
        node->infer_shape(true);

        auto then_sg = then_branch->clone();
        auto else_sg = else_branch->clone();
        std::vector<TensorPtr> then_inputs, else_inputs;
        for (size_t i = 1; i < inputs.size() - else_input_num; ++i)
        {
            then_inputs.push_back(inputs[i]);
        }
        for (size_t i = 1 + then_input_num; i < inputs.size(); ++i)
        {
            else_inputs.push_back(inputs[i]);
        }
        then_sg->replace_symbols(then_inputs);
        else_sg->replace_symbols(else_inputs);

        // Subgraph node create
        auto sgn = IfNode::create(g, node, then_sg, else_sg);
        auto nsg = sgn->subgraph();
        auto then_branch_attrs = then_branch->attrs();
        auto else_branch_attrs = else_branch->attrs();
        nsg->set_attr("then_branch_attrs", then_branch_attrs);
        nsg->set_attr("else_branch_attrs", else_branch_attrs);
        return outputs;
    }
} // namespace ops
} // namespace aipubt
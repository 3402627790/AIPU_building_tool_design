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
 * @Date: 2025-06-23 15:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2025-06-23 15:04:33
 */
#include <core/core.h>
#include <core/loopnode.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtrList loop(TensorPtrList inputs,
                       GraphPtr body,
                       int32_t dependency_inputs_num,
                       int32_t scan_outputs_num,
                       bool keep_out_dim,
                       int32_t max_loop_num)
    {
        // Check paramter
        if ((int32_t)inputs.size() < dependency_inputs_num + 2)
        {
            ERROR() << "inputs size must be equal or greater than (dependency_inputs_num + 2).";
            return {};
        }
        auto iteration_num = inputs[0];
        auto condition = inputs[1];
        if (iteration_num->type() != Tensor::INT32 ||
            iteration_num->shape().size() != 1 ||
            (condition->type() != Tensor::UINT8 && condition->type() != Tensor::BOOL) ||
            condition->shape().size() != 1)
        {
            ERROR() << "input0 and input1 shape rank is 1.\n"
                    << "input0 type must be int32.\n"
                    << "input1 type must be uint8.";
            return {};
        }
        if (dependency_inputs_num + scan_outputs_num + 1 != (int32_t)body->get_output_tensors().size())
        {
            ERROR() << "body output size must be equal (dependency_inputs_num + scan_outputs_num + 1).\n"
                    << "But now body output is " << body->get_output_tensors().size()
                    << ", expect " << dependency_inputs_num + scan_outputs_num + 1;
            return {};
        }
        if (body->get_output_tensors()[0]->type() != Tensor::UINT8 ||
            body->get_output_tensors()[0]->shape().size() != 1)
        {
            ERROR() << "The body output 0 type must be uint8,and shape must be 1.";
            return {};
        }

        // Create node
        auto g = inputs[0]->node()->graph();
        auto node = Node::create(g->get_valid_node_name("loop"), OpType_Loop);
        g->add_node(node);
        for (auto inp : inputs)
        {
            node->add_input(inp);
        }

        TensorPtrList outputs;
        for (int32_t i = 1; i < dependency_inputs_num + 1; ++i)
        {
            auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("loop/output"), body->get_output_tensors()[i]->shape());
            add_tensor_info(body->get_output_tensors()[i], output);
            node->add_output(output);
            outputs.push_back(output);
        }
        for (int32_t i = 1 + dependency_inputs_num; i < 1 + dependency_inputs_num + scan_outputs_num; ++i)
        {
            auto body_output = body->get_output_tensors()[i];
            std::vector<int32_t> dims(body_output->shape().begin(), body_output->shape().end());
            if (iteration_num->node()->type() == OpType_Constant)
            {
                // use iteration_num to infer shape
                auto weight = iteration_num->node()->constants()["weights"];
                int32_t loop_num = ComputeTensor(weight).astype(Tensor::INT32);
                if (keep_out_dim)
                {
                    dims[0] *= loop_num;
                }
                else
                {
                    dims.insert(dims.begin(), loop_num);
                }
            }
            else
            {
                // use max_loop_num to infer shape
                if (keep_out_dim)
                {
                    dims[0] *= max_loop_num;
                }
                else
                {
                    dims.insert(dims.begin(), max_loop_num);
                }
            }
            auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("loop/output"), TensorShape(dims));
            add_tensor_info(body->get_output_tensors()[i], output);
            node->add_output(output);
            outputs.push_back(output);
        }
        auto &params = node->params();
        params["dependency_inputs_num"] = dependency_inputs_num;
        params["scan_outputs_num"] = scan_outputs_num;
        params["body"] = body->name();
        params["loop_default_count"] = max_loop_num;
        node->infer_shape(true);

        auto sg = body->clone();
        sg->replace_symbols(inputs);

        // Subgraph node create
        auto sgn = LoopNode::create(g, node, sg);
        auto nsg = sgn->subgraph();
        for (auto &kv : body->attrs())
        {
            nsg->set_attr(kv.first, kv.second);
        }

        return outputs;
    }
} // namespace ops
} // namespace aipubt
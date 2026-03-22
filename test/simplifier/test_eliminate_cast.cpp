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
 * @Author: Samuel Han
 * @Date: 2022-03-30 15:55:17
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-30 16:39:29
 */

#include "test_simplifier.h"

void test_eliminate_cast(void)
{
    aipubt::GraphPtr graph = aipubt::Graph::create("test_eliminate_cast_graph");

    aipubt::NodePtr i_node = aipubt::Node::create("input_node", aipubt::OpType_Input);
    aipubt::TensorPtr i_o_t = std::make_shared<aipubt::Tensor>("input_node_output", aipubt::TensorShape(1, 1, 3072), aipubt::Tensor::INT32);
    i_node->add_output(i_o_t);

    aipubt::NodePtr c_node = aipubt::Node::create("cast_node", aipubt::OpType_Cast);
    aipubt::TensorPtr c_o_t = std::make_shared<aipubt::Tensor>("cast_node_output", aipubt::TensorShape(1, 1, 3072), aipubt::Tensor::INT32);
    c_node->add_input(i_o_t);
    c_node->add_output(c_o_t);
    c_node->params()["to_dtype"] = aipubt::Tensor::INT32;

    aipubt::NodePtr c_node1 = aipubt::Node::create("cast_node1", aipubt::OpType_Cast);
    aipubt::TensorPtr c_o_t1 = std::make_shared<aipubt::Tensor>("cast_node_output1", aipubt::TensorShape(1, 1, 3072), aipubt::Tensor::UINT32);
    c_node1->add_input(c_o_t);
    c_node1->add_output(c_o_t1);
    c_node1->params()["to_dtype"] = aipubt::Tensor::UINT32;

    aipubt::NodePtr r_node = aipubt::Node::create("reshape_node", aipubt::OpType_Reshape);
    aipubt::TensorPtr r_o_t = std::make_shared<aipubt::Tensor>("reshape_node_output", aipubt::TensorShape(1, 1, 1, 3072), aipubt::Tensor::UINT32);
    r_node->add_input(c_o_t1);
    r_node->add_output(r_o_t);
    aipubt::NodeParamValue shape;
    for (int32_t i = 0; i < 4; i++)
    {
        shape.push_back(r_o_t->shape()[i]);
    }
    r_node->params()["shape"] = shape;

    aipubt::NodePtr c_node2 = aipubt::Node::create("cast_node2", aipubt::OpType_Cast);
    aipubt::TensorPtr c_o_t2 = std::make_shared<aipubt::Tensor>("cast_node_output2", aipubt::TensorShape(1, 1, 1, 3072), aipubt::Tensor::UINT32);
    c_node2->add_input(r_o_t);
    c_node2->add_output(c_o_t2);
    c_node2->params()["to_dtype"] = aipubt::Tensor::UINT32;

    graph->add_node(i_node);
    graph->add_node(c_node);
    graph->add_node(c_node1);
    graph->add_node(r_node);
    graph->add_node(c_node2);

    bool ret = aipubt::eliminate_cast(graph);
    DEBUG() << (int32_t)ret;
    graph->to_png("cast.png");
    graph->serialize("cast.txt", "cast.bin");
}
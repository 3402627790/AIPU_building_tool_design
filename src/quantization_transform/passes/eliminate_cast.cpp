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
 * @Date: 2022-11-08 19:23:03
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-19 10:05:35
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
static bool criteria(aipubt::NodePtr n)
{
    /* qtlib now only try to eliminate the uint8->int8 and int8->uint8 cast in QAT models.*/
    if (n->type() != aipubt::OpType_Cast)
    {
        return false;
    }
    if (n->attrs().count("eliminate_cast"))
    {
        bool mode = n->get_attr<bool>("eliminate_cast");
        if (!mode)
        {
            return false;
        }
    }

    auto in_dtype = n->inputs()[0]->type();
    auto out_dtype = n->outputs()[0]->type();
    if (aipubt::Tensor::is_float(in_dtype) || aipubt::Tensor::is_float(out_dtype))
    {
        return false;
    }

    auto in_bits = aipubt::Tensor::dtype_size(in_dtype);
    auto out_bits = aipubt::Tensor::dtype_size(out_dtype);

    if (in_bits != out_bits)
    {
        return false;
    }

    bool is_up_cast = aipubt::Tensor::is_signed(in_dtype) == true && aipubt::Tensor::is_signed(out_dtype) == false;
    bool is_down_cast = aipubt::Tensor::is_signed(in_dtype) == false && aipubt::Tensor::is_signed(out_dtype) == true;
    if (!is_down_cast && !is_up_cast)
    {
        return false;
    }

    return true;
}

static bool pass_down_quantization(aipubt::NodePtr cast_node)
{
    auto passed_quantization = cast_node->inputs()[0]->quantization();
    auto compared_quantization = cast_node->outputs()[0]->quantization();
    auto cast_input = cast_node->inputs()[0];
    auto cast_output = cast_node->outputs()[0];
    auto cast_output_consumers = cast_output->consumers();
    std::queue<aipubt::NodePtr> node_queue;
    // std::queue<std::pair<aipubt::NodePtr, aipubt::TensorPtr>> node_queue;
    for (auto &consumer_n : cast_output_consumers)
    {
        auto it = std::find(consumer_n->inputs().begin(), consumer_n->inputs().end(), cast_output);
        auto idx = std::distance(consumer_n->inputs().begin(), it);
        auto inp_t = consumer_n->remove_input(idx);
        consumer_n->add_input(cast_input, idx);

        for (auto &consumer_o : consumer_n->outputs())
        {
            if (compared_quantization == consumer_o->quantization() &&
                consumer_n->type() != aipubt::OpType_Concat &&
                consumer_n->type() != aipubt::OpType_Convolution)
            {
                consumer_o->quantization() = passed_quantization;
                for (auto &cc_n : consumer_o->consumers())
                {
                    node_queue.push(cc_n);
                }
            }
        }
    }

    while (node_queue.size())
    {
        auto cur_n = node_queue.front();
        for (auto &cur_n_o : cur_n->outputs())
        {
            if (compared_quantization == cur_n_o->quantization() && cur_n->type() != aipubt::OpType_Concat)
            {
                cur_n_o->quantization() = passed_quantization;
                for (auto &cur_n_o_consumer : cur_n_o->consumers())
                {
                    node_queue.push(cur_n_o_consumer);
                }
            }
        }
        node_queue.pop();
    }
    return true;
}

static bool pass_up_quantization(aipubt::NodePtr cast_node)
{
    if (cast_node->inputs()[0]->consumers().size() > 1)
    {
        WARN() << "now input tensor of cast op has more than one consumers, it is hard to distinguish the quantization pass, so we quit.";
        return false;
    }
    if (cast_node->inputs()[0]->node()->outputs().size() > 1)
    {
        WARN() << " now cast op parent has more than one output tenosrs, it is also hard to distinguish the quantization pass in these output tensors, so we quit";
        return false;
    }
    if (cast_node->inputs()[0]->node()->type() == aipubt::OpType_Input)
    {
        return false;
    }

    auto passed_quantization = cast_node->outputs()[0]->quantization();
    auto compared_quantization = cast_node->inputs()[0]->quantization();
    auto producer = cast_node->inputs()[0]->node();
    auto r_idx = producer->remove_output(cast_node->inputs()[0]);
    producer->add_output(cast_node->outputs()[0], r_idx);

    std::queue<aipubt::NodePtr> node_queue;
    for (auto &p_inpt : producer->inputs())
    {
        if (p_inpt->quantization() == compared_quantization)
        {
            p_inpt->quantization() = passed_quantization;
            node_queue.push(p_inpt->node());
        }
    }
    while (node_queue.size())
    {

        auto cur_n = node_queue.front();
        for (auto &cur_n_inpt : cur_n->inputs())
        {
            if (cur_n_inpt->quantization() == compared_quantization)
            {
                cur_n_inpt->quantization() = passed_quantization;
                node_queue.push(cur_n_inpt->node());
            }
        }
        node_queue.pop();
    }
    return true;
}

void eliminate_cast(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        bool ret = false;
        auto inp_dtype = n->inputs()[0]->type();
        auto out_dtype = n->outputs()[0]->type();
        auto in_scales = n->inputs()[0]->quantization().scales();
        auto out_scales = n->outputs()[0]->quantization().scales();
        bool is_up_cast = aipubt::Tensor::is_signed(inp_dtype) == true && aipubt::Tensor::is_signed(out_dtype) == false;
        bool is_down_cast = aipubt::Tensor::is_signed(inp_dtype) == false && aipubt::Tensor::is_signed(out_dtype) == true;
        if ((n->outputs()[0]->is_io_tensor() || is_up_cast) && in_scales == out_scales)
        {
            ret = ret || pass_up_quantization(n);
        }
        else if (is_down_cast && in_scales == out_scales)
        {
            ret = ret || pass_down_quantization(n);
        }

        if (ret)
        {
            n->clear_inputs();
            n->clear_outputs();
            g->remove_node(n);
        }
    }
}

} // namespace qtlibn

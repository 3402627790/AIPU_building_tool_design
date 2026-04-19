
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
 * @Date: 2021-07-07 16:26:26
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:34:44
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr constant(GraphPtr g,
                       const void *data,
                       const TensorShape &shape,
                       Tensor::Dtype dtype,
                       const Quantization &quant)
    {
        if (data == nullptr)
        {
            ERROR() << "constant data is nullptr!";
            return nullptr;
        }

        if (g == nullptr)
        {
            g = get_default_graph();
        }

        TensorShape output_shape(shape);
        if (output_shape.layout() == DataLayout_Flat)
        {
            set_default_layout(output_shape);
        }

        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("constant/output"), output_shape);
        output->quantization() = quant;
        output->set_type(dtype);

        TensorPtr const_tensor = std::make_shared<Tensor>(g->get_valid_tensor_name("constant/const_tensor"), output_shape);
        const_tensor->quantization() = output->quantization();
        const_tensor->set_type(dtype);
        void *new_data = new char[const_tensor->mem_size()];
        memcpy(new_data, data, const_tensor->mem_size());
        const_tensor->set_data(new_data, const_tensor->mem_size());

        auto node = Node::create(g->get_valid_node_name("constant"), OpType_Constant);
        node->add_output(output);
        node->constants()["weights"] = const_tensor;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    TensorPtr constant(const void *data,
                       const TensorShape &shape,
                       Tensor::Dtype dtype,
                       const Quantization &quant,
                       GraphPtr g)
    {
        return constant(g,
                        data,
                        shape,
                        dtype,
                        quant);
    }

    TensorPtr constant(TensorPtr c,
                       const Quantization &quant,
                       GraphPtr g)
    {
        if (!c->data<void>())
        {
            WARN() << "ops::constant requires a Tensor with data, but got nullptr.";
            return nullptr;
        }
        if (g == nullptr)
        {
            g = get_default_graph();
        }

        TensorPtr weight_tensor = std::make_shared<Tensor>(g->get_valid_tensor_name(c->name() + "/weight"), c->shape());
        c->quantization() = quant;
        weight_tensor->quantization() = quant;
        weight_tensor->set_type(c->type());
        void *new_data = new char[weight_tensor->mem_size()];
        memcpy(new_data, c->data<void>(), weight_tensor->mem_size());
        weight_tensor->set_data(new_data, weight_tensor->mem_size());

        auto node = Node::create(g->get_valid_node_name("constant"), OpType_Constant);
        node->add_output(c);
        node->constants()["weights"] = weight_tensor;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return c;
    }
}; // namespace ops
}; // namespace aipubt

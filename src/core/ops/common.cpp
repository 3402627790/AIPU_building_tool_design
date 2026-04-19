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
 * @Author: Neo Wang
 * @Date: 2021-04-14 11:04:33
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-01 21:20:23
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
namespace aipubt
{
namespace ops
{

    TensorPtr make_weight(const void *data,
                          const TensorShape &shape,
                          Tensor::Dtype dtype,
                          const Quantization &quant,
                          GraphPtr g)
    {
        if (!g)
        {
            g = get_default_graph();
        }

        TensorPtr weight = std::make_shared<Tensor>(g->get_valid_tensor_name("filter"), shape);
        weight->quantization() = quant;
        weight->set_type(dtype);

        uint8_t *weight_data = new uint8_t[weight->mem_size()];
        memcpy(weight_data, data, weight->mem_size());
        weight->set_data(weight_data, weight->mem_size());
        return weight;
    }

    TensorPtr make_weight(GraphPtr g,
                          const void *data,
                          const TensorShape &shape,
                          Tensor::Dtype dtype,
                          const Quantization &quant)
    {
        return make_weight(data, shape, dtype, quant, g);
    }

    TensorPtr make_bias(const void *data,
                        uint32_t data_len,
                        Tensor::Dtype dtype,
                        const Quantization &quant,
                        GraphPtr g)
    {
        if (!g)
        {
            g = get_default_graph();
        }

        TensorShape shape(data_len / Tensor::dtype_size(dtype));
        TensorPtr bias = std::make_shared<Tensor>(g->get_valid_tensor_name("bias"), shape);
        bias->quantization() = quant;
        bias->set_type(dtype);

        uint8_t *bias_data = new uint8_t[data_len];
        memcpy(bias_data, data, data_len);
        bias->set_data(bias_data, bias->mem_size());
        return bias;
    }

    TensorPtr make_bias(GraphPtr g,
                        const void *data,
                        uint32_t data_len,
                        Tensor::Dtype dtype,
                        const Quantization &quant)
    {
        return make_bias(data, data_len, dtype, quant, g);
    }

    TensorPtr make_tensor(GraphPtr g,
                          const void *data,
                          const TensorShape &shape,
                          Tensor::Dtype dtype,
                          const Quantization &quant,
                          DataLayout layout,
                          const std::string &name)
    {
        if (g == nullptr)
        {
            g = get_default_graph();
        }
        TensorPtr t = std::make_shared<Tensor>(g->get_valid_tensor_name(name), shape);
        t->shape().set_layout(layout);
        t->set_type(dtype);
        void *new_data = new char[t->mem_size()];
        memcpy(new_data, data, t->mem_size());
        t->quantization() = quant;
        t->set_data(new_data, t->mem_size());
        return t;
    }

    TensorPtr make_tensor(const void *data,
                          const TensorShape &shape,
                          Tensor::Dtype dtype,
                          const Quantization &quant,
                          DataLayout layout,
                          const std::string &name,
                          GraphPtr g)
    {
        return make_tensor(g, data, shape, dtype, quant, layout, name);
    }
}; // namespace ops
}; // namespace aipubt
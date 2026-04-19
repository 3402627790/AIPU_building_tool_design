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
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-03-14 10:18:21
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    static std::vector<int32_t> get_output_max_dim(TensorPtr input1, TensorPtr input2)
    {
        std::vector<int32_t> output_dims;
        std::vector<int32_t> input1_dims(input1->shape().begin(), input1->shape().end());
        std::vector<int32_t> input2_dims(input2->shape().begin(), input2->shape().end());
        std::reverse(input1_dims.begin(), input1_dims.end());
        std::reverse(input2_dims.begin(), input2_dims.end());
        size_t max_dim = std::max(input1_dims.size(), input2_dims.size());
        for (size_t i = 0; i < max_dim; ++i)
        {
            if (input1_dims.size() > i && input2_dims.size() > i)
            {
                output_dims.insert(output_dims.begin(), std::max(input1_dims[i], input2_dims[i]));
            }
            else if (input1_dims.size() > i)
            {
                output_dims.insert(output_dims.begin(), input1_dims[i]);
            }
            else
            {
                output_dims.insert(output_dims.begin(), input2_dims[i]);
            }
        }
        return output_dims;
    }

    static TensorPtr eltwise(TensorPtr input1, TensorPtr input2, const Activation &with_activation,
                             const Quantization &quant, std::string method)
    {
        auto g = input1->node()->graph();
        std::vector<int32_t> output_dims = get_output_max_dim(input1, input2);
        DataLayout layout = input1->shape().layout();
        if (input1->shape().dim() < input2->shape().dim())
        {
            layout = input2->shape().layout();
        }
        TensorShape output_shape = TensorShape(output_dims, layout);
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("eltwise/" + method), output_shape);
        add_tensor_info(input1, output, quant);
        auto node = Node::create(g->get_valid_node_name("eltwise/" + method), OpType_Eltwise);
        node->add_input(input1);
        node->add_input(input2);
        node->add_output(output);
        g->add_node(node);
        auto &params = node->params();
        params["method"] = method;

        append_activation(node, with_activation);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    bool constant_eltwise_support(TensorPtr input1,
                                  TensorPtr input2,
                                  const Activation &with_activation,
                                  std::string method)
    {
        if ((input1->node()->type() != OpType_Constant) && (input2->node()->type() != OpType_Constant))
        {
            return false;
        }

        if ((method == "MIN") || (method == "MAX") || method == "SUB")
        {
            return false;
        }

        if ((with_activation.type != "NONE") && (with_activation.type != "RELU") && (with_activation.type != "CLIP"))
        {
            return false;
        }
        if (input1->shape().size() != input2->shape().size())
        {
            return false;
        }

        // shape check
        if (input1->node()->type() == OpType_Constant)
        {
            TensorPtr temp = input1;
            input1 = input2;
            input2 = temp;
        }
        if (input2->shape().dim() > input1->shape().dim())
        {
            return false;
        }
        std::vector<int> input1_valid_dim;
        for (int i = 0; i < input1->shape().dim(); ++i)
        {
            if (input1->shape()[i] == 1)
                continue;
            else
                input1_valid_dim.push_back(input1->shape()[i]);
        }
        std::vector<int> input2_valid_dim;
        for (int i = 0; i < input2->shape().dim(); ++i)
        {
            if (input2->shape()[i] == 1)
                continue;
            else
                input2_valid_dim.push_back(input2->shape()[i]);
        }

        if (input1_valid_dim.size() != input2_valid_dim.size())
        {
            return false;
        }
        for (size_t i = 0; i < input1_valid_dim.size(); ++i)
        {
            if (input1_valid_dim[i] != input2_valid_dim[i])
            {
                return false;
            }
        }
        // [1,1,384],[1,384,1] -> [1,384,384] in AIB: mobilebert_quant
        int32_t dim_diff = input1->shape().dim() - input2->shape().dim();
        for (int32_t i = dim_diff, j = 0; i < input1->shape().dim(); ++i, ++j)
        {
            if (input2->shape()[j] != 1 && input2->shape()[j] != input1->shape()[i])
            {
                return false;
            }
        }
        return true;
    }

    TensorPtr constant_eltwise(TensorPtr input1, TensorPtr input2, const Activation &with_activation,
                               const Quantization &quant, std::string method)
    {
        if (input1->node()->type() == OpType_Constant)
        {
            TensorPtr temp = input1;
            input1 = input2;
            input2 = temp;
        }
        input2->shape() = input1->shape();
        TensorPtr const_tensor = input2->node()->constants()["weights"];
        const_tensor->shape() = input1->shape();
        return eltwise(input1, input2, with_activation, quant, method);
    }

    TensorPtr adapt_eltwise(TensorPtr input1, TensorPtr input2, const Activation &with_activation,
                            const Quantization &quant, std::string method)
    {
        if (constant_eltwise_support(input1, input2, with_activation, method))
        {
            return constant_eltwise(input1, input2, with_activation, quant, method);
        }
        else
        {
            return eltwise(input1, input2, with_activation, quant, method);
        }
    }

    TensorPtr elementwise_add(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return adapt_eltwise(input1, input2, with_activation, quant, "ADD");
    }

    TensorPtr elementwise_mul(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return adapt_eltwise(input1, input2, with_activation, quant, "MUL");
    }

    TensorPtr elementwise_max(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return adapt_eltwise(input1, input2, with_activation, quant, "MAX");
    }

    TensorPtr elementwise_min(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return adapt_eltwise(input1, input2, with_activation, quant, "MIN");
    }

    TensorPtr elementwise_sub(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return adapt_eltwise(input1, input2, with_activation, quant, "SUB");
    }

    TensorPtr max(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return adapt_eltwise(input1, input2, with_activation, quant, "MAX");
    }

    TensorPtr min(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return adapt_eltwise(input1, input2, with_activation, quant, "MIN");
    }

    static TensorPtr alone_eltwise(TensorPtr input1, TensorPtr input2,
                                   const Activation &with_activation,
                                   const Quantization &quant, OpType optype)
    {
        if (optype != OpType_Sub &&
            ((input1->shape().size() < input2->shape().size()) ||
             (input1->shape().size() == input2->shape().size() &&
              input1->shape().dim() < input2->shape().dim())))
        {
            TensorPtr temp = input1;
            input1 = input2;
            input2 = temp;
        }

        auto g = input1->node()->graph();
        std::vector<int32_t> output_dims = get_output_max_dim(input1, input2);
        TensorShape output_shape(output_dims, input1->shape().layout());
        std::string name = optype_to_string(optype);
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name(name), output_shape);
        add_tensor_info(input1, output, quant);

        auto node = Node::create(g->get_valid_node_name(name), optype);
        append_activation(node, with_activation);
        node->add_input(input1);
        node->add_input(input2);
        node->add_output(output);
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    //
    TensorPtr add(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return alone_eltwise(input1, input2, with_activation, quant, OpType_Add);
    }

    TensorPtr sub(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return alone_eltwise(input1, input2, with_activation, quant, OpType_Sub);
    }

    TensorPtr mul(TensorPtr input1, TensorPtr input2, const Activation &with_activation, const Quantization &quant)
    {
        return alone_eltwise(input1, input2, with_activation, quant, OpType_Mul);
    }

}; // namespace ops
}; // namespace aipubt

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
 * @Date: 2021-07-14 10:10:54
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-02 16:08:37
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{

    static TensorPtr logical(TensorPtr input1,
                             TensorPtr input2,
                             std::string method,
                             const Quantization &quant = {})
    {
        GraphPtr g = input1->node()->graph();
        if ((input1 == nullptr) || (input2 == nullptr))
        {
            ERROR() << "input1 or input2 is nullptr!";
            return nullptr;
        }
        if (input1->type() == Tensor::INT32)
        {
            input1 = ops::cast(input1, Tensor::INT16, true);
        }
        else if (input1->type() == Tensor::UINT32)
        {
            input1 = ops::cast(input1, Tensor::UINT16, true);
        }

        if (input2->type() == Tensor::INT32)
        {
            input2 = ops::cast(input2, Tensor::INT16, true);
        }
        else if (input2->type() == Tensor::UINT32)
        {
            input2 = ops::cast(input2, Tensor::UINT16, true);
        }
        auto new_inputs = do_multibroadcast({input1, input2});
        if (new_inputs.empty())
        {
            ERROR() << "inputs cannot be multibroadcastable!";
            return nullptr;
        }

        Tensor::Dtype out_type = Tensor::UINT8;
        if (!quant.empty())
        {
            if ((Tensor::dtype_size(quant.aipu_dtype()) == 1) ||
                (Tensor::dtype_size(quant.aipu_dtype()) == 2))
            {
                out_type = quant.aipu_dtype();
            }
        }
        else
        {
            if (Tensor::dtype_size(input1->type()) == 2)
            {
                out_type = Tensor::UINT16;
            }
        }

        TensorShape new_shape = new_inputs[0]->shape();
        std::string method_lower = method;
        std::transform(method_lower.begin(), method_lower.end(), method_lower.begin(), ::tolower);
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("logical_" + method_lower), new_shape);
        add_tensor_info(new_inputs[0], output);
        output->set_type(out_type);
        if (!new_inputs[0]->quantization().empty())
        {
            output->quantization().set_scale(1.f);
            output->quantization().set_offset(0);
            output->quantization().set_aipu_dtype(out_type);
        }

        auto node = Node::create(g->get_valid_node_name("logical_" + method_lower), OpType_Logical);
        node->add_input(new_inputs[0]);
        node->add_input(new_inputs[1]);
        node->add_output(output);
        node->params()["method"] = method;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    TensorPtr less(TensorPtr input1,
                   TensorPtr input2,
                   const Quantization &quant)
    {
        return logical(input1, input2, "LESS", quant);
    }

    TensorPtr less_equal(TensorPtr input1,
                         TensorPtr input2,
                         const Quantization &quant)
    {
        return logical(input1, input2, "LESS_EQUAL", quant);
    }

    TensorPtr equal(TensorPtr input1,
                    TensorPtr input2,
                    const Quantization &quant)
    {
        return logical(input1, input2, "EQUAL", quant);
    }

    TensorPtr not_equal(TensorPtr input1,
                        TensorPtr input2,
                        const Quantization &quant)
    {
        return logical(input1, input2, "NOT_EQUAL", quant);
    }

    TensorPtr greater(TensorPtr input1,
                      TensorPtr input2,
                      const Quantization &quant)
    {
        return logical(input1, input2, "GREATER", quant);
    }

    TensorPtr greater_equal(TensorPtr input1,
                            TensorPtr input2,
                            const Quantization &quant)
    {
        return logical(input1, input2, "GREATER_EQUAL", quant);
    }

    TensorPtr logical_and(TensorPtr input1,
                          TensorPtr input2)
    {
        return logical(input1, input2, "AND");
    }

    TensorPtr logical_or(TensorPtr input1,
                         TensorPtr input2)
    {
        return logical(input1, input2, "OR");
    }

    TensorPtr logical_xor(TensorPtr input1,
                          TensorPtr input2)
    {
        return logical(input1, input2, "XOR");
    }

    TensorPtr logical_not(TensorPtr input)
    {
        GraphPtr g = input->node()->graph();
        if (input == nullptr)
        {
            ERROR() << "input is nullptr!";
            return nullptr;
        }
        Tensor::Dtype out_type = Tensor::UINT8;
        if ((input->type() == Tensor::INT16) || (input->type() == Tensor::UINT16))
        {
            out_type = Tensor::UINT16;
        }

        std::string method = "NOT";
        TensorShape new_shape = input->shape();
        std::string method_lower = method;
        std::transform(method_lower.begin(), method_lower.end(), method_lower.begin(), ::tolower);
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("logical_" + method_lower), new_shape);
        auto node = Node::create(g->get_valid_node_name("logical_" + method_lower), OpType_Logical);
        add_tensor_info(input, output);
        output->set_type(out_type);
        if (!output->quantization().empty())
        {
            output->quantization().set_scale(1.f);
            output->quantization().set_offset(0);
            output->quantization().set_aipu_dtype(out_type);
        }
        node->add_input(input);
        node->add_output(output);
        node->params()["method"] = method;

        g->add_node(node);
        node->infer_shape(true);
        return output;
    }
}; // namespace ops
}; // namespace aipubt
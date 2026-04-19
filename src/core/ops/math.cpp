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
 * @Date: 2021-07-13 10:55:59
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-23 22:35:21
 */

#include "ops_internal.h"
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{

    static TensorPtr _eltwise_unary(TensorPtr input,
                                    OpType optype,
                                    std::string name)
    {
        GraphPtr g = input->node()->graph();
        TensorShape output_shape = input->shape();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name(name), output_shape);
        auto node = Node::create(g->get_valid_node_name(name), optype);
        node->add_input(input);
        node->add_output(output);
        g->add_node(node);
        node->infer_shape(true);
        return output;
    }

    static TensorPtr eltwise_unary(TensorPtr input,
                                   OpType optype,
                                   std::string name)
    {
        auto output = _eltwise_unary(input, optype, name);
        add_tensor_info(input, output);
        quantize(output->node());
        return output;
    }

    static TensorPtr eltwise_unary(TensorPtr input,
                                   OpType optype,
                                   std::string name,
                                   const Quantization &quant)
    {
        auto output = _eltwise_unary(input, optype, name);
        add_tensor_info(input, output, quant);
        quantize(output->node());
        return output;
    }

    static TensorPtr eltwise_binary(TensorPtr input1,
                                    TensorPtr input2,
                                    const Quantization &quant,
                                    OpType type,
                                    const std::string &name,
                                    bool broadcast = true)
    {
        std::vector<TensorPtr> new_inputs = {input1, input2};
        if (broadcast)
        {
            new_inputs = do_multibroadcast({input1, input2});
        }
        GraphPtr g = new_inputs[0]->node()->graph();
        TensorShape output_shape = new_inputs[0]->shape();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name(name), output_shape);
        add_tensor_info(new_inputs[0], output, quant);

        auto node = Node::create(g->get_valid_node_name(name), type);
        node->add_input(new_inputs[0]);
        node->add_input(new_inputs[1]);
        node->add_output(output);
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    TensorPtr rsqrt(TensorPtr input,
                    const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Rsqrt, "rsqrt", quant);
    }

    TensorPtr sqrt(TensorPtr input,
                   const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Sqrt, "sqrt", quant);
    }

    TensorPtr exp(TensorPtr input,
                  const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Exp, "exp", quant);
    }

    TensorPtr abs(TensorPtr input,
                  const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Abs, "abs", quant);
    }

    // FIXME: qtlib will set aipu_dtype, donn't inherit from input.
    TensorPtr negative(TensorPtr input)
    {
        return eltwise_unary(input, OpType_Negative, "negative");
    }

    TensorPtr square(TensorPtr input,
                     const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Square, "square", quant);
    }

    TensorPtr log(TensorPtr input,
                  const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Log, "log", quant);
    }

    TensorPtr sine(TensorPtr input,
                   const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Sine, "sine", quant);
    }

    TensorPtr sinh(TensorPtr input,
                   const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Sinh, "sinh", quant);
    }

    TensorPtr tan(TensorPtr input,
                  const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Tan, "tan", quant);
    }

    TensorPtr atan(TensorPtr input,
                   const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Atan, "atan", quant);
    }

    TensorPtr atanh(TensorPtr input,
                    const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Atanh, "atanh", quant);
    }

    TensorPtr cosine(TensorPtr input,
                     const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Cosine, "cosine", quant);
    }

    TensorPtr sign(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Sign, "sign", quant);
    }

    TensorPtr ceil(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Ceil, "ceil", quant);
    }

    TensorPtr floor(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Floor, "floor", quant);
    }

    TensorPtr mod(TensorPtr input1,
                  TensorPtr input2,
                  bool fmod,
                  const Quantization &quant)
    {
        auto output = eltwise_binary(input1, input2, quant, OpType_Mod, "mod");
        auto n = output->node();
        n->params()["fmod"] = fmod;
        return output;
    }

    TensorPtr pow(TensorPtr input1,
                  TensorPtr input2,
                  const Quantization &quant)
    {
        return eltwise_binary(input1, input2, quant, OpType_Pow, "pow");
    }

    TensorPtr squared_difference(TensorPtr input1,
                                 TensorPtr input2,
                                 const Quantization &quant)
    {
        return eltwise_binary(input1, input2, quant, OpType_SquaredDifference, "squared_difference");
    }

    TensorPtr cosh(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Cosh, "cosh", quant);
    }

    TensorPtr round(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Round, "round", quant);
    }

    TensorPtr asin(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Asin, "asin", quant);
    }

    TensorPtr asinh(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Asinh, "asinh", quant);
    }

    TensorPtr acos(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Acos, "acos", quant);
    }

    TensorPtr acosh(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Acosh, "acosh", quant);
    }

    TensorPtr reciprocal(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Reciprocal, "reciprocal", quant);
    }

    TensorPtr trunc(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Trunc, "trunc", quant);
    }

    TensorPtr erf(TensorPtr input, const Quantization &quant)
    {
        return eltwise_unary(input, OpType_Erf, "erf", quant);
    }
}; // namespace ops
}; // namespace aipubt
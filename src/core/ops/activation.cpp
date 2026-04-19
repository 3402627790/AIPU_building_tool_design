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
 * @Last Modified time: 2022-09-10 21:26:19
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    static NodePtr create_node(TensorPtr input, std::string method, const Quantization &quant = {})
    {
        auto g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("activation/" + method), input->shape());
        add_tensor_info(input, output, quant);

        auto node = Node::create(g->get_valid_node_name("activation/" + method), OpType_Activation);
        node->add_input(input);
        node->add_output(output);
        g->add_node(node);
        node->params()["method"] = method;
        node->infer_shape(true);
        return node;
    }

    TensorPtr clip(TensorPtr input,
                   const NodeParamValue &clip_min,
                   const NodeParamValue &clip_max,
                   const Quantization &quant)
    {
        bool need_reshape = false;
        auto orig_shape = input->shape();

        // layerlib only supports 2~5 dims
        if (orig_shape.dim() == 1)
        {
            auto new_shape = orig_shape;
            new_shape.reshape(1, orig_shape[0]);
            input = reshape(input, new_shape);
            need_reshape = true;
        }
        auto node = create_node(input, "CLIP", quant);
        auto output = node->outputs()[0];
        if (need_reshape)
        {
            output = reshape(output, orig_shape);
        }

        auto &params = node->params();
        params["clip_min"] = clip_min;
        params["clip_max"] = clip_max;
        bool quantized = false;
        if (clip_min.t != NodeParamValue::NodeParamType_float &&
            clip_max.t != NodeParamValue::NodeParamType_float)
        {
            quantized = true;
        }
        node->set_attr("params_quantized", NodeParamValue(quantized));
        quantize(node);
        return output;
    }

    TensorPtr relu(TensorPtr input, const Quantization &quant)
    {
        auto node = create_node(input, "RELU", quant);
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr relu1(TensorPtr input, const Quantization &quant)
    {
        auto node = create_node(input, "RELU1", quant);
        auto &params = node->params();
        params["method"] = "CLIP";
        params["clip_min"] = -1.0f;
        params["clip_max"] = 1.0f;
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr relu6(TensorPtr input, const Quantization &quant)
    {
        auto node = create_node(input, "RELU6", quant);
        auto &params = node->params();
        params["method"] = "RELU6";
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr sigmoid(TensorPtr input, const Quantization &quant)
    {
        auto node = create_node(input, "SIGMOID", quant);
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr logistic(TensorPtr input, const Quantization &quant)
    {
        return sigmoid(input, quant);
    }

    TensorPtr tanh(TensorPtr input, const Quantization &quant)
    {
        TensorPtr ret = input;
        bool need_reshape = false;
        // Layerlib doesn't support rank=1
        // If pad internal, layerlib doesn't accept unknow datalayout.
        if (input->shape().dim() == 1)
        {
            need_reshape = true;
            auto new_shape = reshape_to_4d(input->shape());
            new_shape.set_layout(DataLayout_NHWC);
            ret = reshape(ret, new_shape);
        }

        auto node = create_node(ret, "TANH", quant);
        ret = node->outputs()[0];
        if (need_reshape)
        {
            ret = reshape(ret, input->shape());
        }
        quantize(node);
        return ret;
    }

    TensorPtr leaky_relu(TensorPtr input, NodeParamValue negative_slope, const Quantization &quant)
    {
        auto node = create_node(input, "LEAKYRELU", quant);
        node->params()["negative_slope_value"] = negative_slope;
        bool quantized = false;
        if (negative_slope.t != NodeParamValue::NodeParamType_float)
        {
            quantized = true;
        }
        node->set_attr("params_quantized", NodeParamValue(quantized));
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr thresholded_relu(TensorPtr input, float alpha, const Quantization &quant)
    {
        auto node = create_node(input, "THRESHOLDEDRELU", quant);
        node->params()["alpha"] = alpha;
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr softplus(TensorPtr input, const Quantization &quant)
    {
        auto node = create_node(input, "SOFTPLUS", quant);
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr elu(TensorPtr input, float alpha, const Quantization &quant)
    {
        auto node = create_node(input, "ELU", quant);
        node->params()["alpha"] = alpha;
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr gelu(TensorPtr input, bool approximate, const Quantization &quant)
    {
        auto node = create_node(input, "GELU", quant);
        if (approximate)
        {
            node->params()["approximate"] = "TANH";
        }
        else
        {
            node->params()["approximate"] = "NONE";
        }
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr hard_swish(TensorPtr input, const Quantization &quant)
    {
        TensorPtr ret = input;
        bool need_reshape = false;
        if (ret->shape().dim() == 1)
        {
            need_reshape = true;
            ret = reshape(ret, reshape_to_4d(TensorShape(input->shape())));
        }
        auto node = create_node(ret, "HARDSWISH", quant);
        ret = node->outputs()[0];
        if (need_reshape)
        {
            ret = reshape(ret, TensorShape(input->shape()));
        }
        quantize(node);
        return ret;
    }

    TensorPtr crelu(TensorPtr input, int32_t axis, const Quantization &quant)
    {
        AXIS_CHECK(axis, input);
        auto node = create_node(input, "CRELU", quant);
        auto &shape = node->outputs()[0]->shape();
        shape[axis] *= 2;
        node->params()["axis"] = axis;
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr selu(TensorPtr input, float alpha, float gamma, const Quantization &quant)
    {
        auto node = create_node(input, "SELU", quant);
        node->params()["alpha"] = alpha;
        node->params()["gamma"] = gamma;
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr shrink(TensorPtr input, float lambd, float bias, const Quantization &quant)
    {
        auto node = create_node(input, "SHRINK", quant);
        node->params()["lambd"] = lambd;
        node->params()["bias"] = bias;
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr silu(TensorPtr input, const Quantization &quant)
    {
        auto node = create_node(input, "SILU", quant);
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr softsign(TensorPtr input, const Quantization &quant)
    {
        auto node = create_node(input, "SOFTSIGN", quant);
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr celu(TensorPtr input, float alpha, const Quantization &quant)
    {
        auto node = create_node(input, "CELU", quant);
        node->params()["alpha"] = alpha;
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr mish(TensorPtr input, const Quantization &quant)
    {
        auto node = create_node(input, "MISH", quant);
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr bnll(TensorPtr input, const Quantization &quant)
    {
        GraphPtr g = input->node()->graph();
        TensorShape output_shape = input->shape();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("bnll"), output_shape);
        auto node = Node::create(g->get_valid_node_name("bnll"), OpType_BNLL);
        add_tensor_info(input, output, quant);
        node->add_input(input);
        node->add_output(output);

        g->add_node(node);
        quantize(node);
        return output;
    }

    TensorPtr hard_sigmoid(TensorPtr input, float alpha, float beta, const Quantization &quant)
    {
        auto node = create_node(input, "HARDSIGMOID", quant);
        node->params()["alpha"] = alpha;
        node->params()["beta"] = beta;
        quantize(node);
        return node->outputs()[0];
    }

    TensorPtr swish(TensorPtr input, float alpha, const Quantization &quant)
    {
        auto node = create_node(input, "SWISH", quant);
        node->params()["alpha"] = alpha;
        quantize(node);
        return node->outputs()[0];
    }

}; // namespace ops
}; // namespace aipubt
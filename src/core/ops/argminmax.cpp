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
 * @Last Modified time: 2022-08-13 11:35:37
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr argminmax(TensorPtr input,
                        int32_t axis,
                        std::string method,
                        bool keep_dim,
                        bool select_last_index)
    {
        AXIS_CHECK(axis, input);

        // layerlib doesn't support rank=1
        if (input->shape().dim() == 1)
        {
            input = reshape(input, TensorShape({1, input->shape()[0]}));
            axis = 1;
            keep_dim = false;
        }

        std::vector<int32_t> new_dim;
        std::vector<int32_t> output_dim;
        for (int32_t i = 0; i < input->shape().dim(); ++i)
        {
            if (i != axis)
            {
                output_dim.push_back(input->shape()[i]);
                new_dim.push_back(input->shape()[i]);
            }
            else
            {
                new_dim.push_back(1);
            }
        }
        TensorShape new_shape(new_dim);
        if (static_cast<int>(new_dim.size()) == input->shape().dim()) {
            new_shape.set_layout(input->shape().layout());
        } else {
            set_default_layout(new_shape);
        }

        auto g = input->node()->graph();
        std::transform(method.begin(), method.end(), method.begin(), ::toupper);
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("argmimax/" + method), new_shape);
        output->set_type(Tensor::INT32);
        Quantization quant(1.0, 0, 32, input->quantization().scheme());
        output->quantization() = quant;

        auto node = Node::create(g->get_valid_node_name("argminmax/" + method), OpType_ArgMinMax);
        node->add_input(input);
        node->add_output(output);
        node->params()["axis"] = axis;
        node->params()["method"] = method;
        node->params()["select_last_index"] = select_last_index ? "True" : "False";

        g->add_node(node);
        quantize(node);

        if (!keep_dim)
        {
            TensorShape output_shape(output_dim);
            set_default_layout(output_shape);
            output = ops::reshape(output, output_shape);
        }
        return output;
    }

    TensorPtr argmin(TensorPtr input, int32_t axis, bool keep_dim, bool select_last_index)
    {
        return argminmax(input, axis, "MIN", keep_dim, select_last_index);
    }

    TensorPtr argmax(TensorPtr input, int32_t axis, bool keep_dim, bool select_last_index)
    {
        return argminmax(input, axis, "MAX", keep_dim, select_last_index);
    }
}; //namespace ops
}; // namespace aipubt
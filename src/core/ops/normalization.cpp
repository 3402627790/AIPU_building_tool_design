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
 * @Last Modified time: 2022-08-13 11:32:03
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr normalization(TensorPtr input,
                            std::string method,
                            const std::vector<int32_t> &axis,
                            float epsilon,
                            const Quantization &quant)
    {
        std::vector<int32_t> new_axis;
        for (auto it : axis)
        {
            if (it < 0)
            {
                it = it < 0 ? it + input->shape().dim() : it;
            }
            if (it < 0 || it >= input->shape().dim())
            {
                ERROR() << "axis is invalid, it should be in [-rank, rank)!";
                return nullptr;
            }
            if (std::count(new_axis.begin(), new_axis.end(), it) == 0)
            {
                new_axis.push_back(it);
            }
        }
        if (static_cast<int32_t>(new_axis.size()) > input->shape().dim())
        {
            ERROR() << "axis size is invalid, it should be <= input rank!";
            return nullptr;
        }

        auto g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name(method + "_normalization"), input->shape());
        add_tensor_info(input, output, quant);

        auto node = Node::create(g->get_valid_node_name(method + "_normalization"), OpType_Normalization);
        node->add_input(input);
        node->add_output(output);
        g->add_node(node);

        auto &params = node->params();
        params["method"] = method;
        params["epsilon"] = epsilon;
        params["axis"] = new_axis;
        node->infer_shape(true);
        quantize(node);
        return output;
    }

    TensorPtr l2_normalization(TensorPtr input,
                               const std::vector<int32_t> &axis,
                               float epsilon,
                               const Quantization &quant)
    {
        return normalization(input, "L2", axis, epsilon, quant);
    }

    TensorPtr l1_normalization(TensorPtr input,
                               const std::vector<int32_t> &axis,
                               float epsilon,
                               const Quantization &quant)
    {
        return normalization(input, "L1", axis, epsilon, quant);
    }
} //namespace ops
} // namespace aipubt
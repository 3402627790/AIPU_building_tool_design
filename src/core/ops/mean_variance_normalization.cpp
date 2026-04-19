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
 * @Author: Alex Zheng
 * @Date: 2022-08-16 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2022-08-16 11:04:33
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr mean_variance_normalization(TensorPtr input, const std::vector<int32_t> &axis,
                                          float epsilon, const Quantization &quant)
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
                WARN() << "axis is invalid, it should be in [-rank, rank)!";
                continue;
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
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("mean_variance_normalization"), input->shape());
        add_tensor_info(input, output, quant);
        auto node = Node::create(g->get_valid_node_name("mean_variance_normalization"), OpType_MVN);
        auto &params = node->params();
        params["axis"] = new_axis;
        params["epsilon"] = epsilon;

        node->add_input(input);
        node->add_output(output);
        g->add_node(node);
        quantize(node);

        return output;
    }

}; //namespace ops
}; // namespace aipubt
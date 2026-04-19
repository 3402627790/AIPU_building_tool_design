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
 * @Date: 2022-08-11 17:05:05
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2021-08-11 17:05:05
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> moments(TensorPtr input,
                                   const std::vector<int32_t> &axis,
                                   bool keepdims)
    {
        // convert neg to pos, sort axis, not duplicate
        int32_t in_rank = input->shape().dim();
        std::vector<int32_t> new_axis = axis;
        for (auto &it : new_axis)
        {
            it = it < 0 ? it + in_rank : it;
            if (it < 0 || it >= in_rank)
            {
                ERROR() << "invalid axis, should be in [-rank, rank)!";
                return {};
            }
        }
        std::sort(new_axis.begin(), new_axis.end());
        auto it = std::unique(new_axis.begin(), new_axis.end());
        new_axis.erase(it, new_axis.end());

        // calculate output shape
        std::vector<int> output_dim;
        for (int i = 0; i < input->shape().dim(); ++i)
        {
            if (std::find(new_axis.begin(), new_axis.end(), i) != new_axis.end())
            {
                if (keepdims)
                {
                    output_dim.push_back(1);
                }
            }
            else
            {
                output_dim.push_back(input->shape()[i]);
            }
        }
        if (output_dim.size() == 0)
            output_dim.push_back(1);

        auto g = input->node()->graph();
        TensorShape mean_shape(output_dim);
        mean_shape.set_layout(input->shape().layout());
        TensorPtr mean_output = std::make_shared<Tensor>(g->get_valid_tensor_name("moments/mean"), mean_shape);
        mean_output->set_type(input->type());
        mean_output->quantization() = input->quantization();

        TensorShape variance_shape(output_dim);
        variance_shape.set_layout(input->shape().layout());
        TensorPtr variance_output = std::make_shared<Tensor>(g->get_valid_tensor_name("moments/variance"), variance_shape);
        variance_output->set_type(input->type());
        variance_output->quantization() = input->quantization();

        auto node = Node::create(g->get_valid_node_name("moments"), OpType_Moments);
        node->add_input(input);
        node->add_output(mean_output);
        node->add_output(variance_output);
        node->params()["axis"] = new_axis;
        node->params()["keepdims"] = NodeParamValue(keepdims);
        g->add_node(node);
        quantize(node);

        return {mean_output, variance_output};
    }

}; //namespace ops
}; // namespace aipubt
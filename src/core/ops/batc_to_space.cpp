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
 * @Last Modified time: 2022-08-13 11:35:16
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr batch_to_space(TensorPtr input,
                             const std::vector<uint32_t> &block_size,
                             const std::vector<std::pair<uint32_t, uint32_t>> &crops)
    {
        if (input->shape().dim() != 4)
        {
            ERROR() << "only support input rank=4!";
            return nullptr;
        }
        int32_t in_n = input->shape().n();
        int32_t in_h = input->shape().h();
        int32_t in_w = input->shape().w();
        auto block_size_ = block_size;
        if (block_size_.size() == 1)
        {
            block_size_.push_back(block_size_[0]);
        }
        if (block_size_.size() != 2 || crops.size() != 2)
        {
            ERROR() << "block_shape and crops size() should be 2!";
            return nullptr;
        }
        if (in_n % (block_size_[0] * block_size_[1]) != 0)
        {
            ERROR() << "input[n] cannot be divided by block_size[0] * block_size[1]!";
            return nullptr;
        }
        for (auto it : crops)
        {
            if (it.first < 0 || it.second < 0)
            {
                ERROR() << "crops cannot be < 0!";
                return nullptr;
            }
        }

        auto g = input->node()->graph();
        TensorShape output_shape(input->shape());
        output_shape.n() = in_n / block_size_[0] / block_size_[1];
        output_shape.h() = in_h * block_size_[0] - crops[0].first - crops[0].second;
        output_shape.w() = in_w * block_size_[1] - crops[1].first - crops[1].second;
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("batch_to_space"), output_shape);
        add_tensor_info(input, output);

        auto node = Node::create(g->get_valid_node_name("batch_to_space"), OpType_BatchToSpace);
        node->add_input(input);
        node->add_output(output);
        g->add_node(node);

        auto &params = node->params();
        params["block_size_y"] = block_size_[0];
        params["block_size_x"] = block_size_[1];
        params["crop_top"] = crops[0].first;
        params["crop_bottom"] = crops[0].second;
        params["crop_left"] = crops[1].first;
        params["crop_right"] = crops[1].second;
        node->infer_shape(true);
        quantize(node);
        return output;
    }
}; // namespace ops
}; // namespace aipubt
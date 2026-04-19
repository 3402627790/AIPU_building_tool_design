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
 * @Date: 2025-05-29 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2025-05-29 11:04:33
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr batch_to_space_nd(TensorPtr input,
                                const std::vector<int32_t> &block_size,
                                const std::vector<std::vector<int32_t>> &crops)
    {
        if (block_size.size() != crops.size())
        {
            ERROR() << "block and crops should be the same size!"
                    << "block size: " << block_size.size()
                    << ", crops size: " << crops.size();
            return nullptr;
        }
        if ((int)block_size.size() != (input->shape().dim() - 2))
        {
            ERROR() << "block size should be " << (input->shape().dim() - 2)
                    << ",but got " << block_size.size() << "!";
            return nullptr;
        }
        uint32_t block_len = 1;
        for (auto b : block_size)
        {
            block_len *= b;
        }
        int32_t in_n = input->shape().n();
        if (in_n % block_len != 0)
        {
            ERROR() << "input[n] cannot be divided by block!"
                    << "input[n]: " << in_n << ", block: " << block_len;
            return nullptr;
        }
        for (auto it : crops)
        {
            for (auto crop_value : it)
            {
                if (crop_value < 0)
                {
                    ERROR() << "crops cannot be < 0!";
                    return nullptr;
                }
            }
        }

        auto g = input->node()->graph();
        std::vector<int> output_dim;
        output_dim.push_back(in_n / block_len);
        for (size_t i = 0; i < block_size.size(); i++)
        {
            output_dim.push_back(
                input->shape()[i + 1] * block_size[i] - crops[i][0] - crops[i][1]);
        }
        output_dim.push_back(input->shape()[input->shape().dim() - 1]);

        TensorShape output_shape(output_dim, input->shape().layout());
        TensorPtr output = std::make_shared<Tensor>(
            g->get_valid_tensor_name("batch_to_space_nd"), output_shape);
        add_tensor_info(input, output);

        auto node = Node::create(
            g->get_valid_node_name("batch_to_space_nd"), OpType_BatchToSpaceND);
        node->add_input(input);
        node->add_output(output);
        g->add_node(node);

        auto &params = node->params();
        params["block_size"] = block_size;
        params["crops"] = crops;
        quantize(node);
        return output;
    }

}; // namespace ops
}; // namespace aipubt
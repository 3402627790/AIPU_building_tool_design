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
 * @Last Modified time: 2022-08-13 11:33:14
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr depth_to_space(TensorPtr input, uint32_t block_size, DepthToSpaceMode mode)
    {
        if (input->shape().dim() != 4)
        {
            ERROR() << "only support input rank=4!";
            return nullptr;
        }
        int32_t in_h = input->shape().h();
        int32_t in_w = input->shape().w();
        int32_t in_c = input->shape().c();
        if (block_size <= 0)
        {
            ERROR() << "block_size should be > 0!";
            return nullptr;
        }
        if (in_c % (block_size * block_size) != 0)
        {
            ERROR() << "input[c] cannot be diveded by block_size * block_size!";
            return nullptr;
        }

        auto g = input->node()->graph();
        TensorShape output_shape(input->shape());
        output_shape.h() = in_h * block_size;
        output_shape.w() = in_w * block_size;
        output_shape.c() = in_c / block_size / block_size;
        TensorPtr output = std::make_shared<Tensor>(g->get_valid_tensor_name("d2s"), output_shape);
        add_tensor_info(input, output);

        auto node = Node::create(g->get_valid_node_name("d2s"), OpType_DepthToSpace);
        node->add_input(input);
        node->add_output(output);
        g->add_node(node);

        auto &params = node->params();
        params["block_size_x"] = block_size;
        params["block_size_y"] = block_size;
        if(mode == DepthToSpaceMode::CRD)
        {
            params["mode"] = "CRD";
        }
        else
        {
            params["mode"] = "DCR";
        }
        node->infer_shape(true);
        quantize(node);
        return output;
    }
}; //namespace ops
}; // namespace aipubt
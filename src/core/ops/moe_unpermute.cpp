//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Alex Zheng
 * @Date: 2026-08-20 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time:  2026-08-20 17:14:20
 */

#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>

namespace aipubt
{
namespace ops
{
    TensorPtr moe_unpermute(TensorPtr input,
                            TensorPtr route_pos,
                            TensorPtr top_k_weights)
    {

        auto g = input->node()->graph();
        int32_t H = input->shape()[1];
        int32_t T = route_pos->shape()[0];
        int32_t K = route_pos->shape()[1];
        TensorShape output_shape({T, H});
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("moe_unpermute"), output_shape);
        add_tensor_info(input, output);
        auto node = Node::create(g->get_valid_node_name("moe_unpermute"), OpType_MoEUnPermute);
        node->add_input(input);
        node->add_input(route_pos);
        node->add_input(top_k_weights);
        node->add_output(output);

        auto &params = node->params();
        params["top_k"] = K;

        g->add_node(node);
        node->infer_shape(true);
        quantize(node);
        return output;
    }

} // namespace ops
} // namespace aipubt
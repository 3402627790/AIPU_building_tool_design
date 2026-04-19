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
 * @Last Modified time: 2022-08-14 00:09:32
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr cast(TensorPtr input,
                   Tensor::Dtype to_dtype,
                   bool ignore_scale_zp,
                   ClipMode clip_mode,
                   const Quantization &quant)
    {
        auto g = input->node()->graph();
        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("cast"), input->shape());
        add_tensor_info(input, output, quant);
        output->set_type(to_dtype);
        output->quantization().set_aipu_dtype(to_dtype);

        auto node = Node::create(g->get_valid_node_name("cast"), OpType_Cast);
        node->add_input(input);
        node->add_output(output);
        node->params()["ignore_scale_zp"] = ignore_scale_zp;
        node->params()["clip_mode"] = mode_to_string(clip_mode);
        node->params()["to_dtype"] = to_dtype;
        g->add_node(node);
        node->infer_shape(true);
        quantize(node);

        return output;
    }

}; // namespace ops
}; // namespace aipubt
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
 * @Author: Aaron Gao
 * @Date: 2022-05-17 16:55:01
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-13 11:35:44
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> accidental_hits(TensorPtr true_classes,
                                           TensorPtr sampled_candidates)
    {
        if (true_classes->shape().dim() != 2)
        {
            ERROR() << "Tensor true classes rank must be 2! but now: " << true_classes->shape().dim();
            return {};
        }
        if (sampled_candidates->shape().dim() != 1)
        {
            ERROR() << "Tensor sampled candidates rank must be 1! but now: " << sampled_candidates->shape().dim();
            return {};
        }

        auto g = true_classes->node()->graph();
        int32_t max_matchs[] = {true_classes->shape()[0], true_classes->shape()[1]};
        int32_t max_match = *std::max_element(max_matchs, max_matchs + sizeof(max_matchs)/sizeof(max_matchs[0]));
        TensorShape idx_shape({max_match}, true_classes->shape().layout());
        auto indices = std::make_shared<Tensor>(g->get_valid_tensor_name(true_classes->name() + "/indices"), idx_shape);
        add_tensor_info(true_classes, indices);
        indices->quantization().set_aipu_dtype(Tensor::Dtype::UINT16);

        auto ids = std::make_shared<Tensor>(g->get_valid_tensor_name(true_classes->name() + "/ids"), idx_shape);
        add_tensor_info(true_classes, ids);
        ids->quantization().set_aipu_dtype(Tensor::Dtype::UINT16);

        TensorShape len_shape({1}, true_classes->shape().layout());
        auto length = std::make_shared<Tensor>(g->get_valid_tensor_name(true_classes->name() + "/length"), len_shape);
        add_tensor_info(true_classes, length);
        length->quantization().set_aipu_dtype(Tensor::Dtype::UINT16);

        auto node = Node::create(g->get_valid_node_name(true_classes->node()->name() + "/acchits"), OpType_AccidentalHits);
        node->add_input(true_classes);
        node->add_input(sampled_candidates);
        node->add_output(indices);
        node->add_output(ids);
        node->add_output(length);
        g->add_node(node);
        quantize(node);

        return {indices, ids, length};
    }

}; //namespace ops
}; // namespace aipubt
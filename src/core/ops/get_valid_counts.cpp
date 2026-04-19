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
 * @Last Modified time: 2022-08-13 11:32:47
 */
#include "ops_internal.h"
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> get_valid_counts(TensorPtr input,
                                            float score_threshold,
                                            int32_t id_index,
                                            int32_t score_index)
    {
        auto input_shape = input->shape();
        if (input_shape.dim() != 3)
        {
            ERROR() << " shape only support [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5].";
            return {};
        }
        uint32_t batch_size = input_shape[0];
        uint32_t num_anchors = input_shape[1];
        uint32_t anchors = input_shape[2];
        if ((anchors != 6) && (anchors != 5))
        {
            ERROR() << " shape only support [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5].";
            return {};
        }

        auto g = input->node()->graph();
        TensorShape valid_boxes_shape(batch_size);
        valid_boxes_shape.set_layout(input->shape().layout());
        TensorPtr valid_boxes = std::make_shared<Tensor>(g->get_valid_tensor_name("get_valid_counts/valid_boxes"), valid_boxes_shape);
        valid_boxes->set_type(input->type());
        valid_boxes->quantization() = Quantization(1.0, 0);

        TensorShape rearranged_data_shape(input->shape());
        rearranged_data_shape.set_layout(input->shape().layout());
        TensorPtr rearranged_data = std::make_shared<Tensor>(g->get_valid_tensor_name("get_valid_counts/rearranged_data"), rearranged_data_shape);
        rearranged_data->set_type(input->type());
        rearranged_data->quantization() = input->quantization();

        TensorShape related_index_shape(batch_size, num_anchors);
        related_index_shape.set_layout(input->shape().layout());
        TensorPtr related_index = std::make_shared<Tensor>(g->get_valid_tensor_name("get_valid_counts/related_index"), related_index_shape);
        related_index->set_type(input->type());
        related_index->quantization() = Quantization(1.0, 0);

        auto node = Node::create(g->get_valid_node_name("get_valid_counts"), OpType_GetValidCount);
        node->add_input(input);
        node->add_output(valid_boxes);
        node->add_output(rearranged_data);
        node->add_output(related_index);
        g->add_node(node);

        auto &params = node->params();
        params["score_threshold_value"] = score_threshold;
        params["score_threshold_type"] = Tensor::FP32;
        params["score_threshold"] = score_threshold;
        params["id_index"] = id_index;
        params["score_index"] = score_index;
        quantize(node);
        return {valid_boxes, rearranged_data, related_index};
    }
}; // namespace ops
}; // namespace aipubt
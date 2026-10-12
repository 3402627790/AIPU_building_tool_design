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
 * @Author: Neo Wang
 * @Date: 2021-04-14 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2026-09-11 11:30:37
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtr nms(TensorPtr boxes,
                  TensorPtr scores,
                  int64_t max_output_boxes_per_class,
                  float iou_threshold,
                  float score_threshold,
                  int32_t center_point_box)
    {
        if (boxes->shape().dim() != 3)
        {
            ERROR() << "boxes tensor's rank should be 3, but now is : "
                    << boxes->shape().dim();
            return {};
        }
        if (scores->shape().dim() != 3)
        {
            ERROR() << "scores tensor's rank should be 3, but now is: "
                    << scores->shape().dim();
            return {};
        }
        if (boxes->shape()[0] != scores->shape()[0])
        {
            ERROR() << "All input tensors first dimension which describles batch size should be same.";
            return {};
        }

        auto g = boxes->node()->graph();
        TensorShape indices_shape(boxes->shape()[0] * boxes->shape()[1], 3);
        SymbolShape symbol_shape({boxes->shape().symbol()[0] * boxes->shape().symbol()[1], 3},
                                 indices_shape.layout(), indices_shape.indexing());
        indices_shape.symbol() = symbol_shape;
        Tensor::Dtype indices_type = Tensor::INT32;
        auto selected_indices = std::make_shared<Tensor>(
            g->get_valid_tensor_name("selected_indices"), indices_shape, indices_type);
        selected_indices->quantization() = Quantization(1.0, 0);

        auto node = Node::create(g->get_valid_node_name("nms"), OpType::OpType_NMS);
        node->add_input(boxes);
        node->add_input(scores);
        node->add_output(selected_indices);
        g->add_node(node);

        auto &params = node->params();
        params["center_point_box"] = center_point_box;
        params["iou_threshold"] = iou_threshold;
        params["score_threshold"] = score_threshold;
        params["max_output_boxes_per_class"] = max_output_boxes_per_class;
        quantize(node);
        return selected_indices;
    }

}; // namespace ops
}; // namespace aipubt
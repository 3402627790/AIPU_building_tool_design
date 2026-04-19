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
 * @Last Modified time: 2022-08-13 11:35:09
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
#include "ops_internal.h"
namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> box_with_nms_limit(TensorPtr score,
                                              TensorPtr location,
                                              TensorPtr batch_id,
                                              float score_threshold,
                                              int32_t max_num_detections,
                                              NmsMethod method,
                                              float iou_threshold,
                                              float sigma,
                                              float nms_score_threshold,
                                              int32_t num_output_rois)
    {
        if (score->shape().dim() != 2 ||
            location->shape().dim() != 2 ||
            batch_id->shape().dim() != 1)
        {
            ERROR() << "score,localtion,batch_id tensor rank should be 2,2,1!";
            return {};
        }
        int32_t num_rois = score->shape()[0];
        int32_t num_classes = score->shape()[1];
        if (num_rois != location->shape()[0] ||
            num_rois != batch_id->shape()[0])
        {
            ERROR() << "location[0] & batch_id[0] should be equal to score[0]!";
            return {};
        }
        if (4 * num_classes != location->shape()[1])
        {
            ERROR() << "location[1] should be equal to score[1] * 4!";
            return {};
        }
        // if (num_classes <= 1)
        // {
        //     ERROR() << "score[1] should be > 1!";
        //     return {};
        // }

        auto g = score->node()->graph();
        TensorShape o_score_shape(num_output_rois,1);
        o_score_shape.set_layout(score->shape().layout());
        TensorPtr o_score = std::make_shared<Tensor>(g->get_valid_tensor_name("box_with_nms_limit/o_score"), o_score_shape);
        add_tensor_info(score, o_score);

        TensorShape o_location_shape(num_output_rois, 4);
        o_location_shape.set_layout(location->shape().layout());
        TensorPtr o_location = std::make_shared<Tensor>(g->get_valid_tensor_name("box_with_nms_limit/o_location"), o_location_shape);
        add_tensor_info(location, o_location);

        Quantization quant(1.0, 0, 32);
        TensorShape o_classes_shape(num_output_rois,1);
        o_classes_shape.set_layout(score->shape().layout());
        TensorPtr o_classes = std::make_shared<Tensor>(g->get_valid_tensor_name("box_with_nms_limit/o_classes"), o_classes_shape);
        add_tensor_info(batch_id, o_classes, quant);

        TensorShape o_batch_id_shape(num_output_rois,1);
        o_batch_id_shape.set_layout(score->shape().layout());
        TensorPtr o_batch_index = std::make_shared<Tensor>(g->get_valid_tensor_name("box_with_nms_limit/o_batch_index"), o_batch_id_shape);
        add_tensor_info(batch_id, o_batch_index, quant);

        TensorShape o_valid_box_shape(1,1);
        o_valid_box_shape.set_layout(score->shape().layout());
        TensorPtr valid_num_rois = std::make_shared<Tensor>(g->get_valid_tensor_name("box_with_nms_limit/valid_num_rois"), o_valid_box_shape);
        add_tensor_info(batch_id, valid_num_rois, quant);

        auto node = Node::create(g->get_valid_node_name("box_with_nms_limit"), OpType_BoxWithNmsLimit);
        node->add_input(score);
        node->add_input(location);
        node->add_input(batch_id);
        node->add_output(o_score);
        node->add_output(o_location);
        node->add_output(o_classes);
        node->add_output(o_batch_index);
        node->add_output(valid_num_rois);
        g->add_node(node);

        auto &params = node->params();
        params["scoreThreshold"] = score_threshold;
        params["maxNumDetections"] = max_num_detections;
        params["softNmsKernel"] = static_cast<int32_t>(method);
        if (method == NmsMethod::Hard || method == NmsMethod::Linear)
        {
            params["iouThreshold"] = iou_threshold;
        }
        else if (method == NmsMethod::Gaussian)
        {
            params["sigma"] = sigma;
        }
        params["nmsScoreThreshold"] = nms_score_threshold;
        quantize(node);

        return {o_score, o_location, o_classes, o_batch_index, valid_num_rois};
    }
}; //namespace ops
}; // namespace aipubt
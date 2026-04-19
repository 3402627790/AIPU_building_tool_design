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
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    TensorPtrList generate_proposals(TensorPtrList inputs,
                                     int32_t pre_max_num,
                                     int32_t post_max_num,
                                     float iou_threshold,
                                     float min_size,
                                     int32_t num_output_rois,
                                     std::vector<int32_t> image_info,
                                     std::vector<float> stride_info)
    {
        if (inputs.size() < 3)
        {
            ERROR() << " Required input: scores,deltas,anchors. Optional input: image_info.!";
            return {};
        }

        TensorPtr scores = inputs[0];
        TensorPtr deltas = inputs[1];
        TensorPtr anchors = inputs[2];
        TensorPtr image_info_t = nullptr;
        if (inputs.size() > 3)
        {
            image_info_t = inputs[3];
        }

        if (scores->shape().dim() != 4 ||
            deltas->shape().dim() != 4 ||
            anchors->shape().dim() != 2)
        {
            ERROR() << "scores,deltas,anchors,image_infor tensor rank must be: 4,4,2!";
            return {};
        }
        if (scores->shape().n() != deltas->shape().n() ||
            scores->shape().h() != deltas->shape().h() ||
            scores->shape().w() != deltas->shape().w())
        {
            ERROR() << "n/h/w dimensions size must be equal between scores and deltas!";
            return {};
        }
        if (scores->shape().c() * 4 != deltas->shape().c())
        {
            ERROR() << "deltas[c] should be equal to scores[c] * 4!";
            return {};
        }
        if (anchors->shape()[0] != scores->shape().c() ||
            anchors->shape()[1] != 4)
        {
            ERROR() << "anchors[0] should be equal scores[c], anchors[1] should be equal with 4!";
            return {};
        }
        if ((image_info_t != nullptr) &&
            (image_info_t->shape()[0] != scores->shape().n() ||
             image_info_t->shape()[1] != 3))
        {
            ERROR() << "image_info[0] should be equal scores[n], image_info[1] should be equal with 3!";
            return {};
        }
        if (inputs.size() == 3 && image_info.size() != 2)
        {
            ERROR() << "If inputs number is 3, need to set paramter 'image_info'!";
            return {};
        }

        auto g = scores->node()->graph();
        auto batch_size = scores->shape()[0];
        TensorShape o_scores_shape(batch_size, num_output_rois);
        o_scores_shape.set_layout(scores->shape().layout());
        TensorPtr o_scores = std::make_shared<Tensor>(g->get_valid_tensor_name("generate_proposals/o_scores"), o_scores_shape);
        o_scores->set_type(scores->type());
        o_scores->quantization() = scores->quantization();

        TensorShape o_location_shape(batch_size, num_output_rois, 4);
        o_location_shape.set_layout(anchors->shape().layout());
        TensorPtr o_location = std::make_shared<Tensor>(g->get_valid_tensor_name("generate_proposals/o_location"), o_location_shape);
        o_location->set_type(anchors->type());
        o_location->quantization() = anchors->quantization();

        Quantization quant(1.0, 0, 32);
        TensorShape o_batch_id_shape(batch_size, num_output_rois);
        o_batch_id_shape.set_layout(scores->shape().layout());
        TensorPtr o_batch_index = std::make_shared<Tensor>(g->get_valid_tensor_name("generate_proposals/o_batch_index"), o_batch_id_shape);
        o_batch_index->set_type(Tensor::INT32);
        o_batch_index->quantization() = quant;

        TensorShape o_valid_box_shape(batch_size, 1);
        o_valid_box_shape.set_layout(scores->shape().layout());
        TensorPtr valid_num_rois = std::make_shared<Tensor>(g->get_valid_tensor_name("generate_proposals/valid_num_rois"), o_valid_box_shape);
        valid_num_rois->set_type(Tensor::INT32);
        valid_num_rois->quantization() = quant;

        auto node = Node::create(g->get_valid_node_name("generate_proposals"), OpType_GenerateProposals);
        auto &params = node->params();
        node->add_input(scores);
        node->add_input(deltas);
        node->constants()["weights"] = anchors;
        if (inputs.size() == 3)
        {
            params["image_height"] = image_info[0];
            params["image_width"] = image_info[1];
        }
        else if (inputs.size() == 4)
        {
            float height_stride = 1.0;
            float width_stride = 1.0;
            if (stride_info.size() > 0)
                height_stride = stride_info[0];
            if (stride_info.size() > 1)
                width_stride = stride_info[1];
            node->add_input(image_info_t);
            params["height_stride"] = height_stride;
            params["width_stride"] = width_stride;
        }

        params["pre_nms_topn"] = pre_max_num;
        params["post_nms_topn"] = post_max_num;
        params["iou_threshold"] = iou_threshold;
        params["min_size"] = min_size;

        node->add_output(o_scores);
        node->add_output(o_location);
        node->add_output(o_batch_index);
        node->add_output(valid_num_rois);

        g->add_node(node);
        quantize(node);

        return {o_scores, o_location, o_batch_index, valid_num_rois};
    }
}; // namespace ops
}; // namespace aipubt
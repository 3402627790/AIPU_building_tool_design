//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2023 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Alex Zheng
 * @Date: 2023-11-1 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2023-11-1 11:04:33
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{

    TensorPtrList filter_boxes(TensorPtr boxes,
                               TensorPtr scores,
                               TensorPtr number,
                               TensorPtr label,
                               TensorPtr classes,
                               const NodeParamValue &score_threshold,
                               const int32_t max_box_num,
                               const std::vector<int32_t> &minsize,
                               const std::vector<Quantization> &quants)
    {
        std::vector<int32_t> minsize_loc = minsize;
        if (minsize_loc.size() == 1)
        {
            minsize_loc.push_back(minsize_loc[0]);
        }
        int32_t batch_size = boxes->shape()[0];
        int32_t num_classes = label->shape()[1];
        auto g = boxes->node()->graph();

        // Y1
        TensorShape out_boxes_shape(batch_size, max_box_num, 4);
        out_boxes_shape.set_layout(boxes->shape().layout());
        auto out_boxes = std::make_shared<Tensor>(g->get_valid_tensor_name("filter_boxes/boxes"), out_boxes_shape, boxes->type());
        if (quants.size() > 0)
        {
            add_tensor_info(boxes, out_boxes, quants[0]);
        }
        else
        {
            add_tensor_info(boxes, out_boxes);
        }

        // Y2
        TensorShape out_scores_shape(batch_size, max_box_num);
        out_scores_shape.set_layout(scores->shape().layout());
        auto out_scores = std::make_shared<Tensor>(g->get_valid_tensor_name("filter_boxes/scores"), out_scores_shape, scores->type());
        if (quants.size() > 1)
        {
            add_tensor_info(scores, out_scores, quants[1]);
        }
        else
        {
            add_tensor_info(scores, out_scores);
        }

        // Y3
        TensorShape out_number_shape(batch_size, num_classes);
        out_number_shape.set_layout(number->shape().layout());
        auto out_number = std::make_shared<Tensor>(g->get_valid_tensor_name("filter_boxes/number"), out_number_shape, number->type());
        if (quants.size() > 2)
        {
            add_tensor_info(number, out_number, quants[2]);
        }
        else
        {
            add_tensor_info(number, out_number);
        }

        // Y4
        TensorShape out_label_shape(batch_size, num_classes);
        out_label_shape.set_layout(label->shape().layout());
        auto out_label = std::make_shared<Tensor>(g->get_valid_tensor_name("filter_boxes/label"), out_label_shape, label->type());
        if (quants.size() > 3)
        {
            add_tensor_info(label, out_label, quants[3]);
        }
        else
        {
            add_tensor_info(label, out_label);
        }

        // Y5
        TensorShape out_classes_shape(batch_size, 1);
        out_classes_shape.set_layout(classes->shape().layout());
        auto out_classes = std::make_shared<Tensor>(g->get_valid_tensor_name("filter_boxes/classes"), out_classes_shape, classes->type());
        if (quants.size() > 4)
        {
            add_tensor_info(classes, out_classes, quants[4]);
        }
        else
        {
            add_tensor_info(classes, out_classes);
        }

        auto node = Node::create(g->get_valid_node_name("filter_boxes"), OpType::OpType_FilterBoxes);
        node->add_input(boxes);
        node->add_input(scores);
        node->add_input(number);
        node->add_input(label);
        node->add_input(classes);
        node->add_output(out_boxes);
        node->add_output(out_scores);
        node->add_output(out_number);
        node->add_output(out_label);
        node->add_output(out_classes);
        g->add_node(node);

        auto &params = node->params();
        params["maxnum"] = max_box_num;
        params["min_size"] = minsize_loc;

        bool quantized = false;
        if (score_threshold.t == NodeParamValue::NodeParamType_float)
        {
            node->params()["score_threshold"] = (float)score_threshold;
        }
        else
        {
            node->params()["score_threshold"] = (int)score_threshold;
            quantized = true;
        }
        node->set_attr("params_quantized", NodeParamValue(quantized));

        quantize(node);
        return {out_boxes, out_scores, out_number, out_label, out_classes};
    }

}; // namespace ops
}; // namespace aipubt
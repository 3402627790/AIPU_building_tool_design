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
 * @Last Modified time: 2022-04-29 11:30:37
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> nms(TensorPtr box_coor,
                               TensorPtr box_num_per_class,
                               TensorPtr class_num_per_batch,
                               TensorPtr score,
                               const std::vector<uint32_t> &image_info,
                               NmsMethod nms_method,
                               uint32_t max_output_size,
                               NodeParamValue score_threshold,
                               NodeParamValue iou_threshold,
                               float sigma,
                               bool center_point_box)
    {
        if (box_coor->shape().dim() != 3 || box_coor->shape()[2] != 4)
        {
            ERROR() << "Box coordinate tensor's rank should be 3, and shape[2] should be 4, but now: "
                    << box_coor->shape().dim() << ", " << box_coor->shape()[2];
            return {};
        }

        if (box_num_per_class->shape().dim() != 2 ||
            class_num_per_batch->shape().dim() != 2 ||
            score->shape().dim() != 2)
        {
            ERROR() << "Rank of 'box number per class'/'class number per batch'/'score of each box' tensors all should be 2.";
            return {};
        }

        if (box_coor->shape()[0] != box_num_per_class->shape()[0] ||
            box_coor->shape()[0] != class_num_per_batch->shape()[0] ||
            box_coor->shape()[0] != score->shape()[0])
        {
            ERROR() << "All input tensors first dimension which describles batch size should be same.";
            return {};
        }

        if (image_info.size() != 2)
        {
            ERROR() << "Size of image information vector should be 2.";
            return {};
        }
        std::map<NmsMethod, std::string> method_table = {
            {NmsMethod::Hard, "HARD"},
            {NmsMethod::Gaussian, "GAUSSIAN"},
            // Linear
        };
        if (!method_table.count(nms_method))
        {
            ERROR() << "Nms method only support Hard and GAUSSIAN now!";
            return {};
        }

        auto g = box_coor->node()->graph();
        auto out_coor = std::make_shared<Tensor>(g->get_valid_tensor_name("selected_coordinates"), box_coor->shape(), box_coor->type());
        out_coor->quantization() = box_coor->quantization();
        auto out_box_num = std::make_shared<Tensor>(g->get_valid_tensor_name("selected_box_number"), box_num_per_class->shape(), box_num_per_class->type());
        out_box_num->quantization() = box_num_per_class->quantization();
        auto out_score = std::make_shared<Tensor>(g->get_valid_tensor_name("selected_score"), score->shape(), score->type());
        out_score->quantization() = score->quantization();
        Tensor::Dtype out_indices_type = Tensor::Dtype::UINT16;
        if (Tensor::is_float(box_coor->type()))
        {
            out_indices_type = Tensor::Dtype::INT32;
        }

        auto out_indices = std::make_shared<Tensor>(g->get_valid_tensor_name("selected_indices"), score->shape(), out_indices_type);
        out_indices->quantization() = Quantization(1.0, 0);

        auto node = Node::create(g->get_valid_node_name("nms"), OpType::OpType_NMS);
        node->add_input(box_coor);
        node->add_input(box_num_per_class);
        node->add_input(class_num_per_batch);
        node->add_input(score);
        node->add_output(out_coor);
        node->add_output(out_box_num);
        node->add_output(out_score);
        node->add_output(out_indices);
        g->add_node(node);

        auto &params = node->params();
        params["center_point_box"] = center_point_box;
        params["iou_threshold"] = iou_threshold;
        params["score_threshold"] = score_threshold;
        params["max_output_size"] = max_output_size;
        if (nms_method == NmsMethod::Gaussian)
        {
            params["soft_nms_sigma"] = sigma;
        }
        else
        {
            params["iou_threshold"] = iou_threshold;
        }
        params["method"] = method_table[nms_method];

        bool quantized = false;
        if (score_threshold.t != NodeParamValue::NodeParamType_float)
        {
            quantized = true;
        }
        node->set_attr("params_quantized", NodeParamValue(quantized));
        quantize(node);
        return {out_coor, out_box_num, out_score, out_indices};
    }

}; // namespace ops
}; // namespace aipubt
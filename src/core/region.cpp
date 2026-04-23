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
 * @Author: Alex Zheng
 * @Date: 2023-10-26 11:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2023-10-26 11:04:33
 */
 #include <core/core.h>
 #include <core/node.h>
 #include <core/ops/ops.h>
 #include <core/ops/ops_utils.h>
 namespace aipubt
 {
 namespace ops
 {
     TensorPtrList region(TensorPtr input,
                          const NodeParamValue &obj_threshold,
                          const int32_t max_box_num,
                          const int32_t box_per_grid,
                          const int32_t class_num,
                          const int32_t grid_width,
                          const int32_t grid_height,
                          const bool grid_compensate,
                          const std::vector<float> anchors,
                          const std::vector<Quantization> &quants)
     {
         if ((input->shape().dim() != 5) || (input->shape()[4] < 6))
         {
             ERROR() << "input shape is [batch_size, s, s, anchor_num, num_classes + 5], but now: "
                     << input->shape();
             return {};
            }
    
            int32_t batch_size = input->shape()[0];
            int32_t num_classes = input->shape()[4] - 5;
            if (num_classes != class_num)
            {
                ERROR() << "class_num != num_classes";
                return {};
            }
            std::vector<float> default_anchors =
                {1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071};
            if (anchors.size() == 0)
            {
                WARN() << "The anchors is empty. use default anchors.";
            }
            else
            {
                default_anchors = anchors;
            }
            auto g = input->node()->graph();
            TensorShape scores_shape({batch_size, max_box_num});
            auto scores = std::make_shared<Tensor>(g->get_valid_tensor_name("region/scores"), scores_shape, Tensor::Dtype::UINT8);
            if ((quants.size() > 0) && !quants[0].empty())
            {
                scores->quantization() = quants[0];
            }
            else
            {
                scores->quantization() = input->quantization();
            }
    
            TensorShape coordinates_shape({batch_size, max_box_num, 4});
            auto coordinates = std::make_shared<Tensor>(g->get_valid_tensor_name("region/coordinates"), coordinates_shape, Tensor::Dtype::INT16);
            if ((quants.size() > 1) && !quants[1].empty())
            {
                coordinates->quantization() = quants[1];
            }
            else
            {
                coordinates->quantization() = Quantization(1.0, 0);
            }
    
            TensorShape boxes_number_shape({batch_size, class_num});
            auto boxes_number = std::make_shared<Tensor>(g->get_valid_tensor_name("region/boxes_number"), boxes_number_shape, Tensor::Dtype::INT16);
            boxes_number->quantization() = Quantization(1.0, 0);
    
            TensorShape valid_classes_shape({batch_size, class_num});
            auto valid_classes = std::make_shared<Tensor>(g->get_valid_tensor_name("region/valid_classes"), valid_classes_shape, Tensor::Dtype::INT16);
            valid_classes->quantization() = Quantization(1.0, 0);
    
            TensorShape total_class_shape({batch_size, 1});
            auto total_class = std::make_shared<Tensor>(g->get_valid_tensor_name("region/total_class"), total_class_shape, Tensor::Dtype::INT16);
            total_class->quantization() = Quantization(1.0, 0);
    
            auto node = Node::create(g->get_valid_node_name("region"), OpType::OpType_Region);
            node->add_input(input);
            node->add_output(scores);
            node->add_output(coordinates);
            node->add_output(boxes_number);
            node->add_output(valid_classes);
            node->add_output(total_class);
            g->add_node(node);
    
            auto &params = node->params();
    
            params["obj_thresh"] = obj_threshold;
            params["max_box_num"] = max_box_num;
            params["box_per_grid"] = box_per_grid;
            params["class_num"] = class_num;
            params["grid_width"] = grid_width;
            params["grid_height"] = grid_height;
            params["grid_compensate"] = grid_compensate;
            params["anchors"] = default_anchors;
            bool quantized = false;
            if (obj_threshold.t != NodeParamValue::NodeParamType_float)
            {
                quantized = true;
            }
            node->set_attr("params_quantized", NodeParamValue(quantized));
    
            quantize(node);
            return {scores, coordinates, boxes_number, valid_classes, total_class};
        }
    
    }; // namespace ops
    }; // namespace aipubt
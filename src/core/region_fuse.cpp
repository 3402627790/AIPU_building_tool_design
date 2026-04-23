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
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-27 16:20:50
 */
 #include <core/core.h>
 #include <core/node.h>
 #include <core/ops/ops.h>
 #include <core/ops/ops_utils.h>
 namespace aipubt
 {
 namespace ops
 {
     std::vector<TensorPtr> region_fuse(std::vector<TensorPtr> inputs,
                                        const int32_t class_num,
                                        const std::vector<Quantization> &quants)
     {
         if (inputs.size() != 10)
         {
             ERROR() << "10 inputs are required, now it is " << inputs.size() << ".";
             return {};
         }
         // X1 X2 batch check
         if ((inputs[0]->shape().dim() != 2) || (inputs[1]->shape().dim() != 2) || (inputs[0]->shape()[0] != inputs[1]->shape()[0]))
         {
             ERROR() << "X1 X2 shape error.";
             return {};
         }
         // X3 X4 shape check
         if ((inputs[2]->shape().dim() != 3) || (inputs[3]->shape().dim() != 3) || (inputs[2]->shape()[0] != inputs[3]->shape()[0]))
         {
             ERROR() << "X3 X4 shape error.";
             return {};
         }
 
         auto g = inputs[0]->node()->graph();
         int32_t batch_size = inputs[0]->shape()[0];
         int32_t max_box_num = inputs[0]->shape()[1] + inputs[1]->shape()[1];
         TensorShape scores_shape({batch_size, max_box_num});
         auto scores = std::make_shared<Tensor>(g->get_valid_tensor_name("region_fuse/scores"), scores_shape, Tensor::Dtype::UINT8);
         if ((quants.size() > 0) && !quants[0].empty())
         {
             scores->quantization() = quants[0];
         }
         else
         {
             scores->quantization() = inputs[0]->quantization();
         }
 
         TensorShape coordinates_shape({batch_size, max_box_num, 4});
         auto coordinates = std::make_shared<Tensor>(g->get_valid_tensor_name("region_fuse/coordinates"), coordinates_shape, Tensor::Dtype::INT16);
         if ((quants.size() > 1) && !quants[1].empty())
         {
             coordinates->quantization() = quants[1];
         }
         else
         {
             coordinates->quantization() = Quantization(1.0, 0);
         }
 
         TensorShape boxes_number_shape({batch_size, class_num});
         auto boxes_number = std::make_shared<Tensor>(g->get_valid_tensor_name("region_fuse/boxes_number"), boxes_number_shape, Tensor::Dtype::INT16);
         boxes_number->quantization() = Quantization(1.0, 0);
 

         TensorShape valid_classes_shape({batch_size, class_num});
         auto valid_classes = std::make_shared<Tensor>(g->get_valid_tensor_name("region_fuse/valid_classes"), valid_classes_shape, Tensor::Dtype::INT16);
         valid_classes->quantization() = Quantization(1.0, 0);
 
         TensorShape maximum_shape({batch_size, 1});
         auto maximum_class = std::make_shared<Tensor>(g->get_valid_tensor_name("region_fuse/maximum_class"), maximum_shape, Tensor::Dtype::INT16);
         maximum_class->quantization() = Quantization(1.0, 0);
 
         auto node = Node::create(g->get_valid_node_name("region_fuse"), OpType::OpType_RegionFuse);
         for (auto input : inputs)
         {
             node->add_input(input);
         }
         node->add_output(scores);
         node->add_output(coordinates);
         node->add_output(boxes_number);
         node->add_output(valid_classes);
         node->add_output(maximum_class);
         g->add_node(node);
 
         auto &params = node->params();
         params["class_num"] = class_num;
         quantize(node);
         return {scores, coordinates, boxes_number, valid_classes, maximum_class};
     }
 }; //namespace ops
 }; // namespace aipubt
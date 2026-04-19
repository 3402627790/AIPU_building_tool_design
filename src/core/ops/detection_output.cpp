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
    std::vector<TensorPtr> detection_output(TensorPtr scores,
                                            TensorPtr coordinates,
                                            TensorPtr anchor,
                                            const int32_t width,
                                            const int32_t height,
                                            const NodeParamValue score_threshold,
                                            const int32_t max_output_size,
                                            std::vector<float> variance,
                                            const std::vector<Quantization> &quants)
    {
        int32_t batch_size = scores->shape()[0];
        auto g = scores->node()->graph();

        // Y1
        TensorShape Y1_shape(batch_size, max_output_size);
        auto Y1 = std::make_shared<Tensor>(g->get_valid_tensor_name("detection_output/Y1"), Y1_shape, scores->type());
        if (quants.size() > 0)
        {
            add_tensor_info(scores, Y1, quants[0]);
        }
        else
        {
            add_tensor_info(scores, Y1);
        }

        // Y2
        TensorShape Y2_shape(batch_size, max_output_size, 4);
        auto Y2 = std::make_shared<Tensor>(g->get_valid_tensor_name("detection_output/Y2"), Y2_shape, anchor->type());
        if (quants.size() > 1)
        {
            add_tensor_info(anchor, Y2, quants[1]);
        }
        else
        {
            add_tensor_info(anchor, Y2);
        }

        // Y3
        TensorShape Y3_shape(batch_size, max_output_size);
        auto Y3 = std::make_shared<Tensor>(g->get_valid_tensor_name("detection_output/Y3"), Y3_shape, anchor->type());
        if (quants.size() > 2)
        {
            add_tensor_info(anchor, Y3, quants[2]);
        }
        else
        {
            add_tensor_info(anchor, Y3);
        }

        // Y4
        TensorShape Y4_shape(batch_size, max_output_size);
        auto Y4 = std::make_shared<Tensor>(g->get_valid_tensor_name("detection_output/Y4"), Y4_shape, anchor->type());
        if (quants.size() > 3)
        {
            add_tensor_info(anchor, Y4, quants[3]);
        }
        else
        {
            add_tensor_info(anchor, Y4);
        }

        // Y5
        TensorShape Y5_shape(batch_size, 1);
        auto Y5 = std::make_shared<Tensor>(g->get_valid_tensor_name("detection_output/Y5"), Y5_shape, anchor->type());
        if (quants.size() > 4)
        {
            add_tensor_info(anchor, Y5, quants[4]);
        }
        else
        {
            add_tensor_info(anchor, Y5);
        }
        auto node = Node::create(g->get_valid_node_name("detection_output"),
                                 OpType::OpType_DetectionOutput);
        node->add_input(scores);
        node->add_input(coordinates);
        node->add_input(anchor);
        node->add_output(Y1);
        node->add_output(Y2);
        node->add_output(Y3);
        node->add_output(Y4);
        node->add_output(Y5);
        g->add_node(node);
        auto &params = node->params();
        params["score_threshold"] = score_threshold;
        params["max_output_size"] = max_output_size;
        params["width"] = width;
        params["height"] = height;
        params["variance"] = variance;
        bool quantized = false;
        if (score_threshold.t != NodeParamValue::NodeParamType_float)
        {
            quantized = true;
        }
        node->set_attr("params_quantized", NodeParamValue(quantized));
        // TODO: Enable after qtlib support
        //  quantize(node);
        return {Y1, Y2, Y3, Y4, Y5};
    }
}; // namespace ops
}; // namespace aipubt
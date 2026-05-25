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
 * @Author: Samuel Han
 * @Date: 2023-04-17 12:59:04
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-17 17:57:44
 */

#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"
#include "quantization_transform/utils/data_type.h"

namespace aipubt
{
namespace engine
{

    static void pooling_l2_forward(NodePtr node)
    {
        int kernel_x = node->params()["kernel_x"];
        int kernel_y = node->params()["kernel_y"];
        int stride_x = node->params()["stride_x"];
        int stride_y = node->params()["stride_y"];
        // int dilation_x = node->params()["dilation_x"];
        // int dilation_y = node->params()["dilation_y"];
        int pad_left = node->params()["pad_left"];
        int pad_right = node->params()["pad_right"];
        int pad_top = node->params()["pad_top"];
        int pad_bottom = node->params()["pad_bottom"];

        auto dtype = node->inputs()[0]->type();
        ComputeTensor inp = node->inputs()[0];
        auto pad = ComputeTensor::zeros({inp->shape()[0], inp->shape()[1] + pad_top + pad_bottom,
                                         inp->shape()[2] + pad_left + pad_right, inp->shape()[3]},
                                        dtype);
        pad[{{0, inp->shape()[0]}, {pad_top, pad_top + inp->shape()[1]}, {pad_left, pad_left + inp->shape()[2]}, {0, inp->shape()[3]}}] = inp;
        int out_h = (pad->shape()[1] - kernel_y) / stride_y + 1;
        int out_w = (pad->shape()[2] - kernel_x) / stride_x + 1;
        auto out = ComputeTensor::zeros({inp->shape()[0], out_h, out_w, inp->shape()[3]}, dtype);
        for (int h = 0; h < out_h; h++)
        {
            for (int w = 0; w < out_w; w++)
            {
                ComputeTensor t = pad[{{0, -1}, {h * stride_y, h * stride_y + kernel_y}, {w * stride_x, w * stride_x + kernel_x}, {0, -1}}];
                t = t.pow(2);
                t = t.sum({1, 2}, true).astype(dtype);
                t = t.sqrt();
                out[{{0, -1}, {h, h + 1}, {w, w + 1}, {0, -1}}] = t;
            }
        }
        auto placeholder = out.pow(2);
        placeholder->set_type(dtype);
        node->placeholders().push_back(placeholder);
        node->outputs()[0]->set_type(dtype);
        node->outputs()[0]->share_data_from(out);
    }

    EXEOP_REGISTER(OpType_Pooling,
                   pooling_l2_forward,
                   DTCriteria::FP_ONLY,
                   [](NodePtr node) -> bool { return node->params()["method"] == "L2"; });
} // namespace engine

} // namespace aipubt

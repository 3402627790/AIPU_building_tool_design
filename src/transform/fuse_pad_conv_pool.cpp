//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-06-16 14:07:21
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-06-30 15:14:21
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"

namespace aipubt
{
static inline bool criteria(NodePtr n1, NodePtr n2)
{
    auto is_pads_postive = [](std::vector<std::vector<int32_t>> pads) -> bool
    {
        for (auto pad : pads)
        {
            if (std::all_of(pad.begin(), pad.end(), [](int32_t p)
                            { return p < 0; }))
            {
                return false;
            }
        }
        return true;
    };
    auto is_quantized_constant_value = [n1, n2](float constant_value) -> bool
    {
        if (constant_value + n1->outputs()[0]->quantization().offset() == 0)
            return true;
        else
            return false;
    };

    auto is_spatial_pad = [](std::vector<std::vector<int32_t>> pads) -> bool
    {
        bool spatial = true;
        for (size_t i = 0; i < pads.size(); ++i)
        {
            if (i == 0 || i == pads.size() - 1)
            {
                bool allZero = std::all_of(pads[i].begin(), pads[i].end(), [](const int &num)
                                           { return num == 0; });
                if (!allZero)
                {
                    spatial = false;
                    break;
                }
            }
        }
        return spatial;
    };

    auto is_pad_calculated = [n1, n2](std::vector<std::vector<int32_t>> pads) -> bool
    {
        for (const auto &pad : pads)
        {
            if (pad[0] != 0)
            {
                return true;
            }
        }
        auto params = n2->params();
        auto i_shape = n1->inputs()[0]->shape();
        bool ceil_mode = params["ceil_mode"];
        int kernel_y = params["kernel_y"];
        int kernel_x = params["kernel_x"];
        int stride_y = params["stride_y"];
        int stride_x = params["stride_x"];
        int dilation_y = params["dilation_y"];
        int dilation_x = params["dilation_x"];

        if (n2->type() == OpType_Pooling)
        {
            if (kernel_y < pads[1][1] || kernel_x < pads[2][1])
            {
                return true;
            }

            int pad_left = params["pad_left"];
            int pad_right = params["pad_right"];
            int pad_top = params["pad_top"];
            int pad_bottom = params["pad_bottom"];

            if ((i_shape[1] + pad_top + pad_bottom + pads[1][1] - ((kernel_y - 1) * dilation_y + 1)) % stride_y != 0)
            {
                if (ceil_mode)
                {
                    return true;
                }
            }
            else
            {
                return true;
            }

            if ((i_shape[2] + pad_left + pad_right + pads[2][1] - ((kernel_x - 1) * dilation_x + 1)) % stride_x != 0)
            {
                if (ceil_mode)
                {
                    return true;
                }
            }
            else
            {
                return true;
            }
        }
        else if (n2->type() == OpType_Pooling3D)
        {
            int kernel_z = params["kernel_z"];
            int stride_z = params["stride_z"];
            int dilation_z = params["dilation_z"];

            int pad_x_begin = params["pad_x_begin"];
            int pad_x_end = params["pad_x_end"];
            int pad_y_begin = params["pad_y_begin"];
            int pad_y_end = params["pad_y_end"];
            int pad_z_begin = params["pad_z_begin"];
            int pad_z_end = params["pad_z_end"];

            if (kernel_z < pads[1][1] || kernel_y < pads[2][1] || kernel_x < pads[3][1])
            {
                return true;
            }

            if ((i_shape[1] + pad_z_begin + pad_z_end + pads[1][1] - ((kernel_z - 1) * dilation_z + 1)) % stride_z != 0)
            {
                if (ceil_mode)
                {
                    return true;
                }
            }
            else
            {
                return true;
            }

            if ((i_shape[2] + pad_y_begin + pad_y_end + pads[2][1] - ((kernel_y - 1) * dilation_y + 1)) % stride_y != 0)
            {
                if (ceil_mode)
                {
                    return true;
                }
            }
            else
            {
                return true;
            }

            if ((i_shape[3] + pad_x_begin + pad_x_end + pads[3][1] - ((kernel_x - 1) * dilation_x + 1)) % stride_x != 0)
            {
                if (ceil_mode)
                {
                    return true;
                }
            }
            else
            {
                return true;
            }
        }
        return false;
    };

    bool ret = true;

    std::string mode = n1->params()["mode"];
    float constant_value = n1->params()["constant_value"];
    std::vector<std::vector<int32_t>> pads = n1->params()["pads"];

    if ((int32_t)pads.size() != n2->inputs()[0]->shape().dim())
    {
        return false;
    }

    if (n2->type() == OpType_Pooling || n2->type() == OpType_Pooling3D)
    {
        std::string method = n2->params()["method"];
        if ((method == "MAX") && is_pad_calculated(pads))
        {
            return false;
        }
        if (method == "AVG")
        {
            bool count_pad = n2->params()["count_include_pad"];
            if (!count_pad)
            {
                return false;
            }
        }
    }
    ret = ret && (!n1->outputs()[0]->is_io_tensor());
    ret = ret && (mode == "CONSTANT");
    ret = ret && is_spatial_pad(pads);
    ret = ret && ((constant_value == 0) || is_quantized_constant_value(constant_value));
    ret = ret && is_pads_postive(pads);
    // The expression cannot perform size comparison, so it is excluded.
    if (n1->params().count("ds_pads"))
    {
        std::vector<std::vector<tongue::Expr>> ds_pads = n1->params()["ds_pads"];
        for (auto exprs : ds_pads)
        {
            if (std::any_of(exprs.begin(), exprs.end(), [](tongue::Expr e)
                            { return !e->is_constant(); }))
            {
                return false;
            }
        }
    }

    return ret;
}

bool fuse_pad_conv_pool(GraphPtr g)
{
    // TODO: maybe pad + convolution serial.
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Pad}, {"1", OpType_Pooling}}, {{"0", "1"}}},
        {{{"0", OpType_Pad}, {"1", OpType_Pooling3D}}, {{"0", "1"}}},
        {{{"0", OpType_Pad}, {"1", OpType_Convolution}}, {{"0", "1"}}},
        {{{"0", OpType_Pad}, {"1", OpType_Convolution3D}}, {{"0", "1"}}},
        {{{"0", OpType_Pad}, {"1", OpType_DepthwiseConv}}, {{"0", "1"}}},
    };
    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m[0], m[1]))
                {
                    continue;
                }
                std::vector<std::vector<int32_t>> pads = m[0]->params()["pads"];
                auto pad_size = pads.size();
                auto params = m[1]->params();
                if (pad_size == 5)
                {
                    // pads=[n, d, h, w, c]
                    m[1]->params()["pad_x_begin"] = params["pad_x_begin"] + pads[3][0];
                    m[1]->params()["pad_x_end"] = params["pad_x_end"] + pads[3][1];
                    m[1]->params()["pad_y_begin"] = params["pad_y_begin"] + pads[2][0];
                    m[1]->params()["pad_y_end"] = params["pad_y_end"] + pads[2][1];
                    m[1]->params()["pad_z_begin"] = params["pad_z_begin"] + pads[1][0];
                    m[1]->params()["pad_z_end"] = params["pad_z_end"] + pads[1][1];
                }
                else if (pad_size == 4)
                {
                    m[1]->params()["pad_top"] = params["pad_top"] + pads[1][0];
                    m[1]->params()["pad_bottom"] = params["pad_bottom"] + pads[1][1];
                    m[1]->params()["pad_left"] = params["pad_left"] + pads[2][0];
                    m[1]->params()["pad_right"] = params["pad_right"] + pads[2][1];
                }
                auto inpt = m[0]->remove_input(0);
                m[1]->clear_inputs();
                m[1]->add_input(inpt);

                m[0]->clear_inputs();
                m[0]->clear_outputs();
                g->remove_node(m[0]);
                is_changed = true;
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}

} // namespace aipubt

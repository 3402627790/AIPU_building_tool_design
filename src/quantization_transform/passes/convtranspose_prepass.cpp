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
 * @Date: 2026-04-08 15:31:51
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2026-04-08 17:07:26
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/location.h"
#include "core/ops/ops.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
static std::vector<aipubt::ComputeTensor> get_scales(aipubt::NodePtr node)
{
    aipubt::ComputeTensor input_scales = node->inputs()[0]->quantization().scales();
    aipubt::ComputeTensor act_scales;
    if (node->constants().count("quantize_scale"))
    {
        act_scales = node->constants()["quantize_scale"];
    }
    else if (node->params().count("quantize_scale"))
    {
        act_scales = std::vector<float>(node->params()["quantize_scale"]);
    }
    std::vector<aipubt::ComputeTensor> ret = {input_scales, act_scales};
    return ret;
}
static std::vector<aipubt::ComputeTensor> get_zps(aipubt::NodePtr node)
{
    aipubt::ComputeTensor input_zps = node->inputs()[0]->quantization().offsets();
    input_zps = input_zps.astype(aipubt::Tensor::FP32);
    aipubt::ComputeTensor act_zps;
    if (node->constants().count("quantize_zp"))
    {
        act_zps = node->constants()["quantize_zp"];
    }
    else if (node->params().count("quantize_zp"))
    {
        act_zps = std::vector<float>(node->params()["quantize_zp"]);
    }
    std::vector<aipubt::ComputeTensor> ret = {input_zps, act_zps};
    return ret;
}

static int32_t criteria(aipubt::NodePtr node)
{
    if (node->type() != aipubt::OpType_DeQuantize)
    {
        return 0;
    }
    auto scales = get_scales(node);
    auto input_scales = scales[0];
    auto act_scales = scales[1];
    auto zps = get_zps(node);
    auto input_zps = zps[0];
    auto act_zps = zps[1];

    if ((node->constants().count("quantize_scale") == 0) &&
        (node->params().count("quantize_scale") == 0))
    {
        return 0;
    }

    if (act_zps->shape().size() != input_zps->shape().size() ||
        act_scales->shape().size() != input_scales->shape().size())
    {
        return 0;
    }

    if (input_scales != act_scales && input_zps != act_zps)
    {
        return 1;
    }
    else if (input_scales != act_scales)
    {
        return 2;
    }
    else if (input_zps != act_zps)
    {
        return 3;
    }
    return 0;
}
void dequantize_preprocess(aipubt::GraphPtr g)
{
    // CSM-1005, dequantize has different node.inputs[0].quantization.scale with node.params['quantize_scale']
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        auto ret = criteria(node);
        if (ret == 0)
        {
            continue;
        }

        auto scales = get_scales(node);
        auto input_scales = scales[0];
        auto act_scales = scales[1];

        // auto zps = get_zps(node);
        // auto input_zps = zps[0];
        // auto act_zps = zps[1];
        if (ret == 2)
        {
            // only need to compensate scale
            if (node->params().count("quantize_scale"))
            {
                node->params()["quantize_scale"] = aipubt::NodeParamValue(input_scales);
            }
            else
            {
                node->constants()["quantize_scale"] = input_scales;
            }

            auto deq_out = node->outputs()[0];
            auto consumers = deq_out->consumers();
            auto compensate_scale = act_scales.astype(aipubt::Tensor::FP32) / input_scales.astype(aipubt::Tensor::FP32);
            auto constant_ot = aipubt::ops::constant(compensate_scale, aipubt::Quantization(), g);
            auto mul_ot = aipubt::ops::mul(deq_out, constant_ot);

            for (auto consumer : consumers)
            {
                auto idx = consumer->remove_input(deq_out);
                consumer->add_input(mul_ot, idx);
            }
        }
        else
        {
            // TODO
            WARN() << "[qtlib] now only support input_scale != params['quantize_scale].";
        }
    }
}
}; // namespace qtlibn
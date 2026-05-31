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
 * @Date: 2023-10-16 14:14:07
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-18 16:03:36
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

static bool clip_criteira(aipubt::NodePtr n)
{
    auto is_int = [](aipubt::NodeParamValue v) -> bool {
        bool ret = false;
        if (v.t == aipubt::NodeParamValue::NodeParamType_int || v.t == aipubt::NodeParamValue::NodeParamType_uint)
        {
            ret = true;
        }
        return ret;
    };

    bool ret = false;
    if ((n->type() == aipubt::OpType_Activation && (std::string)n->params()["method"] == "CLIP") ||
        n->type() == aipubt::OpType_Clip)
    {
        aipubt::NodeParamValue clip_min = n->params()["clip_min"];
        aipubt::NodeParamValue clip_max = n->params()["clip_max"];
        ret = is_int(clip_min) && is_int(clip_max);
    }
    return ret;
}

static bool clip_process(aipubt::NodePtr node)
{
    int32_t clip_min = node->params()["clip_min"];
    int32_t clip_max = node->params()["clip_max"];

    auto out_scale = node->outputs()[0]->quantization().scale();
    auto out_zp = node->outputs()[0]->quantization().offset();
    float f_clip_min = DEQUANTIZE(clip_min, out_scale, out_zp);
    float f_clip_max = DEQUANTIZE(clip_max, out_scale, out_zp);
    node->params()["clip_min"] = f_clip_min;
    node->params()["clip_max"] = f_clip_max;
    return true;
}

static std::vector<std::vector<std::function<bool(aipubt::NodePtr)>>> _processes = {
    {clip_criteira, clip_process},
};

void set_param_quantized(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        for (auto process : _processes)
        {
            if (process[0](node))
            {
                process[1](node);
            }
        }
    }
}

} // namespace qtlibn

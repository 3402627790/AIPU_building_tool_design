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
 * @Author: Kyle Huang
 * @Date: 2023-10-12
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2023-10-12
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static double int_dtype_max(const Tensor::Dtype dtype)
{
    double ret = 0.0;
    switch (dtype)
    {
    case Tensor::INT8:
        ret = INT8_MAX;
        break;
    case Tensor::INT16:
        ret = INT16_MAX;
        break;
    case Tensor::INT32:
        ret = INT32_MAX;
        break;
    // // -Werror,-Wimplicit-const-int-float-conversion
    // case Tensor::INT64:
    //     ret = INT64_MAX;
    //     break;
    case Tensor::UINT8:
        ret = UINT8_MAX;
        break;
    case Tensor::UINT16:
        ret = UINT16_MAX;
        break;
    case Tensor::UINT32:
        ret = UINT32_MAX;
        break;
    // // -Werror,-Wimplicit-const-int-float-conversion
    // case Tensor::UINT64:
    //     ret = UINT64_MAX;
    //     break;
    default:
        ERROR() << "Cannot support this type: " << Tensor::dtype_to_string(dtype);
        break;
    }
    return ret;
}

static double int_dtype_min(const Tensor::Dtype dtype)
{
    double ret = 0.0;
    switch (dtype)
    {
    case Tensor::INT8:
        ret = INT8_MIN;
        break;
    case Tensor::INT16:
        ret = INT16_MIN;
        break;
    case Tensor::INT32:
        ret = INT32_MIN;
        break;
    // // -Werror,-Wimplicit-const-int-float-conversion
    // case Tensor::INT64:
    //     ret = INT64_MIN;
    //     break;
    case Tensor::UINT8:
    case Tensor::UINT16:
    case Tensor::UINT32:
    // // -Werror,-Wimplicit-const-int-float-conversion
    // case Tensor::UINT64:
        ret = 0.0;
        break;
    default:
        ERROR() << "Cannot support this type: " << Tensor::dtype_to_string(dtype);
        break;
    }
    return ret;
}

static inline bool creteria(NodePtr n1, NodePtr n2)
{
    if (n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (n1->children().size() > 1)
    {
        return false;
    }
    if (!n1->constants().count("weights"))
    {
        return false;
    }
    auto weight = n1->constants()["weights"];
    if (!weight->data<void>())
    {
        return false;
    }
    if (n2->outputs()[0]->type() == Tensor::INT64 ||
        n2->outputs()[0]->type() == Tensor::UINT64 ||
        n2->outputs()[0]->type() == Tensor::FP64)
    {
        return false;
    }
    return true;
}

bool fuse_constant_cast(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Constant}, {"1", OpType_Cast}},
        {{"0", "1"}}};
    GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!creteria(m[0], m[1]))
            {
                continue;
            }
            auto f = m[1]->inputs()[0];
            auto t = m[1]->outputs()[0];
            auto to_dtype = t->type();

            std::string clip_mode = "saturation";
            if (m[1]->params().count("clip_mode"))
            {
                clip_mode = m[1]->params()["clip_mode"].s;
            }
            transform(clip_mode.begin(), clip_mode.end(), clip_mode.begin(), ::tolower);

            bool ignore_scale_zp = false;
            if (m[1]->params().count("ignore_scale_zp"))
            {
                ignore_scale_zp = m[1]->params()["ignore_scale_zp"];
            }

            int scale_value = -1;
            if (m[1]->params().count("scale_value"))
            {
                scale_value = m[1]->params()["scale_value"];
            }

            int shift_value = -1;
            if (m[1]->params().count("shift_value"))
            {
                shift_value = m[1]->params()["shift_value"];
            }

            ComputeTensor weight = m[0]->constants()["weights"];
            // refer: AIPU_building_tool_design/python/src/AIPUBuilder/executor/ops/cast.py
            if (Tensor::is_float(f->type()))
            {
                // fp->fp: just astype
                // fp->int: clamp + astype
                if (!Tensor::is_float(to_dtype))
                {
                    weight = weight.clamp(int_dtype_min(to_dtype), int_dtype_max(to_dtype));
                }
            }
            else
            {
                // int->fp: just astype
                // int->int: calculation + clamp + astype
                if (!Tensor::is_float(to_dtype))
                {
                    int in_zp = f->quantization().offset();
                    int out_zp = t->quantization().offset();
                    weight = weight.astype(Tensor::FP64);
                    if (scale_value != -1 && shift_value != -1)
                    {
                        // if has scale shift, ignore clip_mode amd ignore_scale_zp
                        // y = (x + in_zp) * quant - out_zp
                        double quant = (double)scale_value / (double)((int64_t)1 << shift_value);
                        weight = (weight + in_zp) * quant;
                        weight = weight.round();
                        weight = weight - out_zp;
                        weight = weight.clamp(int_dtype_min(to_dtype), int_dtype_max(to_dtype));
                    }
                    else
                    {
                        // if no scale shift
                        if (clip_mode == "saturation")
                        {
                            if (ignore_scale_zp)
                            {
                                in_zp = 0;
                                out_zp = 0;
                            }
                            weight = weight + in_zp - out_zp;
                            weight = weight.clamp(int_dtype_min(to_dtype), int_dtype_max(to_dtype));
                        }
                        // truncation just astype, no need clamp
                    }
                }
            }
            ComputeTensor new_weight = weight.astype(to_dtype);

            new_weight->monopolize_data();
            m[0]->constants()["weights"] = new_weight;
            m[0]->constants()["weights"]->quantization() = t->quantization();
            m[1]->clear_outputs();
            m[0]->clear_outputs();
            m[0]->add_output(t);

            g->remove_node(m[1]);

            is_merged = true;
        }
        if (!is_merged)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}
}; // namespace aipubt

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
/*a
 * @Author: Samuel Han
 * @Date: 2022-03-30 13:38:32
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-03-30 14:21:45
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace aipubt
{

static inline bool criteria(NodePtr n, NodePtr c, TensorPtr &d)
{
    if ((std::string)n->params()["method"] == "MIN" && (std::string)n->params()["with_activation"] == "RELU")
    {
        auto w = c->constants()["weights"];
        //only support  int
        if (!(w->type() == Tensor::Dtype::FP32 ||
              (Tensor::Dtype::UINT8 <= w->type() && w->type() <= Tensor::Dtype::INT64)))
        {
            return false;
        }
        d = w;
        int axis = 0;
        for (int i = 0; i < w->shape().dim(); ++i)
        {
            if (w->shape()[i] == 1)
            {
                axis += 1;
                continue;
            }
            d = shrink_tensor_data(d, axis);
            if (!d)
            {
                return false;
            }
        }
        return true;
    }

    return false;
}

bool fuse_clamp(GraphPtr g)
{
    GraphMatcherPattern pattern = {
        {{"0", OpType_Constant}, {"1", OpType_Eltwise}},
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
            TensorPtr d = nullptr;
            if (!criteria(m[1], m[0], d))
            {
                continue;
            }
            auto in = m[1]->inputs()[0];
            if (in == m[0]->outputs()[0])
            {
                in = m[1]->inputs()[1];
            }
            auto dt = d->type();
            TensorPtr o = m[1]->outputs()[0];
            if (dt == Tensor::Dtype::FP32)
            {
                float upper = d->data<float>()[0];
                o = ops::clip(in, 0., upper);
            }
            else
            {
                int64_t upper = 0;
                if (dt == Tensor::Dtype::UINT8)
                {
                    upper = d->data<uint8_t>()[0];
                }
                else if (dt == Tensor::Dtype::UINT16)
                {
                    upper = d->data<uint16_t>()[0];
                }
                else if (dt == Tensor::Dtype::UINT32)
                {
                    upper = d->data<uint32_t>()[0];
                }
                else if (dt == Tensor::Dtype::INT8)
                {
                    upper = d->data<int8_t>()[0];
                }
                else if (dt == Tensor::Dtype::INT16)
                {
                    upper = d->data<int16_t>()[0];
                }
                else if (dt == Tensor::Dtype::INT32)
                {
                    upper = d->data<int32_t>()[0];
                }
                else
                {
                    WARN() << "un-support dtype = " << dt << " in fuse_clamp pass.";
                }

                if (o->quantization().empty())
                {
                    o = ops::clip(in, 0, upper, o->quantization());
                }
                else
                {
                    auto o_scale = m[0]->outputs()[0]->quantization().scale();
                    auto o_zp = m[0]->outputs()[0]->quantization().offset();
                    float f_upper = (upper + o_zp) / o_scale;
                    o = ops::clip(in, 0., f_upper, o->quantization());
                }
            }
            auto clip = o->node();
            clip->clear_outputs();
            clip->add_output(m[1]->outputs()[0]);
            m[0]->clear_outputs();
            m[1]->clear_inputs();
            m[1]->clear_inputs();
            g->remove_node(m[0]);
            g->remove_node(m[1]);
            is_changed = true;
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
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-03-30 13:38:32
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-07-05 10:40:45
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/fp16/half.h"
#include "core/ops/ops.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

// if putting into lowering or gsim, constant of ADD may be fp32->fp16(flush by qtlib)->fp32, biases has precision loss
static inline bool criteria(std::vector<aipubt::NodePtr> m)
{
    auto add_consts = m[0];
    auto add = m[1];

    auto params = add->params();
    if (m[1]->type() == aipubt::OpType_Eltwise &&
        params["method"] != "ADD")
    {
        return false;
    }
    if (params.count("with_activation") && params["with_activation"] != "NONE")
    {
        return false;
    }

    if (!aipubt::Tensor::is_float(add->inputs()[0]->type()) ||
        !aipubt::Tensor::is_float(add->inputs()[1]->type()))
    {
        return false;
    }

    if (add->inputs()[0]->shape().dim() < 2 ||
        add->inputs()[0]->shape().dim() > 6)
    {
        return false;
    }

    // relax this limitation
    if (add->inputs()[0]->type() != add->inputs()[1]->type())
    {
        return false;
    }

    auto const_shape = add_consts->outputs()[0]->shape();

    const bool is_scalar_const = const_shape.dim() == 0;
    if (!is_scalar_const && const_shape.size() != (size_t)const_shape[-1])
    {
        return false;
    }

    if (!is_scalar_const && const_shape.size() != 1 &&
        add->inputs()[0]->shape()[-1] != add->inputs()[1]->shape()[-1])
    {
        return false;
    }

    // aipubt::Tensor::Dtype preferred_dtype = quant_utils::preferred_fp_dtype(add);
    std::vector<aipubt::Tensor::Dtype> preferred_dtype = quant_utils::preferred_fp_dtype(add);
    if (preferred_dtype[0] == aipubt::Tensor::VOID && preferred_dtype[1] == aipubt::Tensor::VOID)
    {
        return false;
    }

    for (auto &ts : {m[1]->inputs(), m[1]->outputs()})
    {
        for (auto &t : ts)
        {
            if (t->is_io_tensor())
            {
                return false;
            }
        }
    }

    return true;
}

void fuse_fp_add_constant_to_bn(aipubt::GraphPtr g)
{
    static std::vector<aipubt::GraphMatcherPattern> patterns = {
        {{{"0", aipubt::OpType_Constant}, {"1", aipubt::OpType_Add}}, {{"0", "1"}}},
        {{{"0", aipubt::OpType_Constant}, {"1", aipubt::OpType_Eltwise}}, {{"0", "1"}}},
    };
    for (auto &pattern : patterns)
    {
        aipubt::GraphMatcher matcher(pattern);
        matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);

        for (auto &&m : matched)
        {
            if (!criteria(m))
            {
                continue;
            }

            aipubt::Tensor::Dtype wht_dtype = quant_utils::preferred_fp_dtype(m[1])[1];
            int32_t dim = m[1]->outputs()[0]->shape()[-1];
            std::vector<float> weight_d(dim, 1.0f);
            aipubt::ComputeTensor weight_t(weight_d);
            weight_t = weight_t.astype(wht_dtype);

            aipubt::ComputeTensor bias_t = m[0]->constants()["weights"];
            bias_t = bias_t.reshape(std::vector<int32_t>{dim});
            bias_t = bias_t.astype(aipubt::Tensor::FP32);

            auto inp = m[1]->inputs()[0];
            if (inp == m[0]->outputs()[0])
            {
                inp = m[1]->inputs()[1];
            }

            bool unquantifiable = inp->quantization().unquantifiable();
            inp->quantization().set_unquantifiable(true);
            auto output = aipubt::ops::batch_normalization(inp, weight_t, bias_t, -1, 0.f);
            auto bn = output->node();
            bn->remove_output(0);
            bn->add_output(m[1]->outputs()[0]);
            bn->inputs()[0]->quantization().set_unquantifiable(unquantifiable);
            bn->copy_attrs(m[1]);

            for (auto _m : m)
            {
                g->remove_node(_m);
            }
        }
    }
    return;
}
}; // namespace qtlibn
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
/*
 * =====================================================================================
 *
 *       Filename:  fuse_fc_bn.cpp
 *
 *    Description:  fuse fc and bn
 *
 *        Version:  1.0
 *        Created:  04/18/23 11:59:49
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "transform/passes.h"

namespace aipubt
{

static void merge_fc_weight(ComputeTensor &fc_weight, ComputeTensor &fc_bias, ComputeTensor &bn_weight, ComputeTensor &bn_bias)
{
    fc_weight = fc_weight * bn_weight.reshape((*bn_weight).shape()[0], 1);
    fc_bias = fc_bias * bn_weight + bn_bias;
}

static bool is_float_node(NodePtr node)
{
    for (const auto &tensors : {node->inputs(), node->outputs()})
    {
        for (auto tensor : tensors)
        {
            if (!Tensor::is_float(tensor->type()))
            {
                return false;
            }
        }
    }
    return true;
}

static bool criteria(NodePtrList nodes)
{
    bool is_float_ir = false;
    for (auto node : nodes)
    {
        is_float_ir |= is_float_node(node);
    }
    if (!is_float_ir)
    {
        return false;
    }

    auto fc_node = nodes[0];
    auto fc_out = fc_node->outputs()[0];
    if ((fc_out->is_io_tensor()) || (fc_out->consumers().size() > 1))
    {
        return false;
    }

    auto bn_node = nodes[1];
    int axis = bn_node->params()["axis"];
    if (axis != 1)
    {
        return false;
    }

    return true;
}

bool fuse_fc_bn(GraphPtr graph)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_FullyConnected}, {"1", OpType_BatchNorm}},
        {{"0", "1"}},
    };
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::RelaxedMode);

    auto matched = matcher.match(graph);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_fused = false;
        for (auto &&m : matched)
        {
            if (!criteria(m))
            {
                continue;
            }

            auto fc_node = m[0];
            auto bn_node = m[1];

            DEBUG() << "Fuse fc: " << fc_node->name()
                    << " bn: " << bn_node->name();

            auto origin_fc_weight = fc_node->constants()["weights"];
            int32_t ic = origin_fc_weight->shape()[-1];
            int32_t oc = origin_fc_weight->shape().size() / ic;

            ComputeTensor fc_weight = origin_fc_weight;
            auto fc_weight_zp = origin_fc_weight->quantization().offsets();
            bool is_quant_weight = false;
            if (fc_weight_zp.size() > 0)
            {
                int32_t group_num = fc_weight_zp->shape().size() / oc;
                fc_weight_zp = fc_weight_zp.reshape(std::vector<int32_t>{(int32_t)oc, group_num}).permute(std::vector<int32_t>{1, 0});
                std::vector<ComputeTensor> zp_vec(ic / group_num, fc_weight_zp);
                ComputeTensor new_fc_weight_zp = ComputeTensor::concat(zp_vec);
                new_fc_weight_zp = new_fc_weight_zp.reshape(std::vector<int32_t>{ic / group_num, group_num, oc}).permute(std::vector<int32_t>{2, 1, 0}).reshape(fc_weight->shape());
                fc_weight = fc_weight.astype(Tensor::FP32) + new_fc_weight_zp.astype(Tensor::FP32);
                is_quant_weight = true;
            }
            auto fc_weight_scale = origin_fc_weight->quantization().scales();
            if (fc_weight_scale.size() > 0)
            {
                int32_t group_num = fc_weight_scale->shape().size() / oc;
                fc_weight_scale = fc_weight_scale.reshape(std::vector<int32_t>{oc, group_num}).permute(std::vector<int32_t>{1, 0});
                std::vector<ComputeTensor> scale_vec(ic / group_num, fc_weight_scale);
                ComputeTensor new_fc_weight_scale = ComputeTensor::concat(scale_vec);
                new_fc_weight_scale = new_fc_weight_scale.reshape(std::vector<int32_t>{ic / group_num, group_num, (int32_t)oc}).permute(std::vector<int32_t>{2, 1, 0}).reshape(fc_weight->shape());
                fc_weight = fc_weight.astype(Tensor::FP32) / new_fc_weight_scale.astype(Tensor::FP32);
                is_quant_weight = true;
            }
            ComputeTensor fc_bias = fc_node->constants()["biases"];
            ComputeTensor bn_weight = bn_node->constants()["weights"];
            ComputeTensor bn_bias = bn_node->constants()["biases"];
            merge_fc_weight(fc_weight, fc_bias, bn_weight, bn_bias);
            if (is_quant_weight)
            {
                // NOTE: AIFF only supports 16bit float point
                fc_weight = fc_weight.astype(Tensor::FP16);
            }
            fc_node->constants()["weights"] = fc_weight;
            fc_node->constants()["biases"] = fc_bias;
            fc_node->replace_output_by(fc_node->outputs()[0], bn_node->outputs()[0]);

            graph->remove_node(bn_node);

            is_changed = true;
            is_fused = true;
        }
        if (!is_fused)
        {
            break;
        }
        matched = matcher.match(graph);
    }

    return is_changed;
}
} // namespace aipubt

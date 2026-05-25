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
 * @Date: 2023-03-23 15:29:14
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-04-08 09:31:00
 */

#ifndef __ENGINE_OPS_GROUPNORM_H__
#define __ENGINE_OPS_GROUPNORM_H__

#include "engine/ops/norm.h"
#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"

namespace aipubt
{
namespace engine
{

    bool criteria_float(NodePtr node)
    {
        auto _check_fp32_dtype = [](TensorPtr t) -> bool {
            if (t->type() != Tensor::FP32)
            {
                return false;
            }
            return true;
        };

        bool ret = true;
        std::vector<std::string> _const_keys = {"weights", "biases"};
        for (auto &ck : _const_keys)
        {
            ret = ret && _check_fp32_dtype(node->constants()[ck]);
        }
        return ret;
    }

    void norm_forward(NodePtr node)
    {
        float eps = node->params()["epsilon"];
        int32_t group = 1;
        if (node->params().count("group"))
        {
            group = node->params()["group"];
        }
        ComputeTensor input = node->inputs()[0];
        input = input.astype(Tensor::FP32);

        int32_t dim_num = input->shape().dim();
        bool is_groupnorm = node->type() == OpType_GroupNorm;

        std::vector<int> axes;
        int32_t channel_axis = dim_num - 1;
        if (is_groupnorm || !node->params().count("axis"))
        {
            int32_t axis = node->params().count("axis") ? (int32_t)node->params()["axis"] : -1;
            axis = axis < 0 ? axis + dim_num : axis;
            channel_axis = axis;
            axes.clear();
            for (int i = 1; i < dim_num; i++)
            {
                axes.push_back(i);
            }
        }
        else
        {
            axes = node->params()["axis"];
        }

        std::vector<ComputeTensor> in_chunks;
        std::vector<int> axis_shape(dim_num, 1);
        if (is_groupnorm)
        {
            // GroupNorm
            in_chunks = input.split(group, channel_axis);
            axis_shape[channel_axis] = input->shape()[channel_axis];
        }
        else
        {
            // Instance/Layer Norm
            in_chunks.push_back(input);
            if (node->params().count("axis_shape"))
            {
                axis_shape = node->params()["axis_shape"];
            }
            else
            {
                for (auto &a : axes)
                {
                    axis_shape[a] = input->shape()[a];
                }
            }
        }
        std::vector<ComputeTensor> mean_chunks;
        std::vector<ComputeTensor> ngamma_chunks;
        for (size_t i = 0; i < in_chunks.size(); i++)
        {
            auto t = in_chunks[i];
            auto t_mean = t.mean(axes, true).astype(t->type());
            ComputeTensor t_var = (t - t_mean);
            t_var = (t_var * t_var).mean(axes, true).astype(t->type());
            ComputeTensor t_std = (t_var + eps);
            t_std = t_std.sqrt();
            for (int j = 0; j < t_std->shape().size(); j++)
            {
                if (t_std->data<float_t>()[j] == 0.0)
                {
                    WARN() << "input std contain zero val, please check axis or set non-zero epsilon";
                }
            }
            ComputeTensor t_ngamma = 1.0f / t_std;

            if (t_mean->shape()[channel_axis] != t->shape()[channel_axis])
            {
                std::vector<ComputeTensor> t_mean_v;
                std::vector<ComputeTensor> t_ngamma_v;
                for (int j = 0; j < t->shape()[channel_axis]; j++)
                {
                    t_mean_v.push_back(t_mean);
                    t_ngamma_v.push_back(t_ngamma);
                }
                t_mean = ComputeTensor::concat(t_mean_v, channel_axis);
                t_ngamma = ComputeTensor::concat(t_ngamma_v, channel_axis);
            }

            mean_chunks.push_back(t_mean);
            ngamma_chunks.push_back(t_ngamma);
        }
        auto mean = ComputeTensor::concat(mean_chunks, channel_axis);
        auto ngamma = ComputeTensor::concat(ngamma_chunks, channel_axis);

        ComputeTensor normalized = (input - mean) * ngamma;
        // for (auto as : axis_shape)
        // {
        //     INFO() << "as =" << as;
        // }
        ComputeTensor gamma = ComputeTensor::zeros(axis_shape, Tensor::FP32) + 1.0;
        ComputeTensor beta = ComputeTensor::zeros(axis_shape, Tensor::FP32);
        if (node->constants().count("weights"))
        {
            gamma = node->constants()["weights"];
            // gamma = gamma.reshape(axis_shape);
        }
        if (node->constants().count("biases"))
        {
            beta = node->constants()["biases"];
            // beta = beta.reshape(axis_shape);
        }
        ComputeTensor output = gamma * normalized + beta;
        node->outputs()[0]->share_data_from(output);

        node->placeholders().push_back(ngamma);
        node->placeholders().push_back(normalized);
    }

} // namespace engine

} // namespace aipubt

#endif //!__ENGINE_OPS_GROUPNORM_H__

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
 * @Author: Neo Wang
 * @Date: 2021-04-14 11:04:33
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-04-28 18:24:58
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    // TODO: "WITHOUT_GAMMA_BETA" needs gt and IR checker update: CLT-11
    std::vector<TensorPtr> layer_normalization(TensorPtr input,
                                               TensorPtr gamma,
                                               TensorPtr beta,
                                               const std::vector<int32_t> &axis,
                                               float epsilon,
                                               bool stash_type,
                                               bool out_mean_invstddev,
                                               const std::vector<Quantization> &quants)
    {
        if (input->shape().dim() < 2 || input->shape().dim() > 4)
        {
            ERROR() << "input rank should be in [2,4]!";
            return {};
        }

        std::vector<int32_t> new_axis;
        for (auto it : axis)
        {
            if (it < 0)
            {
                it = it < 0 ? it + input->shape().dim() : it;
            }
            if (it < 0 || it >= input->shape().dim())
            {
                ERROR() << "axis is invalid, it should be in [-rank, rank)!";
                return {};
            }
            if (std::count(new_axis.begin(), new_axis.end(), it) == 0)
            {
                new_axis.push_back(it);
            }
        }
        if (static_cast<int32_t>(new_axis.size()) > input->shape().dim())
        {
            ERROR() << "axis size is invalid, it should be <= input rank!";
            return {};
        }

        std::string method = "WITH_GAMMA_BETA";
        if (gamma == nullptr && beta == nullptr)
        {
            method = "WITHOUT_GAMMA_BETA";
        }
        else
        {
            auto in_dtype_size = Tensor::dtype_size(input->type());
            auto w_dtype_size = Tensor::dtype_size(gamma->type());
            if (in_dtype_size != w_dtype_size && quants.size() == 0)
            {
                ERROR() << "input data type width should be same with gamma data type width! \n"
                        << "input:" << input->type() << "gamma:" << gamma->type();
                return {};
            }
            for (uint32_t i = 0; i < new_axis.size(); ++i)
            {
                if (gamma->shape()[i] != input->shape()[new_axis[i]] ||
                    beta->shape()[i] != input->shape()[new_axis[i]])
                {
                    ERROR() << "gamma or beta shape cannot match input! \n"
                            << "input[" << new_axis[i] << "]:" << input->shape()[new_axis[i]]
                            << "gamma[" << i << "]:" << gamma->shape()[i]
                            << "beta[" << i << "]:" << beta->shape()[i];
                    return {};
                }
            }
        }
        bool is_reshape = false;
        if (input->shape().dim() == 2)
        {
            is_reshape = true;
        }
        auto g = input->node()->graph();
        TensorPtr new_input = input;
        if (is_reshape)
        {
            /* if origin input dim is 2-dim, and n.params['axis'] = [0,1],
             * the input will expand(or unsuqeeze) in dim=0,
             * so the new_axis should increase + 1
             */
            std::transform(new_axis.begin(), new_axis.end(), new_axis.begin(), [](int32_t na) { return na + 1; });
            TensorShape new_shape(input->shape());
            new_shape.reshape(1, input->shape()[0], input->shape()[1]);
            new_input = reshape(input, new_shape);
            new_input->node()->infer_shape(true);
        }

        auto output = std::make_shared<Tensor>(g->get_valid_tensor_name("layer_norm/output"), new_input->shape());
        if (quants.size() > 0)
        {
            add_tensor_info(new_input, output, quants[0]);
        }
        else
        {
            add_tensor_info(new_input, output);
        }

        auto node = Node::create(g->get_valid_node_name("layer_norm"), OpType_LayerNorm);
        node->add_input(new_input);
        node->add_output(output);

        std::vector<TensorPtr> outputs;
        outputs.push_back(output);
        if (out_mean_invstddev)
        {
            TensorShape mean_shape(new_input->shape());
            for (auto ax : new_axis)
            {
                mean_shape[ax] = 1;
            }
            auto mean = std::make_shared<Tensor>(g->get_valid_tensor_name("layer_norm/mean"), mean_shape);
            if (quants.size() > 0)
            {
                add_tensor_info(new_input, mean, quants[0]);
            }
            else
            {
                add_tensor_info(new_input, mean);
            }

            auto invstddev = std::make_shared<Tensor>(g->get_valid_tensor_name("layer_norm/invstddev"), mean_shape);
            if (quants.size() > 1)
            {
                add_tensor_info(new_input, invstddev, quants[1]);
            }
            else
            {
                add_tensor_info(new_input, invstddev, Quantization(1.0, 0));
            }

            node->add_output(mean);
            node->add_output(invstddev);
            outputs.push_back(mean);
            outputs.push_back(invstddev);
        }

        auto &params = node->params();
        params["method"] = method;
        params["axis"] = new_axis;
        params["epsilon"] = epsilon;
        params["stash_type"] = stash_type;
        if (method == "WITH_GAMMA_BETA")
        {
            node->constants()["weights"] = gamma;
            node->constants()["biases"] = beta;
        }
        node->infer_shape(true);
        g->add_node(node);
        quantize(node);

        if (is_reshape)
        {
            std::vector<TensorPtr> outputs_ln = outputs;
            outputs.clear();
            for (auto o : outputs_ln)
            {
                std::vector<int32_t> cur_shape = o->shape();
                cur_shape.erase(cur_shape.begin());
                TensorShape new_shape(cur_shape);
                auto new_output = reshape(o, new_shape);
                new_output->node()->infer_shape(true);
                outputs.push_back(new_output);
            }
        }
        return outputs;
    }

} // namespace ops
} // namespace aipubt
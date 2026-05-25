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
 * @Author: Kejia Qiang
 * @Date: 2023-05-05 16:18:46
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-05-22 15:39:08
 */
#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"
namespace aipubt
{
namespace engine
{

    void basiclstm_forward(NodePtr node)
    {
        // float forward only
        ComputeTensor input = node->inputs()[0];
        ComputeTensor h_cell = node->inputs()[1];
        ComputeTensor c_cell = node->inputs()[2];
        auto batch_size = input->shape()[0];
        int time_step = node->params()["time_steps"];
        int input_size = node->params()["input_size"];
        int cell_size = node->params()["cell_size"];
        std::string direction = node->params()["direction"];
        int h_init_batch = h_cell->shape()[0];
        int c_init_batch = c_cell->shape()[0];

        std::transform(direction.begin(), direction.end(), direction.begin(), ::toupper);
        if (direction == "REVERSE")
        {
            // TODO...
            // implement torch.flip in computeTensor
            ERROR() << "[engine] basiclstm forward now doesnot support direction == reverse";
        }

        int current_batch_idx = 0;
        if (node->attrs().count("current_batch_idx"))
        {
            current_batch_idx = node->get_attr("current_batch_idx");
        }
        int start_data_idx = current_batch_idx * batch_size;
        int start_h_init_idx = start_data_idx % h_init_batch;
        int start_c_init_idx = start_data_idx % c_init_batch;

        ComputeTensor h_init = h_cell[start_h_init_idx];
        ComputeTensor c_init = c_cell[start_c_init_idx];
        h_init = h_init.unsqueeze();
        c_init = c_init.unsqueeze();
        for (int i = 1; i < batch_size; i++)
        {
            int current_h_init_idx = (start_data_idx + i) % h_init_batch;
            int current_c_init_idx = (start_data_idx + i) % c_init_batch;
            ComputeTensor h_cell_t = h_cell[current_h_init_idx];
            ComputeTensor c_cell_t = c_cell[current_c_init_idx];
            if (h_cell_t->shape().dim() < h_init->shape().dim())
            {
                h_cell_t = h_cell_t.unsqueeze();
                c_cell_t = c_cell_t.unsqueeze();
            }
            h_init = ComputeTensor::concat({h_init, h_cell_t});
            c_init = ComputeTensor::concat({c_init, c_cell_t});
        }

        ComputeTensor weight = node->constants()["weights"];
        weight = weight.permute({1, 0});
        ComputeTensor bias = node->constants()["biases"];

        std::vector<std::string> out_sequence = node->params()["out_sequence"];
        std::vector<std::string> activations_list = {"SIGMOID", "TANH", "TANH"};
        auto h_batch = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);
        auto c_batch = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);
        auto h_last = ComputeTensor::zeros({batch_size, cell_size}, Tensor::FP32);
        auto c_last = ComputeTensor::zeros({batch_size, cell_size}, Tensor::FP32);

        float threshold = HUGE_VALF;
        float cell_clip = HUGE_VALF;
        float forget_bias = 0.0;
        if (node->params().count("threshold"))
        {
            threshold = node->params()["threshold"];
        }
        if (node->params().count("cell_clip"))
        {
            cell_clip = node->params()["cell_clip"];
        }
        if (node->params().count("forget_bias"))
        {
            forget_bias = node->params()["forget_bias"];
        }

        auto f_lut_in = ComputeTensor::zeros({batch_size, time_step, 3 * cell_size}, Tensor::FP32);
        auto f_lut_out = ComputeTensor::zeros({batch_size, time_step, 3 * cell_size}, Tensor::FP32);
        auto g_lut_in = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);
        auto g_lut_out = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);
        auto h_lut_in = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);
        auto h_lut_out = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);

        for (int b = 0; b < batch_size; b++)
        {
            auto h_all = ComputeTensor::zeros({1, cell_size}, Tensor::FP32);
            auto c_all = ComputeTensor::zeros({1, cell_size}, Tensor::FP32);
            ComputeTensor h_prev = h_init[{{b}, ComputeTensor::Ellipse}];
            h_prev = h_prev.unsqueeze();
            ComputeTensor c_prev = c_init[{{b}, ComputeTensor::Ellipse}];
            // c_prev = c_prev.unsqueeze();

            for (int ts = 0; ts < time_step; ts++)
            {
                ComputeTensor in_ts = input[b][ts];
                in_ts = in_ts.reshape(1, input_size);

                std::vector<ComputeTensor> tensors = {in_ts, h_prev};
                auto inputs = ComputeTensor::concat(tensors, 1);
                auto sum_0 = inputs.matmul(weight);
                sum_0 = sum_0 + bias.unsqueeze();
                ComputeTensor i_tmp = sum_0[{ComputeTensor::Ellipse, {0, cell_size}}];
                ComputeTensor g_tmp = sum_0[{ComputeTensor::Ellipse, {cell_size, cell_size * 2}}];
                ComputeTensor f_tmp = sum_0[{ComputeTensor::Ellipse, {cell_size * 2, cell_size * 3}}];
                ComputeTensor o_tmp = sum_0[{ComputeTensor::Ellipse, {cell_size * 3, cell_size * 4}}];

                i_tmp = i_tmp.clamp(-threshold, threshold);
                g_tmp = g_tmp.clamp(-threshold, threshold);
                o_tmp = o_tmp.clamp(-threshold, threshold);
                f_tmp = (f_tmp + forget_bias).clamp(-threshold, threshold);

                f_lut_in[{{b, b + 1}, {ts, ts + 1}}] = ComputeTensor::concat({i_tmp, f_tmp, o_tmp}, 1).unsqueeze();
                g_lut_in[{{b, b + 1}, {ts, ts + 1}}] = g_tmp.unsqueeze();

                auto rnn_activation_function = [](const std::string &activation_name, ComputeTensor x) -> ComputeTensor {
                    if (activation_name == "SIGMOID")
                    {
                        return x.sigmoid();
                    }
                    else if (activation_name == "TANH")
                    {
                        return x.tanh();
                    }
                    else
                    {
                        ERROR() << "not support activation " << activation_name;
                    }
                    return x.sigmoid();
                };

                auto i = rnn_activation_function(activations_list[0], i_tmp);
                auto g = rnn_activation_function(activations_list[1], g_tmp);
                auto f = rnn_activation_function(activations_list[0], f_tmp);
                auto o = rnn_activation_function(activations_list[0], o_tmp);

                f_lut_out[{{b, b + 1}, {ts, ts + 1}}] = ComputeTensor::concat({i, f, o}, 1).unsqueeze();
                g_lut_out[{{b, b + 1}, {ts, ts + 1}}] = g.unsqueeze();

                c_prev = f * c_prev + i * g;
                c_prev = c_prev.clamp(-cell_clip, cell_clip);
                h_lut_in[{{b, b + 1}, {ts, ts + 1}}] = c_prev.unsqueeze();
                auto c_lut = rnn_activation_function(activations_list[2], c_prev);
                h_lut_out[{{b, b + 1}, {ts, ts + 1}}] = c_lut.unsqueeze();
                h_prev = o * c_lut;

                h_all = ts == 0 ? h_prev : ComputeTensor::concat({h_all, h_prev});
                c_all = ts == 0 ? c_prev : ComputeTensor::concat({c_all, c_prev});
            }
            h_last = b == 0 ? h_prev : ComputeTensor::concat({h_last, h_prev});

            h_all = h_all.unsqueeze();
            h_batch = b == 0 ? h_all : ComputeTensor::concat({h_batch, h_all});

            c_last = b == 0 ? c_prev : ComputeTensor::concat({c_last, c_prev});

            c_all = c_all.unsqueeze();
            c_batch = b == 0 ? c_all : ComputeTensor::concat({c_batch, c_all});
        }
        // merge activation lut to meet AIFF requirements
        if (activations_list[0] == activations_list[1])
        {
            auto merged_in = ComputeTensor::concat({f_lut_in, g_lut_in}, 2);
            auto merged_out = ComputeTensor::concat({f_lut_out, g_lut_out}, 2);
            f_lut_in = g_lut_in = merged_in;
            f_lut_out = g_lut_out = merged_out;
        }
        if (activations_list[0] == activations_list[2])
        {
            auto merged_in = ComputeTensor::concat({f_lut_in, h_lut_in}, 2);
            auto merged_out = ComputeTensor::concat({f_lut_out, h_lut_out}, 2);
            f_lut_in = h_lut_in = merged_in;
            f_lut_out = h_lut_out = merged_out;
        }
        if (activations_list[1] == activations_list[2])
        {
            auto merged_in = ComputeTensor::concat({g_lut_in, h_lut_in}, 2);
            auto merged_out = ComputeTensor::concat({g_lut_out, h_lut_out}, 2);
            h_lut_in = g_lut_in = merged_in;
            h_lut_out = g_lut_out = merged_out;
        }

        std::vector<ComputeTensor> lut_ph_list = {
            f_lut_in,
            f_lut_out,
            g_lut_in,
            g_lut_out,
            h_lut_in,
            h_lut_out,
            h_batch};

        if (node->placeholders().size() < 1)
        {
            for (size_t i = 0; i < lut_ph_list.size() + c_batch->shape()[1]; i++)
            {
                node->placeholders().push_back(ComputeTensor::zeros({1}));
            }
        }
        for (size_t i = 0; i < lut_ph_list.size(); i++)
        {
            node->placeholders()[i] = lut_ph_list[i];
        }
        for (int ts = 0; ts < c_batch->shape()[1]; ts++)
        {
            auto batch_dim = c_batch->shape()[0];
            ComputeTensor ph = c_batch[{{0, batch_dim}, {ts, ts + 1}}];
            node->placeholders()[7 + ts] = ph;
        }

        for (size_t i = 0; i < out_sequence.size(); i++)
        {
            std::string item = out_sequence[i];
            if (item == "Y")
            {
                if (direction == "REVERSE")
                {
                    // TODO... reverse h_batch at 1 dim
                }
                h_batch = h_batch.astype(node->outputs()[i]->type());
                node->outputs()[i]->share_data_from(h_batch);
            }
            if (item == "H")
            {
                h_last = h_last.astype(node->outputs()[i]->type());
                node->outputs()[i]->share_data_from(h_last);
            }
            if (item == "C")
            {
                c_last = c_last.astype(node->outputs()[i]->type());
                node->outputs()[i]->share_data_from(c_last);
            }
        }
    }

    EXEOP_REGISTER(OpType_BasicLSTM, basiclstm_forward, DTCriteria::FP32_ONLY);
} // namespace engine
} // namespace aipubt
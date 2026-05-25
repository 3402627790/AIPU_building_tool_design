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
 * @Date: 2023-08-05 16:18:46
 * @Last Modified by: Kejia Qiang
 * @Last Modified time: 2023-08-05 16:18:46
 */
#include "core/compute_tensor.h"
#include "core/node.h"
#include "engine/engine.h"

using aipubt::ComputeTensor;
using aipubt::Tensor;
using std::string;

namespace aipubt
{
namespace engine
{
    using activation_function = aipubt::ComputeTensor (*)(aipubt::ComputeTensor);
    static aipubt::ComputeTensor rnn_sigmoid(aipubt::ComputeTensor x)
    {
        return x.sigmoid();
    }

    static aipubt::ComputeTensor rnn_tanh(aipubt::ComputeTensor x)
    {
        return x.tanh();
    }
    static std::map<std::string, activation_function> rnn_activation_function = {
        {"SIGMOID", rnn_sigmoid},
        {"TANH", rnn_tanh},
    };
    void gru(NodePtr node)
    {
        ComputeTensor inp0 = node->inputs()[0];
        ComputeTensor inp1 = node->inputs()[1];
        inp0 = inp0.astype(Tensor::FP32);

        int init_batch = inp1->shape()[0];
        int batch_size = inp0->shape()[0];
        int time_step = node->params()["time_steps"];
        int input_size = node->params()["input_size"];
        int cell_size = node->params()["cell_size"];
        string direction = node->params()["direction"];
        float_t threshold = std::numeric_limits<float_t>::max();
        if (node->params().count("threshold"))
        {
            threshold = node->params()["threshold"];
        }
        if (direction == "REVERSE")
        {
            // TODO...
            // implement torch.flip in computeTensor
            ERROR() << "[engine] gruv1/3 forward now doesnot support direction == reverse";
        }

        ComputeTensor init_H = inp1.astype(Tensor::FP32);
        int start_data_idx = 0;
        int start_init_idx = start_data_idx % init_batch;
        ComputeTensor in_state = init_H[{{start_init_idx, start_init_idx+1}}];
        for (int i = 1; i < batch_size; i++)
        {
            int current_init_idx = (start_data_idx + i) % init_batch;
            in_state = ComputeTensor::concat({in_state, init_H[{{current_init_idx, current_init_idx + 1}}]});
        }

        ComputeTensor weight = node->constants()["weights"];
        weight = weight.astype(Tensor::FP32);
        ComputeTensor bias = node->constants()["biases"];
        bias = bias.astype(Tensor::FP32);
        std::vector<string> out_seq = node->params()["out_sequence"];
        std::vector<string> activation_list;
        if (node->params().count("activations"))
        {
            activation_list = node->params()["activations"];
        }
        else
        {
            activation_list = {"SIGMOID", "TANH"};
        }

        auto state_batch = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);
        auto state_last = ComputeTensor::zeros({batch_size, cell_size}, Tensor::FP32);

        weight = weight.permute({1,0});
        int w1 = weight->shape()[0];
        int w2 = weight->shape()[1];
        ComputeTensor gates_kernel = weight[{{0,w1},{0,2*cell_size}}];
        ComputeTensor candidate_kernel = weight[{{0,w1},{2*cell_size,w2}}];

        auto f_lut_in = ComputeTensor::zeros({batch_size, time_step, 2*cell_size}, Tensor::FP32);
        auto f_lut_out = ComputeTensor::zeros({batch_size, time_step, 2*cell_size}, Tensor::FP32);
        auto g_lut_in = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);
        auto g_lut_out = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);
        auto hidden_gate_out = ComputeTensor::zeros({batch_size, time_step, cell_size}, Tensor::FP32);

        ComputeTensor wx_gk = gates_kernel[{{0,input_size}}];
        wx_gk = wx_gk.permute({1,0});
        ComputeTensor wh_gk = gates_kernel[{{input_size, w1}}];
        wh_gk = wh_gk.permute({1,0});
        ComputeTensor wx_ck = gates_kernel[{{0,input_size}}];
        wx_ck = wx_ck.permute({1,0});
        ComputeTensor wh_ck = gates_kernel[{{input_size, w1}}];
        wh_ck = wh_ck.permute({1,0});

        std::vector<string> weight_names = {"wx_gk","wh_gk","wx_ck","wh_ck"};
        std::vector<ComputeTensor> weights = {wx_gk, wh_gk, wx_ck, wh_ck};
        for (int i = 0; i < 4; i++)
        {
            auto weight_name = weight_names[i];
            if (!node->constants().count(weight_name))
            {
                node->constants()[weight_name] = weights[i];
            }
        }

        ComputeTensor gates_bias = bias[{{0, 2*cell_size}}];
        ComputeTensor candidate_bias = bias[{{2*cell_size, 3*cell_size}}];
        for (int b = 0; b < batch_size; b++)
        {
            ComputeTensor::Slice b_slice(b);
            ComputeTensor state = in_state[b_slice];
            state = state.unsqueeze();
            auto state_all = ComputeTensor::zeros({1, cell_size}, Tensor::FP32);
            for (int ts = 0; ts < time_step; ts++)
            {
                ComputeTensor in_ts = inp0[{{b,b+1},{ts,ts+1},{0,input_size}}];
                int in_ts_size = in_ts->size();
                in_ts = in_ts.reshape(in_ts_size / input_size, input_size);
                if (node->type() == OpType_GRUv1)
                {
                    std::vector<ComputeTensor> gate_kernel_f = (gates_kernel+0.0).split(2,0);
                    ComputeTensor gate_kernel_x_f = (gate_kernel_f[0]).astype(aipubt::Tensor::FP32);
                    ComputeTensor gate_kernel_h_f = (gate_kernel_f[1]).astype(aipubt::Tensor::FP32);
                    std::vector<ComputeTensor> candidate_kernel_f = (candidate_kernel+0.0).split(2, 0);
                    ComputeTensor candidate_kernel_x_f = (candidate_kernel_f[0]).astype(aipubt::Tensor::FP32);
                    ComputeTensor candidate_kernel_h_f = (candidate_kernel_f[1]).astype(aipubt::Tensor::FP32);

                    ComputeTensor hidden_bias = bias[{3*cell_size, bias->shape()[0]}];
                    auto h_by_wg_f = state.matmul(gate_kernel_h_f);
                    auto x_by_wg_f = in_ts.matmul(gate_kernel_x_f) + gates_bias.unsqueeze();
                    auto mat_sum_f = x_by_wg_f + h_by_wg_f;
                    mat_sum_f = mat_sum_f.clamp(-threshold, threshold);
                    std::vector<ComputeTensor::Slice> f_slices = {ComputeTensor::Slice(b), ComputeTensor::Slice(ts),
                        ComputeTensor::Slice(0,2*cell_size)};
                    f_lut_in[f_slices] = mat_sum_f.squeeze();
                    auto sig_lut_out_f = rnn_activation_function[activation_list[0]](mat_sum_f);
                    f_lut_out[f_slices] = sig_lut_out_f.squeeze();
                    std::vector<ComputeTensor> r_u = sig_lut_out_f.split(2, 1);
                    auto r_f = r_u[0];
                    auto u_f = r_u[1];

                    std::vector<ComputeTensor::Slice> gh_slices = {ComputeTensor::Slice(b), ComputeTensor::Slice(ts),
                        ComputeTensor::Slice(0,cell_size)};
                    auto h_by_wc_f = state.matmul(candidate_kernel_h_f);
                    h_by_wc_f = h_by_wc_f + hidden_bias.unsqueeze();
                    hidden_gate_out[gh_slices] = h_by_wc_f.squeeze();
                    auto r_hwc_f = r_f * h_by_wc_f;
                    auto x_by_wc_f = in_ts.matmul(candidate_kernel_x_f) + candidate_bias.unsqueeze();
                    auto mat_sum2_f = x_by_wc_f + r_hwc_f;
                    mat_sum2_f = mat_sum2_f.clamp(-threshold, threshold);
                    g_lut_in[gh_slices] = mat_sum2_f.squeeze();
                    auto c_f = rnn_activation_function[activation_list[1]](mat_sum2_f);
                    g_lut_out[gh_slices] = c_f.squeeze();
                    state = (1.0f - u_f) * c_f + u_f * state;
                }
                else
                {
                    auto gate_input = ComputeTensor::concat({in_ts, state}, 1);
                    auto sum0 = gate_input.matmul(gates_kernel) + gates_bias.unsqueeze();
                    sum0 = sum0.clamp(-threshold, threshold);
                    std::vector<ComputeTensor::Slice> f_slices = {ComputeTensor::Slice(b), ComputeTensor::Slice(ts),
                        ComputeTensor::Slice(0,2*cell_size)};
                    std::vector<ComputeTensor::Slice> gh_slices = {ComputeTensor::Slice(b), ComputeTensor::Slice(ts),
                        ComputeTensor::Slice(0,cell_size)};
                    f_lut_in[f_slices] = sum0.squeeze();
                    auto f_out = rnn_activation_function[activation_list[0]](sum0);
                    f_lut_out[f_slices] = f_out.squeeze();
                    auto r_u = f_out.split(2,1);
                    auto r = r_u[0];
                    auto u = r_u[1];
                    auto state_r = state * r;

                    auto candidate_input = ComputeTensor::concat({in_ts, state_r}, 1);
                    auto sum1 = candidate_input.matmul(candidate_kernel) + candidate_bias.unsqueeze();
                    sum1 = sum1.clamp(-threshold, threshold);
                    g_lut_in[gh_slices] = sum1.squeeze();
                    auto c = rnn_activation_function[activation_list[1]](sum1);
                    g_lut_out[gh_slices] = c.squeeze();
                    state = ((1.0f-u) * c) + (u * state);
                }
                if (ts == 0)
                {
                    state_all = state;
                }
                else
                {
                    state_all = ComputeTensor::concat({state_all, state});
                }
            }
            if (b == 0)
            {
                state_last = state;
            }
            else
            {
                state_all = ComputeTensor::concat({state_all, state});
            }
            state_all = state_all.unsqueeze();
            if (b == 0)
            {
                state_batch = state_all;
            }
            else
            {
                state_batch = ComputeTensor::concat({state_batch, state_all});
            }
        }
        std::vector<string> ph_names = {"state_batch", "f_lut_in", "f_lut_out", "g_lut_in", "g_lut_out", "hidden_gate_out"};
        std::vector<ComputeTensor> ph_list = {state_batch, f_lut_in, f_lut_out, g_lut_in, g_lut_out, hidden_gate_out};
        if (node->placeholders().size() < ph_list.size())
        {
            for(unsigned int i = 0; i < ph_list.size(); i++)
            {
                ph_list[i]->set_name(ph_names[i]);
                node->placeholders().push_back(ph_list[i]);
            }
        }

        if (out_seq[0] == "Hn")
        {
            node->outputs()[0]->share_data_from(state_last);
        }
        else
        {
            node->outputs()[0]->share_data_from(state_batch);
            if (out_seq.size() > 1)
            {
                node->outputs()[1]->share_data_from(state_last);
            }
        }
    }

    EXEOP_REGISTER(OpType_GRUv1, gru, DTCriteria::FP32_ONLY);
    EXEOP_REGISTER(OpType_GRUv3, gru, DTCriteria::FP32_ONLY);
}
}
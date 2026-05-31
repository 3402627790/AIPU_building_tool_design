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
 * @Date: 2023-02-21 14:24:51
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-06-02 11:19:14
 */

#include "core/compute_tensor.h"
#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
static void load_crnn_ph(aipubt::NodePtr node)
{
    int32_t batch_size = 1;
    int32_t time_step = node->params()["time_steps"];

    std::vector<float> c_ts_scale, h_batch_scale, split_scale, tanh_scale, bias_scale, sigmoid_scale, ig_mul_scale, fcprev_mul_scale;
    std::vector<int32_t> c_ts_zerop, h_batch_zerop, split_zerop, tanh_zerop, bias_zerop, sigmoid_zerop, ig_mul_zerop, fcprev_mul_zerop;
    if (node->params().count("activations_scale") && node->params().count("activations_zp"))
    {
        std::vector<float> activations_scale = node->params()["activations_scale"];
        std::vector<int32_t> activations_zerop = node->params()["activations_zp"];
        int32_t plh_size = activations_scale.size() / time_step;
        for (int32_t ts = 0; ts < time_step; ts++)
        {
            split_scale.push_back(activations_scale[plh_size * ts + 0]);
            split_zerop.push_back(activations_zerop[plh_size * ts + 0]);
            sigmoid_scale.push_back(activations_scale[plh_size * ts + 1]);
            sigmoid_zerop.push_back(activations_zerop[plh_size * ts + 1]);
            tanh_scale.push_back(activations_scale[plh_size * ts + 2]);
            tanh_zerop.push_back(activations_zerop[plh_size * ts + 2]);
            ig_mul_scale.push_back(activations_scale[plh_size * ts + 5]);
            ig_mul_zerop.push_back(activations_zerop[plh_size * ts + 5]);
            fcprev_mul_scale.push_back(activations_scale[plh_size * ts + 6]);
            fcprev_mul_zerop.push_back(activations_zerop[plh_size * ts + 6]);
            c_ts_scale.push_back(activations_scale[plh_size * ts + 8]);
            c_ts_zerop.push_back(activations_zerop[plh_size * ts + 8]);
            h_batch_scale.push_back(activations_scale[plh_size * ts + 9]);
            h_batch_zerop.push_back(activations_zerop[plh_size * ts + 9]);
            bias_scale.push_back(plh_size == 11 ? activations_scale[plh_size * ts + 10] : activations_scale[plh_size * ts + 0]);
            bias_zerop.push_back(plh_size == 11 ? activations_zerop[plh_size * ts + 10] : activations_zerop[plh_size * ts + 0]);
        }
    }
    else
    {
        ERROR() << "cannot find activations_scale and activations_zp in parameter of basiclstm.";
    }
    std::vector<std::string> _plhs_name = {
        "split_plh",
        "bias_plh",
        "sigmoid_plh",
        "sigmoid_plh",
        "sigmoid_plh",
        "tanh_plh",
        "c_ts_plh",
        "tanh_plh",
        "h_batch_plh",
        "ig_mul_plh",
        "fcprev_mul_plh",
    };
    std::vector<std::pair<std::vector<float>, std::vector<int32_t>>> _plhs_scale_zp = {
        {split_scale, split_zerop},           //fc out
        {bias_scale, bias_zerop},             // forgetbias out
        {sigmoid_scale, sigmoid_zerop},       // f sigmoid out
        {sigmoid_scale, sigmoid_zerop},       // i sigmoid out
        {sigmoid_scale, sigmoid_zerop},       // o sigmoid out
        {tanh_scale, tanh_zerop},             // g tanh out
        {c_ts_scale, c_ts_zerop},             // c tanh in
        {tanh_scale, tanh_zerop},             // c tanh out
        {h_batch_scale, h_batch_zerop},       // h batch out
        {ig_mul_scale, ig_mul_zerop},         // i*g out
        {fcprev_mul_scale, fcprev_mul_zerop}, // f*cprev out
    };

    int32_t plh_size = _plhs_name.size();
    // for (auto i = 0; i < time_step; i++)
    {
        for (int32_t p = 0; p < plh_size; p++)
        {
            auto scales = _plhs_scale_zp[p].first;
            auto zerops = _plhs_scale_zp[p].second;
            auto plh_tensor = aipubt::ComputeTensor::zeros({batch_size, time_step, 1}, aipubt::Tensor::INT8);
            set_default_qinfo_to_tensor(plh_tensor);
            plh_tensor->quantization().set_scales(scales);
            plh_tensor->quantization().set_offsets(zerops);
            node->placeholders().push_back(plh_tensor);
            plh_tensor->quantization().set_data_unchanged(true);
        }
    }
}

void basiclstm_placeholder_loading_preprocess(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        if (node->type() == aipubt::OpType_BasicLSTM)
        {
            if (node->placeholders().size() == 0 &&
                node->attrs().count("set_default_placeholder_info"))
            {
                if (node->constants()["weights"]->quantization().scales().size() == 2)
                {
                    load_crnn_ph(node);
                }
            }
        }
    }
}

} // namespace qtlibn
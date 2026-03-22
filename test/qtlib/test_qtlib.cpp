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
 * @Date: 2020-05-17 20:10:08
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-03-17 16:16:58
 */

#include "test_qtlib.h"

static std::vector<float> generate_random_vector(size_t n, float min, float max)
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(min, max);

    std::vector<float> result(n);
    for (auto &val : result)
    {
        val = dis(gen);
    }
    return result;
}

static void test_tanh(void)
{
    auto _g_test = [](float *in_data, float *out_data, int32_t num) {
        for (int i = 0; i < num; i++)
        {
            float value = in_data[i];
            float in_clamp_min = 0.0, in_clamp_max = 10.0;
            float out_clamp_min = 0.0, out_clamp_max = 1.0;
            if (value < 0)
            {
                in_clamp_min = -10.0;
                in_clamp_max = 10.0;
                out_clamp_min = -1.0;
                out_clamp_max = 1.0;
            }
            auto min_value = value < in_clamp_min ? in_clamp_min : value;
            value = min_value > in_clamp_max ? in_clamp_max : min_value;
            value = (exp(value) - exp(-value)) / (exp(value) + exp(-value));

            auto max_value = value < out_clamp_min ? out_clamp_min : value;
            value = max_value > out_clamp_max ? out_clamp_max : max_value;

            out_data[i] = value;
            INFO() << "i = " << i << ", value = " << value;
        }
    };
    auto _ct_test = [](float *in_data, float *out_data, int32_t num) {
        aipubt::TensorPtr t = std::make_shared<aipubt::Tensor>("t", aipubt::TensorShape(num), aipubt::Tensor::FP32);
        t->set_data(in_data, num * 4);
        auto value = aipubt::ComputeTensor(t);
        float pos_in_clip_min = 0.0, pos_in_clip_max = 10.0;
        float pos_out_clip_min = 0.0, pos_out_clip_max = 1.0;
        float neg_in_clip_min = -10.0, neg_in_clip_max = 10.0;
        float neg_out_clip_min = -1.0, neg_out_clip_max = 1.0;

        auto neg_mask = value < 0;
        auto neg_value_clip = (value * neg_mask).clamp(neg_in_clip_min, neg_in_clip_max);
        auto pos_mask = 1 - neg_mask;
        auto pos_value_clip = (value * pos_mask).clamp(pos_in_clip_min, pos_in_clip_max);
        auto cur_value = neg_value_clip + pos_value_clip;

        auto exp_value = cur_value.exp();
        auto exp_neg_value = (0.f - cur_value).exp();
        cur_value = (exp_value - exp_neg_value) / (exp_value + exp_neg_value);

        neg_value_clip = (cur_value * neg_mask).clamp(neg_out_clip_min, neg_out_clip_max);
        pos_value_clip = (cur_value * pos_mask).clamp(pos_out_clip_min, pos_out_clip_max);

        value = neg_value_clip + pos_value_clip;
        INFO() << value;
        memcpy(out_data, ((aipubt::TensorPtr)value)->data<char>(), num * 4);
    };

    float *data = new float[6]{1.233, 0.222, -0.987, 2.45, -8.9, -12.78};
    float *g_out_data = new float[6]{0};
    float *ct_out_data = new float[6]{0};
    _g_test(data, g_out_data, 6);
    _ct_test(data, ct_out_data, 6);
    auto ret = memcmp(g_out_data, ct_out_data, 6 * 4);
    if (ret != 0)
    {
        ERROR() << "tanh failed";
    }
}

static void test_selu(void)
{
    float alpha = 1.0;
    float gamma = 2.0;
    auto _g_test = [&alpha, &gamma](float *in_data, float *out_data, int32_t num) {
        for (int i = 0; i < num; i++)
        {
            float value = in_data[i];
            value = std::max(0.f, value) + std::min(0.f, alpha * (exp(value) - 1));
            value *= gamma;
            out_data[i] = value;
            INFO() << "i = " << i << ", value = " << value;
        }
    };
    auto _ct_test = [&alpha, &gamma](float *in_data, float *out_data, int32_t num) {
        aipubt::TensorPtr t = std::make_shared<aipubt::Tensor>("t", aipubt::TensorShape(num), aipubt::Tensor::FP32);
        t->set_data(in_data, num * 4);
        auto value = aipubt::ComputeTensor(t);
        auto max_value = value.maximum(0.f);
        auto min_value = ((value.exp() - 1.f) * alpha).minimum(0.f);
        value = (max_value + min_value) * gamma;
        INFO() << value;
        memcpy(out_data, ((aipubt::TensorPtr)value)->data<char>(), num * 4);
    };

    float *data = new float[6]{1.233, 0.222, -0.987, 2.45, -8.9, -12.78};
    float *g_out_data = new float[6]{0};
    float *ct_out_data = new float[6]{0};
    _g_test(data, g_out_data, 6);
    _ct_test(data, ct_out_data, 6);
    auto ret = memcmp(g_out_data, ct_out_data, 6 * 4);
    if (ret != 0)
    {
        ERROR() << "selu failed";
    }
}

static void test_hardswish(void)
{
    float hard_swish_opt_tag = 2;
    auto _g_test = [&hard_swish_opt_tag](float *in_data, float *out_data, int32_t num) {
        for (int i = 0; i < num; i++)
        {
            float value = in_data[i];
            if (hard_swish_opt_tag != 0)
            {
                value = value - 3;
            }

            if (value <= -3)
            {
                value = 0;
            }
            else if (value > -3 && value < 3)
            {
                value = value * (value + 3) / 6;
            }
            if (hard_swish_opt_tag == 2)
            {
                value = value + 0.375;
            }
            out_data[i] = value;
            INFO() << "i = " << i << ", value = " << value;
        }
    };
    auto _ct_test = [&hard_swish_opt_tag](float *in_data, float *out_data, int32_t num) {
        aipubt::TensorPtr t = std::make_shared<aipubt::Tensor>("t", aipubt::TensorShape(num), aipubt::Tensor::FP32);
        t->set_data(in_data, num * 4);
        auto value = aipubt::ComputeTensor(t);
        if (hard_swish_opt_tag != 0)
        {
            value = value - 3;
        }

        auto mask = (value > -3);
        value = value * mask;

        mask = (value > -3) * (value < 3);

        auto masked_value = value * mask;
        masked_value = masked_value * (masked_value + 3) / 6;
        auto unmasked_value = (value * (1 - mask));
        auto new_value = masked_value + unmasked_value;

        if (hard_swish_opt_tag == 2)
        {
            new_value = new_value + 0.375f;
        }
        INFO() << new_value;
        memcpy(out_data, ((aipubt::TensorPtr)new_value)->data<char>(), num * sizeof(float));
    };

    // float *data = new float[6]{1.233, 0.222, -0.987, 2.45, -8.9, -12.78};
    int32_t num = 12;
    float minv = -10.f;
    float maxv = 10.f;
    std::vector<float> random_data = generate_random_vector(num, minv, maxv);
    float *in_data = new float[num];
    memcpy(in_data, random_data.data(), num * 4);
    float *g_out_data = new float[num]{0};
    float *ct_out_data = new float[num]{0};
    _g_test(in_data, g_out_data, num);
    _ct_test(in_data, ct_out_data, num);
    auto ret = memcmp(g_out_data, ct_out_data, num * sizeof(float));
    if (ret != 0)
    {
        ERROR() << "hardswish failed";
        for (int i = 0; i < 6; i++)
        {
            INFO() << "golden data[" << i << "] = " << g_out_data[i] << ", ct data = " << ct_out_data[i];
        }
    }
}

void test_qtlib(void)
{
    // test_qtlibn_relu();
    // test_qtlibn_depthwise();
    // test_qtlibn_abs();
    // test_qtlibn_gruv3();
    // test_qtlibn_basiclstm();
    // test_q();
    test_tanh();
    test_selu();
    test_hardswish();
}
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
 * @Date: 2022-03-21 18:43:10
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-04 16:21:10
 */

#include "quantization_transform/utils/calibrate_utils.h"
#include <typeinfo>

namespace calibrate_utils
{
/* TODO:
1.implement to statisitic max/min values along dim
2.like per-channel statistic max/min
*/
#define DEQUANTIZE(x, s, z) ((x) - (z)) * (s)

void statistic_max_min_from_float_tensor(aipubt::TensorPtr tensor,
                                         std::vector<float> &statistic_max,
                                         std::vector<float> &statistic_min)
{
    using T = float;
    const std::vector<aipubt::Quantization::Granularity> _sp_grand = {
        aipubt::Quantization::Per_Tensor,
        aipubt::Quantization::Per_Channel,
        aipubt::Quantization::Per_Group,
    };
    auto granularity = tensor->quantization().granularity();
    if (!aipubt::contains(_sp_grand, granularity))
    {
        WARN() << "unsupported granularity = " << aipubt::Quantization::granularity_to_string(granularity) << ", we only support [Per_Tensor, Per_Channel, Per_Group]";
        granularity = aipubt::Quantization::Per_Tensor;
    }
    if (granularity == aipubt::Quantization::Per_Tensor)
    {
        T f_min = tensor->min<T>();
        T f_max = tensor->max<T>();
        statistic_min.push_back(f_min);
        statistic_max.push_back(f_max);
    }
    else if (granularity == aipubt::Quantization::Per_Channel || granularity == aipubt::Quantization::Per_Group)
    {
        auto group = tensor->quantization().group();
        auto permuted_tensor = permute_utils::permute(tensor);
        auto datas = permuted_tensor->data<T>();
        auto step = tensor->mem_size() / sizeof(T) / group;
        T *tmp_datas = new T[step];
        for (auto i = 0; i < group; i++)
        {
            memset(tmp_datas, 0, sizeof(T) * step);
            memcpy(tmp_datas, datas + i * step, step * sizeof(T));
            T f_min = *std::min_element(tmp_datas, tmp_datas + step);
            T f_max = *std::max_element(tmp_datas, tmp_datas + step);
            statistic_min.push_back(f_min);
            statistic_max.push_back(f_max);
        }
        delete[] tmp_datas;

        // auto permuted_tensor = permute_utils::permute(tensor);
        // auto group_num = tensor->quantization().group();
        // aipubt::ComputeTensor permuted_ct = permuted_tensor;
        // int32_t shape_size = permuted_tensor->shape().size();
        // int32_t group_size = shape_size / group_num;
        // std::vector<int32_t> new_shape = {group_num, group_size};
        // permuted_ct = permuted_ct.astype(aipubt::Tensor::FP32).reshape(new_shape);
        // auto permuted_ct_min = permuted_ct.min(std::vector<int32_t>({1}));
        // auto permuted_ct_max = permuted_ct.max(std::vector<int32_t>({1}));
        // statistic_min = permuted_ct_min;
        // statistic_max = permuted_ct_max;
    }
    else
    {
    }
}

template <typename T>
static void statistic_max_min(aipubt::TensorPtr tensor,
                              std::vector<float> &statistic_max,
                              std::vector<float> &statistic_min)
{
    auto scale_num = tensor->quantization().scales().size();
    auto zp_num = tensor->quantization().offsets().size();
    if (scale_num > 0 && zp_num > 0)
    {
        std::vector<float> scales = tensor->quantization().scales();
        std::vector<int32_t> zerops = tensor->quantization().offsets();
        // auto group = tensor->quantization().group();
        auto permuted_tensor = permute_utils::permute(tensor);
        auto datas = permuted_tensor->data<T>();
        // per-group or per-channel
        // now we have already change per-group to per-channel in update_tensor_group_and_granularity pass
        int32_t scale_size = scales.size();
        auto step = tensor->mem_size() / sizeof(T) / scale_size;
        T *tmp_datas = new T[step];
        for (auto i = 0; i < scale_size; i++)
        {
            memset(tmp_datas, 0, sizeof(T) * step);
            memcpy(tmp_datas, datas + i * step, step * sizeof(T));
            T min_v = *std::min_element(tmp_datas, tmp_datas + step);
            T max_v = *std::max_element(tmp_datas, tmp_datas + step);
            float f_min = DEQUANTIZE(float(min_v), scales[i], zerops[i]);
            float f_max = DEQUANTIZE(float(max_v), scales[i], zerops[i]);

            statistic_min.push_back(f_min);
            statistic_max.push_back(f_max);
        }
        delete[] tmp_datas;
    }
    else
    {
        ERROR() << "tensor:" << tensor << ", scale_size = " << scale_num << ", zp_size = " << zp_num << ", should be size > 0.";
    }
}

static std::map<aipubt_dtype, std::function<void(aipubt::TensorPtr, std::vector<float> &, std::vector<float> &)>> STATISTIC_MAX_MIN = {
    {aipubt::Tensor::INT8, statistic_max_min<int8_t>},
    {aipubt::Tensor::UINT8, statistic_max_min<uint8_t>},
    {aipubt::Tensor::INT16, statistic_max_min<int16_t>},
    {aipubt::Tensor::UINT16, statistic_max_min<uint16_t>},
    {aipubt::Tensor::INT32, statistic_max_min<int32_t>},
    {aipubt::Tensor::UINT32, statistic_max_min<uint32_t>},
    {aipubt::Tensor::INT64, statistic_max_min<int64_t>},
    {aipubt::Tensor::UINT64, statistic_max_min<uint64_t>},
    // {aipubt::Tensor::FP32, statistic_max_min<float>},
    {aipubt::Tensor::FP32, statistic_max_min_from_float_tensor},
};

void statistic_max_min(aipubt::TensorPtr tensor,
                       std::vector<float> &statistic_max,
                       std::vector<float> &statistic_min)
{
    auto dtype = tensor->type();
    if (!STATISTIC_MAX_MIN.count(dtype))
    {
        DEBUG() << "Tensor " << tensor->name() << " primitive dtype is " << dtype << ", not support.";
        return;
    }
    STATISTIC_MAX_MIN[dtype](tensor, statistic_max, statistic_min);
}

}; // namespace calibrate_utils

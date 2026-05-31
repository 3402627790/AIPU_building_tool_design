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
 * @Date: 2022-03-13 22:13:08
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-10 10:37:57
 */

#include "quantization_transform/utils/quantize_utils.h"

namespace quant_utils
{
float roundHalfToEven(float f)
{
    const float r = round(f); // Result is round-half-away-from-zero
    const float d = r - f;    // Difference

    // Result is not half, RHAFZ result same as RHTE
    if ((d != 0.5f) && (d != -0.5f))
    {
        return r;
    }

    // Check if RHAFZ result is even, then RHAFZ result same as RHTE
    if (fmod(r, 2.0f) == 0.0f)
    {
        return r;
    }

    // Switch to even value
    return f - d;
}

void get_scale_approximation_params(const float &total_scale,
                                    float &do_scale,
                                    int8_t &do_shift,
                                    const int32_t mult_bits,
                                    const std::string &msg,
                                    const bool limit,
                                    const int32_t mult_bits_ceil,
                                    const int32_t shift_bits_ceil,
                                    bool force_positive_shift)
{
    // AIFF max support multiplier value is 32767, so we limit mbits to less equal than 15
    auto real_bits = std::min(mult_bits, mult_bits_ceil);
    float real_factor = pow(2, real_bits) - 1;
    int32_t shift_bits = static_cast<int32_t>(floor(log2(real_factor / total_scale)));

    shift_bits = limit ? std::min(shift_bits, mult_bits) : shift_bits;
    // AIFF max support shift value is 31, so we limit shift to less equal than 31
    if (shift_bits > shift_bits_ceil)
    {
        DEBUG() << msg << ",shift_bits :" << shift_bits << ", is larger than " << shift_bits_ceil;
    }
    do_shift = static_cast<int8_t>(std::min(shift_bits, shift_bits_ceil));
    do_scale = floor(total_scale * pow(2, do_shift) + 0.5);

    if (force_positive_shift && do_shift < 0)
    {
        WARN() << msg << ", now doshift(" << (int32_t)do_shift << ") < 0, so qtlib will clip doshift to zero for meeting lib requirement.";
        do_shift = 0;
        do_scale = std::min(pow(2, mult_bits) - 1, pow(2, mult_bits_ceil) - 1);
    }
}

void get_scale_approximation_params(const std::vector<float> &total_scale,
                                    std::vector<float> &do_scale,
                                    std::vector<int8_t> &do_shift,
                                    const int32_t mult_bits,
                                    const std::string &msg,
                                    const bool limit,
                                    const int32_t mult_bits_ceil,
                                    const int32_t shift_bits_ceil,
                                    bool force_positive_shift)
{
    aipu_assert(total_scale.size() > 0 && "scales items should > 0;");
                                    const bool limit,
                                    const int32_t mult_bits_ceil,
                                    const int32_t shift_bits_ceil,
                                    bool force_positive_shift)
{
    // AIFF max support multiplier value is 32767, so we limit mbits to less equal than 15
    auto real_bits = std::min(mult_bits, mult_bits_ceil);
    float real_factor = pow(2, real_bits) - 1;
    int32_t shift_bits = static_cast<int32_t>(floor(log2(real_factor / total_scale)));

    shift_bits = limit ? std::min(shift_bits, mult_bits) : shift_bits;
    // AIFF max support shift value is 31, so we limit shift to less equal than 31
    if (shift_bits > shift_bits_ceil)
    {
        DEBUG() << msg << ",shift_bits :" << shift_bits << ", is larger than " << shift_bits_ceil;
    }
    do_shift = static_cast<int8_t>(std::min(shift_bits, shift_bits_ceil));
    do_scale = floor(total_scale * pow(2, do_shift) + 0.5);

    if (force_positive_shift && do_shift < 0)
    {
        WARN() << msg << ", now doshift(" << (int32_t)do_shift << ") < 0, so qtlib will clip doshift to zero for meeting lib requirement.";
        do_shift = 0;
        do_scale = std::min(pow(2, mult_bits) - 1, pow(2, mult_bits_ceil) - 1);
    }
}

void get_scale_approximation_params(const std::vector<float> &total_scale,
                                    std::vector<float> &do_scale,
                                    std::vector<int8_t> &do_shift,
                                    const int32_t mult_bits,
                                    const std::string &msg,
                                    const bool limit,
                                    const int32_t mult_bits_ceil,
                                    const int32_t shift_bits_ceil,
                                    bool force_positive_shift)
{
    aipu_assert(total_scale.size() > 0 && "scales items should > 0;");
    if (force_positive_shift && std::any_of(do_shift_list.begin(), do_shift_list.end(), [](int8_t ds) { return ds < 0; }))
    {
        std::vector<int32_t> do_scale_list = do_scale;
        for (uint32_t k = 0; k < do_shift_list.size(); k++)
        {
            auto dshift = do_shift_list[k];
            if (dshift < 0)
            {
                WARN() << msg << ", now doshift(" << (int32_t)do_shift << ") < 0, so qtlib will clip doshift to zero for meeting lib requirement.";
                do_shift_list[k] = 0;
                do_scale_list[k] = std::min(pow(2, mult_bits) - 1, pow(2, mult_bits_ceil) - 1);
            }
        }
        do_shift = aipubt::ComputeTensor(do_shift_list);
        do_scale = aipubt::ComputeTensor(do_scale_list);
    }
}
/**
 * @brief quantize min_val and max_val to get linear quantization info: scale/zp.
 *
 * @return linear quantization scale and zerop.
 */
scale_zp_pair cal_scale_zp(const float &max_val,
                           const float &min_val,
                           const int8_t q_bits,
                           const bool is_sign,
                           const bool is_sym_quant,
                           const bool is_full_range,
                           const bool is_power_two_scale)
{
    // TODO support restricted_range
    // INFO() << "is_sym_quant " << is_sym_quant << ", is_full_range = " << is_full_range << ",  is_signe = " << is_sign;
    auto bits = q_bits;
    float scale = 0.0;
    zp_qdtype zp = 0;
    int32_t q_min = 0;
    int32_t q_max = pow(2, bits) - 1;
    float scale_max = pow(2, 2 * 12) - 1;
    bool is_signed = is_sign;
    if (is_signed)
    {
        if (is_full_range)
        {
            q_max = pow(2, bits - 1) - 1;
            q_min = -1 - q_max;
        }
        else
        {
            q_max = pow(2, bits - 1) - 1;
            q_min = -q_max;
        }
    }
    int32_t q_range = q_max - q_min;
    // INFO() << "q_range = " << q_range;
    if (is_sym_quant)
    {
        float f_range = std::max(fabs(min_val), fabs(max_val));
        if (is_signed)
        {
            f_range = 2 * f_range;
        }
        f_range = f_range > EPSINON ? f_range : q_range;
        scale = q_range / f_range;
        zp = 0;
    }
    else
    {
        float f_range = max_val - min_val;
        // INFO() << "f_range = "<< f_range << ", max _val= " << max_val << ", min_val = " << min_val;
        float f_zfactor = min_val;
        int32_t f_zoffet = q_min;
        f_range = f_range > EPSINON ? f_range : q_range;
        scale = q_range / f_range;
        zp = clip<float, float>(roundHalfToEven(scale * f_zfactor) - f_zoffet, 1 - pow(2, (bits - 1)), pow(2, (bits - 1)) - 1);
    }
    scale = (scale > scale_max) ? scale_max : scale;
    scale_zp_pair scale_zp = std::make_pair(scale, zp);

    return scale_zp;
}

void get_quant_params_from_statistic_value(const std::vector<float> &max_val,
                                           const std::vector<float> &min_val,
                                           const int8_t q_bits,
                                           const bool is_sign,
                                           const bool is_sym_quant,
                                           const bool is_full_range,
                                           const bool is_power_two_scale,
                                           std::vector<float> &scales,
                                           std::vector<zp_qdtype> &zerops)
{
    auto max_val_size = max_val.size();
    auto min_val_size = min_val.size();
    if (max_val_size != min_val_size)
    {
        ERROR() << "please check the max_val and min_val size. which now max_val size = " << max_val_size
                << ", and min_val size = " << min_val_size;
        return;
    }
    for (uint32_t i = 0; i < max_val_size; i++)
    {
        scale_zp_pair scale_zp = cal_scale_zp(max_val[i], min_val[i], q_bits, is_sign, is_sym_quant, is_full_range, is_power_two_scale);
        // DEBUG() << "max_val: " << max_val[i] << ", min_val: " << min_val[i];
        // DEBUG() << "scale: " << scale_zp.first << ", zp: " << (float)scale_zp.second;
        scales.push_back(scale_zp.first);
        zerops.push_back(scale_zp.second);
    }
}

void get_quant_params_from_statistic_value(const std::vector<float> &max_val,
                                           const std::vector<float> &min_val,
                                           const int8_t q_bits,
                                           const bool is_sign,
                                           const bool is_sym_quant,
                                           const bool is_full_range,
                                           std::vector<float> &scales,
                                           std::vector<zp_qdtype> &zerops)
{

    get_quant_params_from_statistic_value(max_val,
                                          min_val,
                                          q_bits,
                                          is_sign,
                                          is_sym_quant,
                                          is_full_range,
                                          false,
                                          scales,
                                          zerops);
}

const int32_t optimized_target_level(std::string target_name)
{
    std::vector<std::string> targets = {"X1", "Z2", "Z3"};
    transform(target_name.begin(), target_name.end(), target_name.begin(), ::toupper);
    auto splitted_target = utils::split_string(target_name, "_");
    return aipubt::contains(targets, splitted_target[0]) ? 0 : 1;
}

/*
    [quantization param]:
    now we only dequantize quantization scale/zp of android/armnn side to float_min/float_max.

    [constant data]:
    now we only support primitive_dequantize in android/armnn side, and aipu_quantize in aipu side.

                |-> android/armnn side function name: primitive_dequantize --> per_tensor/per_channel
    dequantize -
                |-> aipu side function name: aipu_dequantize --> per_tensor/per_channel

              |-> android/armnn side function name: primitive_quantize --> per_tensor/per_channel
    quantize -
              |-> aipu side function name: aipu_quantize --> per_tensor/per_channel
*/
float dequantize_formula_android(float v, float s, int32_t z)
{
    return (v - z) * s;
}

float dequantize_formula_aipu(float v, float s, int32_t z)
{
    return (v + z) / s;
}
/**
 * @brief from one quantized info to original data distribute (like [statistic_min, statistic_max]).
 *
 * @return statistic_min, statistic_max
 */
void dequantize_from_primitive_quant_info(const aipubt::Tensor::Dtype &src_dtype,
                                          const scale_zp_pair &src_scale_zp,
                                          float &float_max,
                                          float &float_min,
                                          std::function<float(float, float, int32_t)> &dequantize_formula)
{
    float src_scale = src_scale_zp.first;
    zp_qdtype src_zp = src_scale_zp.second;
    float range_min = dtype_utils::dtype_to_range_min(src_dtype);
    float range_max = dtype_utils::dtype_to_range_max(src_dtype);
    float_min = dequantize_formula(range_min, src_scale, src_zp);
    float_max = dequantize_formula(range_max, src_scale, src_zp);
}

void dequantize_from_primitive_quant_info(const aipubt_dtype &src_dtype,
                                          const std::vector<float> &src_scales,
                                          const std::vector<zp_qdtype> &src_zerops,
                                          std::vector<float> &statistic_max,
                                          std::vector<float> &statistic_min,
                                          std::function<float(float, float, int32_t)> &dequantize_formula)
{
    auto s_size = src_scales.size();
    auto z_size = src_zerops.size();
    if (s_size != z_size)
    {
        ERROR() << "please check the scale and zerop size. which now scale size = " << s_size
                << ", and zerop size = " << z_size;
        return;
    }

    for (uint32_t i = 0; i < s_size; i++)
    {
        float statis_min;
        float statis_max;
        scale_zp_pair scale_zp = std::make_pair(src_scales[i], src_zerops[i]);
        dequantize_from_primitive_quant_info(src_dtype,
                                             scale_zp,
                                             statis_max,
                                             statis_min,
                                             dequantize_formula);
        statistic_max.push_back(statis_max);
        statistic_min.push_back(statis_min);
    }
}

template <typename ST>
void dequantize_per_tensor(aipubt::TensorPtr tensor, formula_func &dequantize_formula)
{
    // quantization --> float
    std::vector<float> scales = tensor->quantization().scales();
    std::vector<int32_t> zerops = tensor->quantization().offsets();
    ST *datas = tensor->data<ST>();
    auto num = tensor->mem_size() / sizeof(ST);
    float *new_data = new float[num];
    for (uint32_t i = 0; i < num; i++)
    {
        new_data[i] = dequantize_formula(datas[i], scales[0], zerops[0]);
    }
    tensor->set_type(aipubt::Tensor::FP32);
    tensor->set_data(new_data, num * sizeof(float));
}

template <typename ST>
void dequantize_per_channel(aipubt::TensorPtr tensor, formula_func &dequantize_formula)
{
    // quantization --> float
    // trans_tensor just aranges the data, so the quantization info still in tensor.
    auto trans_tensor = permute_utils::permute(tensor);
    ST *datas = trans_tensor->data<ST>();

    std::vector<float> scales = tensor->quantization().scales();
    std::vector<int32_t> zerops = tensor->quantization().offsets();
    auto group = tensor->quantization().group();

    auto data_num = trans_tensor->mem_size() / sizeof(ST);
    // per-group or per-channel
    auto step = data_num / group;
    // DEBUG() << "data_num = " << data_num << ", step="<< step << ", group="<<group;
    float *new_datas = new float[data_num];
    for (uint32_t i = 0; i < data_num; i++)
    {
        new_datas[i] = dequantize_formula((float)datas[i], scales[i / step], zerops[i / step]);
    }

    // need unpermute the new_data to raw data arange
    trans_tensor->set_type(aipubt::Tensor::FP32);
    trans_tensor->set_data(new_datas, data_num * sizeof(float));

    tensor->set_type(aipubt::Tensor::FP32);
    permute_utils::unpermute(trans_tensor);
    if (trans_tensor != tensor)
    {
        float *unpermute_data = trans_tensor->data<float>();
        tensor->set_data(unpermute_data, data_num * sizeof(float));
    }
}

static std::map<aipubt_dtype, std::function<void(aipubt::TensorPtr, formula_func &)>> HTG_DEQUANTIZATION_PER_TENSOR_MAP = {
    {aipubt::Tensor::INT8, dequantize_per_tensor<int8_t>},
    {aipubt::Tensor::UINT8, dequantize_per_tensor<uint8_t>},
    {aipubt::Tensor::INT16, dequantize_per_tensor<int16_t>},
    {aipubt::Tensor::UINT16, dequantize_per_tensor<uint16_t>},
    {aipubt::Tensor::INT32, dequantize_per_tensor<int32_t>},
    {aipubt::Tensor::UINT32, dequantize_per_tensor<uint32_t>},
    {aipubt::Tensor::INT64, dequantize_per_tensor<int64_t>},
    {aipubt::Tensor::UINT64, dequantize_per_tensor<uint64_t>},
    {aipubt::Tensor::FP32, dequantize_per_tensor<float_t>},
};

static std::map<aipubt_dtype, std::function<void(aipubt::TensorPtr, formula_func &)>> HTG_DEQUANTIZATION_PER_CHANNEL_MAP = {
    {aipubt::Tensor::INT8, dequantize_per_channel<int8_t>},
    {aipubt::Tensor::UINT8, dequantize_per_channel<uint8_t>},
    {aipubt::Tensor::INT16, dequantize_per_channel<int16_t>},
    {aipubt::Tensor::UINT16, dequantize_per_channel<uint16_t>},
    {aipubt::Tensor::INT32, dequantize_per_channel<int32_t>},
    {aipubt::Tensor::UINT32, dequantize_per_channel<uint32_t>},
    {aipubt::Tensor::INT64, dequantize_per_tensor<int64_t>},
    {aipubt::Tensor::UINT64, dequantize_per_tensor<uint64_t>},
    {aipubt::Tensor::FP32, dequantize_per_tensor<float_t>},
};

template <typename ST, typename DT>
void aipu_quantize_per_channel(aipubt::TensorPtr tensor)
{
    // float --> quantization
    // trans_tensor just aranges the data, so the quantization info still in tensor.
    auto trans_tensor = permute_utils::permute(tensor);
    ST *datas = trans_tensor->data<ST>();

    std::vector<float> scales = tensor->quantization().scales();
    std::vector<int32_t> zerops = tensor->quantization().offsets();
    auto bits = tensor->quantization().bits();
    auto dtype = tensor->quantization().aipu_dtype();
    // auto group = tensor->quantization().group();

    // INFO() << "tensor: " << tensor->name() << " quantization_bits = " << (int32_t)bits;
    auto issigned = aipubt::Tensor::is_signed(dtype);
    float qmin = dtype_utils::bits_to_range_min(bits, issigned);
    float qmax = dtype_utils::bits_to_range_max(bits, issigned);

    auto data_num = tensor->mem_size() / sizeof(ST);
    // per-group or per-channel
    auto step = data_num / scales.size();
    // DEBUG() << "data_num = " << data_num << ", q_min = " << qmin << ", qmax = " << qmax << ", step = " << step << ", group = " << group;
    DT *new_datas = new DT[data_num];
    for (uint32_t i = 0; i < data_num; i++)
    {
        float input = static_cast<float>(datas[i]);
        float out = AIPU_QUANTIZE(input, scales[i / step], zerops[i / step]);
        out = clip<float, float>(roundHalfToEven(out), qmin, qmax);

        // DEBUG() << "input = " << input << ", scale = " << scales[i / step] << ", zp = " << zerops[i / step] << ", out = " << out;
        new_datas[i] = static_cast<DT>(out);
    }

    // need unpermute the new_data to raw data arange
    trans_tensor->set_type(dtype);
    trans_tensor->set_data(new_datas, data_num * sizeof(DT));

    permute_utils::unpermute(trans_tensor);
    tensor->set_type(dtype);
    if (trans_tensor != tensor)
    {
        DT *unpermute_data = trans_tensor->data<DT>();
        tensor->set_data(unpermute_data, data_num * sizeof(DT));
    }
}

template <typename ST, typename DT>
void aipu_quantize_per_tensor(aipubt::TensorPtr tensor)
{
    // float --> quantization
    std::vector<float> scales = tensor->quantization().scales();
    std::vector<int32_t> zerops = tensor->quantization().offsets();
    auto bits = tensor->quantization().bits();
    auto dtype = tensor->quantization().aipu_dtype();
    auto issigned = aipubt::Tensor::is_signed(dtype);
    if (sizeof(DT) * 8 != bits)
    {
        WARN() << "now sizeof(DT) * 8 = " << sizeof(DT) * 8 << ", and bits = " << bits;
    }
    if ((int32_t)(aipubt::Tensor::dtype_bits(dtype)) != bits)
    {
        WARN() << "now dtype bits = " << aipubt::Tensor::dtype_bits(dtype) << ", and bits = " << bits;
    }
    float qmin = static_cast<float>(dtype_utils::bits_to_range_min(bits, issigned));
    float qmax = static_cast<float>(dtype_utils::bits_to_range_max(bits, issigned));
    ST *datas = tensor->data<ST>();
    auto num = tensor->mem_size() / sizeof(ST);
    DT *new_data = new DT[num];
    // INFO() << "scale.sizne = " << scales.size() << ", offset size = " << zerops.size();
    // INFO() << "scale = " << scales[0] << ", zp = " << (float)zerops[0] << ", qmin = " << qmin << ", qmax = " << qmax;
    for (uint32_t i = 0; i < num; i++)
    {
        float t = AIPU_QUANTIZE(datas[i], scales[0], zerops[0]);
        t = clip<float, float>(roundHalfToEven(t), qmin, qmax);
        new_data[i] = static_cast<DT>(t);
    }
    tensor->set_type(dtype);
    tensor->set_data(new_data, num * sizeof(DT));
}

static std::map<std::vector<aipubt_dtype>, std::function<void(aipubt::TensorPtr)>> QUANTIZATION_PER_TENSOR_MAP = {
    {{aipubt::Tensor::FP32, aipubt::Tensor::INT8}, aipu_quantize_per_tensor<float, int8_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::UINT8}, aipu_quantize_per_tensor<float, uint8_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::INT16}, aipu_quantize_per_tensor<float, int16_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::UINT16}, aipu_quantize_per_tensor<float, uint16_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::INT32}, aipu_quantize_per_tensor<float, int32_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::UINT32}, aipu_quantize_per_tensor<float, uint32_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::INT64}, aipu_quantize_per_tensor<float, int64_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::UINT64}, aipu_quantize_per_tensor<float, uint64_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::ALIGNED_INT4}, aipu_quantize_per_tensor<float, int8_t>},
};

static std::map<std::vector<aipubt_dtype>, std::function<void(aipubt::TensorPtr)>> QUANTIZATION_PER_CHANNEL_MAP = {
    {{aipubt::Tensor::FP32, aipubt::Tensor::INT8}, aipu_quantize_per_channel<float, int8_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::UINT8}, aipu_quantize_per_channel<float, uint8_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::INT16}, aipu_quantize_per_channel<float, int16_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::UINT16}, aipu_quantize_per_channel<float, uint16_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::INT32}, aipu_quantize_per_channel<float, int32_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::UINT32}, aipu_quantize_per_channel<float, uint32_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::INT64}, aipu_quantize_per_channel<float, int64_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::UINT64}, aipu_quantize_per_channel<float, uint64_t>},
    {{aipubt::Tensor::FP32, aipubt::Tensor::ALIGNED_INT4}, aipu_quantize_per_channel<float, int8_t>},
};
void aipu_fpx_quantize(aipubt::TensorPtr tensor)
{
    auto aipu_dtype = tensor->quantization().aipu_dtype();
    float float_range_min = dtype_utils::dtype_to_range_min(aipu_dtype);
    float float_range_max = dtype_utils::dtype_to_range_max(aipu_dtype);
    auto scales = tensor->quantization().scales();
    auto zps = tensor->quantization().offsets();
    auto shape = tensor->shape();
    auto quantization_dim = tensor->quantization().quantization_dim();
    if (quantization_dim != 0)
    {
        ERROR() << "qtlib now only support quantization_dim = 0 fpx quantization, but now quantization_dim = " << quantization_dim;
        return;
    }
    // INFO() << "scales.shape = " << scales->shape() << ", and tensor.shape = " << tensor->shape();
    aipubt::ComputeTensor tct = tensor;
    tct = tct.astype(aipubt::Tensor::FP32);
    auto oc = shape[0];
    int32_t scale_size = scales.size();
    bool rearrage = false;
    if (scale_size > oc)
    {
        if (shape.dim() == 1)
        {
            ERROR() << "blockwise quantization now does not support tensor.shape.dim == 1";
            return;
        }
        else if (shape.dim() != 2)
        {
            if (shape.size() != shape[0] * shape[shape.dim() - 1])
            {
                ERROR() << "blockwise quantization now does not support tensor.shape.dim != 2, with shape.size != shape[0] * shape[-1], now shape = " << shape;
                return;
            }
        }

        auto new_shape_size = shape.dim() + 1;
        std::vector<int32_t> re_shape(new_shape_size, 1);
        auto ic = shape[-1];
        int32_t grp_num = scale_size / oc;
        int32_t grp_size = ic / grp_num;
        re_shape[0] = oc;
        re_shape[new_shape_size - 2] = grp_num;
        scales = scales.reshape(re_shape);
        zps = zps.reshape(re_shape);

        std::vector<int32_t> tct_re_shape(new_shape_size, 1);
        tct_re_shape[0] = oc;
        tct_re_shape[new_shape_size - 2] = grp_num;
        tct_re_shape[new_shape_size - 1] = grp_size;
        tct = tct.reshape(tct_re_shape);
        rearrage = true;
    }
    else if (scale_size == oc)
    {
        std::vector<int32_t> re_shape(shape.dim(), 1);
        re_shape[0] = oc;
        scales = scales.reshape(re_shape);
        zps = zps.reshape(re_shape);
    }
    else if (scale_size > 1)
    {
        ERROR() << "qtlib does not support 1 < scales.size(= " << scale_size << ") but not scales.size == oc or scales.size == oc * group_size";
        return;
    }
    // INFO() << "qtlib fpx quantize scales's shape =  " << aipubt::ComputeTensor(re_shape);
    tct = tct * scales - zps;
    if (rearrage)
    {
        tct = tct.reshape(shape);
    }
    tct = tct.minimum(aipubt::ComputeTensor(float_range_max)).maximum(aipubt::ComputeTensor(float_range_min));
    // INFO() << "tensor after fpx quantization, its mins = " << tct.min() << ", its maxs = " << tct.max();
    tct = tct.astype(aipu_dtype);
    tensor->set_type(aipu_dtype);
    tensor->share_data_from(tct);
    // INFO() << "tensor after fpx quantization, its dtype = " << tensor->type() << ", its mem_size =  " << tensor->mem_size();
}

void aipu_quantize(aipubt::TensorPtr tensor)
{
    // this is for OpsAPI usage.
    auto t_dtype = tensor->type();
    if (!aipubt::Tensor::is_float(t_dtype) || (aipubt::Tensor::is_float(t_dtype) && aipubt::Tensor::dtype_bits(t_dtype) <= 8))
    {
        DEBUG() << "tensor name = " << tensor->name() << " has quantized. so no need to quantize any more.";
        return;
    }

    if (tensor->data<float>())
    {
        auto aipu_dtype = tensor->quantization().aipu_dtype();
        if (aipubt::Tensor::is_float(aipu_dtype))
        {
            aipu_fpx_quantize(tensor);
            return;
        }
        auto t_dtype = tensor->type();
        auto granularity = tensor->quantization().granularity();
        auto scales = tensor->quantization().scales();
        std::vector<aipubt_dtype> src_dst_dtype = {t_dtype, aipu_dtype};
        if (QUANTIZATION_PER_TENSOR_MAP.count(src_dst_dtype) == 0)
        {
            ERROR() << tensor << " cannot find the corresponding quantize map. src dtype: " << t_dtype << ", dst dtype: " << aipu_dtype;
            return;
        }

        if (scales.size() > 1 && granularity == aipubt::Quantization::Per_Tensor)
        {
            WARN() << "please check the quantization granularity(=per_tensor), but scales size > 1, and qtlib will change per_tensor to per_channel.";
            granularity = aipubt::Quantization::Per_Channel;
            tensor->quantization().set_granularity(granularity);
        }

        if (scales.size() == 1)
        {
            QUANTIZATION_PER_TENSOR_MAP[src_dst_dtype](tensor);
        }
        else if (scales.size() > 1)
        {
            auto group = tensor->quantization().group();
            auto qdim = tensor->quantization().quantization_dim();
            int32_t qdim_value = tensor->shape()[qdim];
            int32_t scales_num = scales.size();
            if (qdim_value != scales_num && group != scales_num)
            {
                ERROR() << "tensor_name = " << tensor->name() << ": quantization scales num(=" << scales_num
                        << ") != per_channel num(=" << qdim_value << ") and != per-group num = " << group;
            }
            QUANTIZATION_PER_CHANNEL_MAP[src_dst_dtype](tensor);
        }
        else
        {
            ERROR() << "scales.size = " << (int)scales.size() << " is not supported by aipu_quantize";
            return;
        }
    }
    else
    {
        ERROR() << "when quantize the constant data of " << tensor->name() << ", it has null data, please check it.";
        return;
    }
}

void primitive_dequantize(aipubt::TensorPtr tensor, formula_func &formula)
{
    auto tensor_dtype = tensor->type();
    if (tensor->data<int8_t>() && !aipubt::Tensor::is_float(tensor_dtype))
    {
        auto dtype = tensor->type();
        auto granularity = tensor->quantization().granularity();
        auto scales = tensor->quantization().scales();
        if (HTG_DEQUANTIZATION_PER_TENSOR_MAP.count(dtype) == 0)
        {
            ERROR() << "Cannot find the corresponding dequantize map. dtype is: " << dtype;
            return;
        }
        if (scales.size() == 0)
        {
            ERROR() << "when dequantize the tensor(" << tensor << "), its's scales.size == 0, and canot dequantize.";
        }
        if (granularity == aipubt::Quantization::Per_Tensor)
        {
            if (scales.size() > 1)
            {
                WARN() << tensor->name() << ", please check the quantization granularity(=per_tensor), but scales size > 1.";
            }
            HTG_DEQUANTIZATION_PER_TENSOR_MAP[dtype](tensor, formula);
        }
        else if (granularity == aipubt::Quantization::Per_Channel || granularity == aipubt::Quantization::Per_Group)
        {
            auto group = tensor->quantization().group();
            int32_t scales_num = scales.size();
            auto qdim = tensor->quantization().quantization_dim();
            int32_t qdim_value = tensor->shape()[qdim];

            if (group != scales_num && qdim_value != scales_num)
            {
                ERROR() << "quantization group num(=" << group << ") != scale num(" << scales_num
                        << ")  and scale num != per-channel out = " << qdim_value << " in per-channel or per-group.";
            }
            HTG_DEQUANTIZATION_PER_CHANNEL_MAP[dtype](tensor, formula);
        }
        else
        {
            ERROR() << "Unsupported granularity, only supported:[per_tensor, per_channel].";
            return;
        }
    }
    else
    {
        if (tensor->data<float>() == nullptr)
        {
            ERROR() << "when dequantize the " << tensor->name() << ", it has null data, please check it.";
        }
        else
        {
            DEBUG() << "when dequantize the " << tensor->name() << ", it has float data, so we will not dequantize this data.";
        }

        return;
    }
}

template <typename Ori_T, typename Dst_T>
void transform_data_type_to_tensor(aipubt::TensorPtr tensor, Ori_T *data, const int32_t len)
{
    auto new_data = new Dst_T[len];
    for (int32_t l = 0; l < len; l++)
    {
        *(new_data + l) = static_cast<Dst_T>(*(data + l));
    }
    tensor->set_data(new_data, len * sizeof(Dst_T));
}

template <typename Ori_T>
std::map<aipubt_dtype, std::function<void(aipubt::TensorPtr, Ori_T *, int32_t)>> DATA_TYPE_TRANSFORM_MAP = {
    {aipubt::Tensor::INT8, transform_data_type_to_tensor<Ori_T, int8_t>},
    {aipubt::Tensor::UINT8, transform_data_type_to_tensor<Ori_T, uint8_t>},
    {aipubt::Tensor::INT16, transform_data_type_to_tensor<Ori_T, int16_t>},
    {aipubt::Tensor::UINT16, transform_data_type_to_tensor<Ori_T, uint16_t>},
    {aipubt::Tensor::INT32, transform_data_type_to_tensor<Ori_T, int32_t>},
    {aipubt::Tensor::UINT32, transform_data_type_to_tensor<Ori_T, uint32_t>},
    {aipubt::Tensor::INT64, transform_data_type_to_tensor<Ori_T, int64_t>},
};

template <typename Ori_T>
void convert_data_to_tensor(aipubt::TensorPtr tensor,
                            Ori_T *data,
                            const int32_t len)
{
    if (!data)
    {
        ERROR() << "please check the doscale/doshift data, which now is nullptr.";
        return;
    }
    auto output_dtype = tensor->type();
    if (DATA_TYPE_TRANSFORM_MAP<Ori_T>.count(output_dtype) == 0)
    {
        ERROR() << "cannot find the corresponding data type transform map, dtype is: " << output_dtype;
    }
    DATA_TYPE_TRANSFORM_MAP<Ori_T>[output_dtype](tensor, data, len);
}
//must initialize the template in .cpp, or must put the implementation in .h
template void convert_data_to_tensor<float_t>(aipubt::TensorPtr tensor,
                                              float_t *data,
                                              const int32_t len);
template void convert_data_to_tensor<int32_t>(aipubt::TensorPtr tensor,
                                              int32_t *data,
                                              const int32_t len);
template void convert_data_to_tensor<signed char>(aipubt::TensorPtr tensor,
                                                  signed char *data,
                                                  const int32_t len);

float linear_quantize(const float value, const float scale, const float zerop)
{
    return roundHalfToEven(value * scale - zerop);
}
float linear_quantize_clip(const float value, const float scale, const float zerop, const float clip_min, const float clip_max)
{
    return std::max(std::min((float)roundHalfToEven(value * scale - zerop), clip_max), clip_min);
}

void linear_quantize_clip(aipubt::ComputeTensor &x, aipubt::ComputeTensor &scale,
                          aipubt::ComputeTensor &zerop, float clip_min, float clip_max)
{
    x = x * scale + zerop;
    float_t *data = x->data<float_t>();
    for (int i = 0; i < x->shape().size(); i++)
    {
        data[i] = std::min<float_t>(data[i], clip_max);
        data[i] = std::max<float_t>(data[i], clip_min);
    }
}

void linear_quantize_clip(float *dataptr, const size_t size,
                          const float scales, const float zerops,
                          const float clip_min, const float clip_max)
{
    for (unsigned int i = 0; i < size; i++)
    {
        float value = dataptr[i];
        float new_value;
        new_value = linear_quantize_clip(value, scales, zerops, clip_min, clip_max);
        dataptr[i] = new_value;
    }
}

float linear_dequantize(const float value, const float scale, const float zerop)
{
    return (value + zerop) / scale;
}

void get_fpx_params_from_statistic_value(aipubt::TensorPtr t)
{
    aipubt::ComputeTensor scale;
    aipubt::ComputeTensor zp;
    auto aipu_dtype = t->quantization().aipu_dtype();
    auto fpx_range_min = dtype_utils::dtype_to_range_min(aipu_dtype);
    auto fpx_range_max = dtype_utils::dtype_to_range_max(aipu_dtype);
    aipubt::ComputeTensor fval_mins = t->quantization().mins();
    aipubt::ComputeTensor fval_maxs = t->quantization().maxs();

    // auto granularity = t->quantization().granularity();
    auto scheme = t->quantization().scheme();
    // auto qrange = t->quantization().quantized_range();
    auto is_signed = aipubt::Tensor::is_signed(aipu_dtype);
    // DEBUG() << "granularity = " << (int)granularity;
    // DEBUG() << "scheme = " << (int)scheme;
    bool is_symmetric = scheme == aipubt::Quantization::Symmetric
                            ? true
                            : false;
    // bool is_full_range = qrange == aipubt::Quantization::Full_Range
    //                          ? true
    //                          : false;

    auto fpx_range = fpx_range_max - fpx_range_min;
    if (is_symmetric)
    {
        auto fval_range = fval_maxs.abs().maximum(fval_mins.abs());

        if (is_signed)
        {
            fval_range = fval_range * 2;
        }

        //f_range = f_range > EPSINON ? f_range : q_range;
        scale = fpx_range / fval_range;
        zp = aipubt::ComputeTensor::zeros_like(scale);
    }
    else
    {
        auto fval_range = fval_maxs - fval_mins;
        // INFO() << "f_range = "<< f_range << ", max _val= " << max_val << ", min_val = " << min_val;
        auto f_zfactor = fval_mins;
        auto f_zoffset = fpx_range_min;
        float fval_range_min = fval_range.min();
        if (fval_range_min <= EPSINON)
        {
            WARN() << "fval_range_min = " << fval_range_min << " is hard to quantize, scale should be 1.0, zp = 0";
        }
        // fval_range = fval_range.maximum(aipubt::ComputeTensor(EPSINON));
        // zp = clip<float, float>(roundHalfToEven(scale * f_zfactor) - f_zoffet, 1 - pow(2, (bits - 1)), pow(2, (bits - 1)) - 1);
        scale = fpx_range / fval_range;
        zp = scale * f_zfactor - f_zoffset;
        zp = zp.astype(aipu_dtype);
    }
    // scale = scale.minimum(aipubt::ComputeTensor(scale_max));
    t->quantization().set_scales(scale.astype(aipubt::Tensor::FP32));
    t->quantization().set_offsets(zp);
    t->quantization().set_quantized(true);
}

void calculate_scale_zp(aipubt::TensorPtr t, bool force_to_recalculate)
{
    /*
     *this quantization mode is for OpsAPI:
     *  -if tensor with min/max, qtlib will calculate the scale/zp by itself.
     *  -if tensor with scales/zps(without min/max), qtlib will directly use existed scales/zps for quantization.
     *   (we defaultly use these scales/zps as aipu scales/zps)
     *so if you want to use qtlib to quantize, either providing min/max of tensor or providing scales/zps.
    */

    /*
    if constant data is quantized data, we assume the aipu scale/zp has provided.
    if t doesnot have mins/maxs:
        1. check whether t has data, and we require data type is float, if ok, we will auto statistic mins/maxs of t
        2. t has scales/zps, we will use these scales/zps as aipu scales/zps
        3. if t has data but not float data, scales/zps must be provided to qtlib.
    */
    auto scale_zp_flag = t->quantization().scales().size() && t->quantization().offsets().size();
    if (scale_zp_flag && !force_to_recalculate)
    {
        // t already has scales and zps, so we directly uses it even if t also has mins and maxs.
        auto scales = t->quantization().scales();
        auto zps = t->quantization().offsets();
        if (!(scales[0] == 1.0 && zps[0] == 0))
        {
            return;
        }
    }
    auto flag = t->quantization().maxs().size() && t->quantization().mins().size();
    if (!flag)
    {
        auto ttype = t->type();
        if (aipubt::Tensor::is_float(ttype) && t->data<float>())
        {
            // statistic constant data float max/min. and qtlib only receive float data defaultly
            std::vector<float> statistic_max;
            std::vector<float> statistic_min;
            calibrate_utils::statistic_max_min_from_float_tensor(t, statistic_max, statistic_min);
            t->quantization().set_mins(statistic_min);
            t->quantization().set_maxs(statistic_max);
            flag = t->quantization().maxs().size() && t->quantization().mins().size();
        }
        else if (t->quantization().scales().size() == 0)
        {
            WARN() << "please provide the aipu scales/zps or min/max of tensor : " << t->name() << ", node = " << t->node();
            return;
        }
    }

    if (flag)
    {
        auto statistic_max = t->quantization().maxs();
        auto statistic_min = t->quantization().mins();
        // DEBUG() << "t.name" << t->name() << ", stat_max=" << statistic_max[0];
        // DEBUG() << "t.name" << t->name() << ", stat_min=" << statistic_min[0];
        auto dtype = t->quantization().aipu_dtype();
        // DEBUG() << "t.name " << dtype;
        auto bits = t->quantization().bits();
        bool is_signed = aipubt::Tensor::is_signed(dtype);
        auto granularity = t->quantization().granularity();
        auto scheme = t->quantization().scheme();
        auto qrange = t->quantization().quantized_range();
        // DEBUG() << "granularity = " << (int)granularity;
        // DEBUG() << "scheme = " << (int)scheme;
        bool is_symmetric = scheme == aipubt::Quantization::Symmetric
                                ? true
                                : false;
        bool is_full_range = qrange == aipubt::Quantization::Full_Range
                                 ? true
                                 : false;

        auto num = statistic_max.size();
        if (num != statistic_min.size())
        {
            ERROR() << "please check the max_val and min_val size. which now max_val size = "
                    << num << ", and min_val size = " << statistic_min.size();
            return;
        }

        // weak check
        if (granularity == aipubt::Quantization::Per_Tensor && num > 1)
        {
            DEBUG() << "please check the quantization scheme(=per_tensor), \
                    but quantization scales size > 1. ";
        }

        if (aipubt::Tensor::is_float(dtype))
        {
            get_fpx_params_from_statistic_value(t);
        }
        else
        {
            if (bits == -1)
            {
                if (dtype != aipubt::Tensor::VOID)
                {
                    bits = aipubt::Tensor::dtype_bits(dtype);
                    t->quantization().set_bits(bits);
                }
            }
            std::vector<float> scales;
            std::vector<zp_qdtype> zerops;
            get_quant_params_from_statistic_value(statistic_max,
                                                  statistic_min,
                                                  bits,
                                                  is_signed,
                                                  is_symmetric,
                                                  is_full_range,
                                                  scales,
                                                  zerops);
            t->quantization().set_scales(scales);
            t->quantization().set_offsets(zerops);
            t->quantization().set_quantized(true);
        }
    }

    // t->quantization().set_qinvariant(false);
}

void calculate_scale_zp(aipubt::NodePtr node, bool force_to_recalculate)
{
    // when one node of OpsAPI calls the qtlib, the input tensor of this node
    // may not quantize.
    for (auto it : node->inputs())
    {
        if (!it->quantization().quantized())
        {
            calculate_scale_zp(it);
        }
    }
    for (auto ot : node->outputs())
    {
        if (!ot->quantization().quantized())
        {
            calculate_scale_zp(ot);
        }
    }
    for (auto ct : node->constants())
    {
        if (!ct.second->quantization().quantized())
        {
            calculate_scale_zp(ct.second, force_to_recalculate);
        }
    }
    for (auto pt : node->placeholders())
    {
        if (!pt->quantization().quantized())
        {
            calculate_scale_zp(pt);
        }
    }
}

std::vector<aipubt::Tensor::Dtype> preferred_fp_dtype(aipubt::NodePtr node)
{
    static const std::map<std::string, std::vector<aipubt::Tensor::Dtype>> fp_dtype_table =
        {
            // mode, activation dtype, weight dtype
            {"DISABLE", {aipubt::Tensor::VOID, aipubt::Tensor::VOID}},
            {"AUTOMATIC", {aipubt::Tensor::FP16, aipubt::Tensor::FP16}},
            {"FLOAT16_PREFERRED", {aipubt::Tensor::FP16, aipubt::Tensor::FP16}},
            {"BFLOAT16_PREFERRED", {aipubt::Tensor::BFP16, aipubt::Tensor::FP16}},
            {"FLOAT32_PREFERRED", {aipubt::Tensor::FP32, aipubt::Tensor::FP32}},
            {"FLOAT16", {aipubt::Tensor::FP16, aipubt::Tensor::FP16}},
            {"FLOAT32", {aipubt::Tensor::FP32, aipubt::Tensor::FP32}},
            {"BFLOAT16", {aipubt::Tensor::BFP16, aipubt::Tensor::BFP16}},

            {"FP8E4M3FN", {aipubt::Tensor::FP8E4M3FN, aipubt::Tensor::FP8E4M3FN}},
            {"FP8E4M3FN_ACT_FP8E4M3FN_WHT", {aipubt::Tensor::FP8E4M3FN, aipubt::Tensor::FP8E4M3FN}},
            {"FLOAT8_E4M3FN", {aipubt::Tensor::FP8E4M3FN, aipubt::Tensor::FP8E4M3FN}},
            {"FLOAT16_ACT_FP8E4M3FN_WHT", {aipubt::Tensor::FP16, aipubt::Tensor::FP8E4M3FN}},

            {"FLOAT32_ACT_FP4E2M1FN_WHT", {aipubt::Tensor::FP32, aipubt::Tensor::FP4E2M1FN}},
            {"FLOAT16_ACT_FP4E2M1FN_WHT", {aipubt::Tensor::FP16, aipubt::Tensor::FP4E2M1FN}},
            {"FLOAT16_ACT_FLOAT4_E2M1FN_WHT", {aipubt::Tensor::FP16, aipubt::Tensor::FP4E2M1FN}},
            {"FLOAT16_ACT_INT4_WHT", {aipubt::Tensor::FP16, aipubt::Tensor::INT8}},
            {"FLOAT16_ACT_ALIGNED_INT4_WHT", {aipubt::Tensor::FP16, aipubt::Tensor::INT8}},
            {"FLOAT16_ACT_INT8_WHT", {aipubt::Tensor::FP16, aipubt::Tensor::INT8}},
            {"FLOAT16_ACT_INT_WHT", {aipubt::Tensor::FP16, aipubt::Tensor::INT8}},

            {"BFLOAT16_ACT_FP4E2M1FN_WHT", {aipubt::Tensor::BFP16, aipubt::Tensor::FP4E2M1FN}},
            {"BFLOAT16_ACT_FLOAT4_E2M1FN_WHT", {aipubt::Tensor::BFP16, aipubt::Tensor::FP4E2M1FN}},
            {"BFLOAT16_ACT_INT4_WHT", {aipubt::Tensor::BFP16, aipubt::Tensor::INT8}},
            {"BFLOAT16_ACT_ALIGNED_INT4_WHT", {aipubt::Tensor::BFP16, aipubt::Tensor::INT8}},
            {"BFLOAT16_ACT_INT8_WHT", {aipubt::Tensor::BFP16, aipubt::Tensor::INT8}},
            {"BFLOAT16_ACT_INT_WHT", {aipubt::Tensor::BFP16, aipubt::Tensor::INT8}},
        };

    if (node->attrs().count(Constants::trigger_float_op) != 0)
    {
        std::string trigger_mode = node->get_attr<aipubt::NodeParamValue>(Constants::trigger_float_op);
        std::transform(trigger_mode.begin(), trigger_mode.end(), trigger_mode.begin(), ::toupper);
        if (fp_dtype_table.count(trigger_mode))
        {
            return fp_dtype_table.at(trigger_mode);
        }
    }
    return {aipubt::Tensor::VOID, aipubt::Tensor::VOID};
}

void update_q_scheme(aipubt::TensorPtr t, aipubt::TensorPtr src_t)
{
    if (t->quantization().scheme() == aipubt::Quantization::Scheme_Void && !src_t->quantization().empty())
    {
        t->quantization().set_scheme(src_t->quantization().scheme());
    }
}
}; // namespace quant_utils

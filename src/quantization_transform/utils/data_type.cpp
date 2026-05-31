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
 * @Date: 2022-03-14 10:24:39
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-02 10:56:44
 */

#include "quantization_transform/utils/data_type.h"

namespace dtype_utils
{

static std::map<aipubt_dtype, std::pair<int32_t, uint32_t>> DTYPE_TO_RANGE = {
    {aipubt::Tensor::INT8, {-128, 127}},
    {aipubt::Tensor::UINT8, {0, 255}},
    {aipubt::Tensor::INT16, {-32768, 32767}},
    {aipubt::Tensor::UINT16, {0, 65535}},
    {aipubt::Tensor::INT32, {-2147483648, 2147483647}},
    {aipubt::Tensor::UINT32, {0, 4294967295}},
};

static std::set<aipubt_dtype> SIGNED_DTYPE = {
    aipubt::Tensor::INT8,
    aipubt::Tensor::INT16,
    aipubt::Tensor::INT32,
    aipubt::Tensor::INT64,
    aipubt::Tensor::ALIGNED_INT4,
    aipubt::Tensor::ALIGNED_INT12,
    aipubt::Tensor::FP16,
    aipubt::Tensor::FP32,
    aipubt::Tensor::FP64,
    aipubt::Tensor::BFP16,
};

static std::map<uint8_t, aipubt::Tensor::Dtype> BTYPES_TO_SIGNED_QDTYPE = {
    {1, aipubt::Tensor::INT8},
    {2, aipubt::Tensor::INT16},
    {4, aipubt::Tensor::INT32},
    {8, aipubt::Tensor::INT64},
};

static std::map<uint8_t, aipubt::Tensor::Dtype> BTYPES_TO_UNSIGNED_QDTYPE = {
    {1, aipubt::Tensor::UINT8},
    {2, aipubt::Tensor::UINT16},
    {4, aipubt::Tensor::UINT32},
    {8, aipubt::Tensor::UINT64},
};

const size_t dtype_size(const aipubt_dtype &dtype)
{
    return aipubt::Tensor::dtype_size(dtype);
}

static int32_t _int_dtype_to_range_min(const aipubt_dtype &dtype)
{
    return DTYPE_TO_RANGE[dtype].first;
}

static uint32_t _int_dtype_to_range_max(const aipubt_dtype &dtype)
{
    return DTYPE_TO_RANGE[dtype].second;
}

double dtype_to_range_min(const aipubt_dtype &dtype)
{
    if (aipubt::Tensor::is_float(dtype))
    {
        if (dtype == aipubt::Tensor::FP4E2M1FN)
        {
            return float(-6.0f);
        }
        else if (dtype == aipubt::Tensor::FP8E4M3FN)
        {
            return float(-448.0);
        }
        else if (dtype == aipubt::Tensor::FP16)
        {
            return float(-65504.0);
        }
        else
        {
            ERROR() << "please support this float dtype's range min: dtype = " << dtype;
        }
    }
    else
    {
        return _int_dtype_to_range_min(dtype);
    }
    return 1.0;
}

double dtype_to_range_max(const aipubt_dtype &dtype)
{
    if (aipubt::Tensor::is_float(dtype))
    {
        if (dtype == aipubt::Tensor::FP4E2M1FN)
        {
            return float(6.0f);
        }
        else if (dtype == aipubt::Tensor::FP8E4M3FN)
        {
            return float(448.0);
        }
        else if (dtype == aipubt::Tensor::FP16)
        {
            return float(65504.0);
        }
        else
        {
            ERROR() << "please support this float dtype's range max: dtype = " << dtype;
        }
    }
    else
    {
        return _int_dtype_to_range_max(dtype);
    }
    return 1.0;
}

int64_t bits_to_range_min(const int32_t bits, const bool is_signed)
{
    if (bits > 32)
    {
        DEBUG() << "this bits_to_range_min now inputs bits = " << bits;
    }
    if (is_signed)
    {
        return static_cast<int64_t>(-pow(2, bits - 1));
    }
    else
    {
        return static_cast<int64_t>(0);
    }
}

uint64_t bits_to_range_max(const int32_t bits, const bool is_signed)
{
    if (bits > 32)
    {
        DEBUG() << "this bits_to_range_min now inputs bits = " << bits;
    }
    if (is_signed)
    {
        return static_cast<uint64_t>(pow(2, bits - 1) - 1);
    }
    else
    {
        return static_cast<uint64_t>(pow(2, bits) - 1);
    }
}

aipubt_dtype bits_to_dtype(const int32_t bits, const bool is_signed)
{
    aipubt_dtype dtype;
    if (bits <= 8)
    {
        dtype = is_signed ? aipubt::Tensor::INT8 : aipubt::Tensor::UINT8;
    }
    else if (bits <= 16)
    {
        dtype = is_signed ? aipubt::Tensor::INT16 : aipubt::Tensor::UINT16;
    }
    else if (bits <= 32)
    {
        dtype = is_signed ? aipubt::Tensor::INT32 : aipubt::Tensor::UINT32;
    }
    else // 32 < bits <= 64
    {
        dtype = is_signed ? aipubt::Tensor::INT64 : aipubt::Tensor::UINT64;
    }
    return dtype;
}

int32_t dtype_to_bits(aipubt_dtype dtype)
{
    return aipubt::Tensor::dtype_bits(dtype);
}

aipubt::Tensor::Dtype bytes_to_signed_qdtype(uint8_t bytes)
{
    return BTYPES_TO_SIGNED_QDTYPE[bytes];
}

aipubt::Tensor::Dtype bytes_to_unsigned_qdtype(uint8_t bytes)
{
    return BTYPES_TO_UNSIGNED_QDTYPE[bytes];
}

void set_aipu_dtype(aipubt::TensorPtr tensor, bool is_signed, bool is_change_tensor_type)
{
    std::vector<aipubt::Tensor::Dtype> _need_setted_dtypes = {
        aipubt::Tensor::Dtype::VOID,
        aipubt::Tensor::Dtype::FP16,
        aipubt::Tensor::Dtype::FP32,
        aipubt::Tensor::Dtype::FP64,
    };

    // In Fasym mode, aipu_dtype should be same with primitive_dtype, and set by user side.
    aipubt::Tensor::Dtype dtype = tensor->quantization().aipu_dtype();
    if (contains(_need_setted_dtypes, dtype))
    {
        auto bits = tensor->quantization().bits();
        dtype = bits_to_dtype(bits, is_signed);
        tensor->quantization().set_aipu_dtype(dtype);
    }

    if (is_change_tensor_type)
    {
        tensor->set_type(dtype);
    }
}

uint32_t range_to_bits(const float vmin, const float vmax, const bool force_int)
{
    uint32_t qbits = 8;
    bool out_signed = true;
    if (vmin >= 0)
    {
        out_signed = false;
    }
    out_signed = (out_signed || force_int);
    if (out_signed)
    {
        if ((vmax > 127 && vmax <= 32767) || (vmin < -128 && vmin >= -32768))
        {
            qbits = 16;
        }
        else if (vmax > 32767 || vmin < -32768)
        {
            qbits = 32;
        }
    }
    else
    {
        if (vmax > 255 && vmax <= 65535)
        {
            qbits = 16;
        }
        else if (vmax > 65535)
        {

            qbits = 32;
        }
    }
    return qbits;
}

aipubt_dtype range_to_dtype(float vmin, float vmax, bool is_signed)
{
    auto bits = range_to_bits(vmin, vmax, false);
    auto dtype = bits_to_dtype(bits, is_signed);
    return dtype;
}

aipubt::ComputeTensor to_fp24(aipubt::ComputeTensor t)
{
    auto t_ori_dtype = ((aipubt::TensorPtr)t)->type();
    aipubt::ComputeTensor mask_ct = aipubt::ComputeTensor((int32_t)0xFFFFFF00);
    auto ct = t.view(aipubt::Tensor::INT32);
    ct = ct & mask_ct;
    return ct.view(t_ori_dtype);
}

}; // namespace dtype_utils
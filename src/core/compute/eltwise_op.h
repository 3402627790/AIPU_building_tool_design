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
 * @Date: 2022-11-09 11:30:17
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-08-30 15:49:45
 */
#ifndef CORE_COMPUTE_ELTWISE_OP_H
#define CORE_COMPUTE_ELTWISE_OP_H

#include "core/compute/multitypes.h"
#include "core/tensor.h"
#include <cmath>
namespace aipubt
{
template <typename T>
void eltwise_sqrt(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::sqrt(a[i]);
    }
}

template <typename T>
void eltwise_abs(const T *a, T *b, size_t dsize)
{
    // #pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::abs(a[i]);
    }
}

template <typename T>
void eltwise_sin(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::sin(a[i]);
    }
}

template <typename T>
void eltwise_cos(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::cos(a[i]);
    }
}

template <typename T>
void eltwise_tan(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::tan(a[i]);
    }
}

template <typename T>
void eltwise_sinh(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::sinh(a[i]);
    }
}

template <typename T>
void eltwise_cosh(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::cosh(a[i]);
    }
}

template <typename T>
void eltwise_tanh(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::tanh(a[i]);
    }
}

template <typename T>
void eltwise_log(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::log(a[i]);
    }
}
template <typename T>
void eltwise_log2(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::log2(a[i]);
    }
}

template <typename T>
void eltwise_log10(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::log10(a[i]);
    }
}

template <typename T>
void eltwise_exp(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::exp(a[i]);
    }
}

template <typename T>
void eltwise_celu(const T *a, T *b, size_t dsize, float alpha)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        float min_v = alpha * (std::exp(a[i] / alpha) - 1);
        b[i] = std::max(0.f, float(a[i])) + std::min(0.f, min_v);
    }
}

template <typename T>
void eltwise_elu(const T *a, T *b, size_t dsize, float alpha)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = a[i] > static_cast<T>(0.f) ? a[i] : static_cast<T>(alpha * (std::exp(a[i]) - 1));
    }
}

template <typename T>
void eltwise_sigmoid(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::exp(a[i]) / (std::exp(a[i]) + 1);
    }
}

template <typename T>
void eltwise_erf(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::erf(a[i]);
    }
}
template <typename T>
void eltwise_pow(const T *a, T *b, double c, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = std::pow(a[i], (T)c);
    }
}

template <typename T>
void eltwise_max(const T *a, const T *b, T *c, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        c[i] = std::max(a[i], b[i]);
    }
}

template <typename T>
void eltwise_min(const T *a, const T *b, T *c, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        c[i] = std::min(a[i], b[i]);
    }
}

template <typename T>
void eltwise_ceil(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = ceil(a[i]);
    }
}

template <typename T>
void eltwise_floor(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = floor(a[i]);
    }
}

template <typename T>
void eltwise_round(const T *a, T *b, size_t dsize)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = round(a[i]);
    }
}

template <typename T>
void eltwise_clamp(const T *a, T *b, size_t dsize, double min, double max)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        b[i] = a[i];
        if (static_cast<double>(a[i]) > max)
        {
            b[i] = max;
        }
        if (static_cast<double>(a[i]) < min)
        {
            b[i] = min;
        }
    }
}

#define OMP_PARA_INTERNAL _Pragma("omp parallel for")
#define ELTWISE_FUNCTION(name, func)                    \
    template <typename T>                               \
    void eltwise_##name(const T *a, T *b, size_t dsize) \
    {                                                   \
        OMP_PARA_INTERNAL                               \
        for (uint32_t i = 0; i < dsize; ++i)            \
        {                                               \
            b[i] = func(a[i]);                          \
        }                                               \
    }

ELTWISE_FUNCTION(atan, atan);
ELTWISE_FUNCTION(atanh, atanh);
ELTWISE_FUNCTION(asin, asin);
ELTWISE_FUNCTION(asinh, asinh);
ELTWISE_FUNCTION(acos, acos);
ELTWISE_FUNCTION(acosh, acosh);

MULTITYPES_COMPUTE_OP_FLOAT(eltwise_atan);
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_atanh);
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_asin);
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_asinh);
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_acos);
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_acosh);

template <typename T>
void eltwise_op_unary(const T *a, T *b, size_t dsize, std::function<void(uint8_t *, uint8_t *)> func)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        func((uint8_t *)&a[i], (uint8_t *)&b[i]);
    }
}

template <typename T>
void eltwise_op_binary(const T *a, const T *b, T *c, size_t dsize, std::function<void(uint8_t *, uint8_t *, uint8_t *)> func)
{
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        func((uint8_t *)&a[i], (uint8_t *)&b[i], (uint8_t *)&c[i]);
    }
}

MULTITYPES_COMPUTE_OP_FLOAT(eltwise_sqrt)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_sin)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_cos)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_tan)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_sinh)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_cosh)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_exp)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_log)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_log2)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_log10)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_pow)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_tanh)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_erf)
MULTITYPES_COMPUTE_OP_FLOAT(eltwise_sigmoid)

MULTITYPES_COMPUTE_OP(eltwise_clamp)
MULTITYPES_COMPUTE_OP(eltwise_max)
MULTITYPES_COMPUTE_OP(eltwise_min)
MULTITYPES_COMPUTE_OP(eltwise_ceil)
MULTITYPES_COMPUTE_OP(eltwise_celu)
MULTITYPES_COMPUTE_OP(eltwise_elu)
MULTITYPES_COMPUTE_OP(eltwise_floor)
MULTITYPES_COMPUTE_OP(eltwise_round)
MULTITYPES_COMPUTE_OP(eltwise_op_unary)
MULTITYPES_COMPUTE_OP(eltwise_op_binary)

MULTITYPES_COMPUTE_OP_SINGED(eltwise_abs)

} // namespace aipubt

#endif //!CORE_COMPUTE_ELTWISE_OP_H
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
 * @Date: 2020-02-09 12:16:08
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-01-07 14:26:33
 */
#ifndef CORE_COMPUTE_BROADCAST_OP_H
#define CORE_COMPUTE_BROADCAST_OP_H

#include "core/compute/multitypes.h"
#include "core/tensor.h"
#include <vector>

namespace aipubt
{

// c=a op b
inline static void broadcast_shape_to_stride(
    const std::vector<int> &a_shape,
    const std::vector<int> &b_shape,
    const std::vector<int> &c_shape,
    std::vector<uint32_t> &a_stride,
    std::vector<uint32_t> &b_stride,
    std::vector<uint32_t> &c_stride)
{
    a_stride.resize(c_shape.size());
    b_stride.resize(c_shape.size());
    c_stride.resize(c_shape.size());
    uint32_t a = 1, b = 1, c = 1;
    for (uint32_t rid = 0; rid < c_shape.size(); rid++)
    {
        // # 2,3,3,4
        // cstride=[36,12,4,1]
        // # 2,1,3,1
        // astride=[3,0,1,0]
        // # 3,1,4
        // bstride=[0,4,0,1]
        int32_t i = c_shape.size() - rid - 1;
        c_stride[i] = c;
        if (a_shape.size() == 0 || a_shape.size() - 1 < rid || a_shape[a_shape.size() - rid - 1] == 1)
        {
            a_stride[i] = 0;
        }
        else
        {
            a_stride[i] = a;
            if (a_shape.size() > rid)
            {
                a *= a_shape[a_shape.size() - rid - 1];
            }
        }
        if (b_shape.size() == 0 || b_shape.size() - 1 < rid || b_shape[b_shape.size() - rid - 1] == 1)
        {
            b_stride[i] = 0;
        }
        else
        {
            b_stride[i] = b;
            if (b_shape.size() > rid)
            {
                b *= b_shape[b_shape.size() - rid - 1];
            }
        }
        c *= c_shape[i];
    }
}

template <typename T>
void broadcast_add(const T *a, const T *b, T *c, size_t dsize,
                   const std::vector<int> &a_shape,
                   const std::vector<int> &b_shape,
                   const std::vector<int> &c_shape)
{
    std::vector<uint32_t> a_stride;
    std::vector<uint32_t> b_stride;
    std::vector<uint32_t> c_stride;
    broadcast_shape_to_stride(a_shape, b_shape, c_shape, a_stride, b_stride, c_stride);
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        auto idx = offset_to_index(i, c_stride);
        c[i] = a[index_to_offset(idx, a_stride)] + b[index_to_offset(idx, b_stride)];
    }
}

MULTITYPES_COMPUTE_OP(broadcast_add);

template <typename T>
void broadcast_mul(const T *a, const T *b, T *c, size_t dsize,
                   const std::vector<int> &a_shape,
                   const std::vector<int> &b_shape,
                   const std::vector<int> &c_shape)
{
    std::vector<uint32_t> a_stride;
    std::vector<uint32_t> b_stride;
    std::vector<uint32_t> c_stride;
    broadcast_shape_to_stride(a_shape, b_shape, c_shape, a_stride, b_stride, c_stride);
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        auto idx = offset_to_index(i, c_stride);
        auto aof = index_to_offset(idx, a_stride);
        auto bof = index_to_offset(idx, b_stride);
        c[i] = a[aof] * b[bof];
    }
}
MULTITYPES_COMPUTE_OP(broadcast_mul);

template <typename T>
void broadcast_sub(const T *a, const T *b, T *c, size_t dsize,
                   const std::vector<int> &a_shape,
                   const std::vector<int> &b_shape,
                   const std::vector<int> &c_shape)
{
    std::vector<uint32_t> a_stride;
    std::vector<uint32_t> b_stride;
    std::vector<uint32_t> c_stride;
    broadcast_shape_to_stride(a_shape, b_shape, c_shape, a_stride, b_stride, c_stride);
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        auto idx = offset_to_index(i, c_stride);
        c[i] = a[index_to_offset(idx, a_stride)] - b[index_to_offset(idx, b_stride)];
    }
}
MULTITYPES_COMPUTE_OP(broadcast_sub);

template <typename T>
void broadcast_div(const T *a, const T *b, T *c, size_t dsize,
                   const std::vector<int> &a_shape,
                   const std::vector<int> &b_shape,
                   const std::vector<int> &c_shape)
{
    std::vector<uint32_t> a_stride;
    std::vector<uint32_t> b_stride;
    std::vector<uint32_t> c_stride;
    broadcast_shape_to_stride(a_shape, b_shape, c_shape, a_stride, b_stride, c_stride);
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        auto idx = offset_to_index(i, c_stride);
        c[i] = a[index_to_offset(idx, a_stride)] / b[index_to_offset(idx, b_stride)];
    }
}
MULTITYPES_COMPUTE_OP(broadcast_div);

template <typename T>
void broadcast_bit_and(const T *a, const T *b, T *c, size_t dsize,
                       const std::vector<int> &a_shape,
                       const std::vector<int> &b_shape,
                       const std::vector<int> &c_shape)
{
    std::vector<uint32_t> a_stride;
    std::vector<uint32_t> b_stride;
    std::vector<uint32_t> c_stride;
    broadcast_shape_to_stride(a_shape, b_shape, c_shape, a_stride, b_stride, c_stride);
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        auto idx = offset_to_index(i, c_stride);
        c[i] = a[index_to_offset(idx, a_stride)] & b[index_to_offset(idx, b_stride)];
    }
}
MULTITYPES_COMPUTE_OP_INT(broadcast_bit_and);

template <typename T>
void broadcast_bit_or(const T *a, const T *b, T *c, size_t dsize,
                      const std::vector<int> &a_shape,
                      const std::vector<int> &b_shape,
                      const std::vector<int> &c_shape)
{
    std::vector<uint32_t> a_stride;
    std::vector<uint32_t> b_stride;
    std::vector<uint32_t> c_stride;
    broadcast_shape_to_stride(a_shape, b_shape, c_shape, a_stride, b_stride, c_stride);
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        auto idx = offset_to_index(i, c_stride);
        c[i] = a[index_to_offset(idx, a_stride)] | b[index_to_offset(idx, b_stride)];
    }
}
MULTITYPES_COMPUTE_OP_INT(broadcast_bit_or);

template <typename T>
void unique(const T *a, const size_t size, T *out, uint32_t &uniqued_size)
{
    std::vector<T> v(a, a + size);
    std::sort(v.begin(), v.end());
    auto ip = std::unique(v.begin(), v.end());
    uniqued_size = std::distance(v.begin(), ip);
    std::copy(v.begin(), v.begin() + uniqued_size, out);
}
MULTITYPES_COMPUTE_OP(unique);

#define OMP_PARA_INTERNAL _Pragma("omp parallel for")
#define BROADCAST_FUNCTION(name, func)                                                         \
    template <typename T>                                                                      \
    void broadcast_##name(const T *a, const T *b, T *c, size_t dsize,                          \
                          const std::vector<int> &a_shape,                                     \
                          const std::vector<int> &b_shape,                                     \
                          const std::vector<int> &c_shape)                                     \
    {                                                                                          \
        std::vector<uint32_t> a_stride;                                                        \
        std::vector<uint32_t> b_stride;                                                        \
        std::vector<uint32_t> c_stride;                                                        \
        broadcast_shape_to_stride(a_shape, b_shape, c_shape, a_stride, b_stride, c_stride);    \
                                                                                               \
        OMP_PARA_INTERNAL                                                                      \
        for (uint32_t i = 0; i < dsize; ++i)                                                   \
        {                                                                                      \
            auto idx = offset_to_index(i, c_stride);                                           \
            c[i] = func(a[index_to_offset(idx, a_stride)], b[index_to_offset(idx, b_stride)]); \
        }                                                                                      \
    }

BROADCAST_FUNCTION(max, std::max);
BROADCAST_FUNCTION(min, std::min);
BROADCAST_FUNCTION(pow, std::pow);

MULTITYPES_COMPUTE_OP(broadcast_pow);
MULTITYPES_COMPUTE_OP(broadcast_max);
MULTITYPES_COMPUTE_OP(broadcast_min);

#define OMP_PARA_INTERNAL _Pragma("omp parallel for")
#define BROADCAST_FUNCTION_OPERATION(name, func)                                                       \
    template <typename T>                                                                              \
    void broadcast_##name(const T *a, const T *b, T *c, size_t dsize,                                  \
                          const std::vector<int> &a_shape,                                             \
                          const std::vector<int> &b_shape,                                             \
                          const std::vector<int> &c_shape)                                             \
    {                                                                                                  \
        std::vector<uint32_t> a_stride;                                                                \
        std::vector<uint32_t> b_stride;                                                                \
        std::vector<uint32_t> c_stride;                                                                \
        broadcast_shape_to_stride(a_shape, b_shape, c_shape, a_stride, b_stride, c_stride);            \
                                                                                                       \
        OMP_PARA_INTERNAL                                                                              \
        for (uint32_t i = 0; i < dsize; ++i)                                                           \
        {                                                                                              \
            auto idx = offset_to_index(i, c_stride);                                                   \
            c[i] = (a[index_to_offset(idx, a_stride)])func(b[index_to_offset(idx, b_stride)]) ? 1 : 0; \
        }                                                                                              \
    }

BROADCAST_FUNCTION_OPERATION(lt, <);
BROADCAST_FUNCTION_OPERATION(gt, >);

MULTITYPES_COMPUTE_OP(broadcast_lt);
MULTITYPES_COMPUTE_OP(broadcast_gt);

};     // namespace aipubt
#endif //!CORE_COMPUTE_BROADCAST_OP_H

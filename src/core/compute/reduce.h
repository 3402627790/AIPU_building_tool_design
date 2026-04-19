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
 * @Last Modified time: 2023-05-16 18:20:22
 */
#ifndef CORE_COMPUTE_REDUCE_H
#define CORE_COMPUTE_REDUCE_H

#include "core/compute/multitypes.h"
#include "core/tensor.h"
namespace aipubt
{
template <typename T>
void reduce_sum(T *input, T *output, size_t head, size_t tail, size_t length)
{

#pragma omp parallel for
    for (uint32_t i = 0; i < head; ++i)
    {
        for (uint32_t j = 0; j < tail; ++j)
        {
            T *ptr = input + i * tail * length + j;
            T s(0);
            for (uint32_t k = 0; k < length; ++k)
            {
                s += *ptr;
                ptr += tail;
            }
            output[i * tail + j] = s;
        }
    }
}

template <typename T>
void reduce_prod(T *input, T *output, size_t head, size_t tail, size_t length)
{

#pragma omp parallel for
    for (uint32_t i = 0; i < head; ++i)
    {
        for (uint32_t j = 0; j < tail; ++j)
        {
            T *ptr = input + i * tail * length + j;
            T s(1.);
            for (uint32_t k = 0; k < length; ++k)
            {
                s *= *ptr;
                ptr += tail;
            }
            output[i * tail + j] = s;
        }
    }
}

template <typename T>
void reduce_mean(T *input, T *output, size_t head, size_t tail, size_t length)
{

#pragma omp parallel for
    for (uint32_t i = 0; i < head; ++i)
    {
        for (uint32_t j = 0; j < tail; ++j)
        {
            T *ptr = input + i * tail * length + j;
            T s(0);
            for (uint32_t k = 0; k < length; ++k)
            {
                s += *ptr;
                ptr += tail;
            }
            output[i * tail + j] = s / length;
        }
    }
}

template <typename T>
void reduce_min(T *input, T *output, size_t head, size_t tail, size_t length)
{
    T s = std::numeric_limits<T>::max();
#pragma omp parallel for
    for (uint32_t i = 0; i < head; ++i)
    {
        for (uint32_t j = 0; j < tail; ++j)
        {
            T *ptr = input + i * tail * length + j;
            for (uint32_t k = 0; k < length; ++k)
            {
                s = std::min(s, *ptr);
                ptr += tail;
            }
            output[i * tail + j] = s;
        }
    }
}

template <typename T>
void reduce_max(T *input, T *output, size_t head, size_t tail, size_t length)
{
    T s = std::numeric_limits<T>::min();
#pragma omp parallel for
    for (uint32_t i = 0; i < head; ++i)
    {
        for (uint32_t j = 0; j < tail; ++j)
        {
            T *ptr = input + i * tail * length + j;
            for (uint32_t k = 0; k < length; ++k)
            {
                s = std::max(s, *ptr);
                ptr += tail;
            }
            output[i * tail + j] = s;
        }
    }
}

MULTITYPES_COMPUTE_OP(reduce_sum)
MULTITYPES_COMPUTE_OP(reduce_prod)
MULTITYPES_COMPUTE_OP(reduce_mean)
MULTITYPES_COMPUTE_OP(reduce_min)
MULTITYPES_COMPUTE_OP(reduce_max)

};     // namespace aipubt
#endif //!CORE_COMPUTE_REDUCE_H

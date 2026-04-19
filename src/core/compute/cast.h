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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-11-09 11:42:11
 */
#ifndef CORE_COMPUTE_CAST_H
#define CORE_COMPUTE_CAST_H

#include "core/compute/multitypes.h"
#include "core/tensor.h"
#include <cmath>
namespace aipubt
{
#define MULTITYPES_COMPUTE_ITEM_T0_T1(t, func) \
    {t, reinterpret_cast<decltype(&func<T, int8_t>)>(&func<T, tensor_data_type<t>::type>)},

namespace internal
{
    template <typename T, typename T1>
    void cast_from_t_to_t1(const T *a, T1 *b, size_t dsize)
    {
#pragma omp parallel for
        for (uint32_t i = 0; i < dsize; ++i)
        {
            b[i] = a[i];
        }
    }
    template <typename T>
    void cast_to_t1(Tensor::Dtype t, const T *a, int8_t *b, size_t dsize)
    {
        static std::map<Tensor::Dtype, decltype(&cast_from_t_to_t1<T, int8_t>)>
            my_map = {_TENSOR_TYPE_MAP_INIT(MULTITYPES_COMPUTE_ITEM_T0_T1, cast_from_t_to_t1)};
        if (my_map.count(t))
        {
            my_map.at(t)(a, b, dsize);
        }
    }
    MULTITYPES_COMPUTE_OP(cast_to_t1);

} // namespace internal
template <typename... T>
bool cast_all(T &&... arg)
{
    return internal::cast_to_t1_all(std::forward<T>(arg)...);
}

} // namespace aipubt

#endif //!CORE_COMPUTE_CAST_H
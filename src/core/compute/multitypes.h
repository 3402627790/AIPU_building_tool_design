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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2025-10-17 17:40:08
 */
#ifndef CORE_COMPUTE_MULTITYPES_H
#define CORE_COMPUTE_MULTITYPES_H

#include "core/fp16.h"
#include "core/tensor.h"
namespace aipubt
{

#define _TENSOR_TYPE_MAP_INIT_INT(Op, ...)                                                \
    Op(Tensor::Dtype::UINT8, __VA_ARGS__)                                                 \
        Op(Tensor::Dtype::INT8, __VA_ARGS__)                                              \
            Op(Tensor::Dtype::UINT16, __VA_ARGS__)                                        \
                Op(Tensor::Dtype::INT16, __VA_ARGS__)                                     \
                    Op(Tensor::Dtype::UINT32, __VA_ARGS__)                                \
                        Op(Tensor::Dtype::INT32, __VA_ARGS__)                             \
                            Op(Tensor::Dtype::UINT64, __VA_ARGS__)                        \
                                Op(Tensor::Dtype::INT64, __VA_ARGS__)                     \
                                    Op(Tensor::Dtype::ALIGNED_INT4, __VA_ARGS__)          \
                                        Op(Tensor::Dtype::ALIGNED_UINT4, __VA_ARGS__)     \
                                            Op(Tensor::Dtype::ALIGNED_INT12, __VA_ARGS__) \
                                                Op(Tensor::Dtype::ALIGNED_UINT12, __VA_ARGS__)

#define _TENSOR_TYPE_MAP_INIT_FP(Op, ...)                       \
    Op(Tensor::Dtype::FP32, __VA_ARGS__)                        \
        Op(Tensor::Dtype::FP64, __VA_ARGS__)                    \
            Op(Tensor::Dtype::FP16, __VA_ARGS__)                \
                Op(Tensor::Dtype::BFP16, __VA_ARGS__)           \
                    Op(Tensor::Dtype::FP8E4M3FN, __VA_ARGS__)   \
                        Op(Tensor::Dtype::FP8E5M2, __VA_ARGS__) \
                            Op(Tensor::Dtype::FP4E2M1FN, __VA_ARGS__)

#define _TENSOR_TYPE_MAP_INIT(Op, ...)         \
    _TENSOR_TYPE_MAP_INIT_INT(Op, __VA_ARGS__) \
    _TENSOR_TYPE_MAP_INIT_FP(Op, __VA_ARGS__)

#define _TENSOR_TYPE_MAP_SINGED(Op, ...)              \
    Op(Tensor::Dtype::INT8, __VA_ARGS__)              \
        Op(Tensor::Dtype::INT16, __VA_ARGS__)         \
            Op(Tensor::Dtype::INT32, __VA_ARGS__)     \
                Op(Tensor::Dtype::INT64, __VA_ARGS__) \
                    _TENSOR_TYPE_MAP_INIT_FP(Op, __VA_ARGS__)

#define MULTITYPES_COMPUTE_ITEM(t, func) \
    {t, reinterpret_cast<decltype(&func<int8_t>)>(&func<tensor_data_type<t>::type>)},

#define MULTITYPES_COMPUTE_OP(func)                                             \
    template <typename... T>                                                    \
    bool func##_all(Tensor::Dtype t, T &&... arg)                               \
    {                                                                           \
        static std::map<Tensor::Dtype, decltype(&func<int8_t>)>                 \
            my_map = {_TENSOR_TYPE_MAP_INIT(MULTITYPES_COMPUTE_ITEM, func)};    \
        if (my_map.count(t))                                                    \
        {                                                                       \
            my_map.at(t)(std::forward<T>(arg)...);                              \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            WARN() << "Unsupport Tensor Type:" << t << " for the op." << #func; \
            return false;                                                       \
        }                                                                       \
        return true;                                                            \
    }

#define MULTITYPES_COMPUTE_OP_FLOAT(func)                                       \
    template <typename... T>                                                    \
    bool func##_all(Tensor::Dtype t, T &&... arg)                               \
    {                                                                           \
        static std::map<Tensor::Dtype, decltype(&func<int8_t>)>                 \
            my_map = {_TENSOR_TYPE_MAP_INIT_FP(MULTITYPES_COMPUTE_ITEM, func)}; \
        if (my_map.count(t))                                                    \
        {                                                                       \
            my_map.at(t)(std::forward<T>(arg)...);                              \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            WARN() << "Unsupport Tensor Type:" << t << " for the op." << #func; \
            return false;                                                       \
        }                                                                       \
        return true;                                                            \
    }
#define MULTITYPES_COMPUTE_OP_INT(func)                                          \
    template <typename... T>                                                     \
    bool func##_all(Tensor::Dtype t, T &&... arg)                                \
    {                                                                            \
        static std::map<Tensor::Dtype, decltype(&func<int8_t>)>                  \
            my_map = {_TENSOR_TYPE_MAP_INIT_INT(MULTITYPES_COMPUTE_ITEM, func)}; \
        if (my_map.count(t))                                                     \
        {                                                                        \
            my_map.at(t)(std::forward<T>(arg)...);                               \
        }                                                                        \
        else                                                                     \
        {                                                                        \
            WARN() << "Unsupport Tensor Type:" << t << " for the op." << #func;  \
            return false;                                                        \
        }                                                                        \
        return true;                                                             \
    }

#define MULTITYPES_COMPUTE_OP_SINGED(func)                                      \
    template <typename... T>                                                    \
    bool func##_all(Tensor::Dtype t, T &&... arg)                               \
    {                                                                           \
        static std::map<Tensor::Dtype, decltype(&func<int8_t>)>                 \
            my_map = {_TENSOR_TYPE_MAP_SINGED(MULTITYPES_COMPUTE_ITEM, func)};  \
        if (my_map.count(t))                                                    \
        {                                                                       \
            my_map.at(t)(std::forward<T>(arg)...);                              \
        }                                                                       \
        else                                                                    \
        {                                                                       \
            WARN() << "Unsupport Tensor Type:" << t << " for the op." << #func; \
            return false;                                                       \
        }                                                                       \
        return true;                                                            \
    }
};     // namespace aipubt
#endif //! CORE_COMPUTE_MULTITYPES_H

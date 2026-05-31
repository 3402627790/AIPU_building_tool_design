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
 * @Date: 2022-03-18 13:27:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-11-01 23:45:14
 */

#include "quantization_transform/utils/permute_utils.h"

namespace permute_utils
{

std::vector<int> get_transpose_perm(const int32_t &axes_num, const int32_t &dim)
{
    int32_t perm_dim = dim < 0 ? axes_num + dim : dim;

    std::vector<int> perms;
    perms.push_back(perm_dim);

    for (int32_t i = 0; i < dim; i++)
    {
        perms.push_back(i);
    }

    for (int32_t i = dim + 1; i < axes_num; i++)
    {
        perms.push_back(i);
    }

    return perms;
}

template <typename T>
static aipubt::TensorPtr _permute(aipubt::TensorPtr t, const std::vector<int> &perms)
{
    auto t_shape = t->shape();
    auto t_dtype = t->type();
    aipubt::TensorPtr permuted_tensor = std::make_shared<aipubt::Tensor>(t->name() + "qtlibn_permutes", t_shape, t_dtype);
    int dim = t->shape().dim();
    if ((uint32_t)dim != perms.size())
    {
        ERROR() << "please check permute perms.";
        return nullptr;
    }

    uint32_t data_size = t->mem_size() / sizeof(T);
    T *constant_data = t->data<T>();
    T *permute_data = new T[data_size];

    std::vector<int> dim_stride;
    for (int idx = 0; idx < dim; idx++)
    {
        int s = 1;
        for (int o = idx + 1; o < dim; o++)
        {
            s *= t->shape()[o];
        }
        dim_stride.push_back(s);
    }

    int idx = 0;
    if (dim == 2)
    {
        for (int c = 0; c < t->shape()[perms[0]]; c++)
        {
            for (int h = 0; h < t->shape()[perms[1]]; h++)
            {
                int os = h * dim_stride[perms[1]] +
                         c * dim_stride[perms[0]];
                permute_data[idx++] = constant_data[os];
            }
        }
    }
    else if (dim == 3)
    {
        for (int c = 0; c < t->shape()[perms[0]]; c++)
        {
            for (int h = 0; h < t->shape()[perms[1]]; h++)
            {
                for (int w = 0; w < t->shape()[perms[2]]; w++)
                {
                    int os = w * dim_stride[perms[2]] +
                             h * dim_stride[perms[1]] +
                             c * dim_stride[perms[0]];
                    permute_data[idx++] = constant_data[os];
                }
            }
        }
    }
    else if (dim == 4)
    {
        for (int n = 0; n < t->shape()[perms[0]]; n++)
        {
            for (int c = 0; c < t->shape()[perms[1]]; c++)
            {
                for (int h = 0; h < t->shape()[perms[2]]; h++)
                {
                    for (int w = 0; w < t->shape()[perms[3]]; w++)
                    {
                        int os = w * dim_stride[perms[3]] +
                                 h * dim_stride[perms[2]] +
                                 c * dim_stride[perms[1]] +
                                 n * dim_stride[perms[0]];
                        permute_data[idx++] = constant_data[os];
                    }
                }
            }
        }
    }
    else
    {
        INFO() << "dim = " << dim << " has not implemented.";
        return t;
    }

    permuted_tensor->set_data(permute_data, data_size * sizeof(T));
    std::vector<int> perm_ts;
    for (auto ts : t->shape())
    {
        perm_ts.push_back(ts);
    }
    for (int i = 0; i < dim; i++)
    {
        permuted_tensor->shape()[i] = perm_ts[perms[i]];
    }
    aipubt::NodeParamValue ps;
    for (uint32_t i = 0; i < perms.size(); i++)
    {
        ps.push_back(perms[i]);
    }
    permuted_tensor->set_attr(Constants::adjust_quantization_dim, ps);
    return permuted_tensor;
}

static std::map<aipubt::Tensor::Dtype, std::function<aipubt::TensorPtr(aipubt::TensorPtr, std::vector<int>)>> PERMUTE_MAP = {
    {aipubt::Tensor::INT8, _permute<int8_t>},
    {aipubt::Tensor::UINT8, _permute<uint8_t>},
    {aipubt::Tensor::INT16, _permute<int16_t>},
    {aipubt::Tensor::UINT16, _permute<uint16_t>},
    {aipubt::Tensor::INT32, _permute<int32_t>},
    {aipubt::Tensor::UINT32, _permute<uint32_t>},
    {aipubt::Tensor::INT64, _permute<int64_t>},
    {aipubt::Tensor::UINT64, _permute<uint64_t>},
    {aipubt::Tensor::FP32, _permute<float>},
    {aipubt::Tensor::FP64, _permute<double>},
};

static aipubt::TensorPtr permute(aipubt::TensorPtr t, const std::vector<int> &perms)
{
    auto dtype = t->type();
    return PERMUTE_MAP[dtype](t, perms);
}

aipubt::TensorPtr permute(aipubt::TensorPtr t)
{
    if (t->quantization().quantization_dim() == 0)
    {
        DEBUG() << t->name() << "'s quantization_dim == 0, so no need to permute the tensor data.";
        return t;
    }

    auto qdim = t->quantization().quantization_dim();
    auto dim_size = t->shape().dim();
    auto perms = get_transpose_perm(dim_size, qdim);
    auto permuted_tensor = permute(t, perms);
    return permuted_tensor;
}

static std::vector<int> get_transpose_perm(std::vector<int> target_axes)
{
    uint32_t size = target_axes.size();
    std::vector<int> origin_axes(size, 0);
    for (uint32_t i = 0; i < size; i++)
    {
        origin_axes[target_axes[i]] = i;
    }
    return origin_axes;
}

template <typename T>
static void _unpermute(aipubt::TensorPtr t, std::vector<int> perms)
{
    int dim = t->shape().dim();
    if ((uint32_t)dim != perms.size())
    {
        ERROR() << "please check target axes.";
        return;
    }

    // auto t_shape = t->shape();
    // auto t_dtype = t->type();
    // aipubt::TensorPtr unpermuted_tensor = std::make_shared<aipubt::Tensor>(t->name() + "qtlibn_unpermutes", t_shape, t_dtype);
    uint32_t data_size = t->mem_size() / sizeof(T);
    T *weight_data = t->data<T>();
    T *permute_data = new T[data_size];

    std::vector<int> dim_stride;
    for (int idx = 0; idx < dim; idx++)
    {
        int s = 1;
        for (int o = idx + 1; o < dim; o++)
        {
            s *= t->shape()[o];
        }
        dim_stride.push_back(s);
    }

    int idx = 0;
    if (dim == 2)
    {
        for (int c = 0; c < t->shape()[perms[0]]; c++)
        {
            for (int h = 0; h < t->shape()[perms[1]]; h++)
            {
                int os = h * dim_stride[perms[1]] +
                         c * dim_stride[perms[0]];
                permute_data[idx++] = weight_data[os];
            }
        }
    }
    else if (dim == 3)
    {
        for (int c = 0; c < t->shape()[perms[0]]; c++)
        {
            for (int h = 0; h < t->shape()[perms[1]]; h++)
            {
                for (int w = 0; w < t->shape()[perms[2]]; w++)
                {
                    int os = w * dim_stride[perms[2]] +
                             h * dim_stride[perms[1]] +
                             c * dim_stride[perms[0]];
                    permute_data[idx++] = weight_data[os];
                }
            }
        }
    }
    else if (dim == 4)
    {
        for (int n = 0; n < t->shape()[perms[0]]; n++)
        {
            for (int c = 0; c < t->shape()[perms[1]]; c++)
            {
                for (int h = 0; h < t->shape()[perms[2]]; h++)
                {
                    for (int w = 0; w < t->shape()[perms[3]]; w++)
                    {
                        int os = w * dim_stride[perms[3]] +
                                 h * dim_stride[perms[2]] +
                                 c * dim_stride[perms[1]] +
                                 n * dim_stride[perms[0]];
                        permute_data[idx++] = weight_data[os];
                    }
                }
            }
        }
    }
    else
    {
        INFO() << "dim = " << dim << " has not implemented.";
        return;
    }

    t->set_data(permute_data, data_size * sizeof(T));
    std::vector<int> perm_ts;
    for (auto ts : t->shape())
    {
        perm_ts.push_back(ts);
    }
    for (int i = 0; i < dim; i++)
    {
        t->shape()[i] = perm_ts[perms[i]];
    }
}

static std::map<aipubt::Tensor::Dtype, std::function<void(aipubt::TensorPtr, std::vector<int>)>> UNPERMUTE_MAP = {
    {aipubt::Tensor::INT8, _unpermute<int8_t>},
    {aipubt::Tensor::UINT8, _unpermute<uint8_t>},
    {aipubt::Tensor::INT16, _unpermute<int16_t>},
    {aipubt::Tensor::UINT16, _unpermute<uint16_t>},
    {aipubt::Tensor::INT32, _unpermute<int32_t>},
    {aipubt::Tensor::UINT32, _unpermute<uint32_t>},
    {aipubt::Tensor::INT64, _unpermute<int64_t>},
    {aipubt::Tensor::UINT64, _unpermute<uint64_t>},
    {aipubt::Tensor::FP32, _unpermute<float>},
    {aipubt::Tensor::FP64, _unpermute<double>},
};

static void unpermute(aipubt::TensorPtr t, const std::vector<int> &perms)
{
    auto dtype = t->type();
    UNPERMUTE_MAP[dtype](t, perms);
}

void unpermute(aipubt::TensorPtr t)
{
    if (t->attrs().count(Constants::adjust_quantization_dim))
    {
        aipubt::NodeParamValue tmp = t->get_attr(Constants::adjust_quantization_dim);
        std::vector<int> perms;
        for (uint32_t i = 0; i < tmp.arr.size(); i++)
        {
            perms.push_back(tmp.arr[i]);
        }
        auto original_perms = get_transpose_perm(perms);
        unpermute(t, original_perms);
    }
}

}; // namespace permute_utils

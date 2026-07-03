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
 * @Author: Alex Zheng
 * @Date: 2026-04-14 14:24:30
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2026-04-14 14:15:49
 */

#pragma once

#include <gcompiler_api.h>
#include <string>
namespace cpsruntime
{

using DeleteFunc = void (*)(void *);
using CreateFunc = void *(*)();

// tensor information
struct CrtTensorInfo
{
public:
    CrtTensorInfo() {}
    CrtTensorInfo(std::string name, aipubt::Tensor::Dtype tensor_type, std::vector<int32_t> dims) : name_(name), tensor_type_(tensor_type), dims_(dims) {}
    ~CrtTensorInfo() = default;

    std::string &name()
    {
        return name_;
    }
    aipubt::Tensor::Dtype &tensor_type()
    {
        return tensor_type_;
    }
    std::vector<int32_t> &get_dims()
    {
        return dims_;
    }

private:
    std::string name_;
    aipubt::Tensor::Dtype tensor_type_;
    std::vector<int32_t> dims_;
};

// input/output data value
struct CrtValue
{
public:
    CrtValue() : data_(nullptr) {}
    ~CrtValue() = default;

    CrtValue(void *pData, size_t size, CrtTensorInfo tensor_info, cpsruntime::DeleteFunc deleter)
    {
        Init(pData, size, tensor_info, deleter);
    }

    CrtValue(void *pData, size_t size, cpsruntime::DeleteFunc deleter)
    {
        Init(pData, size, deleter);
    }

    void Init(void *pData, size_t size, CrtTensorInfo tensor_info, cpsruntime::DeleteFunc deleter)
    {
        raw_data_ = pData;
        if (deleter != nullptr)
            data_.reset(pData, deleter);
        size_ = size;
        tensor_info_ = tensor_info;
    }

    void Init(void *pData, size_t size, CrtTensorInfo tensor_info, const std::function<void(void *)> &deleter)
    {
        raw_data_ = pData;
        if (deleter != nullptr)
            data_.reset(pData, deleter);
        size_ = size;
        tensor_info_ = tensor_info;
    }

    void Init(void *pData, size_t size, cpsruntime::DeleteFunc deleter)
    {
        raw_data_ = pData;
        if (deleter != nullptr)
            data_.reset(pData, deleter);
        size_ = size;
    }

    bool IsAllocated() const
    {
        return data_ != nullptr;
    }

    template <typename T>
    T *get() const
    {
        return static_cast<T *>(raw_data_);
    }

    size_t &memsize()
    {
        return size_;
    }

    size_t size()
    {
        auto dims = tensor_info_.get_dims();
        if (dims.size() == 0)
            return 0;
        size_t s = 1;
        for (auto d : dims)
        {
            if (d == -1)
            {
                return 0;
            }
            else
            {
                s *= d;
            }
        }
        return s;
    }
    struct CrtTensorInfo &tensor_info()
    {
        return tensor_info_;
    }
    std::string &name()
    {
        return tensor_info_.name();
    }
    aipubt::Tensor::Dtype &tensor_type()
    {
        return tensor_info_.tensor_type();
    }
    std::vector<int32_t> &dims()
    {
        return tensor_info_.get_dims();
    }

private:
    std::shared_ptr<void> data_;
    void *raw_data_ = nullptr;
    size_t size_; //  data size
    struct CrtTensorInfo tensor_info_;
};

} // namespace cpsruntime
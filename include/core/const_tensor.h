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
 * @Last Modified time: 2025-10-17 17:43:59
 */

#pragma once

#include "tensor.h"

namespace aipubt
{

class ConstTensor;
class ConstTensorPtr;
/**
 * @brief a class for named DataTensor descriptor, the tensor may have data, may not if it just a descriptor.
 *
 */
class ConstTensor : public Tensor
{
public:
    ConstTensor(const TensorShape &shape, Dtype type, void *data, size_t data_size);
    ConstTensor(const std::string &name, const TensorShape &shape, Dtype type, void *data, size_t data_size) : ConstTensor(shape, type, data, data_size)
    {
        m_name = name;
    };
    ConstTensor() = delete;
    /**
     * @brief Construct a new ConstTensor object
     *        construct a ConstTensor by copy other Tensor, if the tensor has no data, the ConstTensor will have a data with all zero.
     *  If the tensor has data, the ConstTensor will share the data with other Tensor.
     *  please note that the node owner(m_node) and node consumer(m_consumer) will not be copy
     *
     * @param t other Tensor
     */
    ConstTensor(TensorPtr t);
    ConstTensor(ConstTensorPtr t);

    /**
     * @brief get the tensor's data if available else nullptr will be return,
     * the data's memory is owned by tensor, client only can read/write it, no create/delete operation
     *
     * @tparam T the data type
     * @return T* the data pointer
     */
    template <typename T>
    const T *data()
    {
        return (T *)m_data;
    }

    /**
     * @brief get/set the tensor shape
     *
     * @return TensorShape& the tensor shape
     */
    const TensorShape &shape() const
    {
        return m_shape;
    }

private:
    using Tensor::compact;
    using Tensor::quantization;
    using Tensor::set_data;
    using Tensor::shape;
    using Tensor::share_data_from;
    friend class ConstTensorPtr;
};

class ConstTensorPtr
{
public:
    ConstTensorPtr() : m_ptr(nullptr) {}
    ConstTensorPtr(std::nullptr_t p) : m_ptr(p) {}
    ConstTensorPtr(const TensorShape &shape, Tensor::Dtype type, void *data, size_t data_size) : m_ptr(std::make_shared<ConstTensor>(shape, type, data, data_size)) {}
    ConstTensorPtr(const std::string &name, const TensorShape &shape, Tensor::Dtype type, void *data, size_t data_size) : m_ptr(std::make_shared<ConstTensor>(name, shape, type, data, data_size)) {}
    ConstTensorPtr(TensorPtr p) : m_ptr(std::make_shared<ConstTensor>(p)) {}
    ConstTensorPtr(std::shared_ptr<ConstTensor> p) : m_ptr(p) {}

    operator std::shared_ptr<ConstTensor>() const
    {
        return m_ptr;
    }

    ConstTensor &operator*()
    {
        return *m_ptr;
    }

    ConstTensor *operator->()
    {
        return m_ptr.get();
    }

    const ConstTensor *operator->() const
    {
        return m_ptr.get();
    }

    ConstTensor *get()
    {
        return m_ptr.get();
    }

    const ConstTensor *get() const
    {
        return m_ptr.get();
    }

    operator bool() const
    {
        return m_ptr != nullptr;
    }

protected:
    operator TensorPtr() const
    {
        return std::static_pointer_cast<Tensor>(m_ptr);
    }

private:
    std::shared_ptr<ConstTensor> m_ptr;
};

} // namespace aipubt
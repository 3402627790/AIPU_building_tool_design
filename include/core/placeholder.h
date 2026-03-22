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
 * @Last Modified time: 2022-03-24 16:40:41
 */
#ifndef CORE_PLACEHOLDER_H
#define CORE_PLACEHOLDER_H

#include <core/constants.h>
#include <core/tensor.h>
#include <tongue/symbolic.h>
using namespace tongue;

namespace aipubt
{

class PlaceholderPtr;
using MemPtr = PlaceholderPtr;
/**
 * @brief a placeholder for generating RO or descriptor, 
 * when generating a RO/descriptor, some value is is not known,
 * It may depend by later value. In this case, a placeholder can help you,
 * a placeholder can hold both a address(tensor RO) and a constants value(uint32_t RO).
 */
class Placeholder : public Tensor
{
public:
    Placeholder();
    Placeholder(MemoryObjectPtr);
    Placeholder(TensorPtr t);
    Placeholder(PlaceholderPtr t);
    Placeholder(uint32_t);
    Placeholder(MemoryObjectPtr, uint32_t);
    Placeholder(TensorPtr, uint32_t);
    Placeholder &operator=(uint32_t v);
    Placeholder &operator=(MemoryObjectPtr v);
    Placeholder &operator=(TensorPtr v);
    Placeholder &operator=(const Placeholder &v);
    Placeholder &operator+=(uint32_t v);
    Placeholder operator+(uint32_t v);
    friend class PlaceholderPtr;

protected:
private:
    // hide some functions
    using Tensor::compact;
    using Tensor::compact_mem_size;
    using Tensor::consumers;
    using Tensor::data;
    using Tensor::is_compact;
    using Tensor::is_io_tensor;
    using Tensor::max;
    using Tensor::mem_size;
    using Tensor::min;
    using Tensor::node;
    using Tensor::quantization;
    using Tensor::set_data;
    using Tensor::set_io_tensor;
    using Tensor::set_type;
    using Tensor::shape;
    using Tensor::size;
    using Tensor::type;
    using Tensor::type_s;
    using Tensor::operator=;
};

class PlaceholderPtr
{
public:
    PlaceholderPtr() : m_ptr(nullptr) {}
    PlaceholderPtr(std::nullptr_t p) : m_ptr(p) {}
    PlaceholderPtr(TensorPtr p) : m_ptr(std::static_pointer_cast<Placeholder>(p)) {}
    PlaceholderPtr(std::shared_ptr<Placeholder> p) : m_ptr(p) {}

    operator TensorPtr() const
    {
        return std::static_pointer_cast<Tensor>(m_ptr);
    }

    operator std::shared_ptr<Placeholder>() const
    {
        return m_ptr;
    }

    Placeholder &operator*()
    {
        return *m_ptr;
    }

    Placeholder *operator->()
    {
        return m_ptr.get();
    }

    const Placeholder *operator->() const
    {
        return m_ptr.get();
    }

    Placeholder *get()
    {
        return m_ptr.get();
    }

    const Placeholder *get() const
    {
        return m_ptr.get();
    }

    operator bool() const
    {
        return m_ptr != nullptr;
    }

private:
    std::shared_ptr<Placeholder> m_ptr;
};

PlaceholderPtr operator+(PlaceholderPtr p, uint32_t);
inline PlaceholderPtr operator+(uint32_t v, PlaceholderPtr p)
{
    return p + v;
}
PlaceholderPtr operator+(PlaceholderPtr p, Expr o);
inline PlaceholderPtr operator+(Expr o, PlaceholderPtr p)
{
    return p + o;
}
PlaceholderPtr operator-(PlaceholderPtr p, uint32_t);
PlaceholderPtr operator-(PlaceholderPtr p, Expr o);

};     // namespace aipubt
#endif //!CORE_PLACEHOLDER_H

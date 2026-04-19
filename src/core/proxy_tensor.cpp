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
 * @Date: 2020-01-29 14:28:32 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-23 10:03:41
 */
#include "core/proxy_tensor.h"
#include "core/quantization.h"
#include <map>
namespace aipubt
{

ProxyTensor::~ProxyTensor()
{
}

ProxyTensor::ProxyTensor(const TensorPtr t) : Tensor(*t)
{
    m_tensor = t;
    if (t)
    {
        *this = *t;
    }
}

ProxyTensor &ProxyTensor::operator=(const TensorPtr t)
{
    m_tensor = t;
    if (t)
    {
        *this = *t; //copy base data
    }

    return *this;
}

TensorShape &ProxyTensor::shape()
{
    return m_tensor->shape();
}

size_t ProxyTensor::size() const
{
    return m_tensor->size();
}

size_t ProxyTensor::mem_size() const
{
    return m_tensor->mem_size();
}

size_t ProxyTensor::buffer_size() const
{
    return m_tensor->buffer_size();
}

MemoryObjectPtr ProxyTensor::memory_offset()
{
    return m_tensor->memory_offset();
}

uint32_t ProxyTensor::offset()
{
    return m_tensor->offset();
}

void ProxyTensor::set_data(void *d, uint32_t size)
{
    return m_tensor->set_data(d, size);
}

void ProxyTensor::share_data_from(Tensor &t)
{
    m_tensor->share_data_from(t);
}

void ProxyTensor::set_type(Dtype t)
{
    return m_tensor->set_type(t);
}

Tensor::Dtype ProxyTensor::type() const
{
    return m_tensor->type();
}

const std::string &ProxyTensor::type_s() const
{
    return m_tensor->type_s();
}

const std::string &ProxyTensor::name() const
{
    return m_tensor->name();
}

void ProxyTensor::set_name(const std::string &name)
{
    m_tensor->set_name(name);
}

const Quantization ProxyTensor::quantization() const
{
    return m_tensor->quantization();
}

Quantization &ProxyTensor::quantization()
{
    return m_tensor->quantization();
}

void ProxyTensor::compact()
{
    return m_tensor->compact();
}

size_t ProxyTensor::compact_mem_size() const
{
    return m_tensor->compact_mem_size();
}

bool ProxyTensor::is_compact() const
{
    return m_tensor->is_compact();
}

void ProxyTensor::set_io_tensor(bool iot)
{
    return m_tensor->set_io_tensor(iot);
}

bool ProxyTensor::is_io_tensor() const
{
    return m_tensor->is_io_tensor();
}

std::map<std::string, Any> &ProxyTensor::attrs()
{
    return m_tensor->attrs();
}

const std::map<std::string, Any> &ProxyTensor::attrs() const
{
    return m_tensor->attrs();
}

void ProxyTensor::set_attr(const std::string &key, const Any &value)
{
    m_tensor->set_attr(key, value);
}
Any ProxyTensor::get_attr(const std::string &key) const
{
    return m_tensor->get_attr(key);
}
void ProxyTensor::erase_attr(const std::string &key)
{
    m_tensor->erase_attr(key);
}

bool ProxyTensor::operator==(const Tensor &other) const
{
    return m_tensor->operator==(other);
}

bool ProxyTensor::operator!=(const Tensor &other) const
{
    return m_tensor->operator!=(other);
}

} // namespace aipubt

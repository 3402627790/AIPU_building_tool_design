//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
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
 * @Last Modified time: 2025-10-17 17:41:25
 */
#include "builder/target.h"
#include "core.h"
#include "core/quantization.h"
#include "core/wdc/iiwdc.h"
#include "core/wdc/x2_legacy_wdc.h"
#include "core/wdc/x2_new_wdc.h"
#include "core/wdc/x3_wdc.h"
#include <fstream>
#include <iostream>
#include <map>
namespace aipubt
{

static std::map<std::string, Tensor::Dtype> STR_TO_DTYPE_MAP =
    {
        {"void", Tensor::VOID},
        {"bool", Tensor::BOOL},
        {"int8", Tensor::INT8},
        {"uint8", Tensor::UINT8},
        {"int16", Tensor::INT16},
        {"uint16", Tensor::UINT16},
        {"int32", Tensor::INT32},
        {"uint32", Tensor::UINT32},
        {"int64", Tensor::INT64},
        {"uint64", Tensor::UINT64},
        {"float16", Tensor::FP16},
        {"bfloat16", Tensor::BFP16},
        {"float32", Tensor::FP32},
        {"float64", Tensor::FP64},
        {"float8_e4m3fn", Tensor::FP8E4M3FN},
        {"fp8e4m3fn", Tensor::FP8E4M3FN},
        {"float8_e5m2", Tensor::FP8E5M2},
        {"fp8e5m2", Tensor::FP8E5M2},
        {"fp4e2m1fn", Tensor::FP4E2M1FN},
        {"float4_e2m1fn", Tensor::FP4E2M1FN},
        {"aligned_int4", Tensor::ALIGNED_INT4},
        {"aligned_uint4", Tensor::ALIGNED_UINT4},
        {"aligned_int12", Tensor::ALIGNED_INT12},
        {"aligned_uint12", Tensor::ALIGNED_UINT12},
        {"compact_int4", Tensor::COMPACT_INT4},
        {"compact_uint4", Tensor::COMPACT_UINT4},
        {"compact_int12", Tensor::COMPACT_INT12},
        {"compact_uint12", Tensor::COMPACT_UINT12},
        {"compact_fp4e2m1fn", Tensor::COMPACT_FP4E2M1FN},
};

const std::map<std::string, Tensor::Dtype> &get_dtype_dict()
{
    return STR_TO_DTYPE_MAP;
}

static std::map<Tensor::Dtype, std::string> DTYPE_TO_STR_MAP =
    {
        {Tensor::VOID, "void"},
        {Tensor::BOOL, "bool"},
        {Tensor::INT8, "int8"},
        {Tensor::UINT8, "uint8"},
        {Tensor::INT16, "int16"},
        {Tensor::UINT16, "uint16"},
        {Tensor::INT32, "int32"},
        {Tensor::UINT32, "uint32"},
        {Tensor::INT64, "int64"},
        {Tensor::UINT64, "uint64"},
        {Tensor::FP16, "float16"},
        {Tensor::BFP16, "bfloat16"},
        {Tensor::FP32, "float32"},
        {Tensor::FP64, "float64"},
        {Tensor::FP8E4M3FN, "float8_e4m3fn"},
        {Tensor::FP8E5M2, "float8_e5m2"},
        {Tensor::FP4E2M1FN, "float4_e2m1fn"},
        {Tensor::ALIGNED_INT4, "aligned_int4"},
        {Tensor::ALIGNED_UINT4, "aligned_uint4"},
        {Tensor::ALIGNED_INT12, "aligned_int12"},
        {Tensor::ALIGNED_UINT12, "aligned_uint12"},
        {Tensor::COMPACT_INT4, "compact_int4"},
        {Tensor::COMPACT_UINT4, "compact_uint4"},
        {Tensor::COMPACT_INT12, "compact_int12"},
        {Tensor::COMPACT_UINT12, "compact_uint12"},
        {Tensor::COMPACT_FP4E2M1FN, "compact_fp4e2m1fn"},
};

void Tensor::_clean_data()
{
    if (ref_count)
    {
        if (ref_count->load(std::memory_order_acquire) > 0)
        {
            ref_count->fetch_sub(1, std::memory_order_acq_rel);
        }
        if (ref_count->load(std::memory_order_acquire) == 0)
        {
            if (m_data && m_owner_host->load(std::memory_order_acquire))
            {
                delete[] m_data;
            }
            else
            {
                m_attrs.erase("_py");
            }
            m_data = nullptr;

            if (m_device_data && m_owner_device->load(std::memory_order_acquire))
            {
                delete[] m_device_data;
            }
            else
            {
                m_attrs.erase("_py_device");
            }
            m_device_data = nullptr;

            delete ref_count;
            ref_count = nullptr;
            delete m_owner_host;
            delete m_owner_device;
            m_owner_host = nullptr;
            m_owner_device = nullptr;
        }
    }
}
bool Tensor::transfer_ownership(bool is_host)
{
    bool owned = true;
    if (is_host)
    {
        return m_owner_host->compare_exchange_strong(owned, false);
    }
    return m_owner_device->compare_exchange_strong(owned, false);
}

Tensor::~Tensor()
{
    _clean_data();
}
bool Tensor::is_signed(Tensor::Dtype dt)
{
    if (is_float(dt) || dt <= Tensor::BOOL)
    {
        return true;
    }
    return dt & 0b1;
}

bool Tensor::is_int(Tensor::Dtype dt)
{
    return (dt >= Tensor::UINT8 && dt <= Tensor::INT64) || (dt >= Tensor::ALIGNED_UINT4 && dt <= Tensor::ALIGNED_INT12) || (dt >= Tensor::COMPACT_UINT4 && dt <= Tensor::COMPACT_INT12);
}

bool Tensor::is_aligned(Tensor::Dtype dt)
{
    return dt >= Tensor::ALIGNED_UINT4 && dt <= Tensor::FP4E2M1FN;
}

Tensor::Dtype Tensor::dtype_to_signed(Tensor::Dtype dt)
{
    if (is_float(dt) || dt <= Tensor::BOOL)
    {
        return dt;
    }
    return (Tensor::Dtype)(dt | 0b1);
}

Tensor::Dtype Tensor::dtype_to_unsigned(Tensor::Dtype dt)
{
    if (is_float(dt) || dt <= Tensor::BOOL)
    {
        return dt;
    }
    return (Tensor::Dtype)(dt & ~1);
}

const std::string &Tensor::type_s() const
{
    return DTYPE_TO_STR_MAP[m_type];
}

Tensor::Dtype Tensor::string_to_dtype(const std::string &tstr)
{
    return STR_TO_DTYPE_MAP[tstr];
}
const std::string &Tensor::dtype_to_string(Tensor::Dtype dt)
{
    return DTYPE_TO_STR_MAP[dt];
}
Tensor::Dtype Tensor::get_type(const int bits, const bool is_sign)
{

    switch (bits)
    {
    case 8:
        return is_sign ? Tensor::INT8 : Tensor::UINT8;
    case 16:
        return is_sign ? Tensor::INT16 : Tensor::UINT16;
    case 32:
        return is_sign ? Tensor::INT32 : Tensor::UINT32;
    case 64:
        return is_sign ? Tensor::INT64 : Tensor::UINT64;
    default:
        return Tensor::VOID;
    }
}

TensorPtr Tensor::workspace(size_t size)
{
    auto ret = std::make_shared<Tensor>(TensorShape(size), Tensor::UINT8);
    ret->set_attr(Constants::is_workspace_tensor, true);
    return ret;
}

static uint32_t get_tensor_counter()
{
    static uint32_t i = 0;
    return i++;
}
Tensor::Tensor(const TensorShape &shape, Dtype type) : m_shape(shape), m_type(type)
{
    m_name = "_t" + std::to_string(get_tensor_counter());
}
Tensor::Tensor(void *data, size_t size)
{
    m_name = "_t" + std::to_string(get_tensor_counter());
    m_data = static_cast<char *>(data);
    m_shape = TensorShape(size);
    m_data_offset = 0;
    m_offset->size = size;
    if (m_data)
    {
        ref_count = new std::atomic<int>(1);
        m_owner_host = new std::atomic<bool>(true);
        m_owner_device = new std::atomic<bool>(false);
    }
}

Tensor::Tensor(const std::string &name, const TensorShape &shape, Dtype type) : m_name(name), m_shape(shape), m_type(type)
{
}

Tensor::Tensor(const std::string &name, void *data, size_t size) : m_name(name)
{
    m_data = static_cast<char *>(data);
    m_shape = TensorShape(size);
    m_offset->size = size;
    m_data_offset = 0;
    if (m_data)
    {
        ref_count = new std::atomic<int>(1);
        m_owner_host = new std::atomic<bool>(true);
        m_owner_device = new std::atomic<bool>(false);
    }
}
Tensor::Tensor(const Tensor &t)
{
    // clean old data
    _clean_data();
    Tensor *ct = (Tensor *)(&t);
    auto pt = dynamic_cast<ProxyTensor *>(ct);
    if (pt)
    {
        ct = pt->proxied_tensor(true).get();
    }
    m_data = ct->m_data;
    m_device_data = ct->m_device_data;
    m_data_offset = ct->m_data_offset;
    m_shape = ct->m_shape;
    m_type = ct->m_type;
    (*m_offset) = (*ct->m_offset);
    ref_count = ct->ref_count;
    m_owner_host = ct->m_owner_host;
    m_owner_device = ct->m_owner_device;
    m_name = ct->m_name;
    if (ref_count)
    {
        ref_count->fetch_add(1, std::memory_order_relaxed);
    }
    m_flag = ct->m_flag;
    // copy attrs
    m_attrs = ct->m_attrs;
    // Don't copy the owner
    //  m_node = t.m_node;
    //  not copy the consumers!!
}
Tensor &Tensor::operator=(const Tensor &t)
{
    if (this == &t)
    {
        return *this;
    }
    _clean_data();

    Tensor *ct = (Tensor *)(&t);
    auto pt = dynamic_cast<ProxyTensor *>(ct);
    if (pt)
    {
        ct = pt->proxied_tensor(true).get();
    }
    m_data = ct->m_data;
    m_device_data = ct->m_device_data;
    m_data_offset = ct->m_data_offset;
    m_shape = ct->m_shape;
    m_type = ct->m_type;
    (*m_offset) = (*ct->m_offset);
    ref_count = ct->ref_count;
    m_owner_host = ct->m_owner_host;
    m_owner_device = ct->m_owner_device;
    m_name = ct->m_name;
    if (ref_count)
    {
        ref_count->fetch_add(1, std::memory_order_relaxed);
    }
    // copy attrs
    m_attrs = ct->m_attrs;
    return *this;
}

void Tensor::share_data_from(Tensor &t)
{
    Tensor *ct = (Tensor *)(&t);
    auto pt = dynamic_cast<ProxyTensor *>(ct);
    if (pt)
    {
        ct = pt->proxied_tensor(true).get();
    }
    _clean_data();

    m_data = ct->m_data;
    m_device_data = ct->m_device_data;
    m_data_offset = ct->m_data_offset;
    m_offset->size = ct->m_offset->size;
    ref_count = ct->ref_count;
    m_owner_host = ct->m_owner_host;
    m_owner_device = ct->m_owner_device;
    m_name = ct->m_name;
    if (ref_count)
    {
        ref_count->fetch_add(1, std::memory_order_relaxed);
    }
}

void Tensor::set_name(const std::string &name)
{
    auto new_name = name;
    if (new_name.empty())
    {
        new_name = "_t" + std::to_string(get_tensor_counter());
    }
    auto n = m_node.lock();
    if (n)
    {
        auto g = n->graph();
        if (g)
        {
            g->m_tensors.erase(m_name);
            g->m_tensors[new_name] = shared_from_this();
        }
    }
    m_name = new_name;
}

void Tensor::_set_host_device_data(void *d, const size_t &size, const bool owner, const bool is_host)
{
    _clean_data();

    if (is_host)
    {
        m_data = (char *)(d);
        m_device_data = nullptr;
    }
    else
    {
        m_device_data = (char *)(d);
        m_data = nullptr;
    }

    m_data_offset = 0;
    m_offset->size = size;
    if (m_shape.size() * dtype_bits(m_type) / 8.0 != size)
    {
        if (node())
        {
            WARN() << "Setting a data with size = " << size << " to tensor " << m_name << "of node: " << node()->name() << " which mem size should be " << m_shape.size() * dtype_bits(m_type) / 8.0;
        }
        else
        {
            WARN() << "Setting a data with size = " << size << " to tensor " << m_name << " which mem size should be " << m_shape.size() * dtype_bits(m_type) / 8.0;
        }
    }
    if (m_data || m_device_data)
    {
        m_offset->size = size;
void Tensor::share_data_from(Tensor &t)
{
    Tensor *ct = (Tensor *)(&t);
    auto pt = dynamic_cast<ProxyTensor *>(ct);
    if (pt)
    {
        ct = pt->proxied_tensor(true).get();
    }
    _clean_data();

    m_data = ct->m_data;
    m_device_data = ct->m_device_data;
    m_data_offset = ct->m_data_offset;
    m_offset->size = ct->m_offset->size;
    ref_count = ct->ref_count;
    m_owner_host = ct->m_owner_host;
    m_owner_device = ct->m_owner_device;
    m_name = ct->m_name;
    if (ref_count)
    {
        ref_count->fetch_add(1, std::memory_order_relaxed);
    }
}

void Tensor::set_name(const std::string &name)
{
    auto new_name = name;
    if (new_name.empty())
    {
        new_name = "_t" + std::to_string(get_tensor_counter());
    }
    auto n = m_node.lock();
    if (n)
    {
        auto g = n->graph();
        if (g)
        {
            g->m_tensors.erase(m_name);
            g->m_tensors[new_name] = shared_from_this();
        }
    }
    m_name = new_name;
}

void Tensor::_set_host_device_data(void *d, const size_t &size, const bool owner, const bool is_host)
{
    _clean_data();

    if (is_host)
    {
        m_data = (char *)(d);
        m_device_data = nullptr;
    }
    else
    {
        m_device_data = (char *)(d);
        m_data = nullptr;
    }

    m_data_offset = 0;
    m_offset->size = size;
    if (m_shape.size() * dtype_bits(m_type) / 8.0 != size)
    {
        if (node())
        {
            WARN() << "Setting a data with size = " << size << " to tensor " << m_name << "of node: " << node()->name() << " which mem size should be " << m_shape.size() * dtype_bits(m_type) / 8.0;
        }
        else
        {
            WARN() << "Setting a data with size = " << size << " to tensor " << m_name << " which mem size should be " << m_shape.size() * dtype_bits(m_type) / 8.0;
        }
    }
    if (m_data || m_device_data)
    {
        m_offset->size = size;
{
    static std::map<Tensor::Dtype, std::function<void(Tensor *)>> DTYPE_TO_COMPACT_MAP =
        {
            {Tensor::INT8, std::mem_fn(&Tensor::compact_impl<Tensor::INT8>)},
            {Tensor::UINT8, std::mem_fn(&Tensor::compact_impl<Tensor::UINT8>)},
            {Tensor::INT16, std::mem_fn(&Tensor::compact_impl<Tensor::INT16>)},
            {Tensor::UINT16, std::mem_fn(&Tensor::compact_impl<Tensor::UINT16>)},
            {Tensor::INT32, std::mem_fn(&Tensor::compact_impl<Tensor::INT32>)},
            {Tensor::UINT32, std::mem_fn(&Tensor::compact_impl<Tensor::UINT32>)},
            {Tensor::INT64, std::mem_fn(&Tensor::compact_impl<Tensor::INT64>)},
            {Tensor::UINT64, std::mem_fn(&Tensor::compact_impl<Tensor::UINT64>)},
            {Tensor::ALIGNED_INT4, std::mem_fn(&Tensor::compact_impl<Tensor::ALIGNED_INT4>)},
            {Tensor::ALIGNED_UINT4, std::mem_fn(&Tensor::compact_impl<Tensor::ALIGNED_UINT4>)},
            {Tensor::ALIGNED_INT12, std::mem_fn(&Tensor::compact_impl<Tensor::ALIGNED_INT12>)},
            {Tensor::ALIGNED_UINT12, std::mem_fn(&Tensor::compact_impl<Tensor::ALIGNED_UINT12>)},
            {Tensor::FP4E2M1FN, std::mem_fn(&Tensor::compact_impl<Tensor::ALIGNED_UINT4>)},
        };
    auto it = DTYPE_TO_COMPACT_MAP.find(m_type);
    if (it != DTYPE_TO_COMPACT_MAP.end())
    {
        (it->second)(this);
    }
}

void Tensor::decompact()
{
    static std::map<Tensor::Dtype, std::function<void(Tensor *)>> DTYPE_TO_COMPACT_MAP =
        {
            {Tensor::COMPACT_INT4, std::mem_fn(&Tensor::decompact_impl<Tensor::ALIGNED_INT4>)},
            {Tensor::COMPACT_UINT4, std::mem_fn(&Tensor::decompact_impl<Tensor::ALIGNED_UINT4>)},
            {Tensor::COMPACT_INT12, std::mem_fn(&Tensor::decompact_impl<Tensor::ALIGNED_INT12>)},
            {Tensor::COMPACT_UINT12, std::mem_fn(&Tensor::decompact_impl<Tensor::ALIGNED_UINT12>)},
            {Tensor::COMPACT_FP4E2M1FN, std::mem_fn(&Tensor::decompact_impl<Tensor::ALIGNED_UINT4>)},
        };
    auto it = DTYPE_TO_COMPACT_MAP.find(m_type);
    if (it != DTYPE_TO_COMPACT_MAP.end())
    {
        (it->second)(this);
    }
}

size_t Tensor::compact_mem_size() const
{
    return m_shape.size() * dtype_bits(m_type) / 8;
}

bool Tensor::is_compact() const
{
    return is_compact(m_type);
}

const Quantization Tensor::quantization() const
{
    auto q = m_attrs.at("quantization");
    return cast_any<Quantization>(q);
}
Quantization &Tensor::quantization()
{
    if (!m_attrs.count("quantization"))
    {
        m_attrs["quantization"] = Quantization();
    }
    return m_attrs["quantization"];
}

NodePtr Tensor::node() const
{
    return m_node.lock();
}

std::vector<NodePtr> Tensor::consumers() const
{
    std::vector<NodePtr> ret;
    for (auto n : m_consumers)
    {
        ret.push_back(n.lock());
    }
    return ret;
}

void Tensor::add_consumer(NodePtr n)
{
    if (!n)
    {
        return;
    }
    for (auto c : m_consumers)
    {
        if (c.lock() == n)
        {
            return;
        }
    }
    m_consumers.push_back(n);
}

void Tensor::remove_consumer(NodePtr n)
{
    if (!n)
    {
        return;
    }
    for (uint32_t i = 0; i < m_consumers.size(); ++i)
    {

        if (m_consumers[i].lock() == n)
        {
            m_consumers.erase(m_consumers.begin() + i);
            return;
        }
    }
}

bool Tensor::operator==(const Tensor &other) const
{
    if (other.m_shape != m_shape)
    {
        return false;
    }
    else if (other.m_type != m_type)
    {
        return false;
    }
    else if (other.m_device_data != m_device_data)
    {
        return false;
    }
    else if (other.m_data == m_data)
    {
        return true;
    }
    // TODO: if m_device_data is compact
    else if (other.m_data != nullptr && m_data != nullptr)
    {
        if (other.compact_mem_size() != compact_mem_size())
        {
            return false;
        }
        else
        {
            Tensor lhs = *this;
            Tensor rhs = other;
            lhs.compact();
            rhs.compact();
            char *lhs_data = lhs.data<char>();
            char *rhs_data = rhs.data<char>();
            size_t mem_size = lhs.compact_mem_size();
            return (std::memcmp(lhs_data, rhs_data, mem_size) == 0);
        }
    }
    else
    {
        return false;
    }
}

bool Tensor::operator!=(const Tensor &other) const
{
    return !(*this == other);
}

std::ostream &operator<<(std::ostream &os, Tensor::Dtype dtype)
{
    return (os << Tensor::dtype_to_string(dtype));
}
std::ostream &operator<<(std::ostream &os, const Tensor &t)
{
    return os << t.name();
}
std::ostream &operator<<(std::ostream &os, TensorPtr t)
{
    return os << t->name();
}

template <Tensor::Dtype dtype>
void compact_mem_impl(char *new_mem, char *old_mem, int bytes, int bits, int elements)
{
    using data_type = typename tensor_data_type<dtype>::type;

    aipu_assert(bits <= bytes * 8);
    aipu_assert((elements * bits) % 8 == 0);
    auto compact_mem_size = (elements * bits) / 8;
    memset(new_mem, 0, compact_mem_size);

    int group_elements = 1;
    while ((group_elements * bits) % (8 * bytes) != 0)
    {
        group_elements++;
    }
    int groups = (elements + group_elements - 1) / group_elements;

    auto get_mask = [](int bits) {
        return (1 << bits) - 1;
    };

    data_type *new_mem_ptr = reinterpret_cast<data_type *>(new_mem);
    data_type *old_mem_ptr = reinterpret_cast<data_type *>(old_mem);
    int old_mem_idx = 0;
    int new_mem_idx = 0;
    for (auto i = 0; i < groups; ++i)
    {
        int start_pos = 0;
        for (auto j = 0; j < group_elements; j++)
        {
            old_mem_idx = i * group_elements + j;
            data_type current_value = 0;
            if (old_mem_idx < elements)
            {
                current_value = old_mem_ptr[old_mem_idx] & get_mask(bits);
            }
            if (start_pos + bits < 8 * bytes)
            {
                new_mem_ptr[new_mem_idx] |= (current_value << start_pos) & get_mask(start_pos + bits);
                start_pos += bits;
            }
            else
            {
                size_t remain_bits = 8 * bytes - start_pos;
                new_mem_ptr[new_mem_idx] |= (current_value & get_mask(remain_bits)) << start_pos;
                new_mem_idx++;
                start_pos = 0;
                new_mem_ptr[new_mem_idx] |= (current_value >> remain_bits) & get_mask(bits - remain_bits);
                start_pos += bits - remain_bits;
            }
        }
    }
}

template <Tensor::Dtype dtype>
void Tensor::compact_impl()
{
    if (!is_aligned(m_type) || m_data == nullptr)
    {
        return;
    }

    m_type = (Tensor::Dtype)((int)m_type + 0xe); // convert to compact type
    auto bytes = dtype_size(m_type);
    auto bits = dtype_bits(m_type);
    auto elements = m_shape.size();
    aipu_assert((elements * bits) % 8 == 0);
    auto compact_mem_size = elements * bits / 8;

    char *compact_data = new char[compact_mem_size];
    compact_mem_impl<dtype>(compact_data, m_data, bytes, bits, elements);
    set_data(compact_data, compact_mem_size);
}

template <Tensor::Dtype dtype>
void decompact_mem_impl(char *new_mem, char *old_mem, const int bytes, const int bits, const int elements, const bool sign)
{
    using data_type = typename tensor_data_type<dtype>::type;
    aipu_assert(bits <= bytes * 8);
    aipu_assert((elements * bits) % 8 == 0);
    auto decompact_mem_size = (elements * bytes);
    memset(new_mem, 0, decompact_mem_size);

    data_type *new_mem_ptr = reinterpret_cast<data_type *>(new_mem);

    for (auto i = 0; i < elements; ++i)
    {
        auto div = (i * bits) >> 3;
        auto remain = (i * bits) & 0x7;
        unsigned char *ptr = (unsigned char *)old_mem + div;
        data_type v = 0;
        if (remain)
        {
            v = (ptr[0] >> remain);
            remain = 8 - remain;
            ptr++;
        }
        if ((bits - remain) > 8)
        {
            v |= ptr[0] << remain;
            ptr++;
            remain += 8;
        }
        if (remain != bits)
        {
            v |= (ptr[0] & ((1 << (bits - remain)) - 1)) << remain;
        }
        // signed extern
        if (sign && (v & (1 << (bits - 1))))
        {
            v |= ((0xffffffff >> bits) << bits);
        }
        new_mem_ptr[i] = v;
    }
}
template <Tensor::Dtype dtype>
void Tensor::decompact_impl()
{
    if (!is_compact(m_type) || m_data == nullptr)
    {
        return;
    }
    auto bytes = dtype_size(m_type);
    auto bits = dtype_bits(m_type);
    m_type = (Tensor::Dtype)((int)m_type - 0xe); // convert to aligned type
    auto elements = m_shape.size();
    aipu_assert((elements * bits) % 8 == 0);
    auto decompact_mem_size = elements * bytes;

    char *new_data = new char[decompact_mem_size];
    decompact_mem_impl<dtype>(new_data, m_data, bytes, bits, elements, is_signed(m_type) && (!is_float(m_type)));
    set_data(new_data, decompact_mem_size);
}

void Tensor::make_memory_continuous()
{
    if (!m_shape.is_continuous())
    {
        char *data = new char[mem_size()];
        auto eltsize = dtype_size(m_type);
        auto strides = m_shape.strides();
        m_shape = TensorShape(m_shape, m_shape.layout()); // clean strides
        auto m_s = m_shape.strides();
        for (size_t i = 0; i < m_shape.size(); ++i)
        {
            auto offset = index_to_offset(offset_to_index(i, m_s), strides);
            for (uint32_t j = 0; j < eltsize; ++j)
            {
                data[i * eltsize + j] = (m_data + m_data_offset)[offset * eltsize + j];
            }
        }
        set_data(data, mem_size());
    }
}

void Tensor::monopolize_data() // make no tensor share this data
{
    if (!m_data || !ref_count)
    {
        return;
    }
    if (ref_count->load(std::memory_order_acquire) == 1) // no one share
    {
        return;
    }
    if (m_shape.is_continuous() || !*m_owner_host)
    {
        char *data = new char[mem_size()];
        memcpy(data, m_data + m_data_offset, mem_size());
        set_data(data, mem_size());
    }
    else
    {
        make_memory_continuous();
    }
}

bool Tensor::fill_tensors_data(std::vector<TensorPtr> &tensors, std::vector<std::string> &files)
{
    // fill input data
    if (tensors.size() != files.size())
    {
        ERROR() << "Expect " << tensors.size() << " inputs, but got " << files.size() << ".";
        return false;
    }
    for (uint32_t i = 0; i < tensors.size(); ++i)
    {
        std::ifstream infile(files[i], std::ios::binary | std::ios::in);
        if (!infile)
        {
            ERROR() << "Cannot read input files " << files[i];
            infile.close();
            return false;
        }
        infile.seekg(0, std::ios::end);
        size_t file_size = infile.tellg();
        infile.seekg(0);
        size_t mem_size = tensors[i]->mem_size();
        char *data = new char[file_size];
        infile.read(data, file_size);
        if (infile.fail())
        {
            WARN() << "Failed to read input of file" << files[i] << " for tensor " << tensors[i]->name();
            return false;
        }
        if (mem_size > 0)
        {
            if (mem_size != file_size)
            {
                WARN() << "input file:" << tensors[i]->name() << " has a size:" << file_size << ", while the input tensor expects a size:" << mem_size;
            }
            file_size = file_size > mem_size ? mem_size : file_size;
        }
        tensors[i]->set_data(data, file_size);
        infile.close();
    }
    return true;
}

template <Tensor::Dtype dtype>
inline bool compare_data(void *ptr1, size_t offset1, void *ptr2, size_t offset2)
{
    using DType = typename tensor_data_type<dtype>::type;
    DType *p1 = (DType *)ptr1;
    DType *p2 = (DType *)ptr2;
    return p1[offset1] == p2[offset2];
}
template <Tensor::Dtype dtype>
inline void copy_data(void *ptr1, size_t offset1, void *ptr2, size_t offset2)
{
    using DType = typename tensor_data_type<dtype>::type;
    DType *p1 = (DType *)ptr1;
    DType *p2 = (DType *)ptr2;
    p1[offset1] = p2[offset2];
}
TensorPtr shrink_tensor_data(TensorPtr t, int axis, bool keep_dim)
{
    if (!t || !t->data<char>())
    {
        return nullptr;
    }
    auto shape = t->shape();
    if (axis < 0)
    {
        axis += shape.dim();
    }
    if (axis < 0 || axis >= shape.dim())
    {
        return nullptr;
    }

    auto s = shape[axis];
    if (s == 1)
    {
        char *new_data = new char[t->mem_size()];
        memcpy(new_data, t->data<char>(), t->mem_size());
        auto ret = std::make_shared<Tensor>(*t);
        ret->set_data(new_data, t->mem_size());
        if (!keep_dim)
        {
            std::vector<int> new_shape;
            for (int i = 0; i < shape.dim(); ++i)
            {
                if (i != axis)
                {
                    new_shape.push_back(shape[i]);
                }
            }
            ret->shape() = TensorShape(new_shape);
        }
        return ret;
    }
    bool ret = true;
    int bf = 1, af = 1;
    std::vector<int> new_shape;
    for (int i = 0; i < shape.dim(); ++i)
    {
        if (i < axis)
        {
            bf *= shape[i];
            new_shape.push_back(shape[i]);
        }
        else if (i > axis)
        {
            af *= shape[i];
            new_shape.push_back(shape[i]);
        }
        else if (keep_dim)
        {
            new_shape.push_back(1);
        }
    }
    void *ptr = t->data<void>();
    auto elsize = Tensor::dtype_size(t->type());
    char *new_data = new char[bf * af * elsize];
    std::function<bool(void *, size_t, void *, size_t)> compare_func = nullptr;
    std::function<void(void *, size_t, void *, size_t)> copy_func = nullptr;
    if (elsize == 1)
    {
        compare_func = compare_data<Tensor::Dtype::UINT8>;
        copy_func = copy_data<Tensor::Dtype::UINT8>;
    }
    else if (elsize == 2)
    {
        compare_func = compare_data<Tensor::Dtype::UINT16>;
        copy_func = copy_data<Tensor::Dtype::UINT16>;
    }
    else if (elsize == 4)
    {
        compare_func = compare_data<Tensor::Dtype::UINT32>;
        copy_func = copy_data<Tensor::Dtype::UINT32>;
    }
    else if (elsize == 8)
    {
        compare_func = compare_data<Tensor::Dtype::UINT64>;
        copy_func = copy_data<Tensor::Dtype::UINT64>;
    }
    for (int i = 0; (i < bf) && ret; ++i)
    {
        for (int j = 1; (j < s) && ret; ++j)
        {
            for (int k = 0; k < af; ++k)
            {
                if (!compare_func(ptr, (i * s + j) * af + k, ptr, i * s * af + k))
                {
                    ret = false;
                    delete[] new_data;
                    break;
                }
                if (j == 1)
                {
                    copy_func(new_data, i * af + k, ptr, i * s * af + k);
                }
            }
        }
    }
    if (ret)
    {
        auto nt = std::make_shared<Tensor>(*t);
        nt->shape() = TensorShape(new_shape);
        nt->set_data(new_data, bf * af * elsize);
        return nt;
    }
    return nullptr;
}

using WDCEngine = std::unique_ptr<zhouyi::IIWdc>;

static WDCEngine get_wdc_by_method(Tensor::CompressAlgo method)
{
    WDCEngine ret = nullptr;
    switch (method)
    {
    case Tensor::CompressAlgo::I4ALLONE:
        ret = std::make_unique<zhouyi::WdcInt4M1>();
        break;
    case Tensor::CompressAlgo::I4_333331:
        ret = std::make_unique<zhouyi::WdcInt4M2>();
        break;
    case Tensor::CompressAlgo::I4_772:
        ret = std::make_unique<zhouyi::WdcInt4M3>();
        break;
    case Tensor::CompressAlgo::I8_333331:
        ret = std::make_unique<zhouyi::WdcInt8M2>();
        break;
    case Tensor::CompressAlgo::I8ALLONE:
        ret = std::make_unique<zhouyi::WdcInt8M1>();
        break;
    case Tensor::CompressAlgo::LOWBIT16:
        ret = std::make_unique<zhouyi::WdcI16ToNBit>();
        break;
    case Tensor::CompressAlgo::LOWBIT8:
        ret = std::make_unique<zhouyi::WdcI8ToNBit>();
        break;
    case Tensor::CompressAlgo::LEGACY16:
        ret = std::make_unique<zhouyi::WdcLegacy<int16_t>>();
        break;
    case Tensor::CompressAlgo::LEGACY8:
        ret = std::make_unique<zhouyi::WdcLegacy<int8_t>>();
        break;
    case Tensor::CompressAlgo::UNCOMPRESS:
    default:
        break;
    }
    return ret;
}

TensorPtr Tensor::compress(float rate, const std::vector<int> algos)
{
    static const std::vector<int> x2_algo =
        {
            BITMAP8,
            BITMAP16,
            LOWBIT8,
            LOWBIT16,
        };
    static const std::vector<int> x3_algo =
        {
            I8ALLONE,
            I8_333331,
            I4_772,
            I4_333331,
            I4ALLONE,
        };
    const std::vector<int> *available = &x2_algo;
    if (is_compact()) // engine only support aligned data
    {
        auto cp = std::make_shared<Tensor>(*this);
        cp->decompact();
        auto ret = cp->compress(rate, algos);
        if (ret == cp)
        {
            return shared_from_this();
        }
        return ret;
    }
    auto bits = dtype_bits(m_type);
    if (rate == -1.0) // force compress
    {
        rate = 32.0;
    }
    else if (rate >= 1.0 || rate <= 0)
    {
        return shared_from_this();
    }

    if (!(is_int(m_type) || is_aligned(m_type) || bits == 8) || bits > 16)
    {
        return shared_from_this();
    }
    if (is_aligned(m_type))
    {
        rate = rate * (bits - 4) / bits;
        bits -= 4;
    }
    auto target = get_target();
    if (target->isainfo().type >= ISA_ZYv6)
    {
        available = &x3_algo;
    }
    if (algos.size())
    {
        available = &algos;
    }
    int size = mem_size();
    int c_size = ((size + 31) & ~31) << 1;
    char *lut = new char[16];
    char *lut2 = new char[16];
    char *ptr1 = new char[c_size];
    char *ptr2 = new char[c_size];
    long long better_size = size * rate;
    CompressFrameInfo finfo;
    if (m_attrs.count(Constants::compress_frame_info))
    {
        finfo = m_attrs.at(Constants::compress_frame_info);
    }

    CompressInfo info;
    info.ori_size = size;
    info.type = m_type;
    info.shape = m_shape;
    for (auto algo : *available)
    {
        if ((2 << ((int)algo & 0b11)) != (int)bits)
        {
            continue;
        }
        // INFO() << "using algo " << (int)algo;

        auto engine = get_wdc_by_method((CompressAlgo)algo);
        if (!engine)
        {
            continue;
        }
        *(uint64_t *)lut = finfo;
        auto bytes = engine->encode(data(), mem_size(), ptr1, c_size, lut);
        // (CP-5477) the size after compressed need 32Byte align
        // otherwise, it may effect next address when assemble
        bytes = (bytes + 31) / 32 * 32;
        if (bytes < better_size)
        {
            better_size = bytes;
            char *tt = ptr1;
            ptr1 = ptr2;
            ptr2 = tt;
            info.algo = (CompressAlgo)algo;
            tt = lut;
            lut = lut2;
            lut2 = tt;
        }
    }
    delete[] ptr1;
    delete[] lut;
    if (info.algo != UNCOMPRESS)
    {
        for (auto i = 0u; i < 16; ++i)
        {
            info.extra.lut[i] = lut2[i];
        }
        auto ret = std::make_shared<Tensor>(name() + "_cd", ptr2, better_size);
        delete[] lut2;
        ret->set_attr(Constants::compress_info, info);
        return ret;
    }
    delete[] ptr2;
    delete[] lut2;
    return shared_from_this();
}
const Tensor::CompressInfo Tensor::get_compress_info()
{
    if (m_attrs.count(Constants::compress_info))
    {
        return m_attrs.at(Constants::compress_info);
    }
    return CompressInfo();
}
TensorPtr Tensor::decompress()
{
    auto info = get_compress_info();
    if (info.algo != CompressAlgo::UNCOMPRESS)
    {
        auto wdc = get_wdc_by_method(info.algo);
        if (wdc)
        {
            char *new_data = new char[info.ori_size];
            int bytes = wdc->decode(data(), mem_size(),
                                    info.extra.lut,
                                    new_data, info.ori_size);
            if (bytes != (int)(info.ori_size))
            {
                ERROR() << "xxxxxxxxxxxxxxxx";
            }
            auto ret = std::make_shared<Tensor>(name() + "_dc", TensorShape(info.shape), info.type);
            // do signed extern
            ret->set_data(new_data, info.ori_size);
            return ret;
        }
    }
    return shared_from_this();
}

int Tensor::get_ref_count()
{
    int cnt = ref_count->load(std::memory_order_acquire);
    return cnt;
}

} // namespace aipubt

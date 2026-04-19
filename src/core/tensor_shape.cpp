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
 * @Date: 2020-01-29 14:28:26
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-04 13:01:50
 */
#include "core.h"
#include "utils/string_utils.h"
#include <algorithm>
#include <set>
#include <sstream>
#include <string>
namespace aipubt
{
#define MIN(a, b) ((a) > (b) ? (b) : (a))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

SymbolShape::SymbolShape(const std::vector<tongue::Expr> &shape)
{
    m_shape.assign(shape.begin(), shape.end());
}

SymbolShape::SymbolShape(const std::vector<tongue::Expr> &shape, DataLayout layout, ShapeDim idx)
{
    m_shape.assign(shape.begin(), shape.end());
    m_layout = layout;
    m_idx = idx;
}

SymbolShape::SymbolShape(const SymbolShape &shape)
{
    m_idx = shape.m_idx;
    m_layout = shape.m_layout;
    m_shape = shape.m_shape;
}

tongue::Expr SymbolShape::operator[](int index) const
{
    if (index < 0)
    {
        index = m_shape.size() + index;
    }
    aipu_assert(index < static_cast<int>(m_shape.size()));
    return m_shape[index];
}

SymbolShape::operator std::vector<tongue::Expr>() const
{
    return m_shape;
}

bool SymbolShape::operator==(const SymbolShape &other) const
{
    if (layout() != other.layout())
    {
        return false;
    }
    if (dim() != other.dim())
    {
        return false;
    }
    for (auto i = 0; i < dim(); ++i)
    {
        if (!m_shape[i]->equal(other[i]))
        {
            return false;
        }
    }
    return true;
}

bool SymbolShape::operator!=(const SymbolShape &other) const
{
    return !(*this == other);
}

std::vector<tongue::Expr>::iterator SymbolShape::begin()
{
    return m_shape.begin();
}
std::vector<tongue::Expr>::iterator SymbolShape::end()
{
    return m_shape.end();
}

std::vector<tongue::Expr>::const_iterator SymbolShape::begin() const
{
    return m_shape.begin();
}
std::vector<tongue::Expr>::const_iterator SymbolShape::end() const
{
    return m_shape.end();
}

tongue::Expr SymbolShape::h() const
{
    aipu_assert((int)m_shape.size() > m_idx.h);
    return m_shape[m_idx.h];
}

tongue::Expr SymbolShape::w() const
{
    aipu_assert((int)m_shape.size() > m_idx.w);
    return m_shape[m_idx.w];
}

tongue::Expr SymbolShape::c() const
{
    if (m_shape.size() == 3)
    {
        // layout must not be NCHW, FLAT/UNKOWN/NHWC ok
        return m_shape[m_idx.c - 1];
    }
    else if (m_shape.size() == 2)
    {
        // layout must not be NCHW, FLAT/UNKOWN/NHWC ok
        return m_shape[m_idx.c - 2];
    }
    else
    {
        aipu_assert((int)m_shape.size() > m_idx.c);
        return m_shape[m_idx.c];
    }
}

tongue::Expr SymbolShape::n() const
{
    return m_shape[m_idx.n];
}

tongue::Expr SymbolShape::c_prime() const
{
    switch (m_layout)
    {
    case DataLayout_NCHW:
    case DataLayout_NHWC:
    case DataLayout_NHWCSparse:
        return 1;
    case DataLayout_NCHWC4:
        return 4;
    case DataLayout_NCHWC8:
        return 8;
    case DataLayout_NCHWC16:
    case DataLayout_NCHWC16Sparse:
    case DataLayout_NCHWC16T8:
    case DataLayout_NCHWC16T8Sparse:
    case DataLayout_NCHWC16T1:
    case DataLayout_NCHWC16T1Sparse:
    case DataLayout_NDCHWC16:
    case DataLayout_NDCHWC16T8:
    case DataLayout_NDCHWC16T1:
    case DataLayout_NCDHWC16:
        return 16;
    case DataLayout_NCHWC32:
    case DataLayout_NCHWC32Sparse:
    case DataLayout_NCHWC32T8:
    case DataLayout_NCHWC32T8Sparse:
    case DataLayout_NCHWC32T1:
    case DataLayout_NCHWC32T1Sparse:
    case DataLayout_NDCHWC32:
    case DataLayout_NDCHWC32T8:
    case DataLayout_NDCHWC32T1:
    case DataLayout_NCDHWC32:
        return 32;
    default:
        return 1;
    }
}

tongue::Expr SymbolShape::c_tail() const
{
    switch (m_layout)
    {
    case DataLayout_NCHW:
    case DataLayout_NHWC:
    case DataLayout_NDHWC:
    case DataLayout_NHWCSparse:
    case DataLayout_NCHWC4:
    case DataLayout_NCHWC8:
    case DataLayout_NCHWC16:
    case DataLayout_NCHWC16Sparse:
    case DataLayout_NCHWC32:
    case DataLayout_NCHWC32Sparse:
    case DataLayout_NDCHWC16:
    case DataLayout_NDCHWC32:
    case DataLayout_NCDHWC16:
    case DataLayout_NCDHWC32:
        return 0;
    case DataLayout_NCHWC16T8:
    case DataLayout_NCHWC16T8Sparse:
    case DataLayout_NCHWC32T8:
    case DataLayout_NCHWC32T8Sparse:
    case DataLayout_NDCHWC16T8:
    case DataLayout_NDCHWC32T8:
    case DataLayout_NCHWC16T1:
    case DataLayout_NCHWC16T1Sparse:
    case DataLayout_NCHWC32T1:
    case DataLayout_NCHWC32T1Sparse:
    case DataLayout_NDCHWC16T1:
    case DataLayout_NDCHWC32T1:
        return c_total() % c_prime();
    default:
        return 0;
    }
}

tongue::Expr SymbolShape::c_total() const
{
    switch (m_layout)
    {
    case DataLayout_NCHW:
    case DataLayout_NHWC:
    case DataLayout_NDHWC:
    case DataLayout_NHWCSparse:
    case DataLayout_NCHWC4:
    case DataLayout_NCHWC8:
    case DataLayout_NCHWC16:
    case DataLayout_NCHWC16Sparse:
    case DataLayout_NCHWC32:
    case DataLayout_NCHWC32Sparse:
    case DataLayout_NDCHWC16:
    case DataLayout_NDCHWC32:
    case DataLayout_NCDHWC16:
    case DataLayout_NCDHWC32:
        return c_prime() * c();
    case DataLayout_NCHWC16T8:
    case DataLayout_NCHWC16T8Sparse:
    case DataLayout_NCHWC32T8:
    case DataLayout_NCHWC32T8Sparse:
    case DataLayout_NDCHWC16T8:
    case DataLayout_NDCHWC32T8:
    case DataLayout_NCHWC16T1:
    case DataLayout_NCHWC16T1Sparse:
    case DataLayout_NCHWC32T1:
    case DataLayout_NCHWC32T1Sparse:
    case DataLayout_NDCHWC16T1:
    case DataLayout_NDCHWC32T1:
        return c();
    default:
        return 0;
    }
}

tongue::Expr SymbolShape::size() const
{
    tongue::Expr s = 1;
    for (auto d : m_shape)
    {
        s = s * d;
    }
    // sparse
    if (m_layout >= 32)
    {
        // tag=ceil(w,8)*h*ceil(c,16)/16)
        auto tag = s / w() * (((w() + 7) / 8) * 8);
        tag = tag / c_total() * ((c_total() + 31) / 32);
        s = s + tag / 8;
    }
    return s;
}

const std::string SymbolShape::to_string() const
{
    std::ostringstream oss;
    oss << "[";
    if (m_shape.size() > 0)
    {
        for (unsigned int i = 0; i < m_shape.size() - 1; ++i)
        {
            oss << m_shape[i]->to_string() << ",";
        }
        oss << m_shape[m_shape.size() - 1]->to_string();
    }
    oss << "]";
    return oss.str();
}

bool SymbolShape::is_constant() const
{
    for (auto d : m_shape)
    {
        if (!d->is_constant())
        {
            return false;
        }
    }
    return true;
}

std::map<std::string, DataLayout> &get_datalayout_map()
{
    static std::map<std::string, DataLayout> STRING_TO_DATALAYOUT_MAP =
        {
            {"Flat", DataLayout_Flat},
            {"NCHW", DataLayout_NCHW},
            {"NHWC", DataLayout_NHWC},
            {"NDHWC", DataLayout_NDHWC},
            {"NCHWC4", DataLayout_NCHWC4},
            {"NCHWC8", DataLayout_NCHWC8},
            {"NCHWC16", DataLayout_NCHWC16},
            {"NCHWC32", DataLayout_NCHWC32},
            {"NDCHWC16", DataLayout_NDCHWC16},
            {"NDCHWC32", DataLayout_NDCHWC32},
            {"NCDHWC16", DataLayout_NCDHWC16},
            {"NCDHWC32", DataLayout_NCDHWC32},
            {"NCHWC16T8", DataLayout_NCHWC16T8},
            {"NCHWC32T8", DataLayout_NCHWC32T8},
            {"NCHWC16T1", DataLayout_NCHWC16T1},
            {"NCHWC32T1", DataLayout_NCHWC32T1},
            {"NHWCSparse", DataLayout_NHWCSparse},
            {"NCHWC16Sparse", DataLayout_NCHWC16Sparse},
            {"NCHWC32Sparse", DataLayout_NCHWC32Sparse},
            {"NCHWC16T8Sparse", DataLayout_NCHWC16T8Sparse},
            {"NCHWC32T8Sparse", DataLayout_NCHWC32T8Sparse},
            {"NCHWC16T1Sparse", DataLayout_NCHWC16T1Sparse},
            {"NCHWC32T1Sparse", DataLayout_NCHWC32T1Sparse},
            {"NDCHWC16T8", DataLayout_NDCHWC16T8},
            {"NDCHWC32T8", DataLayout_NDCHWC32T8},
            {"NDCHWC16T1", DataLayout_NDCHWC16T1},
            {"NDCHWC32T1", DataLayout_NDCHWC32T1},
        };
    return STRING_TO_DATALAYOUT_MAP;
}
std::ostream &operator<<(std::ostream &os, const DataLayout layout)
{
    return os << datalayout_to_string(layout);
}
std::string datalayout_to_string(DataLayout layout)
{
    static std::map<DataLayout, std::string> m_map = {
        {DataLayout_UnSupport, "UnSupport"},
        {DataLayout_Flat, "Flat"},
        {DataLayout_NCHW, "NCHW"},
        {DataLayout_NHWC, "NHWC"},
        {DataLayout_NDHWC, "NDHWC"},
        {DataLayout_NCHWC4, "NCHWC4"},
        {DataLayout_NCHWC8, "NCHWC8"},
        {DataLayout_NCHWC16, "NCHWC16"},
        {DataLayout_NCHWC32, "NCHWC32"},
        {DataLayout_NDCHWC16, "NDCHWC16"},
        {DataLayout_NDCHWC32, "NDCHWC32"},
        {DataLayout_NCDHWC16, "NCDHWC16"},
        {DataLayout_NCDHWC32, "NCDHWC32"},
        {DataLayout_NCHWC16T8, "NCHWC16T8"},
        {DataLayout_NCHWC32T8, "NCHWC32T8"},
        {DataLayout_NCHWC16T1, "NCHWC16T1"},
        {DataLayout_NCHWC32T1, "NCHWC32T1"},
        {DataLayout_NHWCSparse, "NHWCSparse"},
        {DataLayout_NCHWC16Sparse, "NCHWC16Sparse"},
        {DataLayout_NCHWC32Sparse, "NCHWC32Sparse"},
        {DataLayout_NCHWC16T8Sparse, "NCHWC16T8Sparse"},
        {DataLayout_NCHWC32T8Sparse, "NCHWC32T8Sparse"},
        {DataLayout_NCHWC16T1Sparse, "NCHWC16T1Sparse"},
        {DataLayout_NCHWC32T1Sparse, "NCHWC32T1Sparse"},
        {DataLayout_NDCHWC16T8, "NDCHWC16T8"},
        {DataLayout_NDCHWC32T8, "NDCHWC32T8"},
        {DataLayout_NDCHWC16T1, "NDCHWC16T1"},
        {DataLayout_NDCHWC32T1, "NDCHWC32T1"},
    };
    if (!m_map.count(layout))
    {
        return "unkown";
    }
    return m_map[layout];
}

DataLayout string_to_datalayout(const std::string &str)
{
    if (!get_datalayout_map().count(str))
    {
        return DataLayout_UnSupport;
    }
    return get_datalayout_map().at(str);
}

// TensorShape::TensorShape(int d0, DataLayout dl)
// {
//     m_shape.push_back(d0);
//     m_symbol.m_layout = dl;
//     check_layout(true);
// }

TensorShape::TensorShape(const TensorShape &ts)
{
    m_shape = ts.m_shape;
    m_strides = ts.m_strides;
    m_symbol = ts.m_symbol;
}

TensorShape &TensorShape::operator=(const TensorShape &other)
{
    if (this != &other)
    {
        m_shape = other.m_shape;
        m_strides = other.m_strides;
        m_symbol = other.m_symbol;
    }
    return *this;
}

static std::vector<uint32_t> get_compact_strides(const std::vector<int> &m_shape, DataLayout layout)
{
    auto fn = [&m_shape]() {
        std::vector<uint32_t> ret;
        ret.resize(m_shape.size(), 1);
        uint32_t expected_stride = 1;
        for (auto i = m_shape.size(); i != 0; i--)
        {
            int64_t k = i - 1;
            ret[k] = expected_stride;
            expected_stride = expected_stride * m_shape[k];
        }
        return ret;
    };

    auto fn_align_c = [&m_shape, &fn](int align_c) {
        uint32_t dim = m_shape.size();
        if (dim == 0 || m_shape[dim - 1] != align_c)
        {
            return std::vector<uint32_t>();
        }
        return fn();
    };

    switch (layout)
    {
    case DataLayout_NCHWC4:
        return fn_align_c(4);
    case DataLayout_NCHWC8:
        return fn_align_c(8);
    case DataLayout_NCHWC16:
    case DataLayout_NDCHWC16:
    case DataLayout_NCDHWC16:
        return fn_align_c(16);
    case DataLayout_NCHWC32:
    case DataLayout_NDCHWC32:
    case DataLayout_NCDHWC32:
        return fn_align_c(32);
    case DataLayout_UnSupport:
    case DataLayout_Flat:
    case DataLayout_NCHW:
    case DataLayout_NHWC:
    case DataLayout_NDHWC:
        return fn();
    case DataLayout_NCHWC16T8:
    case DataLayout_NCHWC32T8:
    case DataLayout_NCHWC16T1:
    case DataLayout_NCHWC32T1:
    case DataLayout_NHWCSparse:
    case DataLayout_NCHWC16Sparse:
    case DataLayout_NCHWC32Sparse:
    case DataLayout_NCHWC16T8Sparse:
    case DataLayout_NCHWC32T8Sparse:
    case DataLayout_NCHWC16T1Sparse:
    case DataLayout_NCHWC32T1Sparse:
    case DataLayout_NDCHWC16T8:
    case DataLayout_NDCHWC32T8:
    case DataLayout_NDCHWC16T1:
    case DataLayout_NDCHWC32T1:
    default:
        return std::vector<uint32_t>();
    }
}

TensorShape::TensorShape(const std::string &str)
{
    if (str.empty())
    {
        return;
    }
    std::string s = str;
    s = utils::trim(s);
    s.erase(0, s.find_first_not_of("["));
    s.erase(s.find_last_not_of("]") + 1);
    std::istringstream iss(s);
    for (int i; iss >> i;)
    {
        m_shape.push_back(i);
        if (iss.peek() == ',')
        {
            iss.ignore();
        }
    }
    m_strides = get_compact_strides(m_shape, DataLayout_Flat);
    for (auto d : m_shape)
    {
        m_symbol.m_shape.push_back(d);
    }
}
TensorShape::TensorShape(std::initializer_list<int> l, DataLayout dl)
{
    m_shape.resize(l.size());
    for (auto it = l.begin(); it != l.end(); ++it)
    {
        m_shape[it - l.begin()] = *it;
    }
    set_layout(dl);
    check_layout(true);
    m_strides = get_compact_strides(m_shape, dl);
    for (auto d : m_shape)
    {
        m_symbol.m_shape.push_back(d);
    }
}
TensorShape::TensorShape(const std::vector<int> &shape, DataLayout dl)
{
    m_shape = shape;
    set_layout(dl);
    check_layout(true);
    m_strides = get_compact_strides(m_shape, dl);
    for (auto d : m_shape)
    {
        m_symbol.m_shape.push_back(d);
    }
}

int &TensorShape::operator[](int index)
{
    if (index < 0)
    {
        index = m_shape.size() + index;
    }
    aipu_assert(index < static_cast<int>(m_shape.size()));
    m_shape_change = true;
    return m_shape[index];
}
int TensorShape::operator[](int index) const
{
    if (index < 0)
    {
        index = m_shape.size() + index;
    }
    aipu_assert(index < static_cast<int>(m_shape.size()));
    return m_shape[index];
}

bool TensorShape::operator==(const TensorShape &other) const
{
    bool ret = false;
    if (m_symbol.m_layout == other.m_symbol.m_layout ||
        m_symbol.m_layout == DataLayout_Flat ||
        other.m_symbol.m_layout == DataLayout_Flat)
    {
        ret = true;
    }
    if (m_shape.size() != other.m_shape.size())
    {
        return false;
    }
    for (unsigned int i = 0; i < m_shape.size(); ++i)
    {
        ret = ret && m_shape[i] == other.m_shape[i];
    }
    return ret;
}

bool TensorShape::operator>(const TensorShape &other) const
{
    bool ret = false;
    if (m_symbol.m_layout == other.m_symbol.m_layout ||
        m_symbol.m_layout == DataLayout_Flat ||
        other.m_symbol.m_layout == DataLayout_Flat)
    {
        ret = true;
    }
    if (m_shape.size() != other.m_shape.size())
    {
        return false;
    }
    ret = ret && this->size() > other.size();
    for (unsigned int i = 0; i < m_shape.size(); ++i)
    {
        ret = ret && m_shape[i] >= other.m_shape[i];
    }
    return ret;
}

bool TensorShape::operator<(const TensorShape &other) const
{
    bool ret = false;
    if (m_symbol.m_layout == other.m_symbol.m_layout ||
        m_symbol.m_layout == DataLayout_Flat ||
        other.m_symbol.m_layout == DataLayout_Flat)
    {
        ret = true;
    }
    if (m_shape.size() != other.m_shape.size())
    {
        return false;
    }
    ret = ret && this->size() < other.size();
    for (unsigned int i = 0; i < m_shape.size(); ++i)
    {
        ret = ret && m_shape[i] <= other.m_shape[i];
    }
    return ret;
}
bool TensorShape::operator!=(const TensorShape &other) const
{
    return !(*this == other);
}

std::vector<int>::iterator TensorShape::begin()
{
    return m_shape.begin();
}
std::vector<int>::iterator TensorShape::end()
{
    return m_shape.end();
}

std::vector<int>::const_iterator TensorShape::begin() const
{
    return m_shape.begin();
}
std::vector<int>::const_iterator TensorShape::end() const
{
    return m_shape.end();
}

int &TensorShape::h()
{
    aipu_assert(dim() > m_symbol.m_idx.h);
    m_shape_change = true;
    return m_shape[m_symbol.m_idx.h];
}
int &TensorShape::w()
{
    aipu_assert(dim() > m_symbol.m_idx.w);
    m_shape_change = true;
    return m_shape[m_symbol.m_idx.w];
}
int &TensorShape::c()
{
    if (dim() == 3)
    {
        m_shape_change = true;
        return m_shape[m_symbol.m_idx.c - 1];
    }
    else
    {
        aipu_assert(dim() > m_symbol.m_idx.c);
        m_shape_change = true;
        return m_shape[m_symbol.m_idx.c];
    }
}
int &TensorShape::n()
{
    aipu_assert(dim() > m_symbol.m_idx.n);
    m_shape_change = true;
    return m_shape[m_symbol.m_idx.n];
}
int TensorShape::h() const
{
    aipu_assert(dim() > m_symbol.m_idx.h);
    return m_shape[m_symbol.m_idx.h];
}
int TensorShape::w() const
{
    aipu_assert(dim() > m_symbol.m_idx.w);
    return m_shape[m_symbol.m_idx.w];
}

int TensorShape::c() const
{
    if (dim() == 3)
    {
        // layout must not be NCHW, FLAT/UNKOWN/NHWC ok
        return m_shape[m_symbol.m_idx.c - 1];
    }
    else if (dim() == 2)
    {
        // layout must not be NCHW, FLAT/UNKOWN/NHWC ok
        return m_shape[m_symbol.m_idx.c - 2];
    }
    else
    {
        aipu_assert(dim() > m_symbol.m_idx.c);
        return m_shape[m_symbol.m_idx.c];
    }
}

int TensorShape::n() const
{
    return m_shape[m_symbol.m_idx.n];
}
int TensorShape::size() const
{
    int s = 1;
    for (auto d : m_shape)
    {
        s *= d;
    }
    // sparse
    if (m_symbol.m_layout >= 32)
    {
        // tag=ceil(w,8)*h*ceil(c,16)/16)
        auto tag = s / w() * (((w() + 7) / 8) * 8);
        tag = tag / c_total() * ((c_total() + 31) / 32);
        s = s + tag / 8;
    }
    return s;
}
int TensorShape::dim() const
{
    return m_shape.size();
}

DataLayout TensorShape::layout() const
{
    return m_symbol.m_layout;
}

TensorShape::operator std::vector<int>() const
{
    return m_shape;
}

TensorShape::operator std::vector<uint32_t>() const
{
    std::vector<uint32_t> ret;
    for (auto s : m_shape)
    {
        ret.push_back(s);
    }
    return ret;
}

void TensorShape::push_back(int d)
{
    m_shape.push_back(d);
    m_strides.push_back(1); // assume the memory is continuous
    for (uint32_t i = 0; i < m_strides.size() - 1; ++i)
    {
        m_strides[i] *= d;
    }
    m_symbol.m_shape.push_back(d);
}

bool TensorShape::reshape(std::initializer_list<int> l)
{
    TensorShape ts(l, m_symbol.m_layout);
    auto tsize = ts.size();
    if (tsize < 0)
    {
        int idx = -1;
        for (int i = 0; i < ts.dim(); ++i)
        {
            if (ts[i] == -1)
            {
                if (idx == -1)
                {
                    idx = i;
                }
                else
                {
                    idx = -2;
                    break;
                }
            }
        }
        if (idx >= 0)
        {
            ts[idx] = size() / (-tsize);
        }
    }
    if (ts.size() == size())
    {
        (*this) = ts;
        check_layout(true);
        return true;
    }
    return false;
}
bool TensorShape::reshape(std::vector<int> l)
{
    TensorShape ts(l, m_symbol.m_layout);
    auto tsize = ts.size();
    if (tsize < 0)
    {
        int idx = -1;
        for (int i = 0; i < ts.dim(); ++i)
        {
            if (ts[i] == -1)
            {
                if (idx == -1)
                {
                    idx = i;
                }
                else
                {
                    idx = -2;
                    break;
                }
            }
        }
        if (idx >= 0)
        {
            ts[idx] = size() / (-tsize);
        }
    }
    if (ts.size() == size())
    {
        (*this) = ts;
        check_layout(true);
        return true;
    }
    return false;
}

const std::string TensorShape::to_string() const
{
    std::ostringstream oss;
    oss << "[";
    if (m_shape.size() > 0)
    {
        for (unsigned int i = 0; i < m_shape.size() - 1; ++i)
        {
            oss << m_shape[i] << ",";
        }
        oss << m_shape[m_shape.size() - 1];
    }
    oss << "]";
    return oss.str();
}

void TensorShape::set_layout(DataLayout lo)
{
    m_symbol.m_layout = lo;
    switch (lo)
    {
    case DataLayout_NCHW:
    case DataLayout_NCHWC4:
    case DataLayout_NCHWC8:
    case DataLayout_NCHWC16:
    case DataLayout_NCHWC32:
    case DataLayout_NCHWC16Sparse:
    case DataLayout_NCHWC32Sparse:
        m_symbol.m_idx.n = 0;
        m_symbol.m_idx.c = 1;
        m_symbol.m_idx.h = 2;
        m_symbol.m_idx.w = 3;
        m_symbol.m_idx.d = -1;
        break;
    case DataLayout_NHWC:
    case DataLayout_NHWCSparse:
    case DataLayout_NCHWC32T8:
    case DataLayout_NCHWC32T1:
    case DataLayout_NCHWC32T8Sparse:
    case DataLayout_NCHWC32T1Sparse:
    case DataLayout_NCHWC16T8:
    case DataLayout_NCHWC16T1:
    case DataLayout_NCHWC16T8Sparse:
    case DataLayout_NCHWC16T1Sparse:
        m_symbol.m_idx.n = 0;
        m_symbol.m_idx.h = 1;
        m_symbol.m_idx.w = 2;
        m_symbol.m_idx.c = 3;
        m_symbol.m_idx.d = -1;
        break;
    case DataLayout_NDHWC:
    case DataLayout_NDCHWC16T8:
    case DataLayout_NDCHWC32T8:
    case DataLayout_NDCHWC16T1:
    case DataLayout_NDCHWC32T1:
        m_symbol.m_idx.n = 0;
        m_symbol.m_idx.d = 1;
        m_symbol.m_idx.h = 2;
        m_symbol.m_idx.w = 3;
        m_symbol.m_idx.c = 4;
        break;
    case DataLayout_NDCHWC16:
    case DataLayout_NDCHWC32:
        m_symbol.m_idx.n = 0;
        m_symbol.m_idx.d = 1;
        m_symbol.m_idx.c = 2;
        m_symbol.m_idx.h = 3;
        m_symbol.m_idx.w = 4;
        break;
    case DataLayout_NCDHWC16:
    case DataLayout_NCDHWC32:
        m_symbol.m_idx.n = 0;
        m_symbol.m_idx.c = 1;
        m_symbol.m_idx.d = 2;
        m_symbol.m_idx.h = 3;
        m_symbol.m_idx.w = 4;
        break;
    default:
        break;
    }
    // m_strides = get_compact_strides(m_shape, m_symbol.m_layout);
}

bool TensorShape::check_layout(bool is_correct) const
{
    // NHWC/NHCW dim ==4
    // NCHWCx dim==5
    bool ret = false;
    if (m_symbol.m_layout == DataLayout_NHWC || m_symbol.m_layout == DataLayout_NHWCSparse ||
        m_symbol.m_layout == DataLayout_NCHW || m_symbol.m_layout == DataLayout_NCHWC16T1 ||
        m_symbol.m_layout == DataLayout_NCHWC32T1 || m_symbol.m_layout == DataLayout_NCHWC16T1Sparse ||
        m_symbol.m_layout == DataLayout_NCHWC32T1Sparse)
    {
        ret = dim() == 4;
    }
    else if (m_symbol.m_layout == DataLayout_NCHWC16T8 || m_symbol.m_layout == DataLayout_NCHWC32T8 ||
             m_symbol.m_layout == DataLayout_NCHWC16T8Sparse || m_symbol.m_layout == DataLayout_NCHWC32T8Sparse)
    {
        ret = dim() == 4 && c_tail() % 8 == 0;
    }
    else if (m_symbol.m_layout == DataLayout_NDHWC || m_symbol.m_layout == DataLayout_NDCHWC16T1 ||
             m_symbol.m_layout == DataLayout_NDCHWC32T1)
    {
        ret = dim() == 5;
    }
    else if (m_symbol.m_layout == DataLayout_NDCHWC16T8 || m_symbol.m_layout == DataLayout_NDCHWC32T8)
    {
        ret = dim() == 5 && c_tail() % 8 == 0;
    }
    else if ((m_symbol.m_layout >= DataLayout_NCHWC4 && m_symbol.m_layout <= DataLayout_NCHWC32) ||
             (m_symbol.m_layout >= 32 + DataLayout_NCHWC4 && m_symbol.m_layout <= DataLayout_NCHWC32Sparse))
    {
        ret = (dim() == 5) && ((uint32_t)dim() == m_shape.size()) && (c_prime() == m_shape[4]);
    }
    else if (m_symbol.m_layout == DataLayout_NDCHWC16 || m_symbol.m_layout <= DataLayout_NDCHWC32 ||
             m_symbol.m_layout == DataLayout_NCDHWC16 || m_symbol.m_layout <= DataLayout_NCDHWC32)
    {
        ret = dim() == 6;
    }
    if (is_correct && !ret)
    {
        m_symbol.m_layout = DataLayout_Unkown;
    }
    return ret;
}

bool TensorShape::is_continuous() const
{
    std::set<DataLayout> continuous_layout = {
        DataLayout_NCHWC16T8,
        DataLayout_NCHWC32T8,
        DataLayout_NCHWC16T1,
        DataLayout_NCHWC32T1,
        DataLayout_NHWCSparse,
        DataLayout_NCHWC16Sparse,
        DataLayout_NCHWC32Sparse,
        DataLayout_NCHWC16T8Sparse,
        DataLayout_NCHWC32T8Sparse,
        DataLayout_NCHWC16T1Sparse,
        DataLayout_NCHWC32T1Sparse,
        DataLayout_NDCHWC16T8,
        DataLayout_NDCHWC32T8,
        DataLayout_NDCHWC16T1,
        DataLayout_NDCHWC32T1,
    };
    if (continuous_layout.count(m_symbol.m_layout) != 0)
    {
        return true;
    }

    auto compact_strides = get_compact_strides(m_shape, m_symbol.m_layout);
    if (compact_strides.size() == 0)
    {
        return false;
    }
    return compact_strides == m_strides;
}

std::vector<uint32_t> TensorShape::strides() const
{
    if (m_shape_change)
    {
        m_shape_change = false;
        if (check_strides(m_strides))
        {
            return m_strides;
        }
        m_strides = get_compact_strides(m_shape, m_symbol.m_layout);
    }
    return m_strides;
}

std::vector<uint32_t> TensorShape::reset_strides()
{
    m_strides = get_compact_strides(m_shape, m_symbol.m_layout);
    return m_strides;
}

bool TensorShape::check_strides(const std::vector<uint32_t> &strides) const
{
    if (m_symbol.m_layout != DataLayout_Flat && !check_layout(true))
    {
        return false;
    }

    if (m_symbol.m_layout >= DataLayout_NCHWC16T8 && strides.size() != 0)
    {
        return false;
    }

    if (m_shape.size() != strides.size())
    {
        return false;
    }

    uint32_t lane = 1;
    for (uint32_t i = m_shape.size(); i != 0; i--)
    {
        uint32_t index = i - 1;
        if (strides[index] < lane)
        {
            return false;
        }
        lane = strides[index] * m_shape[index];
    }
    return true;
}

bool TensorShape::change_layout(DataLayout layout)
{
    if (m_symbol.m_layout == layout ||
        m_symbol.m_layout == DataLayout_Unkown ||
        layout == DataLayout_Unkown ||
        !check_layout(true))
    {
        return false;
    }
    int d = 0;
    tongue::Expr symbol_d = 0;
    if (m_symbol.m_idx.d != -1)
    {
        d = m_shape[m_symbol.m_idx.d];
        symbol_d = m_symbol.m_shape[m_symbol.m_idx.d];
    }

    if (((d == 0) && (layout == DataLayout_NDHWC ||
                      layout == DataLayout_NDCHWC32T8 ||
                      layout == DataLayout_NDCHWC16T8 ||
                      layout == DataLayout_NDCHWC32 ||
                      layout == DataLayout_NDCHWC16)) ||
        ((d != 0) && (layout == DataLayout_NHWC ||
                      layout == DataLayout_NHWCSparse ||
                      layout == DataLayout_NCHWC32T8 ||
                      layout == DataLayout_NCHWC32T8Sparse ||
                      layout == DataLayout_NCHWC16T8 ||
                      layout == DataLayout_NCHWC16T8Sparse ||
                      layout == DataLayout_NCHWC32T1 ||
                      layout == DataLayout_NCHWC32T1Sparse ||
                      layout == DataLayout_NCHWC16T1 ||
                      layout == DataLayout_NCHWC16T1Sparse ||
                      layout == DataLayout_NCHWC32 ||
                      layout == DataLayout_NCHWC16 ||
                      layout == DataLayout_NCHW)))
    {
        return false; // D dimension cannot be created or deleted here
    }

    // backup
    auto backup = *this;
    auto backup_strides = m_strides;
    int origin_c = c_total();
    auto symbol_c = m_symbol.c_total();

    if (layout == DataLayout_NHWC ||
        layout == DataLayout_NHWCSparse ||
        layout == DataLayout_NCHWC32T8 ||
        layout == DataLayout_NCHWC32T8Sparse ||
        layout == DataLayout_NCHWC16T8 ||
        layout == DataLayout_NCHWC16T8Sparse ||
        layout == DataLayout_NCHWC32T1 ||
        layout == DataLayout_NCHWC32T1Sparse ||
        layout == DataLayout_NCHWC16T1 ||
        layout == DataLayout_NCHWC16T1Sparse)
    {
        m_shape = {n(), h(), w(), origin_c};
        m_symbol.m_shape = {m_symbol.n(), m_symbol.h(), m_symbol.w(), symbol_c};
    }
    else if (layout == DataLayout_NCHW)
    {
        m_shape = {n(), origin_c, h(), w()};
        m_symbol.m_shape = {m_symbol.n(), symbol_c, m_symbol.h(), m_symbol.w()};
    }
    else if (layout == DataLayout_NDHWC ||
             layout == DataLayout_NDCHWC32T8 ||
             layout == DataLayout_NDCHWC16T8 ||
             layout == DataLayout_NDCHWC32T1 ||
             layout == DataLayout_NDCHWC16T1)
    {
        m_shape = {n(), d, h(), w(), origin_c};
        m_symbol.m_shape = {m_symbol.n(), symbol_d, m_symbol.h(), m_symbol.w(), symbol_c};
    }
    else // NCHWCx,NDCHWCx
    {
        m_symbol.m_layout = layout;
        auto c_ = c_prime();
        if (c_ == 1)
        {
            ERROR() << "The c_prime of layout " << (int)m_symbol.m_layout << "should not be 1";
        }
        if (origin_c % c_ != 0)
        {
            // restore
            *this = backup;
            return false;
        }
        if (d != 0)
        {
            m_shape = {n(), d, origin_c / c_, h(), w(), c_};
            m_symbol.m_shape = {m_symbol.n(), symbol_d, symbol_c.floordiv(c_), m_symbol.h(), m_symbol.w(), c_};
        }
        else
        {
            m_shape = {n(), origin_c / c_, h(), w(), c_};
            m_symbol.m_shape = {m_symbol.n(), symbol_c.floordiv(c_), m_symbol.h(), m_symbol.w(), c_};
        }
    }
    set_layout(layout);
    m_strides = get_compact_strides(m_shape, m_symbol.m_layout);
    auto shape_size = [](const std::vector<int> &s) -> size_t {
        size_t k = 1;
        for (auto i : s)
        {
            k *= i;
        }
        return k;
    };
    if (!check_layout(true) || shape_size(backup.m_shape) != shape_size(m_shape))
    {
        *this = backup;
        m_strides = backup_strides;
        return false;
    }
    // Currently we give a compact stride for new layout
    return true;
}

std::ostream &operator<<(std::ostream &os, const SymbolShape &shape)
{
    return os << shape.to_string();
}

std::ostream &operator<<(std::ostream &os, const TensorShape &shape)
{
    return os << shape.to_string();
}

void TensorShape::set_strides(const std::vector<uint32_t> &strides)
{
    if (check_strides(strides))
    {
        m_strides = strides;
    }
}

int TensorShape::c_prime() const
{
    switch (m_symbol.m_layout)
    {
    case DataLayout_NCHW:
    case DataLayout_NHWC:
    case DataLayout_NHWCSparse:
        return 1;
    case DataLayout_NCHWC4:
        return 4;
    case DataLayout_NCHWC8:
        return 8;
    case DataLayout_NCHWC16:
    case DataLayout_NCHWC16Sparse:
    case DataLayout_NCHWC16T8:
    case DataLayout_NCHWC16T8Sparse:
    case DataLayout_NCHWC16T1:
    case DataLayout_NCHWC16T1Sparse:
    case DataLayout_NDCHWC16:
    case DataLayout_NDCHWC16T8:
    case DataLayout_NDCHWC16T1:
    case DataLayout_NCDHWC16:
        return 16;
    case DataLayout_NCHWC32:
    case DataLayout_NCHWC32Sparse:
    case DataLayout_NCHWC32T8:
    case DataLayout_NCHWC32T8Sparse:
    case DataLayout_NCHWC32T1:
    case DataLayout_NCHWC32T1Sparse:
    case DataLayout_NDCHWC32:
    case DataLayout_NDCHWC32T8:
    case DataLayout_NDCHWC32T1:
    case DataLayout_NCDHWC32:
        return 32;
    default:
        return 1;
    }
}
int TensorShape::c_tail() const
{
    switch (m_symbol.m_layout)
    {
    case DataLayout_NCHW:
    case DataLayout_NHWC:
    case DataLayout_NDHWC:
    case DataLayout_NHWCSparse:
    case DataLayout_NCHWC4:
    case DataLayout_NCHWC8:
    case DataLayout_NCHWC16:
    case DataLayout_NCHWC16Sparse:
    case DataLayout_NCHWC32:
    case DataLayout_NCHWC32Sparse:
    case DataLayout_NDCHWC16:
    case DataLayout_NDCHWC32:
    case DataLayout_NCDHWC16:
    case DataLayout_NCDHWC32:
        return 0;
    case DataLayout_NCHWC16T8:
    case DataLayout_NCHWC16T8Sparse:
    case DataLayout_NCHWC32T8:
    case DataLayout_NCHWC32T8Sparse:
    case DataLayout_NDCHWC16T8:
    case DataLayout_NDCHWC32T8:
    case DataLayout_NCHWC16T1:
    case DataLayout_NCHWC16T1Sparse:
    case DataLayout_NCHWC32T1:
    case DataLayout_NCHWC32T1Sparse:
    case DataLayout_NDCHWC16T1:
    case DataLayout_NDCHWC32T1:
        return c_total() % c_prime();
    default:
        return 0;
    }
}

int TensorShape::c_total() const
{
    switch (m_symbol.m_layout)
    {
    case DataLayout_NCHW:
    case DataLayout_NHWC:
    case DataLayout_NDHWC:
    case DataLayout_NHWCSparse:
    case DataLayout_NCHWC4:
    case DataLayout_NCHWC8:
    case DataLayout_NCHWC16:
    case DataLayout_NCHWC16Sparse:
    case DataLayout_NCHWC32:
    case DataLayout_NCHWC32Sparse:
    case DataLayout_NDCHWC16:
    case DataLayout_NDCHWC32:
    case DataLayout_NCDHWC16:
    case DataLayout_NCDHWC32:
        return c_prime() * c();
    case DataLayout_NCHWC16T8:
    case DataLayout_NCHWC16T8Sparse:
    case DataLayout_NCHWC32T8:
    case DataLayout_NCHWC32T8Sparse:
    case DataLayout_NDCHWC16T8:
    case DataLayout_NDCHWC32T8:
    case DataLayout_NCHWC16T1:
    case DataLayout_NCHWC16T1Sparse:
    case DataLayout_NCHWC32T1:
    case DataLayout_NCHWC32T1Sparse:
    case DataLayout_NDCHWC16T1:
    case DataLayout_NDCHWC32T1:
        return c();
    default:
        return 0;
    }
}

std::function<int(int)> get_datalayout_axis_map(DataLayout from, DataLayout to)
{
    static std::map<DataLayout, int> dimensions = {
        {DataLayout_NCHW, 4},
        {DataLayout_NCHWC4, 5},
        {DataLayout_NCHWC8, 5},
        {DataLayout_NCHWC16, 5},
        {DataLayout_NCHWC32, 5},
        {DataLayout_NCHWC16Sparse, 5},
        {DataLayout_NCHWC32Sparse, 5},
        {DataLayout_NHWC, 4},
        {DataLayout_NHWCSparse, 4},
        {DataLayout_NCHWC32T8, 4},
        {DataLayout_NCHWC32T8Sparse, 4},
        {DataLayout_NCHWC16T8, 4},
        {DataLayout_NCHWC16T8Sparse, 4},
        {DataLayout_NCHWC32T1, 4},
        {DataLayout_NCHWC32T1Sparse, 4},
        {DataLayout_NCHWC16T1, 4},
        {DataLayout_NCHWC16T1Sparse, 4},
        {DataLayout_NDHWC, 5},
        {DataLayout_NDCHWC16T8, 5},
        {DataLayout_NDCHWC32T8, 5},
        {DataLayout_NDCHWC16T1, 5},
        {DataLayout_NDCHWC32T1, 5},
        {DataLayout_NDCHWC16, 6},
        {DataLayout_NDCHWC32, 6},
        {DataLayout_Flat, 0},
    };
    std::vector<int> _t;
    for (int i = 0; i < dimensions[from]; ++i)
    {
        _t.push_back(32);
    }
    TensorShape ts(_t);
    aipu_assert(ts.change_layout(from));
    TensorShape nts = ts;
    if (nts.change_layout(to)) // allow to convert
    {
        auto ori = ts.indexing();
        auto nidx = nts.indexing();
        auto dim = dimensions[from];
        return [ori, nidx, dim](int i) {
            if (i < 0)
            {
                i += dim;
            }
            if (i == ori.n)
            {
                return nidx.n;
            }
            else if (i == ori.c)
            {
                return nidx.c;
            }
            else if (i == ori.h)
            {
                return nidx.h;
            }
            else if (i == ori.w)
            {
                return nidx.w;
            }
            else if (i == ori.d)
            {
                return nidx.d;
            }
            // unkown
            return i;
        };
    }
    else
    {
        return [](int i) { return i; };
    }
}
bool TensorShape::broadcastable(const TensorShape &t) const
{
    uint32_t min_dim = MIN(m_shape.size(), t.m_shape.size());
    for (uint32_t i = 0; i < min_dim; ++i)
    {
        const auto a = m_shape[m_shape.size() - 1 - i];
        const auto b = t.m_shape[t.m_shape.size() - 1 - i];
        if (a != b && a != 1 && b != 1)
        {
            return false;
        }
    }
    return true;
}

TensorShape TensorShape::broadcasting(const TensorShape &t) const
{
    uint32_t min_dim = MIN(m_shape.size(), t.m_shape.size());
    uint32_t max_dim = t.m_shape.size();
    std::vector<int> ret;
    if (m_shape.size() > t.m_shape.size())
    {
        ret = m_shape;
        max_dim = m_shape.size();
    }
    else
    {
        ret = t.m_shape;
    }
    for (uint32_t i = 0; i < min_dim; ++i)
    {
        const auto a = m_shape[m_shape.size() - 1 - i];
        const auto b = t.m_shape[t.m_shape.size() - 1 - i];
        if (a != b && a != 1 && b != 1)
        {
            return TensorShape();
        }
        ret[max_dim - 1 - i] = MAX(a, b);
    }
    return TensorShape(ret);
}

std::vector<uint32_t> offset_to_index(size_t offset, const std::vector<uint32_t> &stride)
{
    std::vector<uint32_t> ret(stride.size());
    for (uint32_t i = 0; i < stride.size(); ++i)
    {
        ret[i] = offset / stride[i];
        offset = offset % stride[i];
    }
    return ret;
}

uint32_t index_to_offset(const std::vector<uint32_t> &index, const std::vector<uint32_t> &stride)
{
    uint32_t offset = 0;
    for (uint32_t i = 0; i < stride.size(); ++i)
    {
        offset += index[i] * stride[i];
    }
    return offset;
}

TensorShape get_base_shape(const TensorShape &shape)
{
    static const std::map<DataLayout, DataLayout> layout_map = {
        {DataLayout_NCHWC4, DataLayout_NHWC},
        {DataLayout_NCHWC8, DataLayout_NHWC},
        {DataLayout_NCHWC16, DataLayout_NHWC},
        {DataLayout_NCHWC32, DataLayout_NHWC},
        {DataLayout_NDCHWC16, DataLayout_NDHWC},
        {DataLayout_NDCHWC32, DataLayout_NDHWC},
        {DataLayout_NCDHWC16, DataLayout_NDHWC},
        {DataLayout_NCDHWC32, DataLayout_NDHWC},
    };

    TensorShape ret = shape;
    if (layout_map.count(shape.layout()))
    {
        ret.change_layout(layout_map.at(shape.layout()));
    }
    if (ret.dim() == 0) // [] -> [1]
    {
        ret = TensorShape({1}, ret.layout());
    }
    return ret;
}
} // namespace aipubt

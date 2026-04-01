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
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-11-06 15:00:53
 */
#include "core/compute_tensor.h"
#include "core/compute/broadcast_op.h"
#include "core/compute/cast.h"
#include "core/compute/eltwise_op.h"
#include "core/compute/permute.h"
#include "core/compute/reduce.h"
#include "core/compute/sgemm.h"
#include "core/quantization.h"
#include "utils/utils.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>
namespace aipubt
{

#define ELTWISE_FUNC_FOR_COMPUTETENSOR(name)                                                                 \
    ComputeTensor ComputeTensor::name() const                                                                \
    {                                                                                                        \
        auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);                                               \
        m_tensor->make_memory_continuous();                                                                  \
        if (eltwise_##name##_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size())) \
        {                                                                                                    \
            return ret;                                                                                      \
        }                                                                                                    \
        return nullptr;                                                                                      \
    }

template <typename Container>
bool inline contains(Container const &c, typename Container::const_reference v)
{
    return std::find(c.begin(), c.end(), v) != c.end();
}

static bool type_casting(Tensor::Dtype t0, Tensor::Dtype t1, Tensor::Dtype &t)
{
    if (t0 == t1)
    {
        t = t1;
        return true;
    }

    using Dtype = Tensor::Dtype;
    int big = t0;
    int small = t1;
    if (small > big)
    {
        big = t1;
        small = t0;
    }
    if (big == Dtype::FP4E2M1FN) // fp4 can cast into all type, and fp4 is the small type
    {
        t = (Dtype)small;
        return true;
    }
    if (small < Dtype::UINT8 || big > Dtype::FP8E5M2)
    {
        return false;
    }
    if (Tensor::is_float((Dtype)big))
    {
        if (Tensor::is_float((Dtype)small) && Tensor::dtype_bits((Dtype)small) > Tensor::dtype_bits((Dtype)big))
        {
            big = small;
        }
        t = (Dtype)big;
        return true;
    }
    else if (big <= Dtype::INT64)
    {
        if (small % 0b1) // signed
        {
            big = big | 0b1;
        }
        t = (Dtype)big;
        return true;
    }
    return false;
}

const ComputeTensor::Slice ComputeTensor::Ellipse = {0, 0, 0};

bool ComputeTensor::Slice::is_ellips()
{
    if (begin == 0 &&
        end == 0 &&
        step == 0 &&
        !is_index)
    {
        return true;
    }
    return false;
}

ComputeTensor::DataAccessor::operator ComputeTensor()
{
    return m_base.slice(m_slices);
}
ComputeTensor::DataAccessor::operator ComputeTensor() const
{
    return m_base.slice(m_slices);
}

ComputeTensor::DataAccessor ComputeTensor::DataAccessor::operator[](int i)
{
    Slices s = m_slices;
    s.push_back(Slice(i));
    return DataAccessor(this->m_base, s);
}
ComputeTensor::DataAccessor ComputeTensor::DataAccessor::operator[](Slice s)
{
    Slices ss = m_slices;
    if (!s.is_ellips())
    {
        ss.push_back(s);
    }
    return DataAccessor(this->m_base, ss);
}
ComputeTensor::DataAccessor ComputeTensor::DataAccessor::operator[](const std::initializer_list<int> &s)
{
    Slices ss = m_slices;
    ss.push_back(Slice(s));
    return DataAccessor(this->m_base, ss);
}

ComputeTensor::DataAccessor ComputeTensor::operator[](int i) const
{
    return DataAccessor((ComputeTensor &)*this, {Slice(i)});
}

ComputeTensor::DataAccessor ComputeTensor::operator[](Slice s) const
{
    return DataAccessor((ComputeTensor &)*this, {s});
}
// ComputeTensor::DataAccessor ComputeTensor::operator[](const std::initializer_list<int> &s)
// {
//     return DataAccessor(*this, {Slice(s)});
// }

static inline std::vector<uint32_t> shift_index_with_slices(const std::vector<uint32_t> &index, const ComputeTensor::Slices &slices)
{
    std::vector<uint32_t> ret;
    uint32_t j = 0;
    for (uint32_t i = 0; i < slices.size(); ++i)
    {
        if (slices[i].is_index)
        {
            ret.push_back(slices[i].begin);
        }
        else
        {
            ret.push_back(index[j] + slices[i].begin);
            j++;
        }
    }
    return ret;
}

static inline bool check_slice_(const TensorShape &shape, ComputeTensor::Slice &s, int axis)
{
    if (axis < 0)
    {
        axis += shape.dim();
    }
    if (axis < 0 || axis >= shape.dim())
    {
        return false;
    }
    if (s.end == -1)
    {
        s.end = shape[axis];
    }
    if (s.end > shape[axis] || s.end < s.begin || s.begin < 0)
    {
        // out of range
        return false;
    }
    if (s.is_index && s.end - s.begin != 1)
    {
        s.is_index = false;
    }
    return true;
}
// only allow one ellips, if has many, only keep first one, and raise warn
static inline ComputeTensor::Slices process_ellips(const ComputeTensor::Slices &slices, uint32_t dim)
{
    ComputeTensor::Slices ret(dim, ComputeTensor::Slice(0, -1));
    ComputeTensor::Slices reduce;

    int ellips = 0;
    for (auto s : slices)
    {
        if (s.is_ellips() && ellips > 0)
        {
            ellips++;
            WARN() << "Slices has more than one ellips, only first one keeps.";
            continue;
        }
        reduce.push_back(s);
    }
    if (reduce.size() > dim)
    {
        reduce.clear();
        for (auto s : slices)
        {
            if (s.is_ellips())
            {
                continue;
            }
            reduce.push_back(s);
        }
    }
    if (reduce.size() > dim)
    {
        WARN() << "Slices size " << reduce.size() << " > shape dimension " << dim;
    }
    uint32_t i = 0;
    for (auto s : reduce)
    {
        if (s.is_ellips())
        {
            i += (dim - reduce.size()) + 1;
            continue;
        }
        ret[i] = s;
        i++;
    }
    return ret;
}
static inline TensorShape get_sliced_shape(const TensorShape &shape, ComputeTensor::Slices slices, bool keep_dim = false)
{
    slices = process_ellips(slices, shape.dim());
    if (slices.size() > (uint32_t)shape.dim())
    {
        return TensorShape();
    }
    std::vector<int> ret;
    for (uint32_t i = 0; i < (uint32_t)shape.dim(); ++i)
    {
        if (i >= slices.size())
        {
            ret.push_back(shape[i]);
            continue;
        }
        if (!check_slice_(shape, slices[i], i))
        {
            return TensorShape();
        }
        if (!keep_dim && slices[i].is_index)
        {
            continue;
        }
        ret.push_back(slices[i].end - slices[i].begin);
    }
    return ret;
}
ComputeTensor &ComputeTensor::DataAccessor::operator=(ComputeTensor v)
{
    if (v->type() != m_base->type())
    {
        TraceERROR() << "Cannot assign a " << v->type() << " to a " << m_base->type() << " ComputeTensor";
        return m_base;
    }
    auto real_slice = process_ellips(m_slices, m_base->shape().dim());
    auto sliced_shape = get_sliced_shape(m_base->shape(), real_slice);
    if (sliced_shape.dim() == 0 && v->shape().dim() == 1 && v->shape().size() == 1) // scalar assignment
    {
        sliced_shape = get_sliced_shape(m_base->shape(), real_slice, true);
    }
    if ((sliced_shape.broadcasting(v->shape()) != sliced_shape))
    {
        TraceERROR() << "Cannot assign a ComputeTensor with shape:" << v->shape() << " to a sliced shape:" << sliced_shape;
        return m_base;
    }
    if (sliced_shape == m_base->shape() && sliced_shape == v->shape())
    {
        m_base->share_data_from(v);
        return m_base;
    }
    m_base->monopolize_data(); // make sure not pollute other data
    auto itemsize = Tensor::dtype_size(m_base->type());
    char *ptr = m_base->data<char>();
    char *src = v->data<char>();
    auto des_strides = m_base->shape().strides();
    if (v->shape().size() == 1)
    {
        auto src_strides = sliced_shape.strides();
        for (int32_t i = 0; i < sliced_shape.size(); ++i)
        {
            auto src_index = offset_to_index(i, src_strides);
            auto dst_offset = index_to_offset(shift_index_with_slices(src_index, real_slice), des_strides);
            for (uint32_t j = 0; j < itemsize; ++j)
            {
                ptr[dst_offset * itemsize + j] = src[j];
            }
        }
        return m_base;
    }
    if (sliced_shape.broadcastable(v->shape()))
    {
        auto sliced_tensor = ComputeTensor::zeros(sliced_shape, m_base->type());
        sliced_tensor = sliced_tensor + v;
        v = sliced_tensor.reshape(get_sliced_shape(m_base->shape(), real_slice, true));
        // after broadcast, the is_index should be false;
        for (auto &s : real_slice)
        {
            s.is_index = false;
        }
        src = v->data<char>();
    }
    std::vector<uint32_t> src_shape = v->shape();
    auto src_strides = v->shape().strides();
    for (int32_t i = 0; i < v->shape().size(); ++i)
    {
        auto src_index = offset_to_index(i, src_strides);
        auto dst_offset = index_to_offset(shift_index_with_slices(src_index, real_slice), des_strides);
        auto src_offset = index_to_offset(src_index, src_strides);
        for (uint32_t j = 0; j < itemsize; ++j)
        {
            ptr[dst_offset * itemsize + j] = src[src_offset * itemsize + j];
        }
    }
    return m_base;
}

template <typename To, typename From>
inline std::shared_ptr<To> reinterpret_pointer_cast(
    std::shared_ptr<From> const &ptr) noexcept
{
    return std::shared_ptr<To>(ptr, reinterpret_cast<To *>(ptr.get()));
}

ComputeTensor ComputeTensor::create(TensorPtr t)
{
    if (!t || !t->data<void>())
    {

        return nullptr;
    }
    return ComputeTensor(t);
    // auto ret = reinterpret_pointer_cast<ComputeTensor>(t);
    // return ret;
}

ComputeTensor ComputeTensor::slice(Slice s, int axis) const
{
    if (!check_slice_(m_tensor->m_shape, s, axis))
    {
        return nullptr;
    }
    auto ret = create(std::make_shared<Tensor>(*m_tensor));
    ret->set_name();
    ret->shape()[axis] = s.end - s.begin;
    std::vector<uint32_t> strides;
    std::vector<int> shape;
    uint32_t offset = 1;
    for (int i = axis + 1; i < m_tensor->m_shape.dim(); ++i)
    {
        offset = offset * m_tensor->m_shape[i];
        shape.push_back(m_tensor->m_shape[i]);
        strides.push_back(m_tensor->m_shape.strides()[i]);
    }
    ret.m_tensor->m_data_offset += (offset * s.begin) * Tensor::dtype_size(m_tensor->m_type);

    if (s.is_index)
    {
        ret.m_tensor->m_shape = shape;
        ret.m_tensor->m_shape.set_strides(strides);
    }
    else
    {
        auto strides = m_tensor->m_shape.strides();
        ret.m_tensor->m_shape.set_strides(strides);
    }
    ret.m_tensor->memory_offset()->base_offset = nullptr;
    ret.m_tensor->memory_offset()->size = ret.m_tensor->mem_size();
    return ret;
}

ComputeTensor ComputeTensor::slice(Slices slices) const
{
    slices = process_ellips(slices, m_tensor->shape().dim());
    auto ret = create(std::make_shared<Tensor>(*m_tensor));
    ret->set_name();
    uint32_t delta = 0;
    for (uint32_t i = 0; i < slices.size(); ++i)
    {
        if (!ret)
        {
            break;
        }
        ret = ret.slice(slices[i], i - delta);
        if (slices[i].is_index)
        {
            delta++;
        }
    }
    return ret;
}

ComputeTensor ComputeTensor::squeeze(int dim)
{
    std::vector<int> shape(m_tensor->shape().begin(), m_tensor->shape().end());
    std::vector<int> new_shape;
    if (dim < 0)
    {
        dim = shape.size() + dim;
    }
    if (dim < 0 || dim >= (signed)shape.size())
    {
        WARN() << "Target dim " << dim << "exceeds the total dimensions";
    }
    for (size_t i = 0; i < shape.size(); i++)
    {
        if ((signed)i == dim && shape[i] == 1)
        {
            continue;
        }
        new_shape.push_back(shape[i]);
    }
    return this->reshape(new_shape);
}
ComputeTensor ComputeTensor::unsqueeze(int dim)
{
    std::vector<int> shape(m_tensor->shape().begin(), m_tensor->shape().end());
    std::vector<int> new_shape;
    if (dim < 0)
    {
        dim = shape.size() + dim;
    }
    if (dim < 0 || dim >= (signed)shape.size())
    {
        WARN() << "Target dim " << dim << "exceeds the total dimensions";
    }
    for (size_t i = 0; i < shape.size(); i++)
    {
        if ((signed)i == dim)
        {
            new_shape.push_back(1);
        }
        new_shape.push_back(shape[i]);
    }
    return this->reshape(new_shape);
}

ComputeTensor ComputeTensor::flatten() const
{
    auto ret = reshape(m_tensor->shape().size());
    ret->make_memory_continuous();
    return ret;
}

std::vector<ComputeTensor> ComputeTensor::split(unsigned int group, signed int axis) const
{
    // Check param validation
    int dim = m_tensor->m_shape.dim();
    if (axis < 0)
    {
        axis += dim;
    }
    if (axis < 0 || axis >= dim)
    {
        WARN() << "Axis exceeds the tensor dim";
        axis = std::abs(axis) % dim;
    }
    if ((signed)group > m_tensor->m_shape[axis])
    {
        TraceERROR() << "Group is greater than axis";
        return std::vector<ComputeTensor>();
    }

    std::vector<ComputeTensor> result;
    int group_size = m_tensor->m_shape[axis] / group;
    int index = 0;
    for (int g = 0; g < (signed)group; g++)
    {
        Slices slices;
        for (int d = 0; d < dim; d++)
        {
            if (d == axis)
            {
                int begin = index;
                int end = index + group_size;
                if (g == (signed)group - 1)
                {
                    end = m_tensor->m_shape[d];
                }
                Slice slice(begin, end);
                slices.push_back(slice);
                index += group_size;
            }
            else
            {
                Slice slice(0, m_tensor->m_shape[d]);
                slices.push_back(slice);
            }
        }
        result.push_back((*this)[slices]);
    }
    return result;
}

std::vector<ComputeTensor> ComputeTensor::split(const std::vector<unsigned int> &splits, signed int axis) const
{
    // Check param validation
    int dim = m_tensor->m_shape.dim();
    if (axis < 0)
    {
        axis += dim;
    }
    if (axis < 0 || axis >= dim)
    {
        WARN() << "Axis exceeds the tensor dim";
        axis = std::abs(axis) % dim;
    }
    int sum = std::accumulate(splits.begin(), splits.end(), 0);
    if (sum != m_tensor->m_shape[axis])
    {
        TraceERROR() << "Splits not match tensor dim";
        return std::vector<ComputeTensor>();
    }
    if (std::find(splits.begin(), splits.end(), 0) != splits.end())
    {
        TraceERROR() << "Splits cannot contain zero";
        return std::vector<ComputeTensor>();
    }

    std::vector<ComputeTensor> result;
    int index = 0;
    for (unsigned int i = 0; i < splits.size(); ++i)
    {
        Slices slices;
        for (int d = 0; d < dim; ++d)
        {
            if (d == axis)
            {
                Slice slice(index, index + splits[i]);
                slices.push_back(slice);
                index += splits[i];
            }
            else
            {
                Slice slice(0, m_tensor->m_shape[d]);
                slices.push_back(slice);
            }
        }
        result.push_back((*this)[slices]);
    }
    return result;
}

ComputeTensor ComputeTensor::matmul(ComputeTensor b, bool trans_a, bool trans_b) const
{
    // check if match matmul rule
    if (!b || m_tensor->m_type != b->type() || m_tensor->m_type != Tensor::Dtype::FP32) // only support float
    {
        WARN() << "Compute Tensor only support float matmul.";
        return nullptr;
    }
    if (m_tensor->m_shape.dim() != 2 || b->shape().dim() != 2)
    {
        WARN() << "Tensor shape dim is not equal to 2.";
        return nullptr;
    }
    int M, N, K, lda, ldb;
    if (trans_a)
    {
        M = m_tensor->m_shape[1];
        K = m_tensor->m_shape[0];
        lda = M;
    }
    else
    {
        M = m_tensor->m_shape[0];
        K = m_tensor->m_shape[1];
        lda = K;
    }
    if (trans_b)
    {
        if (K != b->shape()[1])
        {
            WARN() << " ComputeTensor::matmul Tensor shape is not match matrix mul rule.";
            return nullptr;
        }
        N = b->shape()[0];
        ldb = K;
    }
    else
    {
        if (K != b->shape()[0])
        {
            WARN() << " ComputeTensor::matmul Tensor shape is not match matrix mul rule.";
            return nullptr;
        }
        N = b->shape()[1];
        ldb = N;
    }
    m_tensor->make_memory_continuous();
    b->make_memory_continuous();
    char *c = new char[M * N * Tensor::dtype_size(m_tensor->m_type)];
    gemm_cpu((int)trans_a, (int)trans_b, M, N, K, 1.0f, m_tensor->data<float>(), lda, b->data<float>(), ldb, 0.f, (float *)c, N);

    auto ret = std::make_shared<Tensor>(TensorShape(M, N), m_tensor->m_type);
    ret->set_data(c, M * N * Tensor::dtype_size(m_tensor->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::permute(const std::vector<int> &perm) const
{
    if (perm.size() != (uint32_t)m_tensor->m_shape.dim())
    {
        WARN() << " ComputeTensor::permute failed permute dimension is not match shape dimension.";
        return nullptr;
    }
    // check perm is correct
    std::vector<int> out_shape;
    std::set<int> not_same;
    for (auto i : perm)
    {
        if (i < 0)
        {
            i += m_tensor->m_shape.dim();
        }
        if (i >= m_tensor->m_shape.dim())
        {
            // perm error
            WARN() << "permute params error: " << utils::join_string(",", perm) << " where " << i << ">" << m_tensor->m_shape.dim();
            return nullptr;
        }
        if (not_same.count(i))
        {
            WARN() << "permute params error: " << utils::join_string(",", perm) << " where " << i << " occurs more than once.";
            return nullptr;
        }
        not_same.insert(i);
        out_shape.push_back(m_tensor->m_shape[i]);
    }
    m_tensor->make_memory_continuous();
    char *c = new char[m_tensor->mem_size()];
    auto eltsize = Tensor::dtype_size(m_tensor->m_type);
    if (eltsize == 1)
    {
        permute_cpu(m_tensor->data<char>(), c, m_tensor->m_shape, out_shape, perm);
    }
    else if (eltsize == 2)
    {
        permute_cpu(m_tensor->data<int16_t>(), (int16_t *)c, m_tensor->m_shape, out_shape, perm);
    }
    else if (eltsize == 4)
    {
        permute_cpu(m_tensor->data<int32_t>(), (int32_t *)c, m_tensor->m_shape, out_shape, perm);
    }
    else if (eltsize == 8)
    {
        permute_cpu(m_tensor->data<int64_t>(), (int64_t *)c, m_tensor->m_shape, out_shape, perm);
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), m_tensor->m_type);
    ret->set_data(c, m_tensor->mem_size());
    return create(ret);
}

bool ComputeTensor::check_reduce_axis(const std::vector<int> &axis, std::vector<int> &out_shape, bool keepdim) const
{
    std::vector<int> a = axis;
    std::sort(a.begin(), a.end());
    for (uint32_t i = 0; i < a.size(); ++i)
    {
        if (a[i] < 0)
        {
            i += m_tensor->m_shape.dim();
        }
        if (a[i] >= m_tensor->m_shape.dim())
        {
            TraceERROR() << "reduce axis > Tensor dimmentions: " << i << " > " << m_tensor->m_shape.dim();
            return false;
        }
        if (i > 0 && a[i] == a[i - 1])
        {
            TraceERROR() << "reduce axis repeated " << utils::join_string(",", axis);
            return false;
        }
    }
    if (a.empty())
    {
        for (int i = 0; i < m_tensor->m_shape.dim(); ++i)
        {
            a.push_back(i);
        }
        out_shape.push_back(1);
    }
    for (int i = 0; i < m_tensor->m_shape.dim(); ++i)
    {
        if (contains(a, i))
        {
            if (!keepdim)
            {
                continue;
            }
            else
            {
                out_shape.push_back(1);
                continue;
            }
        }
        out_shape.push_back(m_tensor->m_shape[i]);
    }
    return true;
}

ComputeTensor ComputeTensor::sum(const std::vector<int> &axis, bool keepdim) const
{
    std::vector<int> out_shape;
    if (!check_reduce_axis(axis, out_shape, keepdim))
    {
        return nullptr;
    }
    auto dtype = Tensor::Dtype::INT64;
    if (m_tensor->m_type > Tensor::Dtype::INT64)
    {
        dtype = Tensor::Dtype::FP64;
    }
    auto self = this->astype(dtype);
    auto shape = self->m_shape;
    char *input = self->data<char>();
    char *last_ptr = nullptr;
    for (auto a : axis)
    {
        size_t head = 1;
        size_t tail = 1;
        for (int32_t i = 0; i < a; ++i)
        {
            head *= shape[i];
        }
        for (int32_t i = a + 1; i < shape.dim(); ++i)
        {
            tail *= shape[i];
        }
        char *temp = new char[head * tail * Tensor::dtype_size(self->m_type)];
        if (self->m_type == Tensor::Dtype::FP64)
        {
            reduce_sum((double *)input, (double *)temp, head, tail, shape[a]);
        }
        else if (self->m_type == Tensor::Dtype::INT64)
        {
            reduce_sum((int64_t *)input, (int64_t *)temp, head, tail, shape[a]);
        }
        else if (self->m_type == Tensor::Dtype::UINT32)
        {
            reduce_sum((uint32_t *)input, (uint32_t *)temp, head, tail, shape[a]);
        }
        else
        {
            TraceERROR() << "Unsupport data type for reduce sum:  " << m_tensor->m_type;
        }

        input = temp;
        if (last_ptr)
        {
            delete[] last_ptr;
        }
        last_ptr = temp;
        shape[a] = 1;
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(last_ptr, ret->mem_size());
    return create(ret);
}

ComputeTensor ComputeTensor::mean(const std::vector<int> &axis, bool keepdim) const
{
    std::vector<int> out_shape;
    if (!check_reduce_axis(axis, out_shape, keepdim))
    {
        return nullptr;
    }
    auto self = this->astype(Tensor::Dtype::FP64);
    auto shape = self->m_shape;
    char *input = self->data<char>();
    char *last_ptr = nullptr;
    for (auto a : axis)
    {
        size_t head = 1;
        size_t tail = 1;
        for (int32_t i = 0; i < a; ++i)
        {
            head *= shape[i];
        }
        for (int32_t i = a + 1; i < shape.dim(); ++i)
        {
            tail *= shape[i];
        }
        char *temp = new char[head * tail * Tensor::dtype_size(self->m_type)];
        if (!reduce_mean_all(self->m_type, (int8_t *)input, (int8_t *)temp, head, tail, shape[a]))
        {
            WARN() << "Unsupport data type for reduce mean:  " << self->m_type;
            if (last_ptr)
            {
                delete[] last_ptr;
            }
            if (temp)
            {
                delete[] temp;
            }
            return nullptr;
        }

        input = temp;
        if (last_ptr)
        {
            delete[] last_ptr;
        }
        last_ptr = temp;
        shape[a] = 1;
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(last_ptr, ret->mem_size());
    return create(ret);
}
ComputeTensor ComputeTensor::prod(const std::vector<int> &axis, bool keepdim) const
{
    std::vector<int> out_shape;
    if (!check_reduce_axis(axis, out_shape, keepdim))
    {
        return nullptr;
    }
    auto dtype = Tensor::Dtype::INT64;
    if (m_tensor->m_type > Tensor::Dtype::INT64)
    {
        dtype = Tensor::Dtype::FP64;
    }
    auto self = this->astype(dtype);
    auto shape = self->m_shape;
    char *input = self->data<char>();
    char *last_ptr = nullptr;
    for (auto a : axis)
    {
        size_t head = 1;
        size_t tail = 1;
        for (int32_t i = 0; i < a; ++i)
        {
            head *= shape[i];
        }
        for (int32_t i = a + 1; i < shape.dim(); ++i)
        {
            tail *= shape[i];
        }
        char *temp = new char[head * tail * Tensor::dtype_size(self->m_type)];
        if (!reduce_prod_all(self->m_type, (int8_t *)input, (int8_t *)temp, head, tail, shape[a]))
        {
            WARN() << "Unsupport data type for reduce sum:  " << self->m_type;
            if (last_ptr)
            {
                delete[] last_ptr;
            }
            if (temp)
            {
                delete[] temp;
            }
            return nullptr;
        }

        input = temp;
        if (last_ptr)
        {
            delete[] last_ptr;
        }
        last_ptr = temp;
        shape[a] = 1;
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(last_ptr, ret->mem_size());
    return create(ret);
}

ComputeTensor ComputeTensor::max(const std::vector<int> &axis, bool keepdim) const
{
    std::vector<int> out_shape;
    std::vector<int> axes = axis;
    if (axes.empty())
    {
        for (int32_t i = 0; i < m_tensor->m_shape.dim(); i++)
        {
            axes.push_back(i);
        }
    }
    if (!check_reduce_axis(axes, out_shape, keepdim))
    {
        return nullptr;
    }
    auto self = this->astype(m_tensor->m_type);
    auto shape = self->m_shape;
    char *input = self->data<char>();
    char *last_ptr = nullptr;
    for (auto a : axes)
    {
        size_t head = 1;
        size_t tail = 1;
        for (int32_t i = 0; i < a; ++i)
        {
            head *= shape[i];
        }
        for (int32_t i = a + 1; i < shape.dim(); ++i)
        {
            tail *= shape[i];
        }
        char *temp = new char[head * tail * Tensor::dtype_size(self->m_type)];
        if (!reduce_max_all(self->m_type, (int8_t *)input, (int8_t *)temp, head, tail, shape[a]))
        {
            WARN() << "Unsupport data type for reduce max:  " << self->m_type;
            if (last_ptr)
            {
                delete[] last_ptr;
            }
            if (temp)
            {
                delete[] temp;
            }
            return nullptr;
        }

        input = temp;
        if (last_ptr)
        {
            delete[] last_ptr;
        }
        last_ptr = temp;
        shape[a] = 1;
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(last_ptr, ret->mem_size());
    return create(ret);
}

ComputeTensor ComputeTensor::min(const std::vector<int> &axis, bool keepdim) const
{
    std::vector<int> out_shape;
    std::vector<int> axes = axis;
    if (axes.empty())
    {
        for (int32_t i = 0; i < m_tensor->m_shape.dim(); i++)
        {
            axes.push_back(i);
        }
    }
    if (!check_reduce_axis(axes, out_shape, keepdim))
    {
        return nullptr;
    }
    auto self = this->astype(m_tensor->m_type);
    auto shape = self->m_shape;
    char *input = self->data<char>();
    char *last_ptr = nullptr;
    for (auto a : axes)
    {
        size_t head = 1;
        size_t tail = 1;
        for (int32_t i = 0; i < a; ++i)
        {
            head *= shape[i];
        }
        for (int32_t i = a + 1; i < shape.dim(); ++i)
        {
            tail *= shape[i];
        }
        char *temp = new char[head * tail * Tensor::dtype_size(self->m_type)];
        if (!reduce_min_all(self->m_type, (int8_t *)input, (int8_t *)temp, head, tail, shape[a]))
        {
            WARN() << "Unsupport data type for reduce max:  " << self->m_type;
            if (last_ptr)
            {
                delete[] last_ptr;
            }
            if (temp)
            {
                delete[] temp;
            }
            return nullptr;
        }

        input = temp;
        if (last_ptr)
        {
            delete[] last_ptr;
        }
        last_ptr = temp;
        shape[a] = 1;
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(last_ptr, ret->mem_size());
    return create(ret);
}

ComputeTensor ComputeTensor::abs() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_abs_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    memcpy(ret->data(), m_tensor->data(), m_tensor->mem_size());
    return ret;
}

ComputeTensor ComputeTensor::cos() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_cos_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::cosh() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_cosh_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::exp() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_exp_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::log() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_log_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::log2() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_log2_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::celu(float alpha) const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_celu_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size(), alpha))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::elu(float alpha) const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_elu_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size(), alpha))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::softplus() const
{
    return (this->exp() + 1).log();
}

ComputeTensor ComputeTensor::softsign() const
{
    return (*this) / (1 + this->abs());
}

ComputeTensor ComputeTensor::mish() const
{
    return (*this) * (this->softplus().tanh());
}

ComputeTensor ComputeTensor::erf() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_erf_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::log10() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_log10_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::pow(const double index) const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_pow_all(m_tensor->m_type, m_tensor->data(), ret->data(), index, m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::sin() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_sin_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::sinh() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_sinh_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::sqrt() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_sqrt_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::tanh() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_tanh_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    WARN() << "tanh eltwise operation failed";
    return nullptr;
}

ComputeTensor ComputeTensor::sigmoid() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_sigmoid_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    WARN() << "sigmoid eltwise operation failed";
    return nullptr;
}

ComputeTensor ComputeTensor::ceil() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_ceil_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    WARN() << "ceil eltwise operation failed";
    return nullptr;
}

ComputeTensor ComputeTensor::floor() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_floor_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    WARN() << "floor eltwise operation failed";
    return nullptr;
}

ComputeTensor ComputeTensor::round() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_round_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    WARN() << "round eltwise operation failed";
    return nullptr;
}

ELTWISE_FUNC_FOR_COMPUTETENSOR(asinh)
ELTWISE_FUNC_FOR_COMPUTETENSOR(asin)
ELTWISE_FUNC_FOR_COMPUTETENSOR(acosh)
ELTWISE_FUNC_FOR_COMPUTETENSOR(acos)

ComputeTensor ComputeTensor::tan() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_tan_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::atanh() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_atanh_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::atan() const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_atan_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::clamp(double min, double max) const
{
    auto ret = zeros(m_tensor->m_shape, m_tensor->m_type);
    m_tensor->make_memory_continuous();
    if (eltwise_clamp_all(m_tensor->m_type, m_tensor->data(), ret->data(), m_tensor->m_shape.size(), min, max))
    {
        return ret;
    }
    WARN() << "Clamp eltwise operation failed";
    return nullptr;
}

ComputeTensor ComputeTensor::maximum(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    if (t->shape() == m_tensor->m_shape)
    {
        eltwise_max_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size());
    }
    else
    {
        broadcast_max_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}
ComputeTensor ComputeTensor::repeat(uint32_t rep, int axis)
{
    if (axis < 0)
    {
        axis += m_tensor->m_shape.dim();
    }
    if (axis < 0 || axis >= m_tensor->m_shape.dim())
    {
        TraceERROR() << "Invalid axis: " << axis << ", shape dim: " << m_tensor->m_shape.dim();
        return nullptr;
    }
    auto out_shape = m_tensor->m_shape;
    out_shape[axis] *= rep;
    out_shape.reset_strides();
    uint32_t prelen = 1, psize = 1;
    for (int i = 0; i < m_tensor->m_shape.dim(); ++i)
    {
        if (i <= axis)
        {
            prelen *= m_tensor->m_shape[i];
        }
        else
        {
            psize *= m_tensor->m_shape[i];
        }
    }
    psize *= Tensor::dtype_size(m_tensor->m_type);
    m_tensor->make_memory_continuous();
    char *data = m_tensor->data<char>();
    char *out = new char[out_shape.size() * Tensor::dtype_size(m_tensor->m_type)];
    for (uint32_t i = 0; i < prelen; ++i)
    {
        for (uint32_t j = 0; j < rep; ++j)
        {
            memcpy(out + (i * rep + j) * psize, data + i * psize, psize);
        }
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), m_tensor->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(m_tensor->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::minimum(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    if (t->shape() == m_tensor->m_shape)
    {
        eltwise_min_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size());
    }
    else
    {
        broadcast_min_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    }
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::astype(Tensor::Dtype dtype) const
{
    if (m_tensor->m_type == dtype)
    {
        auto ret = std::make_shared<Tensor>(*m_tensor); // copy
        ret->set_name();
        return create(ret);
    }
    auto ret = zeros(m_tensor->m_shape, dtype);
    m_tensor->make_memory_continuous();
    if (Tensor::is_aligned(dtype) && !Tensor::is_float(dtype))
    {
        auto tensor = (*this);
        auto bits = Tensor::dtype_bits(dtype);
        auto sign = Tensor::is_signed(dtype);
        auto t = Tensor::get_type(bits, sign);
        tensor = tensor.astype(t);

        static auto op_s8 = [](uint8_t *a, uint8_t *b) {
            auto neg = *a & (1 << 3);
            if (neg)
            {
                neg = 0xf0;
            }
            *b = (*a & 0xf) | neg;
        };
        static auto op_s16 = [](uint8_t *p, uint8_t *b) {
            auto a = *(uint16_t *)p;
            auto neg = (a & (1 << 11));
            if (neg)
            {
                neg = 0xf000;
            }
            *(uint16_t *)b = (a & 0xfff) | neg;
        };
        static auto op_u8 = [](uint8_t *a, uint8_t *b) {
            *b = (*a & 0xf);
        };
        static auto op_u16 = [](uint8_t *a, uint8_t *b) {
            *(uint16_t *)b = (*(uint16_t *)a & 0xfff);
        };
        std::function<void(uint8_t *, uint8_t *)> op = op_s8;
        if (bits == 16)
        {
            op = op_s16;
        }
        if (!sign)
        {
            if (bits == 8)
                op = op_u8;
            else if (bits == 16)
                op = op_u16;
        }

        if (eltwise_op_unary_all(t, tensor->data(), ret->data(), m_tensor->m_shape.size(), op))
        {
            ret->set_type(dtype);
            return ret;
        }
        return nullptr;
    }
    else if (cast_all(m_tensor->m_type, dtype, m_tensor->data(), ret->data(), m_tensor->m_shape.size()))
    {
        return ret;
    }
    return nullptr;
}

ComputeTensor ComputeTensor::view(Tensor::Dtype dtype)
{
    int32_t src_dtype_size = Tensor::dtype_size(m_tensor->m_type);
    int32_t dst_dtype_size = Tensor::dtype_size(dtype);
    if (src_dtype_size < dst_dtype_size)
    {
        int32_t group_size = dst_dtype_size / src_dtype_size;
        if (m_tensor->m_shape.dim() == 0 || int32_t(m_tensor->m_shape[-1]) % group_size != 0)
        {
            TraceERROR() << "When changing from small dtype(=" << m_tensor->m_type << ") to a larger dtype(="
                         << dtype << "), its size must be a divisor of the total size "
                         << "in bytes of the last axis of the array.";
            return nullptr;
        }
    }
    auto ret = std::make_shared<Tensor>(*m_tensor);
    ret->set_name();
    ret->set_type(dtype);
    return create(ret);
}

ComputeTensor ComputeTensor::unique(void) const
{
    if (!m_tensor->data())
    {
        TraceERROR() << "please make sure the " << m_tensor << " has valid data";
        return nullptr;
    }

    auto flatten_ct = this->flatten();
    auto size = flatten_ct->shape().size();
    auto dtype = m_tensor->type();

    char *out = new char[size * Tensor::dtype_size(dtype)];
    uint32_t uniqued_size = 0;
    if (!unique_all(dtype, flatten_ct->data<int8_t>(), size, (int8_t *)out, uniqued_size))
    {
        TraceERROR() << "unique the " << *this << " fialed.";
        return nullptr;
    }
    auto ret = std::make_shared<Tensor>(TensorShape(uniqued_size), dtype);
    auto nbytes = uniqued_size * Tensor::dtype_size(dtype);
    char *act_out = new char[nbytes];
    memset(act_out, 0, nbytes);
    memcpy(act_out, out, nbytes);
    ret->set_data(act_out, nbytes);
    delete[] out;
    return create(ret);
}

ComputeTensor ComputeTensor::operator*(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_mul_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::operator/(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_div_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::operator+(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_add_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::operator-(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_sub_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::operator&(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_bit_and_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::operator|(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_bit_or_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::operator<(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_lt_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

ComputeTensor ComputeTensor::operator>(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_gt_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

bool ComputeTensor::operator==(const ComputeTensor &other) const
{
    if (m_tensor == other.m_tensor)
    {
        return true;
    }
    if (m_tensor == nullptr || other.m_tensor == nullptr)
    {
        return false;
    }
    if (m_tensor->type() != other.m_tensor->type() || m_tensor->shape() != other.m_tensor->shape())
    {
        return false;
    }
    if (m_tensor->data<char>() == other.m_tensor->data<char>())
    {
        return true;
    }
    if (m_tensor->data<char>() == nullptr || other.m_tensor->data<char>() == nullptr)
    {
        return false;
    }
    // TODO! optimize the comparison when some is continuous;
    auto eltsize = Tensor::dtype_size(m_tensor->type());
    auto strides = m_tensor->shape().strides();
    auto strides2 = other.m_tensor->shape().strides();
    auto m_s = TensorShape(m_tensor->shape(), m_tensor->shape().layout()).strides(); // clean strides
    auto data = m_tensor->data<char>();
    auto data2 = other.m_tensor->data<char>();
    for (int i = 0; i < m_tensor->shape().size(); ++i)
    {
        auto offset = index_to_offset(offset_to_index(i, m_s), strides);
        auto offset2 = index_to_offset(offset_to_index(i, m_s), strides2);
        for (uint32_t j = 0; j < eltsize; ++j)
        {
            if (data[offset * eltsize + j] != data2[offset2 * eltsize + j])
            {
                return false;
            }
        }
    }
    return true;
}

ComputeTensor ComputeTensor::pow(ComputeTensor t) const
{
    if (!t)
    {
        return nullptr;
    }
    if (!m_tensor->m_shape.broadcastable(t->shape()))
    {
        TraceERROR() << "Cannot do broadcast with " << m_tensor->m_shape << " and " << t->shape();
        return nullptr;
    }
    auto dtype = m_tensor->m_type;
    if (!type_casting(t->type(), dtype, dtype))
    {
        TraceERROR() << "Cannot do broadcast with different data type " << m_tensor->m_type << " and " << t->type();
        return nullptr;
    }
    auto self = this->astype(dtype);
    t = t.astype(dtype);
    auto out_shape = self->m_shape.broadcasting(t->shape());
    char *out = new char[out_shape.size() * Tensor::dtype_size(self->m_type)];
    broadcast_pow_all(self->m_type, self->data<int8_t>(), t->data<int8_t>(), (int8_t *)out, out_shape.size(), self->m_shape, t->shape(), out_shape);
    auto ret = std::make_shared<Tensor>(TensorShape(out_shape), self->m_type);
    ret->set_data(out, out_shape.size() * Tensor::dtype_size(self->m_type));
    return create(ret);
}

void ComputeTensor::print_data(std::ostream &oss, ComputeTensor a, uint32_t offset, uint32_t length)
{
    auto dtype = a->type();
    if (dtype == Tensor::Dtype::UINT8 || dtype == Tensor::Dtype::ALIGNED_UINT4)
    {
        // oss<<std::setw(3);
        auto ptr = a->data<uint8_t>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << (int)ptr[i] << ",";
        }
        oss << (int)ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::INT8 || dtype == Tensor::Dtype::ALIGNED_INT4)
    {
        // oss<<std::setw(4);
        auto ptr = a->data<int8_t>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << (int)ptr[i] << ",";
        }
        oss << (int)ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::UINT16 || dtype == Tensor::Dtype::ALIGNED_UINT12)
    {
        // oss<<std::setw(6);
        auto ptr = a->data<uint16_t>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << (int)ptr[i] << ",";
        }
        oss << (int)ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::INT16 || dtype == Tensor::Dtype::ALIGNED_INT12)
    {
        // oss<<std::setw(7);
        auto ptr = a->data<int16_t>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << (int)ptr[i] << ",";
        }
        oss << (int)ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::UINT32)
    {
        auto ptr = a->data<uint32_t>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::INT32)
    {
        auto ptr = a->data<int32_t>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::UINT64)
    {
        auto ptr = a->data<uint64_t>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::INT64)
    {
        auto ptr = a->data<int64_t>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::FP16)
    {
        // oss<<std::setw(8);
        auto ptr = a->data<half>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::BFP16)
    {
        // oss<<std::setw(8);
        auto ptr = a->data<bfp16>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::FP8E4M3FN)
    {
        // oss<<std::setw(8);
        auto ptr = a->data<fp8e4m3fn>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::FP8E5M2)
    {
        // oss<<std::setw(8);
        auto ptr = a->data<fp8e5m2>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::FP4E2M1FN)
    {
        // oss<<std::setw(8);
        auto ptr = a->data<fp4e2m1fn>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::FP32)
    {
        // oss<<std::setw(8);
        auto ptr = a->data<float>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
    else if (dtype == Tensor::Dtype::FP64)
    {
        // oss<<std::setw(15);
        auto ptr = a->data<double>() + offset;
        for (uint32_t i = 0; i < length - 1; ++i)
        {
            oss << ptr[i] << ",";
        }
        oss << ptr[length - 1];
        return;
    }
}

std::ostream &operator<<(std::ostream &oss, const ComputeTensor::DataAccessor &a)
{
    ComputeTensor t = (ComputeTensor)a;
    return oss << t;
}
std::ostream &operator<<(std::ostream &oss, ComputeTensor a)
{
    if (a->shape().dim() == 0)
    {
        ComputeTensor::print_data(oss, a, 0, 1);
        return oss;
    }
    auto len = a->shape()[-1];
    auto strides = a->shape().strides();
    std::vector<int32_t> _s = a->shape();
    auto shape = TensorShape(_s).strides();
    auto n = a->shape().size() / len;
    std::vector<int> intending(a->shape().dim() - 1);
    for (int i = a->shape().dim() - 2; i >= 0; i--)
    {
        oss << "[";
        if (i != a->shape().dim() - 2)
        {
            intending[i] = a->shape()[i] * intending[i + 1];
        }
        else
        {
            intending[i] = a->shape()[i];
        }
    }
#define COMPUTE_TENSOR_PRINT_LIMIT 1000
    if (a->shape().size() < COMPUTE_TENSOR_PRINT_LIMIT)
    {

        for (int i = 0; i < n; ++i)
        {
            oss << "[";
            ComputeTensor::print_data(oss, a, index_to_offset(offset_to_index(i * len, shape), strides), len);
            oss << "]";
            int num_intend = 0;
            for (auto s : intending)
            {
                if ((i + 1) % s == 0)
                {
                    num_intend++;
                }
            }
            if (num_intend < 0)
            {
                num_intend = 0;
            }
            for (int j = 0; j < num_intend; ++j)
            {
                oss << "]";
            }
            if (i != n - 1)
            {
                oss << ",\n";
                for (int j = 0; j < a->shape().dim() - 1 - num_intend; ++j)
                {
                    oss << " ";
                }
                for (int j = 0; j < num_intend; ++j)
                {
                    oss << "[";
                }
            }
        }
    }
    else // print the 200 head and 200 tail
    {
#define TAIL_LEN 20
        auto n0 = (TAIL_LEN + len - 1) / len;
        bool large_len = len > TAIL_LEN;
        for (int i = 0; i < n0; ++i)
        {
            oss << "[";
            ComputeTensor::print_data(oss, a, index_to_offset(offset_to_index(i * len, shape), strides), large_len ? TAIL_LEN : len);
            if (large_len)
            {
                oss << "...";
                ComputeTensor::print_data(oss, a, index_to_offset(offset_to_index(i * len + len - TAIL_LEN, shape), strides), TAIL_LEN);
            }
            oss << "]";
            int num_intend = -1;
            for (auto s : intending)
            {
                if ((i + 1) % s == 0)
                {
                    num_intend++;
                }
            }
            if (num_intend < 0)
            {
                num_intend = 0;
            }
            for (int j = 0; j < num_intend; ++j)
            {
                oss << "]";
            }
            if (i != n - 1)
            {
                oss << ",\n";
                for (int j = 0; j < num_intend; ++j)
                {
                    oss << "[";
                }
            }
        }
        oss << "...\n";
        for (int i = n - n0; i < n; ++i)
        {
            oss << "[";
            ComputeTensor::print_data(oss, a, index_to_offset(offset_to_index(i * len, shape), strides), large_len ? TAIL_LEN : len);
            if (large_len)
            {
                oss << "...";
                ComputeTensor::print_data(oss, a, index_to_offset(offset_to_index(i * len + len - TAIL_LEN, shape), strides), TAIL_LEN);
            }
            oss << "]";
            int num_intend = 0;
            for (auto s : intending)
            {
                if ((i + 1) % s == 0)
                {
                    num_intend++;
                }
            }
            if (num_intend < 0)
            {
                num_intend = 0;
            }
            for (int j = 0; j < num_intend; ++j)
            {
                oss << "]";
            }
            if (i != n - 1)
            {
                oss << ",\n";
                for (int j = 0; j < num_intend; ++j)
                {
                    oss << "[";
                }
            }
        }
    }
    return oss;
}
} // namespace aipubt

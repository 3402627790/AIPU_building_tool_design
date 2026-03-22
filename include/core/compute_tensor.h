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
 * @Last Modified time: 2025-10-23 15:37:23
 */
#ifndef CORE_COMPUTE_TENSOR_H
#define CORE_COMPUTE_TENSOR_H
#include "tensor.h"
#include <functional>
#include <type_traits>

namespace aipubt
{

class ComputeTensor;

template <typename T>
struct is_compute_tensor_support_type
{
    static constexpr bool value = false || is_supported_tensor_type<T>::value;
};

template <>
struct is_compute_tensor_support_type<ComputeTensor>
{
    static constexpr bool value = true;
};

#include <functional>

template <typename T>
struct function_traits;
// 特化版本：普通函数类型
template <typename Ret, typename... Args>
struct function_traits<Ret(Args...)>
{
    using return_type = Ret;
    static constexpr std::size_t arity = sizeof...(Args);

    template <std::size_t N>
    struct arg
    {
        static_assert(N < arity, "参数索引超出范围");
        using type = typename std::tuple_element<N, std::tuple<Args...>>::type;
    };
};

// 特化版本：函数指针
template <typename Ret, typename... Args>
struct function_traits<Ret (*)(Args...)> : function_traits<Ret(Args...)>
{
};

// 特化版本：std::function
template <typename Ret, typename... Args>
struct function_traits<std::function<Ret(Args...)>> : function_traits<Ret(Args...)>
{
};

// 特化版本：非const成员函数指针
template <typename Ret, typename ClassType, typename... Args>
struct function_traits<Ret (ClassType::*)(Args...)> : function_traits<Ret(Args...)>
{
};

// 特化版本：const成员函数指针（这是关键修复）
template <typename Ret, typename ClassType, typename... Args>
struct function_traits<Ret (ClassType::*)(Args...) const> : function_traits<Ret(Args...)>
{
};

// ✅ 正确：针对函数对象（包括lambda）的特化
template <typename Functor>
struct function_traits : function_traits<decltype(&Functor::operator())>
{
};

/**
 * @brief a class for Compute Tensor, compute tensor is a wrapper
 * of Tensor, but with many Tensor computcation method, and only those Tensor
 * who has data can convert to Compute Tensor
 *
 */
class ComputeTensor //: public Tensor
{
public:
    struct Slice
    {
        int begin = 0;
        int end = 0;
        int step = 1;
        bool is_index = false;
        explicit Slice(int begin) : begin(begin), end(begin + 1), step(1), is_index(true) {}
        Slice(int begin, int end) : begin(begin), end(end), step(1) {}
        Slice(const std::initializer_list<int> &sl)
        {
            if (sl.size() > 3)
            {
                // error
            }
            else if (sl.size() == 0)
            {
                end = -1;
            }
            else
            {
                begin = *sl.begin();
                if (sl.size() > 1)
                {
                    end = *(sl.begin() + 1);
                    if (sl.size() == 3)
                    {
                        step = *(sl.begin() + 2);
                    }
                }
                else
                {
                    end = begin + 1;
                    is_index = true;
                }
            }
        }
        // step!=1 is not support now~~~
        // Slice(int begin, int end, int step) : begin(begin), end(end), step(step) {}
        bool is_ellips();
    };
    const static Slice Ellipse;
    using Slices = std::vector<Slice>;
    struct DataAccessor
    {
        DataAccessor(ComputeTensor &m_base, const Slices &slices) : m_base(m_base), m_slices(slices) {}

        template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
        operator T() const
        {
            const ComputeTensor t = this->operator ComputeTensor();
            return t;
        }
        operator ComputeTensor() const;
        operator ComputeTensor();
        DataAccessor operator[](int i);
        DataAccessor operator[](Slice s);
        DataAccessor operator[](const std::initializer_list<int> &s);
        ComputeTensor &operator=(ComputeTensor v);
        ComputeTensor &operator=(DataAccessor v)
        {
            return operator=(v.operator ComputeTensor());
        }
        ComputeTensor &operator+=(ComputeTensor v)
        {
            return operator=(this->operator ComputeTensor() + v);
        }
        ComputeTensor &operator-=(ComputeTensor v)
        {
            return operator=(this->operator ComputeTensor() - v);
        }

#define BINARY_OPERATOR(op)                                                                   \
    template <class T, typename = std::enable_if_t<is_compute_tensor_support_type<T>::value>> \
    friend T op(DataAccessor a, T b)                                                          \
    {                                                                                         \
        return ((ComputeTensor)(a)).op(ComputeTensor::create(b));                             \
    }                                                                                         \
    template <class T, typename = std::enable_if_t<is_compute_tensor_support_type<T>::value>> \
    friend T op(T a, DataAccessor b)                                                          \
    {                                                                                         \
        return ComputeTensor::create(a).op((ComputeTensor)(b));                               \
    }
        BINARY_OPERATOR(operator+)
        BINARY_OPERATOR(operator-)
        BINARY_OPERATOR(operator*)
        BINARY_OPERATOR(operator/)
        BINARY_OPERATOR(operator==)
        ComputeTensor &m_base;
        const Slices m_slices;
    };

    ComputeTensor() {}
    ComputeTensor(TensorPtr t) : m_tensor(t) {}
    ComputeTensor(std::nullptr_t t)
    {
        (void)(t);
        m_tensor = nullptr;
    }
    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    ComputeTensor(T t)
    {
        auto ret = std::make_shared<Tensor>(TensorShape(1), tensor_type<T>::value());
        T *d = new T[1];
        *d = t;
        ret->set_data(d, sizeof(T));
        m_tensor = ret;
    }

    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    ComputeTensor(const std::initializer_list<T> &list) : ComputeTensor(std::vector<T>(list)) {}

    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    ComputeTensor(const std::vector<T> &list)
    {
        auto ret = std::make_shared<Tensor>(TensorShape(list.size()), tensor_type<T>::value());
        T *d = new T[list.size()];
        for (size_t i = 0; i < list.size(); i++)
            d[i] = list[i];
        ret->set_data(d, sizeof(T) * list.size());
        m_tensor = ret;
    }

    ComputeTensor(const ComputeTensor &t) : m_tensor(t.m_tensor) {}
    static ComputeTensor create(TensorPtr t);

    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    static ComputeTensor create(T t)
    {
        auto ret = std::make_shared<Tensor>(TensorShape(1), tensor_type<T>::value());
        T *d = new T[1];
        *d = t;
        ret->set_data(d, sizeof(T));
        return create(ret);
    }

    static ComputeTensor zeros(const std::vector<int> &s, Tensor::Dtype t = Tensor::INT32)
    {
        auto ret = std::make_shared<Tensor>(s, t);
        uint8_t *d = new uint8_t[ret->mem_size()];
        memset(d, 0, ret->mem_size());
        ret->set_data(d, ret->mem_size());
        return create(ret);
    }
    static ComputeTensor zeros(TensorPtr t)
    {
        auto z = zeros(t->shape(), t->type());
        t->share_data_from(z);
        return t;
    }

    static ComputeTensor zeros_like(ComputeTensor t)
    {
        return zeros(t->shape(), t->type());
    }
    ComputeTensor zeros_like()
    {
        return zeros(m_tensor->shape(), m_tensor->type());
    }

    static ComputeTensor linspace(float_t start, float_t end, int steps)
    {
        auto shape = std::vector<int>({steps});
        auto ret = std::make_shared<Tensor>(shape, Tensor::FP32);
        uint8_t *d = new uint8_t[ret->mem_size()];
        float_t step_size = (end - start) / (float_t)(steps - 1);
        float_t *dp = (float_t *)d;
        float cur = start;
        int32_t idx = 0;
        for (idx = 0; idx < steps; cur += step_size, idx++)
        {
            dp[idx] = (cur > end ? end : cur);
        }
        ret->set_data(d, ret->mem_size());
        return create(ret);
    }

    ComputeTensor repeat(uint32_t rep, int axis = -1);

    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    static ComputeTensor full(const std::vector<int> &s, T fill_value)
    {
        auto z = zeros(s, tensor_type<T>::value());
        return z + fill_value;
    }

    operator TensorPtr() const
    {
        return m_tensor;
    }

    size_t size() const
    {
        size_t s = 0;
        if (m_tensor == nullptr || !m_tensor->data<void>())
        {
            return s;
        }
        s = m_tensor->size();
        return s;
    }

    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    operator std::vector<T>() const
    {
        std::vector<T> list;
        if (m_tensor == nullptr || !m_tensor->data<void>())
        {
            return list;
        }
        list.resize(m_tensor->shape().size());
        auto t = astype(tensor_type<T>::value());
        t->make_memory_continuous();
        memcpy(&list[0], t->data(), list.size() * Tensor::dtype_size(tensor_type<T>::value()));
        return list;
    }

    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    operator std::vector<T>()
    {
        std::vector<T> list;
        if (m_tensor == nullptr || !m_tensor->data<void>())
        {
            return list;
        }
        list.resize(m_tensor->shape().size());
        auto t = astype(tensor_type<T>::value());
        t->make_memory_continuous();
        memcpy(&list[0], t->data(), list.size() * Tensor::dtype_size(tensor_type<T>::value()));
        return list;
    }

    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    operator T() const
    {
        if (m_tensor == nullptr || !m_tensor->data<void>())
        {
            return std::numeric_limits<T>::min();
        }
        if (tensor_type<T>::value() != m_tensor->m_type)
        {
            ComputeTensor nt = (this->astype(tensor_type<T>::value()));
            return nt.m_tensor->data<T>()[0];
        }
        return m_tensor->data<T>()[0];
    }

    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    operator T()
    {
        if (m_tensor == nullptr || !m_tensor->data<void>())
        {
            return std::numeric_limits<T>::min();
        }
        if (tensor_type<T>::value() != m_tensor->m_type)
        {
            ComputeTensor nt = (this->astype(tensor_type<T>::value()));
            return nt.m_tensor->data<T>()[0];
        }
        return m_tensor->data<T>()[0];
    }

    bool operator==(const ComputeTensor &other) const;
    bool operator!=(const ComputeTensor &other) const
    {
        return !operator==(other);
    }
    template <typename T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    bool operator!=(T t) const
    {
        auto other = ComputeTensor::full(m_tensor->shape(), t);
        return !operator==(other);
    }

    Tensor &operator*()
    {
        return *m_tensor;
    }

    Tensor *operator->()
    {
        return m_tensor.get();
    }

    Tensor *operator->() const
    {
        return m_tensor.get();
    }

    template <typename F>
    bool all_of(F &&pred) const
    {
        using traits = function_traits<std::decay_t<F>>;
        using T = typename traits::template arg<0>::type;
        T *data = m_tensor->data<T>();
        for (auto i = 0u; i < size(); ++i)
        {
            if (!pred(data[i]))
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief squeeze or unsqueeze operation
     * squeeze: remove a 1-dim at (dim) position in shape
     * unsqueeze: insert a 1-dim at (dim) position in shape
     * @param dim target index of squeeze/unsqueeze
     * @return ComputeTensor the result
     */
    ComputeTensor squeeze(int dim = 0);
    ComputeTensor unsqueeze(int dim = 0);

    /**
     * @brief Return evenly spaced values within a given interval.
     * @param start: start of inteval
     * @param end: end of interval
     * @param step: spacing between values, and default=1
     * @return ComputeTensor
     */
    static ComputeTensor arange(const int32_t start, const int32_t end, const int32_t step = 1)
    {
        if (start > end)
        {
            WARN() << "please make sure the start value " << start << " is less than the end value " << end << " in ComputeTensor::arange.";
            return ComputeTensor(nullptr);
        }

        if (step <= 0)
        {
            WARN() << "please set the step(now = " << step << ") > 0 in ComputeTensor::arange.";
            return ComputeTensor(nullptr);
        }

        int32_t size = (end - start) / step;
        size = (size * step + start) == end ? size : size + 1;
        std::vector<int32_t> ss = {size};
        ComputeTensor ct = zeros(ss, Tensor::INT64);
        auto data = ct->data<int64_t>();

        for (int32_t i = start, offset = 0; i < end; i += step, offset++)
        {
            data[offset] = i;
        }
        return ct;
    }

    /**
     * @brief Return a list of coordinate matrices from coordinate vectors.
     * @param xi: 1-D arrays representing the coordinated of a grid
     * @param indexing: Cartesian ('xy', default)
     * @return std::vector<ComputeTensor>
     */
    static std::vector<ComputeTensor> meshgrid(std::vector<ComputeTensor> &xi, const std::string indexing = "xy")
    {
        if (indexing != "xy")
        {
            ERROR() << "now meshgrid only supports Cartesian.";
            return std::vector<ComputeTensor>();
        }

        for (auto &x : xi)
        {
            if (x->shape().dim() != 1)
            {
                ERROR() << "now meshgrid only supports 1-D array.";
                return std::vector<ComputeTensor>();
            }
        }

        auto xi_size = xi.size();
        if (xi_size > 2)
        {
            ERROR() << "now meshgrid only supports 2 elements in inputs.";
            return std::vector<ComputeTensor>();
        }

        int32_t x0_len = xi[0]->size();
        int32_t x1_len = xi[1]->size();

        std::vector<ComputeTensor> x0s(x1_len, xi[0]);
        std::vector<ComputeTensor> y0s(x0_len, xi[1]);

        auto xv = ComputeTensor::concat(x0s);
        TensorShape xv_shape(x1_len, x0_len);
        xv->shape() = xv_shape;
        auto yv0 = ComputeTensor::concat(y0s); //.permute(perm);
        yv0->shape() = TensorShape(x0_len, x1_len);
        const std::vector<int32_t> perm = {1, 0};
        auto yv = yv0.permute(perm);

        return std::vector<ComputeTensor>{xv, yv};
    }

    /**
     * @brief return a copy of the tensor collapsed into one dimension
     * @return ComputeTensor
     */
    ComputeTensor flatten() const;

    /**
     * @brief perform a matrix mul: Tensor(this) X Tensor(b)
     *
     * @param b The multiplier Tensor
     * @param trans_a is transpose this
     * @param trans_b is transpose b
     * @return ComputeTensor the result
     */
    ComputeTensor matmul(ComputeTensor b, bool trans_a = false, bool trans_b = false) const;

    ComputeTensor permute(const std::vector<int> &perm) const;
    /**
     * @brief merge multiple tensors along axis.
     *
     * @param tensors is a vector of compute tensors
     * @param dim is the dimension that will concatenate
     * tensors together
     * @return a new compute tensor
     */
    static ComputeTensor concat(std::vector<ComputeTensor> &tensors, int dim = 0)
    {
        if (tensors.size() == 0)
        {
            WARN() << "Failed at ComputeTensor::concat due to tensors is empty.";
            return ComputeTensor(nullptr);
        }
        auto base_shape = tensors[0]->shape();
        int total_dim = base_shape.dim();
        int total_batch = base_shape[dim];
        // check shape consistency
        for (size_t i = 1; i < tensors.size(); i++)
        {
            if (!tensors[i])
            {
                WARN() << "Tensor " << i << " in ComputeTensor::concat list is nullptr.";
                continue;
            }
            auto shape = tensors[i]->shape();
            if (shape.dim() != total_dim)
            {
                WARN() << "ComputeTensor::Concat failed. Tensor dims do not match.";
                return nullptr;
            }
            for (int d = 0; d < shape.dim(); d++)
            {
                if (shape[d] != base_shape[d] && dim != d)
                {
                    WARN() << "ComputeTensor::Concat failed. Tensor Shapes do not match.";
                    return nullptr;
                }
                if (dim == d)
                {
                    total_batch += shape[d];
                }
            }
        }
        // permute target dim to first dim
        std::vector<int> permute_array;
        permute_array.push_back(dim);
        for (int i = 0; i < total_dim; i++)
        {
            if (i != dim)
            {
                permute_array.push_back(i);
            }
        }
        std::vector<ComputeTensor> permuted_tensors;
        for (size_t i = 0; i < tensors.size(); i++)
        {
            if (!tensors[i])
            {
                continue;
            }
            permuted_tensors.push_back(tensors[i].permute(permute_array));
        }
        // Concat first dim
        auto permuted_total_batch_shape = TensorShape(permuted_tensors[0]->shape());
        permuted_total_batch_shape[0] = total_batch;
        auto merged_tensor = zeros(permuted_total_batch_shape, tensors[0]->type());
        int current_batch_idx = 0;
        for (size_t i = 0; i < permuted_tensors.size(); i++)
        {
            int batch = permuted_tensors[i]->shape()[0];
            merged_tensor[{{current_batch_idx, current_batch_idx + batch}}] = permuted_tensors[i];
            current_batch_idx += batch;
        }
        // Reverse to origin dim permutation
        std::vector<int> reverse_array;
        for (int i = 1; i < total_dim; i++)
        {
            reverse_array.push_back(i);
        }
        reverse_array.insert(reverse_array.begin() + dim, 0);
        return merged_tensor.permute(reverse_array);
    }
    static ComputeTensor concat(std::initializer_list<ComputeTensor> tensors, int dim = 0)
    {
        std::vector<ComputeTensor> tensors_v = tensors;
        return concat(tensors_v, dim);
    }

    /**
     * @brief split tensor along axis into multiple tensors
     * @param axis axis to be splited
     * @return vector of splited computeTensor
     */
    std::vector<ComputeTensor> split(unsigned int group = 1, signed int axis = 0) const;

    std::vector<ComputeTensor> split(const std::vector<unsigned int> &splits, signed int axis = 0) const;

    /**
     * @brief reduce sum along axis, if axis is empty, will reduce on all axes.
     *          if axis is illeagle, will return nullptr
     *
     * @param axis axis to be reduce
     * @return ComputeTensor with int64 type for int(8/16/32 singed unsigned) type, float64 for float type.
     */
    ComputeTensor sum(const std::vector<int> &axis = std::vector<int>(), bool keepdim = false) const;
    /**
     * @brief reduce mean along axis, if axis is empty, will reduce on all axes.
     *          if axis is illeagle, will return nullptr
     *
     * @param axis axis to be reduce
     * @return ComputeTensor with  float64 type
     */
    ComputeTensor mean(const std::vector<int> &axis = std::vector<int>(), bool keepdim = false) const;
    /**
     * @brief reduce prod along axis, if axis is empty, will reduce on all axes.
     *          if axis is illeagle, will return nullptr
     *
     * @param axis axis to be reduce
     * @return ComputeTensor with int64 type for int(8/16/32/64 singed unsigned) type, float64 for float type.
     */
    ComputeTensor prod(const std::vector<int> &axis = std::vector<int>(), bool keepdim = false) const;
    /**
     * @brief reduce min along axis, if axis is empty, will reduce on all axes.
     *          if axis is illeagle, will return nullptr
     *
     * @param axis axis to be reduce
     * @return ComputeTensor with int64 type for int(8/16/32/64 singed unsigned) type, float64 for float type.
     */
    ComputeTensor min(const std::vector<int> &axis = std::vector<int>(), bool keepdim = false) const;
    /**
     * @brief reduce max along axis, if axis is empty, will reduce on all axes.
     *          if axis is illeagle, will return nullptr
     *
     * @param axis axis to be reduce
     * @return ComputeTensor with int64 type for int(8/16/32/64 singed unsigned) type, float64 for float type.
     */
    ComputeTensor max(const std::vector<int> &axis = std::vector<int>(), bool keepdim = false) const;

    ComputeTensor abs() const;
    ComputeTensor cos() const;
    ComputeTensor cosh() const;
    ComputeTensor exp() const;
    ComputeTensor log() const;
    ComputeTensor log2() const;
    ComputeTensor log10() const;
    ComputeTensor sin() const;
    ComputeTensor sinh() const;
    ComputeTensor sqrt() const;
    ComputeTensor tan() const;
    ComputeTensor atan() const;
    ComputeTensor atanh() const;
    ComputeTensor asinh() const;
    ComputeTensor asin() const;
    ComputeTensor acosh() const;
    ComputeTensor acos() const;
    ComputeTensor pow(double index) const;
    ComputeTensor sigmoid() const;
    ComputeTensor tanh() const;
    ComputeTensor ceil() const;
    ComputeTensor floor() const;
    ComputeTensor round() const;
    ComputeTensor clamp(double min, double max) const;
    ComputeTensor celu(float alpha) const;
    ComputeTensor elu(float alpha) const;
    ComputeTensor softplus() const;
    ComputeTensor softsign() const;
    ComputeTensor mish() const;
    ComputeTensor erf() const;

    ComputeTensor astype(Tensor::Dtype dtype) const;
    ComputeTensor view(Tensor::Dtype dtype);
    ComputeTensor unique(void) const;

    ComputeTensor pow(ComputeTensor t) const;
    ComputeTensor maximum(ComputeTensor t) const;
    ComputeTensor minimum(ComputeTensor t) const;
    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    ComputeTensor maximum(T b) const
    {
        return (*this).maximum(ComputeTensor::create(b));
    }
    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    ComputeTensor minimum(T b) const
    {
        return (*this).minimum(ComputeTensor::create(b));
    }

    DataAccessor operator[](int i) const;
    DataAccessor operator[](Slice s) const;
    // DataAccessor operator[](const std::initializer_list<int> &s);
    DataAccessor operator[](Slices ss) const
    {
        return DataAccessor((ComputeTensor &)*this, ss);
    }

    DataAccessor operator[](const std::initializer_list<std::initializer_list<int>> &sl) const
    {
        Slices s;
        for (auto arg : sl)
        {
            s.push_back(Slice(arg));
        }
        return DataAccessor((ComputeTensor &)*this, s);
    }

    // template <typename... Args>
    // DataAccessor operator()(Args &&... args)
    // {
    //     Slices s;
    //     for (const auto &arg : {args...})
    //     {
    //         s.push_back(arg);
    //     }
    //     return DataAccessor(*this, s);
    // }

    template <typename... T>
    ComputeTensor reshape(T &&...arg) const
    {
        TensorPtr ret = std::make_shared<Tensor>(*m_tensor);
        ret->set_name();
        if (!ret->shape().reshape(std::forward<T>(arg)...))
        {
            WARN() << "Failed at ComputeTensor::reshape due to shape is not consistent with " << m_tensor->shape() << ".";
            return ComputeTensor(nullptr);
        }
        return create(ret);
    }

    ComputeTensor operator*(ComputeTensor t) const;
    ComputeTensor operator&(ComputeTensor t) const;
    ComputeTensor operator|(ComputeTensor t) const;

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator&(ComputeTensor a, T b)
    {
        return (a)&ComputeTensor::create(b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator|(ComputeTensor a, T b)
    {
        return (a) | ComputeTensor::create(b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator*(ComputeTensor a, T b)
    {
        return (a)*ComputeTensor::create(b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator*(T b, ComputeTensor a)
    {
        return (a)*ComputeTensor::create(b);
    }

    ComputeTensor operator/(ComputeTensor t) const;

    friend ComputeTensor operator/(ComputeTensor a, DataAccessor b)
    {
        return (a) / ((ComputeTensor)b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator/(ComputeTensor a, T b)
    {
        return (a) / ComputeTensor::create(b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator/(T b, ComputeTensor a)
    {
        return ComputeTensor::create(b) / a;
    }

    ComputeTensor operator+(ComputeTensor t) const;

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator+(ComputeTensor a, T b)
    {
        return (a) + ComputeTensor::create(b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator+(T b, ComputeTensor a)
    {
        return (a) + ComputeTensor::create(b);
    }

    ComputeTensor operator-(ComputeTensor t) const;

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator-(ComputeTensor a, T b)
    {
        return (a)-ComputeTensor::create(b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator-(T b, ComputeTensor a)
    {
        return ComputeTensor::create(b) - a;
    }

    ComputeTensor operator<(ComputeTensor t) const;

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator<(ComputeTensor a, T b)
    {
        return (a) < ComputeTensor::create(b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator<(T b, ComputeTensor a)
    {
        return ComputeTensor::create(b) < a;
    }

    ComputeTensor operator>(ComputeTensor t) const;
    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator>(ComputeTensor a, T b)
    {
        return (a) > ComputeTensor::create(b);
    }

    template <class T, typename = std::enable_if_t<is_supported_tensor_type<T>::value>>
    friend ComputeTensor operator>(T b, ComputeTensor a)
    {
        return ComputeTensor::create(b) > a;
    }

    ComputeTensor operator>=(ComputeTensor t) const
    {
        return 1 - ((*this) < t);
    }

    friend std::ostream &operator<<(std::ostream &oss, ComputeTensor a);

    operator bool() const
    {
        return m_tensor.get() != nullptr;
    }
    operator bool()
    {
        return m_tensor.get() != nullptr;
    }
    bool operator!() const
    {
        return !m_tensor;
    }

protected:
    ComputeTensor slice(Slice s, int axis) const;
    ComputeTensor slice(Slices slices) const;
    bool check_reduce_axis(const std::vector<int> &axis, std::vector<int> &out_shape, bool keepdim = false) const;
    static void print_data(std::ostream &oss, ComputeTensor a, uint32_t offset, uint32_t length);
    TensorPtr m_tensor;
};

std::ostream &operator<<(std::ostream &oss, const ComputeTensor::DataAccessor &a);
template <>
struct is_compute_tensor_support_type<ComputeTensor::DataAccessor>
{
    static constexpr bool value = true;
};
} // namespace aipubt
#endif //! CORE_COMPUTE_TENSOR_H

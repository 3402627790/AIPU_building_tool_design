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
#pragma once
#ifndef CORE_QUANTIZATION_H
#define CORE_QUANTIZATION_H
#include "core/tensor.h"
#include <core/compute_tensor.h>
#include <stdint.h>
#include <utility>
#include <vector>

namespace aipubt
{

class ComputeTensor;
class Quantization;
class Quantization
{
public:
    enum Granularity
    {
        Granularity_Void = 0,
        Per_Tensor = 1,
        Per_Group = 2,
        Per_Channel = 3,
    };
    enum Scheme
    {
        Scheme_Void = 0,
        Symmetric = 1,
        Asymmetric = 2,
    };
    enum QuantizedRange
    {
        Range_Void = 0,
        Full_Range = 1,
        Restricted_Range = 2,
    };

    static const std::string scheme_to_string(const Quantization::Scheme &s);
    static const std::string granularity_to_string(const Quantization::Granularity &g);
    static const std::string quantize_range_to_string(const Quantization::QuantizedRange &gr);

    Quantization() {}

    Quantization(float scale,
                 int32_t offset,
                 int bits = -1,
                 Scheme scheme = Scheme::Symmetric)
        : m_bits(bits), m_scheme(scheme)
    {
        set_scale(scale);
        set_offset(offset);
    }

    Quantization(const std::vector<float> &scales,
                 int32_t offset,
                 int32_t bits = -1,
                 Scheme scheme = Scheme::Symmetric)
        : m_bits(bits), m_scheme(scheme)
    {
        set_scales(scales);
        set_offset(offset);
    }

    Quantization(const std::vector<float> &scales,
                 const std::vector<int32_t> &offsets,
                 int32_t bits = -1,
                 Scheme scheme = Scheme::Symmetric)
        : m_bits(bits), m_scheme(scheme)
    {
        set_scales(scales);
        set_offsets(offsets);
    }

    Quantization(float min,
                 float max,
                 int32_t bits = -1,
                 Scheme scheme = Scheme::Symmetric)
        : m_bits(bits), m_scheme(scheme)
    {
        set_min(min);
        set_max(max);
    }

    Quantization(const std::vector<float> &mins,
                 const std::vector<float> &maxs,
                 int32_t bits = -1,
                 Scheme scheme = Scheme::Symmetric)
        : m_bits(bits), m_scheme(scheme)
    {
        set_mins(mins);
        set_maxs(maxs);
    }

    Quantization(const Quantization &other)
        : m_scales(other.m_scales),
          m_offsets(other.m_offsets),
          m_min(other.m_min),
          m_max(other.m_max),
          m_bits(other.m_bits),
          m_scheme(other.m_scheme),
          m_aipu_dtype(other.m_aipu_dtype),
          m_quantization_dim(other.m_quantization_dim),
          m_granularity(other.m_granularity),
          m_qinvariant(other.m_qinvariant),
          m_quantized(other.m_quantized),
          m_quantized_range(other.m_quantized_range),
          m_group(other.m_group),
          m_lut_items_bits(other.m_lut_items_bits),
          m_unquantifiable(other.m_unquantifiable),
          m_data_unchanged(other.m_data_unchanged),
          m_using_qtlib(other.m_using_qtlib)
    {
    }

    ~Quantization() {}

    bool operator==(const Quantization &other) const
    {
        return ((m_scales == other.m_scales) &&
                (m_offsets == other.m_offsets) &&
                (m_bits == other.m_bits) &&
                (m_quantization_dim == other.m_quantization_dim) &&
                (m_aipu_dtype == other.m_aipu_dtype) &&
                (m_min == other.m_min) &&
                (m_max == other.m_max) &&
                (m_scheme == other.m_scheme) &&
                (m_granularity == other.m_granularity) &&
                (m_quantized == other.m_quantized) &&
                (m_quantized_range == other.m_quantized_range) &&
                (m_qinvariant == other.m_qinvariant) &&
                (m_group == other.m_group) &&
                (m_lut_items_bits == other.m_lut_items_bits) &&
                (m_unquantifiable == other.m_unquantifiable) &&
                (m_data_unchanged == other.m_data_unchanged) &&
                (m_using_qtlib == other.m_using_qtlib));
    }

    bool operator!=(const Quantization &other) const
    {
        return !(*this == other);
    }

    Quantization &operator=(const Quantization &other)
    {
        if (this != &other)
        {
            m_scales = other.m_scales;
            m_offsets = other.m_offsets;
            m_bits = other.m_bits;
            m_quantization_dim = other.m_quantization_dim;
            m_aipu_dtype = other.m_aipu_dtype;
            m_min = other.m_min;
            m_max = other.m_max;
            m_scheme = other.m_scheme;
            m_granularity = other.m_granularity;
            m_quantized = other.m_quantized;
            m_quantized_range = other.m_quantized_range;
            m_qinvariant = other.m_qinvariant;
            m_group = other.m_group;
            m_lut_items_bits = other.m_lut_items_bits;
            m_unquantifiable = other.m_unquantifiable;
            m_data_unchanged = other.m_data_unchanged;
            m_using_qtlib = other.m_using_qtlib;
        }
        return *this;
    }

    void set(float scale = 1.0, int32_t offset = 0);
    std::pair<float, int32_t> get() const;

    void set(const std::vector<float> &scales, int32_t offset = 0);
    void set(const std::vector<float> &scales, const std::vector<int32_t> &offsets);

    void set_scale(float scale);
    float scale() const;
    void set_scales(const ComputeTensor &scales = ComputeTensor(std::vector<float>())) { m_scales = ComputeTensor(scales); };

    ComputeTensor scales() const
    {
        return m_scales;
    }

    template <typename T>
    T scales() const
    {
        return m_scales;
    }

    void push_scale(float scale);

    void set_offset(int32_t offset);
    int32_t offset() const;
    // void set_offsets(const std::vector<int32_t> &offsets=std::vector<int32_t>());
    void set_offsets(const ComputeTensor &offsets = ComputeTensor(std::vector<int32_t>())) { m_offsets = ComputeTensor(offsets); };

    ComputeTensor offsets() const
    {
        return m_offsets;
    }

    template <typename T>
    T offsets() const
    {
        return m_offsets;
    }

    void push_offset(int32_t offset);

    void set_min(float f_min);
    void set_mins(const std::vector<float> &f_mins);
    std::vector<float> mins(void) const;

    void set_max(float f_max);
    void set_maxs(const std::vector<float> &f_maxs);
    std::vector<float> maxs(void) const;

    void set_granularity(Quantization::Granularity granu);
    Quantization::Granularity granularity(void) const;

    void set_scheme(Quantization::Scheme scheme);
    Quantization::Scheme scheme(void) const;

    void set_qinvariant(bool qinvariant = false);
    bool qinvariant(void) const;

    void set_unquantifiable(bool unquantifiable = false);
    bool unquantifiable(void) const;

    void set_data_unchanged(bool unchanged = false);
    bool data_unchanged(void) const;

    void set_quantized(bool quantized = false);
    bool quantized(void) const;

    void set_bits(int32_t b);
    int32_t bits() const;

    void set_lut_items_bits(int32_t b);
    int32_t lut_items_bits() const;

    void set_quantized_range(Quantization::QuantizedRange range);
    Quantization::QuantizedRange quantized_range(void) const;

    void set_group(int32_t group);
    int32_t group(void) const;

    void set_quantization_dim(int32_t quantization_dim);
    int32_t quantization_dim() const;

    void set_aipu_dtype(Tensor::Dtype dtype);
    Tensor::Dtype aipu_dtype() const;

    void set_using_qtlib(bool uq);
    bool using_qtlib(void) const;

    bool empty() const;

    friend std::ostream &operator<<(std::ostream &os, const Quantization &qinfo);

private:
    ComputeTensor m_scales;
    ComputeTensor m_offsets;
    ComputeTensor m_min;
    ComputeTensor m_max;
    int32_t m_bits = -1;
    mutable Scheme m_scheme = Scheme::Scheme_Void;
    Tensor::Dtype m_aipu_dtype = Tensor::Dtype::VOID;
    int32_t m_quantization_dim = 0;
    Granularity m_granularity = Granularity::Granularity_Void;
    // m_qinvariant means this tensor can quantize but scale = 1.0, zp = 0, like index.
    bool m_qinvariant = false;
    // m_quantized means this tensor has quantized or not.
    bool m_quantized = false;
    QuantizedRange m_quantized_range = QuantizedRange::Range_Void;
    int32_t m_group = 1;
    int32_t m_lut_items_bits = -1;
    // m_unquantifiable means this tensor will not to quantize, like it will be float32/float16
    bool m_unquantifiable = false;
    // m_data_unchange means the data which this tensor has will not dequantize, or not convert.
    bool m_data_unchanged = false;

    bool m_using_qtlib = true;
};

} // namespace aipubt
#endif //!CORE_QUANTIZATION_H

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
#pragma once
#ifndef CORE_QUANTIZATION_H
#define CORE_QUANTIZATION_H
#include "core/node.h"
#include "core/tensor.h"
#include <core/compute_tensor.h>
#include <stdint.h>
#include <utility>
#include <vector>

namespace aipubt
{

class ComputeTensor;
class Quantization;

/**
 * @class Quantization
 * @brief Describes the quantization configuration and computed scales/offsets
 *        for a single tensor.
 *
 * Usage:
 * @code
 *   Quantization q;
 *   q.set_scheme(Scheme::Symmetric);
 *   .set_bits(8);
 *   .set_granularity(Granularity::PerTensor);
 *   q.set_scales({0.0235f});
 *   q.set_offsets({0});
 * @endcode
 */

class Quantization
{
public:
    enum Scheme
    {
        SchemeNone = 0,
        Symmetric = 1,
        Asymmetric = 2,
        NonLinear = 3,
    };

    enum Granularity
    {
        /*
        1. Per-tensor (per-layer) quantization: y_scale is a scalar.
        2. Per-axis quantization: The scale must be a 1-D tensor, with the length of the quantization axis.
        For an input shape (D0, ..., Di, ..., Dn) and axis=i, y_scale is a 1-D tensor of length Di.
        3. Blocked quantization: The scale’s shape is identical to the input’s shape, except for one dimension,
        in which blocking is performed. Given x shape (D0, ..., Di, ..., Dn), axis=i,
        and block size B: y_scale shape is (D0, ..., ceil(Di/B), ..., Dn).
        */
        GranularityNone = 0,
        PerTensor = 1,
        PerAxis = 2,
        PerBlock = 3,
    };

    enum RoundMode
    {
        RoundToEven = 0,
        Ceil = 1,
        Floor = 2,
        Trunc = 3,
    };

    enum OverflowPolicy
    {
        Saturate = 0,
        Truncate = 1,
    };

    enum CalibrationStrategy
    {
        MinMax = 0,
        Mean = 1,
        KLDivergence = 2,
        MSE = 3,
        Histogram = 4,
        Percentile = 5,
        Custom = 6,
    };

    enum QuantizedRange
    {
        QuantizedRangeNone = 0,
        FullRange = 1,
        RestrictedRange = 2,
    };

    static const std::string scheme_to_string(const Scheme &s);
    static Scheme string_to_scheme(const std::string &scheme);
    static const std::string granularity_to_string(const Granularity &s);
    static Granularity string_to_granularity(const std::string &scheme);
    static const std::string round_mode_to_string(const RoundMode &rm);
    static RoundMode string_to_round_mode(const std::string &rm);
    static const std::string calibration_strategy_to_string(const CalibrationStrategy &rm);
    static CalibrationStrategy string_to_calibration_strategy(const std::string &rm);
    static const std::string overflow_policy_to_string(const OverflowPolicy &op);
    static OverflowPolicy string_to_overflow_policy(const std::string &op);
    static const std::string quantized_range_to_string(const QuantizedRange &qr);
    static QuantizedRange string_to_quantized_range(const std::string &qr);

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
          m_qinvariant(other.m_qinvariant),
          m_quantized(other.m_quantized),
          m_unquantifiable(other.m_unquantifiable),
          m_data_unchanged(other.m_data_unchanged),
          m_using_qtlib(other.m_using_qtlib),
          m_is_dynamic(other.m_is_dynamic),
          m_block_size(other.m_block_size),
          m_lut_items_bits(other.m_lut_items_bits),
          m_bits(other.m_bits),
          m_axis(other.m_axis),
          m_method(other.m_method),
          m_dtype(other.m_dtype),
          m_scheme(other.m_scheme),
          m_granularity(other.m_granularity),
          m_quantized_range(other.m_quantized_range),
          m_round_mode(other.m_round_mode),
          m_overflow_policy(other.m_overflow_policy),
          m_calibration_strategy(other.m_calibration_strategy),
          m_extra_info(other.m_extra_info)
    {
    }

    ~Quantization() {}

    bool operator==(const Quantization &other) const
    {
        return ((m_scales == other.m_scales) &&
                (m_offsets == other.m_offsets) &&
                (m_min == other.m_min) &&
                (m_max == other.m_max) &&
                (m_qinvariant == other.m_qinvariant) &&
                (m_quantized == other.m_quantized) &&
                (m_unquantifiable == other.m_unquantifiable) &&
                (m_data_unchanged == other.m_data_unchanged) &&
                (m_using_qtlib == other.m_using_qtlib) &&
                (m_is_dynamic == other.m_is_dynamic) &&
                (m_block_size == other.m_block_size) &&
                (m_lut_items_bits == other.m_lut_items_bits) &&
                (m_bits == other.m_bits) &&
                (m_axis == other.m_axis) &&
                (m_method == other.m_method) &&
                (m_dtype == other.m_dtype) &&
                (m_scheme == other.m_scheme) &&
                (m_granularity == other.m_granularity) &&
                (m_quantized_range == other.m_quantized_range) &&
                (m_round_mode == other.m_round_mode) &&
                (m_overflow_policy == other.m_overflow_policy) &&
                (m_calibration_strategy == other.m_calibration_strategy) &&
                (m_extra_info == other.m_extra_info));
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
            m_min = other.m_min;
            m_max = other.m_max;
            m_qinvariant = other.m_qinvariant;
            m_quantized = other.m_quantized;
            m_unquantifiable = other.m_unquantifiable;
            m_data_unchanged = other.m_data_unchanged;
            m_using_qtlib = other.m_using_qtlib;
            m_is_dynamic = other.m_is_dynamic;
            m_block_size = other.m_block_size;
            m_lut_items_bits = other.m_lut_items_bits;
            m_bits = other.m_bits;
            m_axis = other.m_axis;
            m_method = other.m_method;
            m_dtype = other.m_dtype;
            m_scheme = other.m_scheme;
            m_granularity = other.m_granularity;
            m_quantized_range = other.m_quantized_range;
            m_round_mode = other.m_round_mode;
            m_calibration_strategy = other.m_calibration_strategy;
            m_overflow_policy = other.m_overflow_policy;
            m_extra_info = other.m_extra_info;
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

    void set_mins(const ComputeTensor &fmins = ComputeTensor(std::vector<float>())) { m_min = ComputeTensor(fmins); }
    ComputeTensor mins() const
    {
        return m_min;
    }
    template <typename T>
    T mins() const
    {
        return m_min;
    }

    void set_max(float f_max);
    // void set_maxs(const std::vector<float> &f_maxs);
    // std::vector<float> maxs(void) const;
    void set_maxs(const ComputeTensor &fmaxs = ComputeTensor(std::vector<float>())) { m_max = ComputeTensor(fmaxs); }
    ComputeTensor maxs() const
    {
        return m_max;
    }
    template <typename T>
    T maxs() const
    {
        return m_max;
    }

    Quantization &set_granularity(Quantization::Granularity granu);
    Quantization &set_granularity(const std::string &granu)
    {
        return set_granularity(string_to_granularity(granu));
    }
    Quantization::Granularity granularity(void) const;

    Quantization &set_scheme(Quantization::Scheme scheme);
    Quantization &set_scheme(const std::string &scheme)
    {
        return set_scheme(string_to_scheme(scheme));
    }
    Quantization::Scheme scheme(void) const;

    Quantization &set_round_mode(Quantization::RoundMode round_mode);
    Quantization &set_round_mode(const std::string &round_mode)
    {
        return set_round_mode(string_to_round_mode(round_mode));
    }
    Quantization::RoundMode round_mode(void) const;

    Quantization &set_calibration_strategy(Quantization::CalibrationStrategy cs);
    Quantization &set_calibration_strategy(const std::string &cs)
    {
        return set_calibration_strategy(string_to_calibration_strategy(cs));
    }
    Quantization::CalibrationStrategy calibration_strategy(void) const;

    Quantization &set_overflow_policy(Quantization::OverflowPolicy cs);
    Quantization &set_overflow_policy(const std::string &cs)
    {
        return set_overflow_policy(string_to_overflow_policy(cs));
    }
    Quantization::OverflowPolicy overflow_policy(void) const;

    Quantization &set_quantized_range(Quantization::QuantizedRange range);
    Quantization &set_quantized_range(const std::string &range)
    {
        return set_quantized_range(string_to_quantized_range(range));
    }
    Quantization::QuantizedRange quantized_range(void) const;

    Quantization &set_qinvariant(const bool &qinvariant = false);
    bool qinvariant(void) const;

    void set_unquantifiable(const bool &unquantifiable = false);
    bool unquantifiable(void) const;

    void set_data_unchanged(const bool &unchanged = false);
    bool data_unchanged(void) const;

    void set_quantized(const bool &quantized = false);
    bool quantized(void) const;

    Quantization &set_dynamic(const bool &is_dynamic);
    bool dynamic(void) const;

    Quantization &set_bits(const int32_t &b);
    int32_t bits() const;

    Quantization &set_lut_items_bits(const int32_t &b);
    int32_t lut_items_bits() const;

    Quantization &set_block_size(const int32_t &group);
    int32_t block_size(void) const;

    Quantization &set_axis(const int32_t &quantization_dim);
    int32_t axis() const;

    Quantization &set_dtype(const Tensor::Dtype &dtype);
    Quantization &set_dtype(const std::string &dtype)
    {
        return set_dtype(aipubt::Tensor::string_to_dtype(dtype));
    }
    Tensor::Dtype dtype() const;

    Quantization &set_using_qtlib(const bool &uq);
    bool using_qtlib(void) const;

    bool empty() const;

    bool has_update() const;
    void erase_extra_info(const std::string &key)
    {
        m_extra_info.erase(key);
    }

    std::map<std::string, NodeParamValue> &extra_info()
    {
        return m_extra_info;
    }

    const std::map<std::string, NodeParamValue> &extra_info() const
    {
        return m_extra_info;
    }

    static NodeParamValue get_value_from_extra_info(const std::map<std::string, NodeParamValue> &m, const std::string &key, NodeParamValue &&default_val = NodeParamValue{})
    {
        auto it = m.find(key);
        return (it != m.end()) ? it->second : default_val;
    }

    Quantization &set_method(const std::string &method);
    const std::string method() const;

    void serialize(std::ostream &oss, std::ostream &wt_oss, std::string key);

    friend std::ostream &operator<<(std::ostream &os, const Quantization &qinfo);
    friend std::ostream &operator<<(std::ostream &os, const Quantization::Scheme &sh);
    friend std::ostream &operator<<(std::ostream &os, const Quantization::Granularity &qcs);
    friend std::ostream &operator<<(std::ostream &os, const Quantization::CalibrationStrategy &qcs);
    friend std::ostream &operator<<(std::ostream &os, const Quantization::RoundMode &rm);

private:
    void do_serialize(std::ostream &oss,
                      std::ostream &wt_oss,
                      std::string key);

    void do_dynamic_serialize(std::ostream &oss,
                              std::ostream &wt_oss,
                              std::string key);

    void do_static_serialize(std::ostream &oss,
                             std::ostream &wt_oss,
                             std::string key);

    ComputeTensor m_scales;
    ComputeTensor m_offsets;
    ComputeTensor m_min;
    ComputeTensor m_max;
    // m_qinvariant means this tensor can quantize but scale = 1.0, zp = 0, like index.
    bool m_qinvariant = false;
    // m_quantized means this tensor has quantized or not.
    bool m_quantized = false;
    // m_unquantifiable means this tensor will not to quantize, like it will be float32/float16
    bool m_unquantifiable = false;
    // m_data_unchange means the data which this tensor has will not dequantize, or not convert.
    bool m_data_unchanged = false;
    bool m_using_qtlib = true;
    bool m_is_dynamic = false;
    int32_t m_block_size = 1;
    int32_t m_lut_items_bits = -1;
    int32_t m_bits = -1;
    int32_t m_axis = 0;
    std::string m_method;
    Tensor::Dtype m_dtype = Tensor::Dtype::VOID;
    mutable Scheme m_scheme = Scheme::SchemeNone;
    Granularity m_granularity = Granularity::GranularityNone;
    QuantizedRange m_quantized_range = QuantizedRange::QuantizedRangeNone;
    RoundMode m_round_mode = RoundMode::RoundToEven;
    OverflowPolicy m_overflow_policy = OverflowPolicy::Saturate;
    CalibrationStrategy m_calibration_strategy = CalibrationStrategy::MinMax;
    std::map<std::string, NodeParamValue> m_extra_info;
};

} // namespace aipubt
#endif //! CORE_QUANTIZATION_H

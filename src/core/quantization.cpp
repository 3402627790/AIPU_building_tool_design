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
 * @Date: 2021-07-01 14:08:19
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2026-07-20 07:17:38
 */

#include <core/quantization.h>
#include <core/tensor.h>

namespace aipubt
{
/**
 * @brief Generic lookup in a static map with error logging on miss.
 *
 * @tparam Map   std::map type (e.g. std::map<Scheme, std::string>).
 * @param m       The static map.
 * @param key     Lookup key.
 * @param def     Default value to return when key is not found.
 * @param name    Human-readable enum name for the error message
 *                (e.g. "scheme", "granularity").
 * @return Found value, or @p def if not found.
 */
template <typename Map>
static std::string lookup_to_string(const Map &m,
                                    const typename Map::key_type &key,
                                    const std::string &def,
                                    const char *name)
{
    auto it = m.find(key);
    if (it != m.end())
        return it->second;
    ERROR() << name << " failed, because unknown value = " << static_cast<int64_t>(key) << ", and return value " << def;
    return def;
}

template <typename Map>
static typename Map::mapped_type lookup_from_string(const Map &m,
                                                    const std::string &key,
                                                    const typename Map::mapped_type &def,
                                                    const char *name)
{
    auto tkey = key;
    std::transform(key.begin(), key.end(), tkey.begin(), ::tolower);
    auto it = m.find(tkey);
    if (it != m.end())
        return it->second;
    ERROR() << name << " failed, because unknown value = " << tkey << ", and return value " << static_cast<int64_t>(def);
    return def;
}

namespace
{
    std::string float_to_string(float f)
    {
        {
            std::stringstream oss;
            oss.precision(10);
            oss << f;
            std::string text = oss.str();
            if ((text.find("e-") == std::string::npos) && (text.find(".") == std::string::npos))
            {
                text += ".0";
            }
            return text;
        };
    }

} // namespace

const std::string Quantization::scheme_to_string(const Quantization::Scheme &s)
{
    static const std::map<Scheme, std::string> m = {
        {Quantization::Symmetric, "symmetric"},
        {Quantization::Asymmetric, "asymmetric"},
        {Quantization::SchemeNone, "scheme_none"},
        {Quantization::NonLinear, "non_linear"},
    };
    const std::string unknown = "unknown";
    return lookup_to_string(m, s, unknown, "scheme_to_string");
}

Quantization::Scheme Quantization::string_to_scheme(const std::string &s)
{
    static const std::map<std::string, Scheme> m = {
        {"symmetric", Quantization::Symmetric},
        {"asymmetric", Quantization::Asymmetric},
        {"scheme_none", Quantization::SchemeNone},
        {"schemenone", Quantization::SchemeNone},
        {"non_linear", Quantization::NonLinear},
        {"nonlinear", Quantization::NonLinear},
    };
    return lookup_from_string(m, s, Quantization::SchemeNone, "string_to_scheme");
}

const std::string Quantization::granularity_to_string(const Quantization::Granularity &s)
{
    static const std::map<Granularity, std::string> m = {
        {Quantization::GranularityNone, "granularity-none"},
        {Quantization::PerTensor, "per-tensor"},
        {Quantization::PerAxis, "per-axis"},
        {Quantization::PerBlock, "per-block"},
    };
    const std::string unknown = "unknown";
    return lookup_to_string(m, s, unknown, "granularity_to_string");
}

Quantization::Granularity Quantization::string_to_granularity(const std::string &s)
{
    static const std::map<std::string, Granularity> m = {
        {"granularity_none", Quantization::GranularityNone},
        {"granularitynone", Quantization::GranularityNone},
        {"per_tensor", Quantization::PerTensor},
        {"pertensor", Quantization::PerTensor},
        {"per-tensor", Quantization::PerTensor},
        {"per_axis", Quantization::PerAxis},
        {"peraxis", Quantization::PerAxis},
        {"per-axis", Quantization::PerAxis},
        {"per_block", Quantization::PerBlock},
        {"perblock", Quantization::PerBlock},
        {"per-block", Quantization::PerBlock},
    };
    return lookup_from_string(m, s, Quantization::GranularityNone, "string_to_granularity");
}

const std::string Quantization::round_mode_to_string(const Quantization::RoundMode &rm)
{
    static const std::map<RoundMode, std::string> m = {
        {Quantization::RoundToEven, "round_to_even"},
        {Quantization::Ceil, "ceil"},
        {Quantization::Floor, "floor"},
        {Quantization::Trunc, "trunc"},
    };
    static const std::string unknown = "unknown";
    return lookup_to_string(m, rm, unknown, "round_mode_to_string");
}

Quantization::RoundMode Quantization::string_to_round_mode(const std::string &s)
{
    static const std::map<std::string, RoundMode> m = {
        {"round_half_to_even", Quantization::RoundToEven},
        {"roundhalftoeven", Quantization::RoundToEven},
        {"roundtoeven", Quantization::RoundToEven},
        {"round_to_even", Quantization::RoundToEven},
        {"ceil", Quantization::Ceil},
        {"floor", Quantization::Floor},
        {"trunc", Quantization::Trunc},
    };
    return lookup_from_string(m, s, Quantization::RoundToEven, "string_to_round_mode");
}

const std::string Quantization::calibration_strategy_to_string(const Quantization::CalibrationStrategy &cs)
{
    static const std::map<CalibrationStrategy, std::string> m = {
        {Quantization::MinMax, "min_max"},
        {Quantization::Mean, "mean"},
        {Quantization::KLDivergence, "kl_divergence"},
        {Quantization::MSE, "mse"},
        {Quantization::Histogram, "histogram"},
        {Quantization::Percentile, "percentile"},
        {Quantization::Custom, "custom"},
    };
    static const std::string unknown = "unknown";
    return lookup_to_string(m, cs, unknown, "calibration_strategy_to_string");
}

Quantization::CalibrationStrategy Quantization::string_to_calibration_strategy(const std::string &s)
{
    static const std::map<std::string, CalibrationStrategy> m = {
        {"min_max", Quantization::MinMax},
        {"minmax", Quantization::MinMax},
        {"mean", Quantization::Mean},
        {"kl_divergence", Quantization::KLDivergence},
        {"kldivergence", Quantization::KLDivergence},
        {"mse", Quantization::MSE},
        {"histogram", Quantization::Histogram},
        {"percentile", Quantization::Percentile},
        {"custom", Quantization::Custom},
    };
    return lookup_from_string(m, s, Quantization::MinMax, "string_to_calibration_strategy");
}

const std::string Quantization::overflow_policy_to_string(const Quantization::OverflowPolicy &op)
{
    static const std::map<OverflowPolicy, std::string> m = {
        {Quantization::Saturate, "saturate"},
        {Quantization::Truncate, "truncate"},
    };
    static const std::string unknown = "unknown";
    return lookup_to_string(m, op, unknown, "overflow_policy_to_string");
}

Quantization::OverflowPolicy Quantization::string_to_overflow_policy(const std::string &s)
{
    static const std::map<std::string, OverflowPolicy> m = {
        {"saturate", Quantization::Saturate},
        {"truncate", Quantization::Truncate},
    };
    return lookup_from_string(m, s, Quantization::Saturate, "string_to_overflow_policy");
}

const std::string Quantization::quantized_range_to_string(const Quantization::QuantizedRange &qr)
{
    static const std::map<QuantizedRange, std::string> m = {
        {Quantization::QuantizedRangeNone, "quantized_range_none"},
        {Quantization::FullRange, "full_range"},
        {Quantization::RestrictedRange, "restricted_range"},
    };
    static const std::string unknown = "unknown";
    return lookup_to_string(m, qr, unknown, "quantized_range_to_string");
}

Quantization::QuantizedRange Quantization::string_to_quantized_range(const std::string &s)
{
    static const std::map<std::string, QuantizedRange> m = {
        {"quantized_range_none", Quantization::QuantizedRangeNone},
        {"quantizedrangenone", Quantization::QuantizedRangeNone},
        {"full_range", Quantization::FullRange},
        {"fullrange", Quantization::FullRange},
        {"restricted_range", Quantization::RestrictedRange},
        {"restrictedrange", Quantization::RestrictedRange},
    };
    return lookup_from_string(m, s, Quantization::QuantizedRangeNone, "string_to_quantized_range");
}

void Quantization::set(float scale, int32_t offset)
{
    set_scale(scale);
    set_offset(offset);
}

std::pair<float, int32_t> Quantization::get() const
{
    std::vector<float> _scales = scales();
    std::vector<int32_t> _offsets = offsets();
    float scale = _scales.size() > 0 ? _scales[0] : 1.0f;
    int32_t offset = _offsets.size() > 0 ? _offsets[0] : 0;
    return std::make_pair(scale, offset);
}

void Quantization::set(const std::vector<float> &scales, int32_t offset)
{
    set_scales(scales);
    set_offset(offset);
}

void Quantization::set(const std::vector<float> &scales, const std::vector<int32_t> &offsets)
{
    set_scales(scales);
    set_offsets(offsets);
}

void Quantization::set_scale(float scale)
{
    if (m_scales.size())
    {
        std::vector<float> cur_scales = m_scales;
        cur_scales[0] = scale;
        m_scales = ComputeTensor(cur_scales);
    }
    else
    {
        m_scales = ComputeTensor(scale);
    }
}

float Quantization::scale() const
{
    std::vector<float> ss = scales();
    return ss.size() > 0 ? ss[0] : 1.0f;
}

void Quantization::push_scale(float scale)
{
    std::vector<float> cur_scale = m_scales;
    cur_scale.push_back(scale);
    m_scales = ComputeTensor(cur_scale);
}

void Quantization::set_offset(int32_t offset)
{
    if (m_offsets.size())
    {
        std::vector<int32_t> cur_offsets = m_offsets;
        cur_offsets[0] = offset;
        m_offsets = ComputeTensor(cur_offsets);
    }
    else
    {
        m_offsets = ComputeTensor(offset);
    }
}

int32_t Quantization::offset() const
{
    int32_t z = 0;
    if (m_offsets.size())
    {
        z = offsets()[0];
    }
    return z;
}

void Quantization::push_offset(int32_t offset)
{
    if (m_offsets.size())
    {
        std::vector<int32_t> cur_offsets = m_offsets;
        cur_offsets.push_back(offset);
        m_offsets = ComputeTensor(cur_offsets);
    }
    else
    {
        m_offsets = ComputeTensor(offset);
    }
}

Quantization &Quantization::set_axis(const int32_t &axis)
{
    m_axis = axis;
    return *this;
}

int32_t Quantization::axis() const
{
    return m_axis;
}

Quantization &Quantization::set_dtype(const Tensor::Dtype &dtype)
{
    m_dtype = dtype;
    return *this;
}

Tensor::Dtype Quantization::dtype() const
{
    return m_dtype;
}

int32_t Quantization::bits() const
{
    return m_bits;
}

Quantization &Quantization::set_bits(const int32_t &b)
{
    m_bits = b;
    return *this;
}

int32_t Quantization::lut_items_bits() const
{
    return m_lut_items_bits;
}

Quantization &Quantization::set_lut_items_bits(const int32_t &b)
{
    m_lut_items_bits = b;
    return *this;
}

void Quantization::set_min(float f_min)
{
    if (m_min.size())
    {
        std::vector<float> cur_min = m_min;
        cur_min[0] = f_min;
        m_min = ComputeTensor(cur_min);
    }
    else
    {
        m_min = ComputeTensor(f_min);
    }
}

void Quantization::set_max(float f_max)
{
    if (m_max.size())
    {
        std::vector<float> cur_max = m_max;
        cur_max[0] = f_max;
        m_max = ComputeTensor(cur_max);
    }
    else
    {
        m_max = ComputeTensor(f_max);
    }
}

Quantization &Quantization::set_granularity(Quantization::Granularity granu)
{
    m_granularity = granu;
    return *this;
}

Quantization::Granularity Quantization::granularity(void) const
{
    return m_granularity;
}

Quantization &Quantization::set_scheme(Quantization::Scheme scheme)
{
    m_scheme = scheme;
    return *this;
}

Quantization::Scheme Quantization::scheme() const
{
    if (m_scheme == SchemeNone) // initialize scheme
    {
        auto ms_size = m_scales.size();
        auto mo_size = m_offsets.size();
        auto mm_size = m_min.size();
        auto mx_size = m_max.size();

        float s = scale();
        int32_t z = offset();

        if (ms_size > 1 || mo_size > 1 || mm_size > 0 or mx_size > 0)
        {
            m_scheme = Scheme::Symmetric;
        }
        else if (ms_size == 1 && mo_size == 1 && (s != 1.0 || z != 0))
        {
            m_scheme = Scheme::Symmetric;
        }
    }
    return m_scheme;
}

Quantization &Quantization::set_round_mode(Quantization::RoundMode round_mode)
{
    m_round_mode = round_mode;
    return *this;
}

Quantization::RoundMode Quantization::round_mode(void) const
{
    return m_round_mode;
}

Quantization &Quantization::set_calibration_strategy(Quantization::CalibrationStrategy cs)
{
    m_calibration_strategy = cs;
    return *this;
}

Quantization::CalibrationStrategy Quantization::calibration_strategy(void) const
{
    return m_calibration_strategy;
}

Quantization &Quantization::set_overflow_policy(Quantization::OverflowPolicy op)
{
    m_overflow_policy = op;
    return *this;
}

Quantization::OverflowPolicy Quantization::overflow_policy(void) const
{
    return m_overflow_policy;
}

Quantization &Quantization::set_quantized_range(Quantization::QuantizedRange range)
{
    m_quantized_range = range;
    return *this;
}

Quantization::QuantizedRange Quantization::quantized_range(void) const
{
    return m_quantized_range;
}

Quantization &Quantization::set_method(const std::string &method)
{
    m_method = method;
    return *this;
}

const std::string Quantization::method() const
{
    return m_method;
}

Quantization &Quantization::set_qinvariant(const bool &qinvariant)
{
    m_qinvariant = qinvariant;
    return *this;
}

bool Quantization::qinvariant(void) const
{
    return m_qinvariant;
}

void Quantization::set_quantized(const bool &quantized)
{
    m_quantized = quantized;
}

bool Quantization::quantized(void) const
{
    return m_quantized;
}

void Quantization::set_unquantifiable(const bool &unquantifiable)
{
    m_unquantifiable = unquantifiable;
}

bool Quantization::unquantifiable(void) const
{
    return m_unquantifiable;
}

void Quantization::set_data_unchanged(const bool &unchanged)
{
    m_data_unchanged = unchanged;
}

bool Quantization::data_unchanged(void) const
{
    return m_data_unchanged;
}

Quantization &Quantization::set_block_size(const int32_t &block_size)
{
    m_block_size = block_size;
    return *this;
}

int32_t Quantization::block_size(void) const
{
    return m_block_size;
}

Quantization &Quantization::set_dynamic(const bool &is_dynamic)
{
    m_is_dynamic = is_dynamic;
    return *this;
}

bool Quantization::dynamic(void) const
{
    return m_is_dynamic;
}

Quantization &Quantization::set_using_qtlib(const bool &uq)
{
    m_using_qtlib = uq;
    return *this;
}

bool Quantization::using_qtlib(void) const
{
    return m_using_qtlib;
}

bool Quantization::has_update() const
{
    auto tq = Quantization();
    return *this != tq;
}

bool Quantization::empty() const
{
    auto ms_size = m_scales.size();
    auto mo_size = m_offsets.size();
    auto mm_size = m_min.size();
    auto mx_size = m_max.size();
    if (ms_size || mo_size || mm_size || mx_size)
        return false;
    else
        return true;
}

template <typename T>
void serialize_value(std::ostream &oss,
                     std::ostream &wt_oss,
                     std::string key,
                     ComputeTensor &value,
                     bool is_int = false)
{
    if (value.size() == 0)
        return;
    if (value.size() > 1)
    {
        TensorPtr value_t = value;
        oss << key + ".dtype=" << Tensor::dtype_to_string(value_t->type()) << "\n";
        oss << key + ".offset=" << wt_oss.tellp() << "\n";
        std::vector<T> value_v = value;
        int32_t v_size = value_v.size();
        int32_t m_size = sizeof(float) * v_size;
        wt_oss.write((char *)(value_v.data()), m_size);
        oss << key + ".size=" << m_size << "\n";

        oss << key + ".shape=" << value_t->shape() << "\n";
    }
    else
    {
        std::vector<T> value_v = value;
        if (is_int)
        {
            oss << key + "=" << value_v[0] << "\n";
        }
        else
        {
            oss << key + "=" << float_to_string(value_v[0]) << "\n";
        }
    }

    return;
}

void Quantization::serialize(std::ostream &oss,
                             std::ostream &wt_oss,
                             std::string key)
{
    std::string prefix = key + ".qtz";

    do_serialize(oss, wt_oss, prefix);
    ComputeTensor scales = m_scales;
    if (scales.contain_quant())
    {
        prefix = prefix + ".scale";
        auto next_quant = scales->quantization();
        ComputeTensor next_scales = next_quant.scales();
        serialize_value<float>(oss, wt_oss, prefix + ".scale", next_scales);
    }
    return;
}

void Quantization::do_serialize(std::ostream &oss,
                                std::ostream &wt_oss,
                                std::string key)
{
    if (m_is_dynamic)
    {
        do_dynamic_serialize(oss, wt_oss, key);
    }
    do_static_serialize(oss, wt_oss, key);

    if (m_scheme != Quantization::SchemeNone)
    {
        oss << key + ".scheme=" << scheme_to_string(m_scheme) << "\n";
    }
    if (m_granularity != Granularity::GranularityNone)
    {
        oss << key + ".granularity=" << granularity_to_string(m_granularity) << "\n";
    }
    if (m_block_size > 1)
    {
        oss << key + ".block_size=" << m_block_size << "\n";
    }
    if (!m_method.empty())
    {
        oss << key + ".method=" << m_method << "\n";
    }
    return;
}

void Quantization::do_dynamic_serialize(std::ostream &oss,
                                        std::ostream &wt_oss,
                                        std::string key)
{
    oss << key + ".dynamic=true\n";
    oss << key + ".dtype=" << Tensor::dtype_to_string(m_dtype) << "\n";
    oss << key + ".dbits=" << m_bits << "\n";
    oss << key + ".round_mode=" << round_mode_to_string(m_round_mode) << "\n";
    return;
}

void Quantization::do_static_serialize(std::ostream &oss,
                                       std::ostream &wt_oss,
                                       std::string key)
{
    serialize_value<float>(oss, wt_oss, key + ".scale", m_scales);
    serialize_value<int>(oss, wt_oss, key + ".zp", m_offsets, true);
    serialize_value<float>(oss, wt_oss, key + ".dmin", m_min);
    serialize_value<float>(oss, wt_oss, key + ".dmax", m_max);
    if (m_scales.size() > 1 && m_block_size == 1)
    {
        oss << key + ".axis=" << m_axis << "\n";
    }
    return;
}

std::ostream &operator<<(std::ostream &os, const Quantization &qinfo)
{
    // return os << t.name();

    os << "quantization info: \n";
    os << "\tcalibration_strategy = " << Quantization::calibration_strategy_to_string(qinfo.calibration_strategy()) << "\n";
    os << "\tquantization dtype = " << qinfo.dtype() << "\n";
    os << "\tscheme = " << Quantization::scheme_to_string(qinfo.scheme()) << "\n";
    os << "\tgranularity = " << Quantization::granularity_to_string(qinfo.granularity()) << "\n";
    os << "\tquantized_range = " << Quantization::quantized_range_to_string(qinfo.quantized_range()) << "\n";
    os << "\tquantization bits = " << qinfo.bits() << ", lut_bits = " << qinfo.lut_items_bits() << "\n";
    os << "\tis_qinvariant = " << qinfo.qinvariant() << "\n";
    os << "\tis_unquantifiable = " << qinfo.unquantifiable() << "\n";
    os << "\tis_data_unchanged = " << qinfo.data_unchanged() << "\n";
    os << "\tquantization method = " << (qinfo.method().size() > 0 ? qinfo.method() : "None") << "\n";
    os << "\tis_dynamic_quantization = " << qinfo.dynamic() << "\n";
    os << "\tblock_size = " << qinfo.block_size() << "\n";
    os << "\tquantization axis = " << qinfo.axis() << "\n";
    os << "\tround mode = " << Quantization::round_mode_to_string(qinfo.round_mode()) << "\n";
    os << "\toverflow_policy = " << Quantization::overflow_policy_to_string(qinfo.overflow_policy()) << "\n";
    os << "\tusing_qtlib = " << qinfo.using_qtlib() << "\n";
    std::map<std::string, NodeParamValue> _extra_info = qinfo.extra_info();
    auto extra_info_size = _extra_info.size();
    if (extra_info_size)
    {
        os << "\textra_info : {"
           << "\n";
        for (auto kv : _extra_info)
        {
            os << "\t\t" << kv.first << " = " << kv.second << "\n";
        }
        os << "\t}\n";
    }

    auto ss = qinfo.scales().size();
    os << "\tscale.size = " << ss;
    if (ss >= 1)
    {
        os << ", scales = " << qinfo.scales() << ", scales.shape = " << qinfo.scales()->shape();
    }
    os << "\n";

    auto offs = qinfo.offsets().size();
    os << "\tzerop.size = " << offs;
    if (offs >= 1)
    {
        os << ", zerops = " << qinfo.offsets() << ", zerops.shape = " << qinfo.offsets()->shape();
    }
    os << "\n";

    auto min_s = qinfo.mins().size();
    os << "\tstatistic_min.size = " << min_s;
    if (min_s > 1)
    {
        os << ", statistic_min[0,...-1] = " << qinfo.mins()[0] << ", ..., " << qinfo.mins()[min_s - 1];
    }
    else if (min_s == 1)
    {
        os << ", statistic_min[0] = " << qinfo.mins()[0];
    }
    os << "\n";

    auto max_s = qinfo.maxs().size();
    os << "\tstatistic_max.size = " << max_s;
    if (max_s > 1)
    {
        os << ", statistic_max[0,...-1] = " << qinfo.maxs()[0] << ", ..., " << qinfo.maxs()[max_s - 1];
    }
    else if (max_s == 1)
    {
        os << ", statistic_max[0] = " << qinfo.maxs()[0];
    }
    os << "\n";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Quantization::CalibrationStrategy &qcs)
{
    os << "quantization calibration_strategy = " << Quantization::calibration_strategy_to_string(qcs) << "\n";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Quantization::Granularity &granu)
{
    os << "quantization granularity = " << Quantization::granularity_to_string(granu) << "\n";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Quantization::Scheme &sh)
{
    os << "quantization scheme = " << Quantization::scheme_to_string(sh) << "\n";
    return os;
}

std::ostream &operator<<(std::ostream &os, const Quantization::RoundMode &rm)
{
    os << "quantization round_mode = " << Quantization::round_mode_to_string(rm) << "\n";
    return os;
}

} // namespace aipubt

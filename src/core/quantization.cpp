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
 * @Date: 2021-07-01 14:08:19
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2025-10-23 15:21:43
 */

#include <core/quantization.h>
#include <core/tensor.h>

namespace aipubt
{

const std::string Quantization::scheme_to_string(const Quantization::Scheme &s)
{
    std::string scm = "unknown";
    if (s == Quantization::Symmetric)
    {
        scm = "symmetric";
    }
    else if (s == Quantization::Asymmetric)
    {
        scm = "asymmetric";
    }
    else if (s == Quantization::Scheme_Void)
    {
        scm = "scheme_void";
    }
    else
    {
        ERROR() << "scheme_to_string failed, because unknown scheme = " << (int)s;
    }
    return scm;
}

const std::string Quantization::granularity_to_string(const Quantization::Granularity &g)
{
    std::string granu = "unknown";
    if (g == Quantization::Per_Channel)
    {
        granu = "per_channel";
    }
    else if (g == Quantization::Per_Group)
    {
        granu = "per_group";
    }
    else if (g == Quantization::Per_Tensor)
    {
        granu = "per_tensor";
    }
    else if (g == Quantization::Granularity_Void)
    {
        granu = "per_void";
    }
    else
    {
        ERROR() << "granularity_to_string failed, because unknown granularity = " << (int)g;
    }
    return granu;
}

const std::string Quantization::quantize_range_to_string(const Quantization::QuantizedRange &gr)
{
    std::string qr_str = "unknown";
    if (gr == Quantization::Full_Range)
    {
        qr_str = "full_range";
    }
    else if (gr == Quantization::Restricted_Range)
    {
        qr_str = "restricted_range";
    }
    else if (gr == Quantization::Range_Void)
    {
        qr_str = "range_void";
    }
    else
    {
        ERROR() << "quantize_range_to_string failed, because unknown quantized_range = " << (int)gr;
    }
    return qr_str;
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

void Quantization::set_quantization_dim(int32_t quantization_dim)
{
    m_quantization_dim = quantization_dim;
}

int32_t Quantization::quantization_dim() const
{
    return m_quantization_dim;
}

void Quantization::set_aipu_dtype(Tensor::Dtype dtype)
{
    m_aipu_dtype = dtype;
    // set_bits(Tensor::dtype_bits(m_aipu_dtype));
}

Tensor::Dtype Quantization::aipu_dtype() const
{
    return m_aipu_dtype;
}

int32_t Quantization::bits() const
{
    return m_bits;
}

void Quantization::set_bits(int32_t b)
{
    m_bits = b;
}

int32_t Quantization::lut_items_bits() const
{
    return m_lut_items_bits;
}

void Quantization::set_lut_items_bits(int32_t b)
{
    m_lut_items_bits = b;
}

void Quantization::set_min(float f_min)
{
    if (m_min.size())
    {
        std::vector<int32_t> cur_min = m_min;
        cur_min[0] = f_min;
        m_min = ComputeTensor(cur_min);
    }
    else
    {
        m_min = ComputeTensor(f_min);
    }
}

void Quantization::set_mins(const std::vector<float> &f_mins)
{
    m_min = ComputeTensor(f_mins);
}

std::vector<float> Quantization::mins(void) const
{
    std::vector<float> mm = m_min;
    return mm;
}

void Quantization::set_max(float f_max)
{
    if (m_max.size())
    {
        std::vector<int32_t> cur_max = m_max;
        cur_max[0] = f_max;
        m_max = ComputeTensor(cur_max);
    }
    else
    {
        m_max = ComputeTensor(f_max);
    }
}

void Quantization::set_maxs(const std::vector<float> &f_maxs)
{
    m_max = ComputeTensor(f_maxs);
}

std::vector<float> Quantization::maxs(void) const
{
    std::vector<float> mx = m_max;
    return mx;
}

void Quantization::set_granularity(Quantization::Granularity granu)
{
    m_granularity = granu;
}

Quantization::Granularity Quantization::granularity(void) const
{
    return m_granularity;
}

void Quantization::set_scheme(Quantization::Scheme scheme)
{
    m_scheme = scheme;
}

Quantization::Scheme Quantization::scheme() const
{
    if (m_scheme == Scheme_Void) // initialize scheme
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

void Quantization::set_qinvariant(bool qinvariant)
{
    m_qinvariant = qinvariant;
}

bool Quantization::qinvariant(void) const
{
    return m_qinvariant;
}

void Quantization::set_quantized(bool quantized)
{
    m_quantized = quantized;
}

bool Quantization::quantized(void) const
{
    return m_quantized;
}

void Quantization::set_unquantifiable(bool unquantifiable)
{
    m_unquantifiable = unquantifiable;
}

bool Quantization::unquantifiable(void) const
{
    return m_unquantifiable;
}

void Quantization::set_data_unchanged(bool unchanged)
{
    m_data_unchanged = unchanged;
}

bool Quantization::data_unchanged(void) const
{
    return m_data_unchanged;
}

void Quantization::set_quantized_range(Quantization::QuantizedRange range)
{
    m_quantized_range = range;
}

Quantization::QuantizedRange Quantization::quantized_range(void) const
{
    return m_quantized_range;
}

void Quantization::set_group(int32_t group)
{
    m_group = group;
}

int32_t Quantization::group(void) const
{
    return m_group;
}

void Quantization::set_using_qtlib(bool uq)
{
    m_using_qtlib = uq;
}

bool Quantization::using_qtlib(void) const
{
    return m_using_qtlib;
}

bool Quantization::empty() const
{
    auto ms_size = m_scales.size();
    auto mo_size = m_offsets.size();
    auto mm_size = m_min.size();
    auto mx_size = m_max.size();
    if ((!ms_size || !mo_size) && (!mm_size || !mx_size))
        return true;
    else
        return false;
}

std::ostream &operator<<(std::ostream &os, const Quantization &qinfo)
{
    // return os << t.name();

    os << "quantization info: \n";
    auto ss = qinfo.scales().size();
    os << "scale.size = " << ss;
    if (ss > 1)
    {
        os << ", scale[0,...-1] = " << qinfo.scales()[0] << ", ..., " << qinfo.scales()[ss - 1];
    }
    else if (ss == 1)
    {
        os << ", scale[0] = " << qinfo.scales()[0];
    }
    os << "\n";

    auto offs = qinfo.offsets().size();
    os << "zerop.size = " << offs;
    if (offs > 1)
    {
        os << ", zerop[0,...-1] = " << qinfo.offsets()[0] << ", ..., " << qinfo.offsets()[ss - 1];
    }
    else if (offs == 1)
    {
        os << ", zerop[0] = " << qinfo.offsets()[0];
    }
    os << "\n";

    os << "quantization bits = " << qinfo.bits() << ", lut_bits = " << qinfo.lut_items_bits() << ", is_unquantifiable = " << qinfo.unquantifiable() << "\n";
    os << "aipu_dtype = " << qinfo.aipu_dtype() << "\n";
    os << "scheme = " << Quantization::scheme_to_string(qinfo.scheme())
       << ", granularity = " << Quantization::granularity_to_string(qinfo.granularity())
       << ", quantized_range = " << Quantization::quantize_range_to_string(qinfo.quantized_range()) << "\n";
    os << "is_data_unchanged = " << qinfo.data_unchanged() << "\n";

    auto min_s = qinfo.mins().size();
    os << "statistic_min.size = " << min_s;
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
    os << "statistic_max.size = " << max_s;
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

} // namespace aipubt

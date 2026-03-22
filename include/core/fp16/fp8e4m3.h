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
 * @Date: 2023-03-09 12:16:08
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2025-10-17 17:40:45
 */
#pragma once
#include "macro.h"
#include <cmath>
#include <limits>
#include <stdint.h>

#define FP8E4M3FUNC1A(func) \
    static inline fp8e4m3fn func(fp8e4m3fn rhs) { return fp8e4m3fn(std::func((float)rhs)); }

#define FP8E4M3FUNC1B(func) \
    static inline bool func(fp8e4m3fn rhs) { return std::func((float)rhs); }

#define FP8E4M3FUNC2A(func) \
    static inline fp8e4m3fn func(fp8e4m3fn lhs, fp8e4m3fn rhs) { return fp8e4m3fn(std::func((float)lhs, (float)rhs)); }

#define FP8E4M3FUNC2B(func) \
    static inline bool func(fp8e4m3fn lhs, fp8e4m3fn rhs) { return std::func((float)lhs, (float)rhs); }

namespace aipubt
{
struct fp8e4m3fn
{
    uint8_t _v = 0;

    inline fp8e4m3fn(float vf) noexcept
    {
        if (std::isnan(vf))
        {
            _v = 0x7f;
        }
        else
        {
            union
            {
                uint32_t U32;
                float F32;
            };
            F32 = vf;
            uint32_t u = U32;
            _v = (u >> 24) & 0x80;
            int32_t e = ((u >> 23) & 0xff) - 127; // subnormal is always zero for fp32
            uint32_t m = (u & 0x7fffff);
            // TODO! using table to accelerate the conversion
            if (e > 8 || (e == 8 && (m > 0x600000)))
            {
                _v |= 0x7f;
            }
            else if (e > -7)
            {
                auto g = (m >> 19) & 1;
                _v |= (((e + 7) << 3) | ((m >> 20) & 0x7)) + g;
            }
            else
            {
                m += (1 << 23);
                auto g = (m >> (13 - e)) & 1;
                _v |= ((m >> (14 - e))) + g; // rounding
            }
        }
    }
    inline operator float() const
    {
        const float values[] = {0.0, 0.001953125, 0.00390625, 0.005859375, 0.0078125, 0.009765625, 0.01171875,
                                0.013671875, 0.015625, 0.017578125, 0.01953125, 0.021484375, 0.0234375, 0.025390625, 0.02734375,
                                0.029296875, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 0.05078125, 0.0546875, 0.05859375,
                                0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.1015625, 0.109375, 0.1171875, 0.125, 0.140625,
                                0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375, 0.375, 0.40625,
                                0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375, 1.5,
                                1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5,
                                8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0,
                                36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0, 64.0, 72.0, 80.0, 88.0, 96.0, 104.0, 112.0, 120.0, 128.0,
                                144.0, 160.0, 176.0, 192.0, 208.0, 224.0, 240.0, 256.0, 288.0, 320.0, 352.0, 384.0, 416.0, 448.0};
        if ((_v & 0x7f) == 0x7f)
        {
            return std::numeric_limits<float>::quiet_NaN();
        }
        auto r = values[_v & 0x7f];
        if (_v >> 7)
        {
            r = -r;
        }
        return r;
    }
    constexpr inline fp8e4m3fn() noexcept = default;
    constexpr inline fp8e4m3fn(fp8e4m3fn const &) noexcept = default;
    constexpr inline fp8e4m3fn(fp8e4m3fn &&) noexcept = default;
    // constexpr inline fp8e4m3fn(std::uint16_t bits) noexcept : data_{bits} {}
    constexpr inline fp8e4m3fn &operator=(fp8e4m3fn const &) noexcept = default;
    constexpr inline fp8e4m3fn &operator=(fp8e4m3fn &&) noexcept = default;
    inline fp8e4m3fn &operator=(float vf) noexcept
    {
        fp8e4m3fn o(vf);
        _v = o._v;
        return *this;
    }

    inline fp8e4m3fn &operator+=(fp8e4m3fn v) noexcept
    {
        return *this += static_cast<float>(v);
    }

    inline fp8e4m3fn &operator-=(fp8e4m3fn v) noexcept
    {
        return *this -= static_cast<float>(v);
    }

    inline fp8e4m3fn &operator*=(fp8e4m3fn v) noexcept
    {
        return *this *= static_cast<float>(v);
    }

    inline fp8e4m3fn &operator/=(fp8e4m3fn v) noexcept
    {
        return *this /= static_cast<float>(v);
    }

    inline fp8e4m3fn &operator+=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 + v;
        return *this;
    }

    inline fp8e4m3fn &operator-=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 - v;
        return *this;
    }

    inline fp8e4m3fn &operator*=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 * v;
        return *this;
    }

    inline fp8e4m3fn &operator/=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 / v;
        return *this;
    }

    inline fp8e4m3fn operator--() noexcept //--f
    {
        *this -= 1.0f;
        return *this;
    }

    inline fp8e4m3fn operator--(int) noexcept // f--
    {
        fp8e4m3fn ans{*this};
        *this -= 1.0f;
        return ans;
    }

    inline fp8e4m3fn operator++() noexcept //++f
    {
        *this += 1.0f;
        return *this;
    }

    inline fp8e4m3fn operator++(int) noexcept // f++
    {
        fp8e4m3fn ans{*this};
        *this += 1.0f;
        return ans;
    }

    // inline fp8e4m3fn operator-() const noexcept
    // {
    //     return fp8e4m3fn{(uint16_t)((_v & (uint16_t)0x7fff) | (_v ^ (uint16_t)0x8000))};
    // }

    inline fp8e4m3fn operator+() const noexcept
    {
        return *this;
    }
};
} // namespace aipubt
using aipubt::fp8e4m3fn;
inline fp8e4m3fn operator+(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    fp8e4m3fn ans = lhs;
    ans += rhs;
    return ans;
}

inline fp8e4m3fn operator-(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    fp8e4m3fn ans = lhs;
    ans -= rhs;
    return ans;
}

inline fp8e4m3fn operator*(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    fp8e4m3fn ans = lhs;
    ans *= rhs;
    return ans;
}

inline fp8e4m3fn operator/(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    return fp8e4m3fn(float(lhs) / float(rhs));
}
template <typename T>
inline fp8e4m3fn operator/(fp8e4m3fn lhs, T rhs) noexcept
{
    return fp8e4m3fn(float(lhs) / float(rhs));
}

inline bool operator<(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    return float(lhs) < float(rhs);
}

inline bool operator==(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    return lhs._v == rhs._v;
}

inline bool operator<=(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    return !(lhs <= rhs);
}

inline bool operator>=(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    return !(lhs < rhs);
}

inline bool operator!=(fp8e4m3fn lhs, fp8e4m3fn rhs) noexcept
{
    return !(lhs == rhs);
}

DECLARE_BUILTIN_FUNCS(FP8E4M3FUNC1A, FP8E4M3FUNC1B, FP8E4M3FUNC2A, FP8E4M3FUNC2B)
/// Extensions to the C++ standard library.
namespace std
{
/// Numeric limits for half-precision floats.
/// Because of the underlying single-precision implementation of many operations, it inherits some properties from
/// `std::numeric_limits<float>`.
template <>
class numeric_limits<aipubt::fp8e4m3fn> : public numeric_limits<float>
{
public:
    /// Supports signed values.
    static constexpr bool is_signed = true;

    /// Is not exact.
    static constexpr bool is_exact = false;

    /// Doesn't provide modulo arithmetic.
    static constexpr bool is_modulo = false;

    /// IEEE conformant.
    static constexpr bool is_iec559 = false;

    /// Supports infinity.
    static constexpr bool has_infinity = false;

    /// Supports quiet NaNs.
    static constexpr bool has_quiet_NaN = true;

    /// Supports subnormal values.
    // static constexpr float_denorm_style has_denorm = denorm_present;

    /// Rounding mode.
    /// Due to the mix of internal single-precision computations (using the rounding mode of the underlying
    /// single-precision implementation) with the rounding mode of the single-to-half conversions, the actual rounding
    /// mode might be `std::round_indeterminate` if the default half-precision rounding mode doesn't match the
    /// single-precision rounding mode.
    static constexpr float_round_style round_style = round_indeterminate;

    /// Significant digits.
    static constexpr int digits = 11;

    /// Significant decimal digits.
    static constexpr int digits10 = 3;

    /// Required decimal digits to represent all possible values.
    static constexpr int max_digits10 = 5;

    /// Number base.
    static constexpr int radix = 2;

    /// One more than smallest exponent.
    static constexpr int min_exponent = -9;

    /// Smallest normalized representable power of 10.
    static constexpr int min_exponent10 = -2;

    /// One more than largest exponent
    static constexpr int max_exponent = 8;

    /// Largest finitely representable power of 10.
    static constexpr int max_exponent10 = 2;

    /// Smallest positive normal value.
    static constexpr aipubt::fp8e4m3fn min() noexcept
    {
        aipubt::fp8e4m3fn ret;
        ret._v = 0x8;
        return ret;
    }

    /// Smallest finite value.
    static constexpr aipubt::fp8e4m3fn lowest() noexcept
    {
        aipubt::fp8e4m3fn ret;
        ret._v = 0xfe;
        return ret;
    }

    /// Largest finite value.
    static constexpr aipubt::fp8e4m3fn max() noexcept
    {
        aipubt::fp8e4m3fn ret;
        ret._v = 0x7e;
        return ret;
    }

    /// Difference between one and next representable value.
    static constexpr aipubt::fp8e4m3fn epsilon() noexcept
    {
        aipubt::fp8e4m3fn ret;
        ret._v = 0x20;
        return ret;
    }

    /// Maximum rounding error.
    static constexpr aipubt::fp8e4m3fn round_error() noexcept
    {
        aipubt::fp8e4m3fn ret;
        ret._v = 0x58;
        return ret;
    }

    /// Quiet NaN.
    static constexpr aipubt::fp8e4m3fn quiet_NaN() noexcept
    {
        aipubt::fp8e4m3fn ret;
        ret._v = 0x7f;
        return ret;
    }

    /// Signalling NaN.
    static constexpr aipubt::fp8e4m3fn signaling_NaN() noexcept
    {
        aipubt::fp8e4m3fn ret;
        ret._v = 0x7f;
        return ret;
    }

    /// Smallest positive subnormal value.
    static constexpr aipubt::fp8e4m3fn denorm_min() noexcept
    {
        aipubt::fp8e4m3fn ret;
        ret._v = 0x01;
        return ret;
    }
};

/// Hash function for half-precision floats.
/// This is only defined if C++11 `std::hash` is supported and enabled.
template <>
struct hash<aipubt::fp8e4m3fn> //: unary_function<aipubt::fp8e4m3fn,size_t>
{
    /// Type of function argument.
    typedef aipubt::fp8e4m3fn argument_type;

    /// Function return type.
    typedef size_t result_type;

    /// Compute hash function.
    /// \param arg half to hash
    /// \return hash value
    result_type operator()(argument_type arg) const
    {
        return hash<uint8_t>()(static_cast<unsigned>(arg._v) & -(arg._v != 0x80));
    }
};
} // namespace std

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
 * @Last Modified time: 2025-10-17 17:40:49
 */
#pragma once
#include "macro.h"
#include <cmath>
#include <limits>
#include <stdint.h>

#define FP8E5M2FUNC1A(func) \
    static inline fp8e5m2 func(fp8e5m2 rhs) { return fp8e5m2(std::func((float)rhs)); }

#define FP8E5M2FUNC1B(func) \
    static inline bool func(fp8e5m2 rhs) { return std::func((float)rhs); }

#define FP8E5M2FUNC2A(func) \
    static inline fp8e5m2 func(fp8e5m2 lhs, fp8e5m2 rhs) { return fp8e5m2(std::func((float)lhs, (float)rhs)); }

#define FP8E5M2FUNC2B(func) \
    static inline bool func(fp8e5m2 lhs, fp8e5m2 rhs) { return std::func((float)lhs, (float)rhs); }

namespace aipubt
{
struct fp8e5m2
{
    uint8_t _v = 0;

    inline fp8e5m2(float vf) noexcept
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
            int32_t e = ((u >> 23) & 0xff) - 127; //subnormal is always zero for fp32
            uint32_t m = (u & 0x7fffff);
            //TODO! using table to accelerate the conversion
            if (e > 15 || (e == 15 && m > 0x600000)) //inf
            {
                _v |= 0x7c;
            }
            else if (e > -14)
            {
                auto g = (m >> 20) & 1;
                _v |= (((e + 15) << 2) | ((m >> 21) & 0x3)) + g;
            }
            else
            {
                m += (1 << 23);
                auto g = (m >> (6 - e)) & 1;
                _v |= ((m >> (7 - e))) + g; //rounding
            }
        }
    }
    inline operator float() const
    {
        const float values[] = {0.0, 1.52587890625e-05, 3.0517578125e-05, 4.57763671875e-05, 6.103515625e-05, 7.62939453125e-05, 9.1552734375e-05,
                                0.0001068115234375, 0.0001220703125, 0.000152587890625, 0.00018310546875, 0.000213623046875,
                                0.000244140625, 0.00030517578125, 0.0003662109375, 0.00042724609375, 0.00048828125, 0.0006103515625,
                                0.000732421875, 0.0008544921875, 0.0009765625, 0.001220703125, 0.00146484375, 0.001708984375, 0.001953125,
                                0.00244140625, 0.0029296875, 0.00341796875, 0.00390625, 0.0048828125, 0.005859375, 0.0068359375, 0.0078125,
                                0.009765625, 0.01171875, 0.013671875, 0.015625, 0.01953125, 0.0234375, 0.02734375, 0.03125, 0.0390625, 0.046875,
                                0.0546875, 0.0625, 0.078125, 0.09375, 0.109375, 0.125, 0.15625, 0.1875, 0.21875, 0.25, 0.3125, 0.375, 0.4375,
                                0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0,
                                16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0,
                                320.0, 384.0, 448.0, 512.0, 640.0, 768.0, 896.0, 1024.0, 1280.0, 1536.0, 1792.0, 2048.0, 2560.0, 3072.0,
                                3584.0, 4096.0, 5120.0, 6144.0, 7168.0, 8192.0, 10240.0, 12288.0, 14336.0, 16384.0, 20480.0, 24576.0,
                                28672.0, 32768.0, 40960.0, 49152.0, 57344.0,
                                std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};

        auto r = values[_v & 0x7f];
        if (_v >> 7)
        {
            r = -r;
        }
        return r;
    }
    constexpr inline fp8e5m2() noexcept = default;
    constexpr inline fp8e5m2(fp8e5m2 const &) noexcept = default;
    constexpr inline fp8e5m2(fp8e5m2 &&) noexcept = default;
    // constexpr inline fp8e5m2(std::uint16_t bits) noexcept : data_{bits} {}
    constexpr inline fp8e5m2 &operator=(fp8e5m2 const &) noexcept = default;
    constexpr inline fp8e5m2 &operator=(fp8e5m2 &&) noexcept = default;
    inline fp8e5m2 &operator=(float vf) noexcept
    {
        fp8e5m2 o(vf);
        _v = o._v;
        return *this;
    }

    inline fp8e5m2 &operator+=(fp8e5m2 v) noexcept
    {
        return *this += static_cast<float>(v);
    }

    inline fp8e5m2 &operator-=(fp8e5m2 v) noexcept
    {
        return *this -= static_cast<float>(v);
    }

    inline fp8e5m2 &operator*=(fp8e5m2 v) noexcept
    {
        return *this *= static_cast<float>(v);
    }

    inline fp8e5m2 &operator/=(fp8e5m2 v) noexcept
    {
        return *this /= static_cast<float>(v);
    }

    inline fp8e5m2 &operator+=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 + v;
        return *this;
    }

    inline fp8e5m2 &operator-=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 - v;
        return *this;
    }

    inline fp8e5m2 &operator*=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 * v;
        return *this;
    }

    inline fp8e5m2 &operator/=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 / v;
        return *this;
    }

    inline fp8e5m2 operator--() noexcept //--f
    {
        *this -= 1.0f;
        return *this;
    }

    inline fp8e5m2 operator--(int) noexcept // f--
    {
        fp8e5m2 ans{*this};
        *this -= 1.0f;
        return ans;
    }

    inline fp8e5m2 operator++() noexcept //++f
    {
        *this += 1.0f;
        return *this;
    }

    inline fp8e5m2 operator++(int) noexcept // f++
    {
        fp8e5m2 ans{*this};
        *this += 1.0f;
        return ans;
    }

    // inline fp8e5m2 operator-() const noexcept
    // {
    //     return fp8e5m2{(uint16_t)((_v & (uint16_t)0x7fff) | (_v ^ (uint16_t)0x8000))};
    // }

    inline fp8e5m2 operator+() const noexcept
    {
        return *this;
    }
};
} // namespace aipubt
using aipubt::fp8e5m2;
inline fp8e5m2 operator+(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    fp8e5m2 ans = lhs;
    ans += rhs;
    return ans;
}

inline fp8e5m2 operator-(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    fp8e5m2 ans = lhs;
    ans -= rhs;
    return ans;
}

inline fp8e5m2 operator*(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    fp8e5m2 ans = lhs;
    ans *= rhs;
    return ans;
}

inline fp8e5m2 operator/(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    return fp8e5m2(float(lhs) / float(rhs));
}
template <typename T>
inline fp8e5m2 operator/(fp8e5m2 lhs, T rhs) noexcept
{
    return fp8e5m2(float(lhs) / float(rhs));
}

inline bool operator<(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    return float(lhs) < float(rhs);
}

inline bool operator==(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    return lhs._v == rhs._v;
}

inline bool operator<=(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    return !(lhs <= rhs);
}

inline bool operator>=(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    return !(lhs < rhs);
}

inline bool operator!=(fp8e5m2 lhs, fp8e5m2 rhs) noexcept
{
    return !(lhs == rhs);
}

DECLARE_BUILTIN_FUNCS(FP8E5M2FUNC1A, FP8E5M2FUNC1B, FP8E5M2FUNC2A, FP8E5M2FUNC2B)
/// Extensions to the C++ standard library.
namespace std
{
/// Numeric limits for half-precision floats.
/// Because of the underlying single-precision implementation of many operations, it inherits some properties from
/// `std::numeric_limits<float>`.
template <>
class numeric_limits<aipubt::fp8e5m2> : public numeric_limits<float>
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
    static constexpr aipubt::fp8e5m2 min() noexcept
    {
        aipubt::fp8e5m2 ret;
        ret._v = 0x8;
        return ret;
    }

    /// Smallest finite value.
    static constexpr aipubt::fp8e5m2 lowest() noexcept
    {
        aipubt::fp8e5m2 ret;
        ret._v = 0xfe;
        return ret;
    }

    /// Largest finite value.
    static constexpr aipubt::fp8e5m2 max() noexcept
    {
        aipubt::fp8e5m2 ret;
        ret._v = 0x7e;
        return ret;
    }

    /// Difference between one and next representable value.
    static constexpr aipubt::fp8e5m2 epsilon() noexcept
    {
        aipubt::fp8e5m2 ret;
        ret._v = 0x20;
        return ret;
    }

    /// Maximum rounding error.
    static constexpr aipubt::fp8e5m2 round_error() noexcept
    {
        aipubt::fp8e5m2 ret;
        ret._v = 0x58;
        return ret;
    }

    /// Quiet NaN.
    static constexpr aipubt::fp8e5m2 quiet_NaN() noexcept
    {
        aipubt::fp8e5m2 ret;
        ret._v = 0x7f;
        return ret;
    }

    /// Signalling NaN.
    static constexpr aipubt::fp8e5m2 signaling_NaN() noexcept
    {
        aipubt::fp8e5m2 ret;
        ret._v = 0x7f;
        return ret;
    }

    /// Smallest positive subnormal value.
    static constexpr aipubt::fp8e5m2 denorm_min() noexcept
    {
        aipubt::fp8e5m2 ret;
        ret._v = 0x01;
        return ret;
    }
};

/// Hash function for half-precision floats.
/// This is only defined if C++11 `std::hash` is supported and enabled.
template <>
struct hash<aipubt::fp8e5m2> //: unary_function<aipubt::fp8e5m2,size_t>
{
    /// Type of function argument.
    typedef aipubt::fp8e5m2 argument_type;

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

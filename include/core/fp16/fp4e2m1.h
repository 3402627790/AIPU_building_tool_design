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
 * @Last Modified time: 2025-10-17 17:43:35
 */
#pragma once
#include "macro.h"
#include <cmath>
#include <limits>
#include <stdint.h>

#define FP4E2M1FUNC1A(func)                      \
    static inline fp4e2m1fn func(fp4e2m1fn rhs)  \
    {                                            \
        return fp4e2m1fn(std::func((float)rhs)); \
    }

#define FP4E2M1FUNC1B(func)                \
    static inline bool func(fp4e2m1fn rhs) \
    {                                      \
        return std::func((float)rhs);      \
    }

#define FP4E2M1FUNC2A(func)                                    \
    static inline fp4e2m1fn func(fp4e2m1fn lhs, fp4e2m1fn rhs) \
    {                                                          \
        return fp4e2m1fn(std::func((float)lhs, (float)rhs));   \
    }

#define FP4E2M1FUNC2B(func)                               \
    static inline bool func(fp4e2m1fn lhs, fp4e2m1fn rhs) \
    {                                                     \
        return std::func((float)lhs, (float)rhs);         \
    }

namespace aipubt
{
struct fp4e2m1fn
{
    uint8_t _v = 0;

    inline fp4e2m1fn(float vf) noexcept
    {

        union
        {
            uint32_t U32;
            float F32;
        };
        F32 = vf;
        if (U32 >> 31)
        {
            _v = 0x8;
        }
        if (std::isnan(vf))
        {
            _v = 0xf;
        }
        else
        {
            if (vf < 0)
            {
                vf = -vf;
            }

            if (vf >= 1.75)
            {
                if (vf >= 3.5)
                {
                    if (vf >= 5)
                    {
                        _v |= 0b111;
                    }
                    else
                    {
                        _v |= 0b110;
                    }
                }
                else
                {
                    if (vf >= 2.5)
                    {
                        _v |= 0b101;
                    }
                    else
                    {
                        _v |= 0b100;
                    }
                }
            }
            else
            {

                if (vf >= 1.25)
                {
                    _v |= 0b11;
                }
                else if (vf >= 0.75)
                {

                    _v |= 0b10;
                }
                else if (vf >= 0.25)
                {
                    _v |= 0b1;
                }
            }
        }
    }
    inline operator float() const
    {
        const float values[] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0f, -0.5, -1, -1.5, -2, -3, -4, -6};

        return values[_v & 0xf];
    }
    constexpr inline fp4e2m1fn() noexcept = default;
    constexpr inline fp4e2m1fn(fp4e2m1fn const &) noexcept = default;
    constexpr inline fp4e2m1fn(fp4e2m1fn &&) noexcept = default;
    // constexpr inline fp4e2m1fn(std::uint16_t bits) noexcept : data_{bits} {}
    constexpr inline fp4e2m1fn &operator=(fp4e2m1fn const &) noexcept = default;
    constexpr inline fp4e2m1fn &operator=(fp4e2m1fn &&) noexcept = default;
    inline fp4e2m1fn &operator=(float vf) noexcept
    {
        fp4e2m1fn o(vf);
        _v = o._v;
        return *this;
    }

    inline fp4e2m1fn &operator+=(fp4e2m1fn v) noexcept
    {
        return *this += static_cast<float>(v);
    }

    inline fp4e2m1fn &operator-=(fp4e2m1fn v) noexcept
    {
        return *this -= static_cast<float>(v);
    }

    inline fp4e2m1fn &operator*=(fp4e2m1fn v) noexcept
    {
        return *this *= static_cast<float>(v);
    }

    inline fp4e2m1fn &operator/=(fp4e2m1fn v) noexcept
    {
        return *this /= static_cast<float>(v);
    }

    inline fp4e2m1fn &operator+=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 + v;
        return *this;
    }

    inline fp4e2m1fn &operator-=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 - v;
        return *this;
    }

    inline fp4e2m1fn &operator*=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 * v;
        return *this;
    }

    inline fp4e2m1fn &operator/=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 / v;
        return *this;
    }

    inline fp4e2m1fn &operator|=(uint32_t v) noexcept
    {
        _v |= v;
        return *this;
    }

    inline fp4e2m1fn operator--() noexcept //--f
    {
        *this -= 1.0f;
        return *this;
    }

    inline fp4e2m1fn operator--(int) noexcept // f--
    {
        fp4e2m1fn ans{*this};
        *this -= 1.0f;
        return ans;
    }

    inline fp4e2m1fn operator++() noexcept //++f
    {
        *this += 1.0f;
        return *this;
    }

    inline fp4e2m1fn operator++(int) noexcept // f++
    {
        fp4e2m1fn ans{*this};
        *this += 1.0f;
        return ans;
    }

    // inline fp4e2m1fn operator-() const noexcept
    // {
    //     return fp4e2m1fn{(uint16_t)((_v & (uint16_t)0x7fff) | (_v ^ (uint16_t)0x8000))};
    // }

    inline fp4e2m1fn operator+() const noexcept
    {
        return *this;
    }
};
} // namespace aipubt
using aipubt::fp4e2m1fn;
inline fp4e2m1fn operator+(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    fp4e2m1fn ans = lhs;
    ans += rhs;
    return ans;
}

inline fp4e2m1fn operator-(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    fp4e2m1fn ans = lhs;
    ans -= rhs;
    return ans;
}

inline fp4e2m1fn operator*(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    fp4e2m1fn ans = lhs;
    ans *= rhs;
    return ans;
}

inline fp4e2m1fn operator/(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    return fp4e2m1fn(float(lhs) / float(rhs));
}
template <typename T>
inline fp4e2m1fn operator/(fp4e2m1fn lhs, T rhs) noexcept
{
    return fp4e2m1fn(float(lhs) / float(rhs));
}

inline bool operator<(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    return float(lhs) < float(rhs);
}

inline bool operator==(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    return lhs._v == rhs._v;
}

inline bool operator<=(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    return !(lhs <= rhs);
}

inline bool operator>=(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    return !(lhs < rhs);
}

inline bool operator!=(fp4e2m1fn lhs, fp4e2m1fn rhs) noexcept
{
    return !(lhs == rhs);
}

DECLARE_BUILTIN_FUNCS(FP4E2M1FUNC1A, FP4E2M1FUNC1B, FP4E2M1FUNC2A, FP4E2M1FUNC2B)
/// Extensions to the C++ standard library.
namespace std
{
/// Numeric limits for half-precision floats.
/// Because of the underlying single-precision implementation of many operations, it inherits some properties from
/// `std::numeric_limits<float>`.
template <>
class numeric_limits<aipubt::fp4e2m1fn> : public numeric_limits<float>
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
    static constexpr bool has_quiet_NaN = false;

    /// Supports subnormal values.
    // static constexpr float_denorm_style has_denorm = denorm_present;

    /// Rounding mode.
    /// Due to the mix of internal single-precision computations (using the rounding mode of the underlying
    /// single-precision implementation) with the rounding mode of the single-to-half conversions, the actual rounding
    /// mode might be `std::round_indeterminate` if the default half-precision rounding mode doesn't match the
    /// single-precision rounding mode.
    static constexpr float_round_style round_style = round_indeterminate;

    /// Significant digits.
    static constexpr int digits = 1;

    /// Significant decimal digits.
    static constexpr int digits10 = 1;

    /// Required decimal digits to represent all possible values.
    static constexpr int max_digits10 = 1;

    /// Number base.
    static constexpr int radix = 2;

    /// One more than smallest exponent.
    static constexpr int min_exponent = -1;

    /// Smallest normalized representable power of 10.
    static constexpr int min_exponent10 = -1;

    /// One more than largest exponent
    static constexpr int max_exponent = 4;

    /// Largest finitely representable power of 10.
    static constexpr int max_exponent10 = 1;

    /// Smallest positive normal value.
    static constexpr aipubt::fp4e2m1fn min() noexcept
    {
        aipubt::fp4e2m1fn ret;
        ret._v = 0x2;
        return ret;
    }

    /// Smallest finite value.
    static constexpr aipubt::fp4e2m1fn lowest() noexcept
    {
        aipubt::fp4e2m1fn ret;
        ret._v = 0x1;
        return ret;
    }

    /// Largest finite value.
    static constexpr aipubt::fp4e2m1fn max() noexcept
    {
        aipubt::fp4e2m1fn ret;
        ret._v = 0x7;
        return ret;
    }

    /// Difference between one and next representable value.
    static constexpr aipubt::fp4e2m1fn epsilon() noexcept
    {
        aipubt::fp4e2m1fn ret;
        ret._v = 0x1;
        return ret;
    }

    /// Maximum rounding error.
    static constexpr aipubt::fp4e2m1fn round_error() noexcept
    {
        aipubt::fp4e2m1fn ret;
        ret._v = 0x2;
        return ret;
    }

    /// Quiet NaN.
    static constexpr aipubt::fp4e2m1fn quiet_NaN() noexcept
    {
        aipubt::fp4e2m1fn ret;
        ret._v = 0xf;
        return ret;
    }

    /// Signalling NaN.
    static constexpr aipubt::fp4e2m1fn signaling_NaN() noexcept
    {
        aipubt::fp4e2m1fn ret;
        ret._v = 0xf;
        return ret;
    }

    /// Smallest positive subnormal value.
    static constexpr aipubt::fp4e2m1fn denorm_min() noexcept
    {
        aipubt::fp4e2m1fn ret;
        ret._v = 0x1;
        return ret;
    }
};

/// Hash function for half-precision floats.
/// This is only defined if C++11 `std::hash` is supported and enabled.
template <>
struct hash<aipubt::fp4e2m1fn> //: unary_function<aipubt::fp4e2m1fn,size_t>
{
    /// Type of function argument.
    typedef aipubt::fp4e2m1fn argument_type;

    /// Function return type.
    typedef size_t result_type;

    /// Compute hash function.
    /// \param arg half to hash
    /// \return hash value
    result_type operator()(argument_type arg) const
    {
        return hash<uint8_t>()(static_cast<unsigned>(arg._v) & -(arg._v != 0xf));
    }
};
} // namespace std

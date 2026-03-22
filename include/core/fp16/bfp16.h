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
 * @Last Modified time: 2023-03-09 12:16:08
 */
#ifndef CORE_COMPUTE_BFP16_H
#define CORE_COMPUTE_BFP16_H
#include <cmath>
#include <limits>
#include <stdint.h>

#define BFP16Func1(func) \
    static inline bfp16 func(bfp16 rhs) { return bfp16(std::func((float)rhs)); }

#define BFP16Func1b(func) \
    static inline bool func(bfp16 rhs) { return std::func((float)rhs); }

#define BFP16Func2(func) \
    static inline bfp16 func(bfp16 lhs, bfp16 rhs) { return bfp16(std::func((float)lhs, (float)rhs)); }

#define BFP16Func2b(func) \
    static inline bool func(bfp16 lhs, bfp16 rhs) { return std::func((float)lhs, (float)rhs); }
namespace aipubt
{
struct bfp16
{
    uint16_t _v = 0;

    inline bfp16(float vf) noexcept
    {
        if (std::isnan(vf))
        {
            _v = 0x7fc0;
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
            u += ((u >> 16) & 1) + (0x7fff);
            _v = u >> 16;
        }
    }
    inline operator float() const
    {
        union
        {
            uint32_t U32;
            float F32;
        };
        U32 = _v << 16;
        return F32;
    }
    constexpr inline bfp16() noexcept = default;
    constexpr inline bfp16(bfp16 const &) noexcept = default;
    constexpr inline bfp16(bfp16 &&) noexcept = default;
    // constexpr inline bfp16(std::uint16_t bits) noexcept : data_{bits} {}
    constexpr inline bfp16 &operator=(bfp16 const &) noexcept = default;
    constexpr inline bfp16 &operator=(bfp16 &&) noexcept = default;
    inline bfp16 &operator=(float vf) noexcept
    {
        if (std::isnan(vf))
        {
            _v = 0x7fc0;
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
            u += ((u >> 16) & 1) + (0x7fff);
            _v = u >> 16;
        }
        return *this;
    }

    inline bfp16 &operator+=(bfp16 v) noexcept
    {
        return *this += static_cast<float>(v);
    }

    inline bfp16 &operator-=(bfp16 v) noexcept
    {
        return *this -= static_cast<float>(v);
    }

    inline bfp16 &operator*=(bfp16 v) noexcept
    {
        return *this *= static_cast<float>(v);
    }

    inline bfp16 &operator/=(bfp16 v) noexcept
    {
        return *this /= static_cast<float>(v);
    }

    inline bfp16 &operator+=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 + v;
        return *this;
    }

    inline bfp16 &operator-=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 - v;
        return *this;
    }

    inline bfp16 &operator*=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 * v;
        return *this;
    }

    inline bfp16 &operator/=(float v) noexcept
    {
        float f16 = *this;
        *this = f16 / v;
        return *this;
    }

    inline bfp16 operator--() noexcept //--f
    {
        *this -= 1.0f;
        return *this;
    }

    inline bfp16 operator--(int) noexcept // f--
    {
        bfp16 ans{*this};
        *this -= 1.0f;
        return ans;
    }

    inline bfp16 operator++() noexcept //++f
    {
        *this += 1.0f;
        return *this;
    }

    inline bfp16 operator++(int) noexcept // f++
    {
        bfp16 ans{*this};
        *this += 1.0f;
        return ans;
    }

    // inline bfp16 operator-() const noexcept
    // {
    //     return bfp16{(uint16_t)((_v & (uint16_t)0x7fff) | (_v ^ (uint16_t)0x8000))};
    // }

    inline bfp16 operator+() const noexcept
    {
        return *this;
    }
};
} // namespace aipubt
using aipubt::bfp16;
inline bfp16 operator+(bfp16 lhs, bfp16 rhs) noexcept
{
    bfp16 ans = lhs;
    ans += rhs;
    return ans;
}

inline bfp16 operator-(bfp16 lhs, bfp16 rhs) noexcept
{
    bfp16 ans = lhs;
    ans -= rhs;
    return ans;
}

inline bfp16 operator*(bfp16 lhs, bfp16 rhs) noexcept
{
    bfp16 ans = lhs;
    ans *= rhs;
    return ans;
}

inline bfp16 operator/(bfp16 lhs, bfp16 rhs) noexcept
{
    return bfp16(float(lhs) / float(rhs));
}
template <typename T>
inline bfp16 operator/(bfp16 lhs, T rhs) noexcept
{
    return bfp16(float(lhs) / float(rhs));
}

inline bool operator<(bfp16 lhs, bfp16 rhs) noexcept
{
    return float(lhs) < float(rhs);
}

inline bool operator==(bfp16 lhs, bfp16 rhs) noexcept
{
    return lhs._v == rhs._v;
}

inline bool operator<=(bfp16 lhs, bfp16 rhs) noexcept
{
    return (lhs < rhs) || (lhs == rhs);
}

inline bool operator>(bfp16 lhs, bfp16 rhs) noexcept
{
    return !(lhs <= rhs);
}

inline bool operator>=(bfp16 lhs, bfp16 rhs) noexcept
{
    return !(lhs < rhs);
}

inline bool operator!=(bfp16 lhs, bfp16 rhs) noexcept
{
    return !(lhs == rhs);
}

BFP16Func1(abs);
BFP16Func1(fabs);
BFP16Func1(sin);
BFP16Func1(cos);
BFP16Func1(tan);
BFP16Func1(asin);
BFP16Func1(acos);
BFP16Func1(atan);
BFP16Func1(sinh);
BFP16Func1(cosh);
BFP16Func1(tanh);
BFP16Func1(asinh);
BFP16Func1(acosh);
BFP16Func1(atanh);
BFP16Func1(erf);
BFP16Func1(erfc);
BFP16Func1(exp);
BFP16Func1(expm1);
BFP16Func1(exp2);
BFP16Func1(log);
BFP16Func1(log10);
BFP16Func1(log1p);
BFP16Func1(log2);
BFP16Func1(sqrt);
BFP16Func1(cbrt);

BFP16Func2(fmod);
BFP16Func2(remainder);
// BFP16Func2(remquo);
// BFP16Func2(fma);
BFP16Func2(fmax);
BFP16Func2(fmin);
BFP16Func2(fdim);

BFP16Func2(hypot);
BFP16Func2(pow);

BFP16Func1(lgamma);
BFP16Func1(tgamma);
BFP16Func1(ceil);
BFP16Func1(floor);
BFP16Func1(trunc);
BFP16Func1(round);
BFP16Func1(lround);
BFP16Func1(nearbyint);
BFP16Func1(rint);
BFP16Func1(lrint);
BFP16Func1(llround);
BFP16Func1(llrint);
BFP16Func1(ilogb);
BFP16Func1(logb);
BFP16Func1b(isfinite);
BFP16Func1b(isinf);
BFP16Func1b(isnan);
BFP16Func1b(isnormal);
BFP16Func1b(signbit);
BFP16Func2(atan2);
BFP16Func2b(isgreater);
BFP16Func2b(isgreaterequal);
BFP16Func2b(isless);
BFP16Func2b(islessequal);
BFP16Func2b(islessgreater);
BFP16Func2b(isunordered);

/// Extensions to the C++ standard library.
namespace std
{
/// Numeric limits for half-precision floats.
/// Because of the underlying single-precision implementation of many operations, it inherits some properties from
/// `std::numeric_limits<float>`.
template <>
class numeric_limits<aipubt::bfp16> : public numeric_limits<float>
{
public:
    /// Supports signed values.
    static constexpr bool is_signed = true;

    /// Is not exact.
    static constexpr bool is_exact = false;

    /// Doesn't provide modulo arithmetic.
    static constexpr bool is_modulo = false;

    /// IEEE conformant.
    static constexpr bool is_iec559 = true;

    /// Supports infinity.
    static constexpr bool has_infinity = true;

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
    static constexpr int min_exponent = -13;

    /// Smallest normalized representable power of 10.
    static constexpr int min_exponent10 = -4;

    /// One more than largest exponent
    static constexpr int max_exponent = 16;

    /// Largest finitely representable power of 10.
    static constexpr int max_exponent10 = 4;

    /// Smallest positive normal value.
    static constexpr aipubt::bfp16 min() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0x0080;
        return ret;
    }

    /// Smallest finite value.
    static constexpr aipubt::bfp16 lowest() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0xff7f;
        return ret;
    }

    /// Largest finite value.
    static constexpr aipubt::bfp16 max() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0x7f7f;
        return ret;
    }

    /// Difference between one and next representable value.
    static constexpr aipubt::bfp16 epsilon() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0x3f81;
        return ret;
    }

    /// Maximum rounding error.
    static constexpr aipubt::bfp16 round_error() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0x3f00;
        return ret;
    }

    /// Positive infinity.
    static constexpr aipubt::bfp16 infinity() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0x7f80;
        return ret;
    }

    /// Quiet NaN.
    static constexpr aipubt::bfp16 quiet_NaN() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0xffc1;
        return ret;
    }

    /// Signalling NaN.
    static constexpr aipubt::bfp16 signaling_NaN() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0xff81;
        return ret;
    }

    /// Smallest positive subnormal value.
    static constexpr aipubt::bfp16 denorm_min() noexcept
    {
        aipubt::bfp16 ret;
        ret._v = 0x0001;
        return ret;
    }
};

/// Hash function for half-precision floats.
/// This is only defined if C++11 `std::hash` is supported and enabled.
template <>
struct hash<aipubt::bfp16> //: unary_function<aipubt::bfp16,size_t>
{
    /// Type of function argument.
    typedef aipubt::bfp16 argument_type;

    /// Function return type.
    typedef size_t result_type;

    /// Compute hash function.
    /// \param arg half to hash
    /// \return hash value
    result_type operator()(argument_type arg) const
    {
        return hash<uint16_t>()(static_cast<unsigned>(arg._v) & -(arg._v != 0x8000));
    }
};
} // namespace std

#endif //! CORE_COMPUTE_BFP16_H
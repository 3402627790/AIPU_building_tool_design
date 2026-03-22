// half - IEEE 754-based half-precision floating point library.
//
// Copyright (c) 2012-2017 Christian Rau <rauy@users.sourceforge.net>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Version 1.12.0

/// \file
/// Main header file for half precision functionality.

#ifndef HALF_HALF_HPP
#define HALF_HALF_HPP

/// Combined gcc version number.
#define HALF_GNUC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)

//check C++11 language features
#if defined(__clang__) //clang
#if __has_feature(cxx_static_assert) && !defined(HALF_ENABLE_CPP11_STATIC_ASSERT)
#define HALF_ENABLE_CPP11_STATIC_ASSERT 1
#endif
#if __has_feature(cxx_constexpr) && !defined(HALF_ENABLE_CPP11_CONSTEXPR)
#define HALF_ENABLE_CPP11_CONSTEXPR 1
#endif
#if __has_feature(cxx_noexcept) && !defined(HALF_ENABLE_CPP11_NOEXCEPT)
#define HALF_ENABLE_CPP11_NOEXCEPT 1
#endif
#if __has_feature(cxx_user_literals) && !defined(HALF_ENABLE_CPP11_USER_LITERALS)
#define HALF_ENABLE_CPP11_USER_LITERALS 1
#endif
#if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L) && !defined(HALF_ENABLE_CPP11_LONG_LONG)
#define HALF_ENABLE_CPP11_LONG_LONG 1
#endif
/*#elif defined(__INTEL_COMPILER)								//Intel C++
	#if __INTEL_COMPILER >= 1100 && !defined(HALF_ENABLE_CPP11_STATIC_ASSERT)		????????
		#define HALF_ENABLE_CPP11_STATIC_ASSERT 1
	#endif
	#if __INTEL_COMPILER >= 1300 && !defined(HALF_ENABLE_CPP11_CONSTEXPR)			????????
		#define HALF_ENABLE_CPP11_CONSTEXPR 1
	#endif
	#if __INTEL_COMPILER >= 1300 && !defined(HALF_ENABLE_CPP11_NOEXCEPT)			????????
		#define HALF_ENABLE_CPP11_NOEXCEPT 1
	#endif
	#if __INTEL_COMPILER >= 1100 && !defined(HALF_ENABLE_CPP11_LONG_LONG)			????????
		#define HALF_ENABLE_CPP11_LONG_LONG 1
	#endif*/
#elif defined(__GNUC__) //gcc
#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L
#if HALF_GNUC_VERSION >= 403 && !defined(HALF_ENABLE_CPP11_STATIC_ASSERT)
#define HALF_ENABLE_CPP11_STATIC_ASSERT 1
#endif
#if HALF_GNUC_VERSION >= 406 && !defined(HALF_ENABLE_CPP11_CONSTEXPR)
#define HALF_ENABLE_CPP11_CONSTEXPR 1
#endif
#if HALF_GNUC_VERSION >= 406 && !defined(HALF_ENABLE_CPP11_NOEXCEPT)
#define HALF_ENABLE_CPP11_NOEXCEPT 1
#endif
#if HALF_GNUC_VERSION >= 407 && !defined(HALF_ENABLE_CPP11_USER_LITERALS)
#define HALF_ENABLE_CPP11_USER_LITERALS 1
#endif
#if !defined(HALF_ENABLE_CPP11_LONG_LONG)
#define HALF_ENABLE_CPP11_LONG_LONG 1
#endif
#endif
#elif defined(_MSC_VER) //Visual C++
#if _MSC_VER >= 1900 && !defined(HALF_ENABLE_CPP11_CONSTEXPR)
#define HALF_ENABLE_CPP11_CONSTEXPR 1
#endif
#if _MSC_VER >= 1900 && !defined(HALF_ENABLE_CPP11_NOEXCEPT)
#define HALF_ENABLE_CPP11_NOEXCEPT 1
#endif
#if _MSC_VER >= 1900 && !defined(HALF_ENABLE_CPP11_USER_LITERALS)
#define HALF_ENABLE_CPP11_USER_LITERALS 1
#endif
#if _MSC_VER >= 1600 && !defined(HALF_ENABLE_CPP11_STATIC_ASSERT)
#define HALF_ENABLE_CPP11_STATIC_ASSERT 1
#endif
#if _MSC_VER >= 1310 && !defined(HALF_ENABLE_CPP11_LONG_LONG)
#define HALF_ENABLE_CPP11_LONG_LONG 1
#endif
#define HALF_POP_WARNINGS 1
#pragma warning(push)
#pragma warning(disable : 4099 4127 4146) //struct vs class, constant in if, negative unsigned
#endif

//check C++11 library features
#include <utility>
#if defined(_LIBCPP_VERSION) //libc++
#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103
#ifndef HALF_ENABLE_CPP11_TYPE_TRAITS
#define HALF_ENABLE_CPP11_TYPE_TRAITS 1
#endif
#ifndef HALF_ENABLE_CPP11_CSTDINT
#define HALF_ENABLE_CPP11_CSTDINT 1
#endif
#ifndef HALF_ENABLE_CPP11_CMATH
#define HALF_ENABLE_CPP11_CMATH 1
#endif
#ifndef HALF_ENABLE_CPP11_HASH
#define HALF_ENABLE_CPP11_HASH 1
#endif
#endif
#elif defined(__GLIBCXX__) //libstdc++
#if defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103
#ifdef __clang__
#if __GLIBCXX__ >= 20080606 && !defined(HALF_ENABLE_CPP11_TYPE_TRAITS)
#define HALF_ENABLE_CPP11_TYPE_TRAITS 1
#endif
#if __GLIBCXX__ >= 20080606 && !defined(HALF_ENABLE_CPP11_CSTDINT)
#define HALF_ENABLE_CPP11_CSTDINT 1
#endif
#if __GLIBCXX__ >= 20080606 && !defined(HALF_ENABLE_CPP11_CMATH)
#define HALF_ENABLE_CPP11_CMATH 1
#endif
#if __GLIBCXX__ >= 20080606 && !defined(HALF_ENABLE_CPP11_HASH)
#define HALF_ENABLE_CPP11_HASH 1
#endif
#else
#if HALF_GNUC_VERSION >= 403 && !defined(HALF_ENABLE_CPP11_CSTDINT)
#define HALF_ENABLE_CPP11_CSTDINT 1
#endif
#if HALF_GNUC_VERSION >= 403 && !defined(HALF_ENABLE_CPP11_CMATH)
#define HALF_ENABLE_CPP11_CMATH 1
#endif
#if HALF_GNUC_VERSION >= 403 && !defined(HALF_ENABLE_CPP11_HASH)
#define HALF_ENABLE_CPP11_HASH 1
#endif
#endif
#endif
#elif defined(_CPPLIB_VER) //Dinkumware/Visual C++
#if _CPPLIB_VER >= 520
#ifndef HALF_ENABLE_CPP11_TYPE_TRAITS
#define HALF_ENABLE_CPP11_TYPE_TRAITS 1
#endif
#ifndef HALF_ENABLE_CPP11_CSTDINT
#define HALF_ENABLE_CPP11_CSTDINT 1
#endif
#ifndef HALF_ENABLE_CPP11_HASH
#define HALF_ENABLE_CPP11_HASH 1
#endif
#endif
#if _CPPLIB_VER >= 610
#ifndef HALF_ENABLE_CPP11_CMATH
#define HALF_ENABLE_CPP11_CMATH 1
#endif
#endif
#endif
#undef HALF_GNUC_VERSION

//support constexpr
#if HALF_ENABLE_CPP11_CONSTEXPR
#define HALF_CONSTEXPR constexpr
#define HALF_CONSTEXPR_CONST constexpr
#else
#define HALF_CONSTEXPR
#define HALF_CONSTEXPR_CONST const
#endif

//support noexcept
#if HALF_ENABLE_CPP11_NOEXCEPT
#define HALF_NOEXCEPT noexcept
#define HALF_NOTHROW noexcept
#else
#define HALF_NOEXCEPT
#define HALF_NOTHROW throw()
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#if HALF_ENABLE_CPP11_TYPE_TRAITS
#include <type_traits>
#endif
#if HALF_ENABLE_CPP11_CSTDINT
#include <cstdint>
#endif
#if HALF_ENABLE_CPP11_HASH
#include <functional>
#endif

/// Default rounding mode.
/// This specifies the rounding mode used for all conversions between [half](\ref half_float::half)s and `float`s as well as
/// for the half_cast() if not specifying a rounding mode explicitly. It can be redefined (before including half.hpp) to one
/// of the standard rounding modes using their respective constants or the equivalent values of `std::float_round_style`:
///
/// `std::float_round_style`         | value | rounding
/// ---------------------------------|-------|-------------------------
/// `std::round_indeterminate`       | -1    | fastest (default)
/// `std::round_toward_zero`         | 0     | toward zero
/// `std::round_to_nearest`          | 1     | to nearest
/// `std::round_toward_infinity`     | 2     | toward positive infinity
/// `std::round_toward_neg_infinity` | 3     | toward negative infinity
///
/// By default this is set to `-1` (`std::round_indeterminate`), which uses truncation (round toward zero, but with overflows
/// set to infinity) and is the fastest rounding mode possible. It can even be set to `std::numeric_limits<float>::round_style`
/// to synchronize the rounding mode with that of the underlying single-precision implementation.
#ifndef HALF_ROUND_STYLE
#define HALF_ROUND_STYLE 1 // = std::round_indeterminate
#endif

/// Tie-breaking behaviour for round to nearest.
/// This specifies if ties in round to nearest should be resolved by rounding to the nearest even value. By default this is
/// defined to `0` resulting in the faster but slightly more biased behaviour of rounding away from zero in half-way cases (and
/// thus equal to the round() function), but can be redefined to `1` (before including half.hpp) if more IEEE-conformant
/// behaviour is needed.
#ifndef HALF_ROUND_TIES_TO_EVEN
#define HALF_ROUND_TIES_TO_EVEN 1 // ties away from zero
#endif

/// Value signaling overflow.
/// In correspondence with `HUGE_VAL[F|L]` from `<cmath>` this symbol expands to a positive value signaling the overflow of an
/// operation, in particular it just evaluates to positive infinity.
#define HUGE_VALH std::numeric_limits<half_float::half>::infinity()

/// Fast half-precision fma function.
/// This symbol is only defined if the fma() function generally executes as fast as, or faster than, a separate
/// half-precision multiplication followed by an addition. Due to the internal single-precision implementation of all
/// arithmetic operations, this is in fact always the case.
#define FP_FAST_FMAH 1

#ifndef FP_ILOGB0
#define FP_ILOGB0 INT_MIN
#endif
#ifndef FP_ILOGBNAN
#define FP_ILOGBNAN INT_MAX
#endif
#ifndef FP_SUBNORMAL
#define FP_SUBNORMAL 0
#endif
#ifndef FP_ZERO
#define FP_ZERO 1
#endif
#ifndef FP_NAN
#define FP_NAN 2
#endif
#ifndef FP_INFINITE
#define FP_INFINITE 3
#endif
#ifndef FP_NORMAL
#define FP_NORMAL 4
#endif

/// Main namespace for half precision functionality.
/// This namespace contains all the functionality provided by the library.
namespace half_float
{
class half;

#if HALF_ENABLE_CPP11_USER_LITERALS
/// Library-defined half-precision literals.
/// Import this namespace to enable half-precision floating point literals:
/// ~~~~{.cpp}
/// using namespace half_float::literal;
/// half_float::half = 4.2_h;
/// ~~~~
namespace literal
{
    half operator""_h(long double);
}
#endif

/// \internal
/// \brief Implementation details.
namespace detail
{
#if HALF_ENABLE_CPP11_TYPE_TRAITS
    /// Conditional type.
    template <bool B, typename T, typename F>
    struct conditional : std::conditional<B, T, F>
    {
    };

    /// Helper for tag dispatching.
    template <bool B>
    struct bool_type : std::integral_constant<bool, B>
    {
    };
    using std::false_type;
    using std::true_type;

    /// Type traits for floating point types.
    template <typename T>
    struct is_float : std::is_floating_point<T>
    {
    };
#else
    /// Conditional type.
    template <bool, typename T, typename>
    struct conditional
    {
        typedef T type;
    };
    template <typename T, typename F>
    struct conditional<false, T, F>
    {
        typedef F type;
    };

    /// Helper for tag dispatching.
    template <bool>
    struct bool_type
    {
    };
    typedef bool_type<true> true_type;
    typedef bool_type<false> false_type;

    /// Type traits for floating point types.
    template <typename>
    struct is_float : false_type
    {
    };
    template <typename T>
    struct is_float<const T> : is_float<T>
    {
    };
    template <typename T>
    struct is_float<volatile T> : is_float<T>
    {
    };
    template <typename T>
    struct is_float<const volatile T> : is_float<T>
    {
    };
    template <>
    struct is_float<float> : true_type
    {
    };
    template <>
    struct is_float<double> : true_type
    {
    };
    template <>
    struct is_float<long double> : true_type
    {
    };
#endif

    /// Type traits for floating point bits.
    template <typename T>
    struct bits
    {
        typedef unsigned char type;
    };
    template <typename T>
    struct bits<const T> : bits<T>
    {
    };
    template <typename T>
    struct bits<volatile T> : bits<T>
    {
    };
    template <typename T>
    struct bits<const volatile T> : bits<T>
    {
    };

#if HALF_ENABLE_CPP11_CSTDINT
    /// Unsigned integer of (at least) 16 bits width.
    typedef std::uint_least16_t uint16;

    /// Unsigned integer of (at least) 32 bits width.
    template <>
    struct bits<float>
    {
        typedef std::uint_least32_t type;
    };

    /// Unsigned integer of (at least) 64 bits width.
    template <>
    struct bits<double>
    {
        typedef std::uint_least64_t type;
    };
#else
    /// Unsigned integer of (at least) 16 bits width.
    typedef unsigned short uint16;

    /// Unsigned integer of (at least) 32 bits width.
    template <>
    struct bits<float> : conditional<std::numeric_limits<unsigned int>::digits >= 32, unsigned int, unsigned long>
    {
    };

#if HALF_ENABLE_CPP11_LONG_LONG
    /// Unsigned integer of (at least) 64 bits width.
    template <>
    struct bits<double> : conditional<std::numeric_limits<unsigned long>::digits >= 64, unsigned long, unsigned long long>
    {
    };
#else
    /// Unsigned integer of (at least) 64 bits width.
    template <>
    struct bits<double>
    {
        typedef unsigned long type;
    };
#endif
#endif

    /// Tag type for binary construction.
    struct binary_t
    {
    };

    /// Tag for binary construction.
    HALF_CONSTEXPR_CONST binary_t binary = binary_t();

    /// Temporary half-precision expression.
    /// This class represents a half-precision expression which just stores a single-precision value internally.
    struct expr
    {
        /// Conversion constructor.
        /// \param f single-precision value to convert
        explicit HALF_CONSTEXPR expr(float f) HALF_NOEXCEPT : value_(f) {}

        /// Conversion to single-precision.
        /// \return single precision value representing expression value
        HALF_CONSTEXPR operator float() const HALF_NOEXCEPT { return value_; }

    private:
        /// Internal expression value stored in single-precision.
        float value_;
    };

    /// SFINAE helper for generic half-precision functions.
    /// This class template has to be specialized for each valid combination of argument types to provide a corresponding
    /// `type` member equivalent to \a T.
    /// \tparam T type to return
    template <typename T, typename, typename = void, typename = void>
    struct enable
    {
    };
    template <typename T>
    struct enable<T, half, void, void>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, half, size_t, void>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, expr, void, void>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, half, half, void>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, half, expr, void>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, expr, half, void>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, expr, expr, void>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, half, half, half>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, half, half, expr>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, half, expr, half>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, half, expr, expr>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, expr, half, half>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, expr, half, expr>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, expr, expr, half>
    {
        typedef T type;
    };
    template <typename T>
    struct enable<T, expr, expr, expr>
    {
        typedef T type;
    };

    /// Return type for specialized generic 2-argument half-precision functions.
    /// This class template has to be specialized for each valid combination of argument types to provide a corresponding
    /// `type` member denoting the appropriate return type.
    /// \tparam T first argument type
    /// \tparam U first argument type
    template <typename T, typename U>
    struct result : enable<expr, T, U>
    {
    };
    template <>
    struct result<half, half>
    {
        typedef half type;
    };

    /// \name Classification helpers
    /// \{

    /// Check for infinity.
    /// \tparam T argument type (builtin floating point type)
    /// \param arg value to query
    /// \retval true if infinity
    /// \retval false else
    template <typename T>
    bool builtin_isinf(T arg)
    {
#if HALF_ENABLE_CPP11_CMATH
        return std::isinf(arg);
#elif defined(_MSC_VER)
        return !::_finite(static_cast<double>(arg)) && !::_isnan(static_cast<double>(arg));
#else
        return arg == std::numeric_limits<T>::infinity() || arg == -std::numeric_limits<T>::infinity();
#endif
    }

    /// Check for NaN.
    /// \tparam T argument type (builtin floating point type)
    /// \param arg value to query
    /// \retval true if not a number
    /// \retval false else
    template <typename T>
    bool builtin_isnan(T arg)
    {
#if HALF_ENABLE_CPP11_CMATH
        return std::isnan(arg);
#elif defined(_MSC_VER)
        return ::_isnan(static_cast<double>(arg)) != 0;
#else
        return arg != arg;
#endif
    }

    /// Check sign.
    /// \tparam T argument type (builtin floating point type)
    /// \param arg value to query
    /// \retval true if signbit set
    /// \retval false else
    template <typename T>
    bool builtin_signbit(T arg)
    {
#if HALF_ENABLE_CPP11_CMATH
        return std::signbit(arg);
#else
        return arg < T() || (arg == T() && T(1) / arg < T());
#endif
    }

    /// \}
    /// \name Conversion
    /// \{

    /// Convert IEEE single-precision to half-precision.
    /// Credit for this goes to [Jeroen van der Zijp](ftp://ftp.fox-toolkit.org/pub/fasthalffloatconversion.pdf).
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \param value single-precision value
    /// \return binary representation of half-precision value
    template <std::float_round_style R>
    uint16 float2half_impl(float value, true_type)
    {
        typedef bits<float>::type uint32;
        uint32 bits; // = *reinterpret_cast<uint32*>(&value);		//violating strict aliasing!
        std::memcpy(&bits, &value, sizeof(float));
        /*			uint16 hbits = (bits>>16) & 0x8000;
			bits &= 0x7FFFFFFF;
			int exp = bits >> 23;
			if(exp == 255)
				return hbits | 0x7C00 | (0x3FF&-static_cast<unsigned>((bits&0x7FFFFF)!=0));
			if(exp > 142)
			{
				if(R == std::round_toward_infinity)
					return hbits | 0x7C00 - (hbits>>15);
				if(R == std::round_toward_neg_infinity)
					return hbits | 0x7BFF + (hbits>>15);
				return hbits | 0x7BFF + (R!=std::round_toward_zero);
			}
			int g, s;
			if(exp > 112)
			{
				g = (bits>>12) & 1;
				s = (bits&0xFFF) != 0;
				hbits |= ((exp-112)<<10) | ((bits>>13)&0x3FF);
			}
			else if(exp > 101)
			{
				int i = 125 - exp;
				bits = (bits&0x7FFFFF) | 0x800000;
				g = (bits>>i) & 1;
				s = (bits&((1L<<i)-1)) != 0;
				hbits |= bits >> (i+1);
			}
			else
			{
				g = 0;
				s = bits != 0;
			}
			if(R == std::round_to_nearest)
				#if HALF_ROUND_TIES_TO_EVEN
					hbits += g & (s|hbits);
				#else
					hbits += g;
				#endif
			else if(R == std::round_toward_infinity)
				hbits += ~(hbits>>15) & (s|g);
			else if(R == std::round_toward_neg_infinity)
				hbits += (hbits>>15) & (g|s);
*/
        static const uint16 base_table[512] = {
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 0x0080, 0x0100,
            0x0200, 0x0400, 0x0800, 0x0C00, 0x1000, 0x1400, 0x1800, 0x1C00, 0x2000, 0x2400, 0x2800, 0x2C00, 0x3000, 0x3400, 0x3800, 0x3C00,
            0x4000, 0x4400, 0x4800, 0x4C00, 0x5000, 0x5400, 0x5800, 0x5C00, 0x6000, 0x6400, 0x6800, 0x6C00, 0x7000, 0x7400, 0x7800, 0x7C00,
            0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
            0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
            0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
            0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
            0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
            0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
            0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00, 0x7C00,
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000,
            0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8001, 0x8002, 0x8004, 0x8008, 0x8010, 0x8020, 0x8040, 0x8080, 0x8100,
            0x8200, 0x8400, 0x8800, 0x8C00, 0x9000, 0x9400, 0x9800, 0x9C00, 0xA000, 0xA400, 0xA800, 0xAC00, 0xB000, 0xB400, 0xB800, 0xBC00,
            0xC000, 0xC400, 0xC800, 0xCC00, 0xD000, 0xD400, 0xD800, 0xDC00, 0xE000, 0xE400, 0xE800, 0xEC00, 0xF000, 0xF400, 0xF800, 0xFC00,
            0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
            0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
            0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
            0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
            0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
            0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00,
            0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00, 0xFC00};
        static const unsigned char shift_table[512] = {
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 13,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13,
            13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24,
            24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 13};
        uint16 hbits = base_table[bits >> 23] + static_cast<uint16>((bits & 0x7FFFFF) >> shift_table[bits >> 23]);
        if (R == std::round_to_nearest)
            hbits += (((bits & 0x7FFFFF) >> (shift_table[bits >> 23] - 1)) | (((bits >> 23) & 0xFF) == 102)) & ((hbits & 0x7C00) != 0x7C00)
#if HALF_ROUND_TIES_TO_EVEN
                     & (((((static_cast<uint32>(1) << (shift_table[bits >> 23] - 1)) - 1) & bits) != 0) | hbits)
#endif
                ;
        else if (R == std::round_toward_zero)
            hbits -= ((hbits & 0x7FFF) == 0x7C00) & ~shift_table[bits >> 23];
        else if (R == std::round_toward_infinity)
            hbits += ((((bits & 0x7FFFFF & ((static_cast<uint32>(1) << (shift_table[bits >> 23])) - 1)) != 0) | (((bits >> 23) <= 102) &
                                                                                                                 ((bits >> 23) != 0))) &
                      (hbits < 0x7C00)) -
                     ((hbits == 0xFC00) & ((bits >> 23) != 511));
        else if (R == std::round_toward_neg_infinity)
            hbits += ((((bits & 0x7FFFFF & ((static_cast<uint32>(1) << (shift_table[bits >> 23])) - 1)) != 0) | (((bits >> 23) <= 358) &
                                                                                                                 ((bits >> 23) != 256))) &
                      (hbits < 0xFC00) & (hbits >> 15)) -
                     ((hbits == 0x7C00) & ((bits >> 23) != 255));
        return hbits;
    }

    /// Convert IEEE double-precision to half-precision.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \param value double-precision value
    /// \return binary representation of half-precision value
    template <std::float_round_style R>
    uint16 float2half_impl(double value, true_type)
    {
        typedef bits<float>::type uint32;
        typedef bits<double>::type uint64;
        uint64 bits; // = *reinterpret_cast<uint64*>(&value);		//violating strict aliasing!
        std::memcpy(&bits, &value, sizeof(double));
        uint32 hi = bits >> 32, lo = bits & 0xFFFFFFFF;
        uint16 hbits = (hi >> 16) & 0x8000;
        hi &= 0x7FFFFFFF;
        int exp = hi >> 20;
        if (exp == 2047)
            return hbits | 0x7C00 | (0x3FF & -static_cast<unsigned>((bits & 0xFFFFFFFFFFFFF) != 0));
        if (exp > 1038)
        {
            if (R == std::round_toward_infinity)
                return hbits | 0x7C00 - (hbits >> 15);
            if (R == std::round_toward_neg_infinity)
                return hbits | 0x7BFF + (hbits >> 15);
            return hbits | 0x7BFF + (R != std::round_toward_zero);
        }
        int g, s = lo != 0;
        if (exp > 1008)
        {
            g = (hi >> 9) & 1;
            s |= (hi & 0x1FF) != 0;
            hbits |= ((exp - 1008) << 10) | ((hi >> 10) & 0x3FF);
        }
        else if (exp > 997)
        {
            int i = 1018 - exp;
            hi = (hi & 0xFFFFF) | 0x100000;
            g = (hi >> i) & 1;
            s |= (hi & ((1L << i) - 1)) != 0;
            hbits |= hi >> (i + 1);
        }
        else
        {
            g = 0;
            s |= hi != 0;
        }
        if (R == std::round_to_nearest)
#if HALF_ROUND_TIES_TO_EVEN
            hbits += g & (s | hbits);
#else
            hbits += g;
#endif
        else if (R == std::round_toward_infinity)
            hbits += ~(hbits >> 15) & (s | g);
        else if (R == std::round_toward_neg_infinity)
            hbits += (hbits >> 15) & (g | s);
        return hbits;
    }

    /// Convert non-IEEE floating point to half-precision.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \tparam T source type (builtin floating point type)
    /// \param value floating point value
    /// \return binary representation of half-precision value
    template <std::float_round_style R, typename T>
    uint16 float2half_impl(T value, ...)
    {
        uint16 hbits = static_cast<unsigned>(builtin_signbit(value)) << 15;
        if (value == T())
            return hbits;
        if (builtin_isnan(value))
            return hbits | 0x7FFF;
        if (builtin_isinf(value))
            return hbits | 0x7C00;
        int exp;
        std::frexp(value, &exp);
        if (exp > 16)
        {
            if (R == std::round_toward_infinity)
                return hbits | (0x7C00 - (hbits >> 15));
            else if (R == std::round_toward_neg_infinity)
                return hbits | (0x7BFF + (hbits >> 15));
            return hbits | (0x7BFF + (R != std::round_toward_zero));
        }
        if (exp < -13)
            value = std::ldexp(value, 24);
        else
        {
            value = std::ldexp(value, 11 - exp);
            hbits |= ((exp + 13) << 10);
        }
        T ival, frac = std::modf(value, &ival);
        hbits += static_cast<uint16>(std::abs(static_cast<int>(ival)));
        if (R == std::round_to_nearest)
        {
            frac = std::abs(frac);
#if HALF_ROUND_TIES_TO_EVEN
            hbits += (frac > T(0.5)) | ((frac == T(0.5)) & hbits);
#else
            hbits += frac >= T(0.5);
#endif
        }
        else if (R == std::round_toward_infinity)
            hbits += frac > T();
        else if (R == std::round_toward_neg_infinity)
            hbits += frac < T();
        return hbits;
    }

    /// Convert floating point to half-precision.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \tparam T source type (builtin floating point type)
    /// \param value floating point value
    /// \return binary representation of half-precision value
    template <std::float_round_style R, typename T>
    uint16 float2half(T value)
    {
        return float2half_impl<R>(value, bool_type < std::numeric_limits<T>::is_iec559 && sizeof(typename bits<T>::type) == sizeof(T) > ());
    }

    /// Convert integer to half-precision floating point.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \tparam S `true` if value negative, `false` else
    /// \tparam T type to convert (builtin integer type)
    /// \param value non-negative integral value
    /// \return binary representation of half-precision value
    template <std::float_round_style R, bool S, typename T>
    uint16 int2half_impl(T value)
    {
#if HALF_ENABLE_CPP11_STATIC_ASSERT && HALF_ENABLE_CPP11_TYPE_TRAITS
        static_assert(std::is_integral<T>::value, "int to half conversion only supports builtin integer types");
#endif
        if (S)
            value = -value;
        uint16 bits = S << 15;
        if (value > 0xFFFF)
        {
            if (R == std::round_toward_infinity)
                bits |= 0x7C00 - S;
            else if (R == std::round_toward_neg_infinity)
                bits |= 0x7BFF + S;
            else
                bits |= 0x7BFF + (R != std::round_toward_zero);
        }
        else if (value)
        {
            unsigned int m = value, exp = 24;
            for (; m < 0x400; m <<= 1, --exp)
                ;
            for (; m > 0x7FF; m >>= 1, ++exp)
                ;
            bits |= (exp << 10) + m;
            if (exp > 24)
            {
                if (R == std::round_to_nearest)
                    bits += (value >> (exp - 25)) & 1
#if HALF_ROUND_TIES_TO_EVEN
                            & (((((1 << (exp - 25)) - 1) & value) != 0) | bits)
#endif
                        ;
                else if (R == std::round_toward_infinity)
                    bits += ((value & ((1 << (exp - 24)) - 1)) != 0) & !S;
                else if (R == std::round_toward_neg_infinity)
                    bits += ((value & ((1 << (exp - 24)) - 1)) != 0) & S;
            }
        }
        return bits;
    }

    /// Convert integer to half-precision floating point.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \tparam T type to convert (builtin integer type)
    /// \param value integral value
    /// \return binary representation of half-precision value
    template <std::float_round_style R, typename T>
    uint16 int2half(T value)
    {
        return (value < 0) ? int2half_impl<R, true>(value) : int2half_impl<R, false>(value);
    }

    /// Convert half-precision to IEEE single-precision.
    /// Credit for this goes to [Jeroen van der Zijp](ftp://ftp.fox-toolkit.org/pub/fasthalffloatconversion.pdf).
    /// \param value binary representation of half-precision value
    /// \return single-precision value
    inline float half2float_impl(uint16 value, float, true_type)
    {
        typedef bits<float>::type uint32;
        /*			uint32 bits = static_cast<uint32>(value&0x8000) << 16;
			int abs = value & 0x7FFF;
			if(abs)
			{
				bits |= 0x38000000 << static_cast<unsigned>(abs>=0x7C00);
				for(; abs<0x400; abs<<=1,bits-=0x800000) ;
				bits += static_cast<uint32>(abs) << 13;
			}
*/
        static const uint32 mantissa_table[2048] = {
            0x00000000, 0x33800000, 0x34000000, 0x34400000, 0x34800000, 0x34A00000, 0x34C00000, 0x34E00000, 0x35000000, 0x35100000, 0x35200000, 0x35300000, 0x35400000, 0x35500000, 0x35600000, 0x35700000,
            0x35800000, 0x35880000, 0x35900000, 0x35980000, 0x35A00000, 0x35A80000, 0x35B00000, 0x35B80000, 0x35C00000, 0x35C80000, 0x35D00000, 0x35D80000, 0x35E00000, 0x35E80000, 0x35F00000, 0x35F80000,
            0x36000000, 0x36040000, 0x36080000, 0x360C0000, 0x36100000, 0x36140000, 0x36180000, 0x361C0000, 0x36200000, 0x36240000, 0x36280000, 0x362C0000, 0x36300000, 0x36340000, 0x36380000, 0x363C0000,
            0x36400000, 0x36440000, 0x36480000, 0x364C0000, 0x36500000, 0x36540000, 0x36580000, 0x365C0000, 0x36600000, 0x36640000, 0x36680000, 0x366C0000, 0x36700000, 0x36740000, 0x36780000, 0x367C0000,
            0x36800000, 0x36820000, 0x36840000, 0x36860000, 0x36880000, 0x368A0000, 0x368C0000, 0x368E0000, 0x36900000, 0x36920000, 0x36940000, 0x36960000, 0x36980000, 0x369A0000, 0x369C0000, 0x369E0000,
            0x36A00000, 0x36A20000, 0x36A40000, 0x36A60000, 0x36A80000, 0x36AA0000, 0x36AC0000, 0x36AE0000, 0x36B00000, 0x36B20000, 0x36B40000, 0x36B60000, 0x36B80000, 0x36BA0000, 0x36BC0000, 0x36BE0000,
            0x36C00000, 0x36C20000, 0x36C40000, 0x36C60000, 0x36C80000, 0x36CA0000, 0x36CC0000, 0x36CE0000, 0x36D00000, 0x36D20000, 0x36D40000, 0x36D60000, 0x36D80000, 0x36DA0000, 0x36DC0000, 0x36DE0000,
            0x38740000, 0x38742000, 0x38744000, 0x38746000, 0x38748000, 0x3874A000, 0x3874C000, 0x3874E000, 0x38750000, 0x38752000, 0x38754000, 0x38756000, 0x38758000, 0x3875A000, 0x3875C000, 0x3875E000,
         static const uint32 exponent_table[64] = {
            0x00000000, 0x00800000, 0x01000000, 0x01800000, 0x02000000, 0x02800000, 0x03000000, 0x03800000, 0x04000000, 0x04800000, 0x05000000, 0x05800000, 0x06000000, 0x06800000, 0x07000000, 0x07800000,
            0x08000000, 0x08800000, 0x09000000, 0x09800000, 0x0A000000, 0x0A800000, 0x0B000000, 0x0B800000, 0x0C000000, 0x0C800000, 0x0D000000, 0x0D800000, 0x0E000000, 0x0E800000, 0x0F000000, 0x47800000,
            0x80000000, 0x80800000, 0x81000000, 0x81800000, 0x82000000, 0x82800000, 0x83000000, 0x83800000, 0x84000000, 0x84800000, 0x85000000, 0x85800000, 0x86000000, 0x86800000, 0x87000000, 0x87800000,
            0x88000000, 0x88800000, 0x89000000, 0x89800000, 0x8A000000, 0x8A800000, 0x8B000000, 0x8B800000, 0x8C000000, 0x8C800000, 0x8D000000, 0x8D800000, 0x8E000000, 0x8E800000, 0x8F000000, 0xC7800000};
        static const unsigned short offset_table[64] = {
            0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024,
            0, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};
        uint32 bits = mantissa_table[offset_table[value >> 10] + (value & 0x3FF)] + exponent_table[value >> 10];
        //			return *reinterpret_cast<float*>(&bits);			//violating strict aliasing!
        float out;
        std::memcpy(&out, &bits, sizeof(float));
        return out;
    }

    /// Convert half-precision to non-IEEE floating point.
    /// \tparam T type to convert to (builtin integer type)
    /// \param value binary representation of half-precision value
    /// \return floating point value
    template <typename T>
    T half2float_impl(uint16 value, T, ...)
    {
        T out;
        int abs = value & 0x7FFF;
        if (abs > 0x7C00)
            out = std::numeric_limits<T>::has_quiet_NaN ? std::numeric_limits<T>::quiet_NaN() : T();
        else if (abs == 0x7C00)
            out = std::numeric_limits<T>::has_infinity ? std::numeric_limits<T>::infinity() : std::numeric_limits<T>::max();
        else if (abs > 0x3FF)
            out = std::ldexp(static_cast<T>((abs & 0x3FF) | 0x400), (abs >> 10) - 25);
        else
            out = std::ldexp(static_cast<T>(abs), -24);
        return (value & 0x8000) ? -out : out;
    }

    /// Convert half-precision to floating point.
    /// \tparam T type to convert to (builtin integer type)
    /// \param value binary representation of half-precision value
    /// \return floating point value
    template <typename T>
    T half2float(uint16 value)
    {
        return half2float_impl(value, T(), bool_type < std::numeric_limits<T>::is_iec559 && sizeof(typename bits<T>::type) == sizeof(T) > ());
    }

    /// Convert half-precision floating point to integer.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \tparam E `true` for round to even, `false` for round away from zero
    /// \tparam T type to convert to (buitlin integer type with at least 16 bits precision, excluding any implicit sign bits)
    /// \param value binary representation of half-precision value
    /// \return integral value
    template <std::float_round_style R, bool E, typename T>
    T half2int_impl(uint16 value)
    {
#if HALF_ENABLE_CPP11_STATIC_ASSERT && HALF_ENABLE_CPP11_TYPE_TRAITS
        static_assert(std::is_integral<T>::value, "half to int conversion only supports builtin integer types");
#endif
        unsigned int e = value & 0x7FFF;
        if (e >= 0x7C00)
            return (value & 0x8000) ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max();
        if (e < 0x3800)
        {
            if (R == std::round_toward_infinity)
                return T(~(value >> 15) & (e != 0));
            else if (R == std::round_toward_neg_infinity)
                return -T(value > 0x8000);
            return T();
        }
        unsigned int m = (value & 0x3FF) | 0x400;
        e >>= 10;
        if (e < 25)
        {
            if (R == std::round_to_nearest)
                m += (1 << (24 - e)) - (~(m >> (25 - e)) & E);
            else if (R == std::round_toward_infinity)
                m += ((value >> 15) - 1) & ((1 << (25 - e)) - 1U);
            else if (R == std::round_toward_neg_infinity)
                m += -(value >> 15) & ((1 << (25 - e)) - 1U);
            m >>= 25 - e;
        }
        else
            m <<= e - 25;
        return (value & 0x8000) ? -static_cast<T>(m) : static_cast<T>(m);
    }

    /// Convert half-precision floating point to integer.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \tparam T type to convert to (buitlin integer type with at least 16 bits precision, excluding any implicit sign bits)
    /// \param value binary representation of half-precision value
    /// \return integral value
    template <std::float_round_style R, typename T>
    T half2int(uint16 value) { return half2int_impl<R, HALF_ROUND_TIES_TO_EVEN, T>(value); }

    /// Convert half-precision floating point to integer using round-to-nearest-away-from-zero.
    /// \tparam T type to convert to (buitlin integer type with at least 16 bits precision, excluding any implicit sign bits)
    /// \param value binary representation of half-precision value
    /// \return integral value
    template <typename T>
    T half2int_up(uint16 value) { return half2int_impl<std::round_to_nearest, 0, T>(value); }

    /// Round half-precision number to nearest integer value.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \tparam E `true` for round to even, `false` for round away from zero
    /// \param value binary representation of half-precision value
    /// \return half-precision bits for nearest integral value
    template <std::float_round_style R, bool E>
    uint16 round_half_impl(uint16 value)
    {
        unsigned int e = value & 0x7FFF;
        uint16 result = value;
        if (e < 0x3C00)
        {
            result &= 0x8000;
            if (R == std::round_to_nearest)
                result |= 0x3C00U & -(e >= (0x3800 + E));
            else if (R == std::round_toward_infinity)
                result |= 0x3C00U & -(~(value >> 15) & (e != 0));
            else if (R == std::round_toward_neg_infinity)
                result |= 0x3C00U & -(value > 0x8000);
        }
        else if (e < 0x6400)
        {
            e = 25 - (e >> 10);
            unsigned int mask = (1 << e) - 1;
            if (R == std::round_to_nearest)
                result += (1 << (e - 1)) - (~(result >> e) & E);
            else if (R == std::round_toward_infinity)
                result += mask & ((value >> 15) - 1);
            else if (R == std::round_toward_neg_infinity)
                result += mask & -(value >> 15);
            result &= ~mask;
        }
        return result;
    }

    /// Round half-precision number to nearest integer value.
    /// \tparam R rounding mode to use, `std::round_indeterminate` for fastest rounding
    /// \param value binary representation of half-precision value
    /// \return half-precision bits for nearest integral value
    template <std::float_round_style R>
    uint16 round_half(uint16 value) { return round_half_impl<R, HALF_ROUND_TIES_TO_EVEN>(value); }

    /// Round half-precision number to nearest integer value using round-to-nearest-away-from-zero.
    /// \param value binary representation of half-precision value
    /// \return half-precision bits for nearest integral value
    inline uint16 round_half_up(uint16 value) { return round_half_impl<std::round_to_nearest, 0>(value); }
    /// \}

    struct functions;
    template <typename>
    struct unary_specialized;
    template <typename, typename>
    struct binary_specialized;
    template <typename, typename, std::float_round_style>
    struct half_caster;
} // namespace detail

/// Half-precision floating point type.
/// This class implements an IEEE-conformant half-precision floating point type with the usual arithmetic operators and
/// conversions. It is implicitly convertible to single-precision floating point, which makes artihmetic expressions and
/// functions with mixed-type operands to be of the most precise operand type. Additionally all arithmetic operations
/// (and many mathematical functions) are carried out in single-precision internally. All conversions from single- to
/// half-precision are done using the library's default rounding mode, but temporary results inside chained arithmetic
/// expressions are kept in single-precision as long as possible (while of course still maintaining a strong half-precision type).
///
/// According to the C++98/03 definition, the half type is not a POD type. But according to C++11's less strict and
/// extended definitions it is both a standard layout type and a trivially copyable type (even if not a POD type), which
/// means it can be standard-conformantly copied using raw binary copies. But in this context some more words about the
/// actual size of the type. Although the half is representing an IEEE 16-bit type, it does not neccessarily have to be of
/// exactly 16-bits size. But on any reasonable implementation the actual binary representation of this type will most
/// probably not ivolve any additional "magic" or padding beyond the simple binary representation of the underlying 16-bit
/// IEEE number, even if not strictly guaranteed by the standard. But even then it only has an actual size of 16 bits if
/// your C++ implementation supports an unsigned integer type of exactly 16 bits width. But this should be the case on
/// nearly any reasonable platform.
///
/// So if your C++ implementation is not totally exotic or imposes special alignment requirements, it is a reasonable
/// assumption that the data of a half is just comprised of the 2 bytes of the underlying IEEE representation.
class half
{
    friend struct detail::functions;
    friend struct detail::unary_specialized<half>;
    friend struct detail::binary_specialized<half, half>;
    template <typename, typename, std::float_round_style>
    friend struct detail::half_caster;
    friend class std::numeric_limits<half>;
#if HALF_ENABLE_CPP11_HASH
    friend struct std::hash<half>;
#endif
#if HALF_ENABLE_CPP11_USER_LITERALS
    friend half literal::operator""_h(long double);
#endif

public:
    /// Default constructor.
    /// This initializes the half to 0. Although this does not match the builtin types' default-initialization semantics
    /// and may be less efficient than no initialization, it is needed to provide proper value-initialization semantics.
    HALF_CONSTEXPR half() HALF_NOEXCEPT : data_() {}

    /// Copy constructor.
    /// \tparam T type of concrete half expression
    /// \param rhs half expression to copy from
    half(detail::expr rhs) : data_(detail::float2half<round_style>(static_cast<float>(rhs))) {}

    /// Conversion constructor.
    /// \param rhs float to convert
    half(float rhs) : data_(detail::float2half<round_style>(rhs)) {}

    // template<typename T>
    // half(T rhs): data_(detail::float2half<round_style>((float)rhs)) {}

    /// Conversion to single-precision.
    /// \return single precision value representing expression value
    operator float() const { return detail::half2float<float>(data_); }

    // template<typename T>
    // operator T() const { return (T)detail::half2float<float>(data_); }
    /// Assignment operator.
    /// \tparam T type of concrete half expression
    /// \param rhs half expression to copy from
    /// \return reference to this half
    half &operator=(detail::expr rhs) { return *this = static_cast<float>(rhs); }

    /// Arithmetic assignment.
    /// \tparam T type of concrete half expression
    /// \param rhs half expression to add
    /// \return reference to this half
    template <typename T>
    typename detail::enable<half &, T>::type operator+=(T rhs) { return *this += static_cast<float>(rhs); }

    /// Arithmetic assignment.
    /// \tparam T type of concrete half expression
    /// \param rhs half expression to subtract
    /// \return reference to this half
    template <typename T>
    typename detail::enable<half &, T>::type operator-=(T rhs) { return *this -= static_cast<float>(rhs); }

    /// Arithmetic assignment.
    /// \tparam T type of concrete half expression
    /// \param rhs half expression to multiply with
    /// \return reference to this half
    template <typename T>
    typename detail::enable<half &, T>::type operator*=(T rhs) { return *this *= static_cast<float>(rhs); }

    /// Arithmetic assignment.
    /// \tparam T type of concrete half expression
    /// \param rhs half expression to divide by
    /// \return reference to this half
    template <typename T>
    typename detail::enable<half &, T>::type operator/=(T rhs) { return *this /= static_cast<float>(rhs); }

    /// Assignment operator.
    /// \param rhs single-precision value to copy from
    /// \return reference to this half
    half &operator=(float rhs)
    {
        data_ = detail::float2half<round_style>(rhs);
        return *this;
    }

    /// Arithmetic assignment.
    /// \param rhs single-precision value to add
    /// \return reference to this half
    half &operator+=(float rhs)
    {
        data_ = detail::float2half<round_style>(detail::half2float<float>(data_) + rhs);
        return *this;
    }

    /// Arithmetic assignment.
    /// \param rhs single-precision value to subtract
    /// \return reference to this half
    half &operator-=(float rhs)
    {
        data_ = detail::float2half<round_style>(detail::half2float<float>(data_) - rhs);
        return *this;
    }

    /// Arithmetic assignment.
    /// \param rhs single-precision value to multiply with
    /// \return reference to this half
    half &operator*=(float rhs)
    {
        data_ = detail::float2half<round_style>(detail::half2float<float>(data_) * rhs);
        return *this;
    }

    /// Arithmetic assignment.
    /// \param rhs single-precision value to divide by
    /// \return reference to this half
    half &operator/=(float rhs)
    {
        data_ = detail::float2half<round_style>(detail::half2float<float>(data_) / rhs);
        return *this;
    }

    /// Prefix increment.
    /// \return incremented half value
    half &operator++() { return *this += 1.0f; }

    /// Prefix decrement.
    /// \return decremented half value
    half &operator--() { return *this -= 1.0f; }

    /// Postfix increment.
    /// \return non-incremented half value
    half operator++(int)
    {
        half out(*this);
        ++*this;
        return out;
    }

    /// Postfix decrement.
    /// \return non-decremented half value
    half operator--(int)
    {
        half out(*this);
        --*this;
        return out;
    }

private:
    /// Rounding mode to use
    static const std::float_round_style round_style = (std::float_round_style)(HALF_ROUND_STYLE);

    /// Constructor.
    /// \param bits binary representation to set half to
    HALF_CONSTEXPR half(detail::binary_t, detail::uint16 bits) HALF_NOEXCEPT : data_(bits) {}

    /// Internal binary representation
    detail::uint16 data_;
};

#if HALF_ENABLE_CPP11_USER_LITERALS
namespace literal
{
    /// Half literal.
    /// While this returns an actual half-precision value, half literals can unfortunately not be constant expressions due
    /// to rather involved conversions.
    /// \param value literal value
    /// \return half with given value (if representable)
    inline half operator""_h(long double value) { return half(detail::binary, detail::float2half<half::round_style>(value)); }
} // namespace literal
#endif

namespace detail
{
    /// Wrapper implementing unspecialized half-precision functions.
    struct functions
    {
        /// Addition implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return Half-precision sum stored in single-precision
        static expr plus(float x, float y) { return expr(x + y); }

        /// Subtraction implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return Half-precision difference stored in single-precision
        static expr minus(float x, float y) { return expr(x - y); }

        /// Multiplication implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return Half-precision product stored in single-precision
        static expr multiplies(float x, float y) { return expr(x * y); }

        /// Division implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return Half-precision quotient stored in single-precision
        static expr divides(float x, float y) { return expr(x / y); }

        /// Output implementation.
        /// \param out stream to write to
        /// \param arg value to write
        /// \return reference to stream
        template <typename charT, typename traits>
        static std::basic_ostream<charT, traits> &write(std::basic_ostream<charT, traits> &out, float arg) { return out << arg; }

        /// Input implementation.
        /// \param in stream to read from
        /// \param arg half to read into
        /// \return reference to stream
        template <typename charT, typename traits>
        static std::basic_istream<charT, traits> &read(std::basic_istream<charT, traits> &in, half &arg)
        {
            float f;
            if (in >> f)
                arg = f;
            return in;
        }

        /// Modulo implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return Half-precision division remainder stored in single-precision
        static expr fmod(float x, float y) { return expr(std::fmod(x, y)); }

        /// Remainder implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return Half-precision division remainder stored in single-precision
        static expr remainder(float x, float y)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::remainder(x, y));
#else
            if (builtin_isnan(x) || builtin_isnan(y))
                return expr(std::numeric_limits<float>::quiet_NaN());
            float ax = std::fabs(x), ay = std::fabs(y);
            if (ax >= 65536.0f || ay < std::ldexp(1.0f, -24))
                return expr(std::numeric_limits<float>::quiet_NaN());
            if (ay >= 65536.0f)
                return expr(x);
            if (ax == ay)
                return expr(builtin_signbit(x) ? -0.0f : 0.0f);
            ax = std::fmod(ax, ay + ay);
            float y2 = 0.5f * ay;
            if (ax > y2)
            {
                ax -= ay;
                if (ax >= y2)
                    ax -= ay;
            }
            return expr(builtin_signbit(x) ? -ax : ax);
#endif
        }

        /// Remainder implementation.
        /// \param x first operand
        /// \param y second operand
        /// \param quo address to store quotient bits at
        /// \return Half-precision division remainder stored in single-precision
        static expr remquo(float x, float y, int *quo)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::remquo(x, y, quo));
#else
            if (builtin_isnan(x) || builtin_isnan(y))
                return expr(std::numeric_limits<float>::quiet_NaN());
            bool sign = builtin_signbit(x), qsign = static_cast<bool>(sign ^ builtin_signbit(y));
            float ax = std::fabs(x), ay = std::fabs(y);
            if (ax >= 65536.0f || ay < std::ldexp(1.0f, -24))
                return expr(std::numeric_limits<float>::quiet_NaN());
            if (ay >= 65536.0f)
                return expr(x);
            if (ax == ay)
                return *quo = qsign ? -1 : 1, expr(sign ? -0.0f : 0.0f);
            ax = std::fmod(ax, 8.0f * ay);
            int cquo = 0;
            if (ax >= 4.0f * ay)
            {
                ax -= 4.0f * ay;
                cquo += 4;
            }
            if (ax >= 2.0f * ay)
            {
                ax -= 2.0f * ay;
                cquo += 2;
            }
            float y2 = 0.5f * ay;
            if (ax > y2)
            {
                ax -= ay;
                ++cquo;
                if (ax >= y2)
                {
                    ax -= ay;
                    ++cquo;
                }
            }
            return *quo = qsign ? -cquo : cquo, expr(sign ? -ax : ax);
#endif
        }

        /// Positive difference implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return Positive difference stored in single-precision
        static expr fdim(float x, float y)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::fdim(x, y));
#else
            return expr((x <= y) ? 0.0f : (x - y));
#endif
        }

        /// Fused multiply-add implementation.
        /// \param x first operand
        /// \param y second operand
        /// \param z third operand
        /// \return \a x * \a y + \a z stored in single-precision
        static expr fma(float x, float y, float z)
        {
#if HALF_ENABLE_CPP11_CMATH && defined(FP_FAST_FMAF)
            return expr(std::fma(x, y, z));
#else
            return expr(x * y + z);
#endif
        }

        /// Get NaN.
        /// \return Half-precision quiet NaN
        static half nanh() { return half(binary, 0x7FFF); }

        /// Exponential implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr exp(float arg) { return expr(std::exp(arg)); }

        /// Exponential implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr expm1(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::expm1(arg));
#else
            return expr(static_cast<float>(std::exp(static_cast<double>(arg)) - 1.0));
#endif
        }

        /// Binary exponential implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr exp2(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::exp2(arg));
#else
            return expr(static_cast<float>(std::exp(arg * 0.69314718055994530941723212145818)));
#endif
        }

        /// Logarithm implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr log(float arg) { return expr(std::log(arg)); }

        /// Common logarithm implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr log10(float arg) { return expr(std::log10(arg)); }

        /// Logarithm implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr log1p(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::log1p(arg));
#else
            return expr(static_cast<float>(std::log(1.0 + arg)));
#endif
        }

        /// Binary logarithm implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr log2(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::log2(arg));
#else
            return expr(static_cast<float>(std::log(static_cast<double>(arg)) * 1.4426950408889634073599246810019));
#endif
        }

        /// Square root implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr sqrt(float arg) { return expr(std::sqrt(arg)); }

        /// Cubic root implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr cbrt(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::cbrt(arg));
#else
            if (builtin_isnan(arg) || builtin_isinf(arg))
                return expr(arg);
            return expr(builtin_signbit(arg) ? -static_cast<float>(std::pow(-static_cast<double>(arg), 1.0 / 3.0)) : static_cast<float>(std::pow(static_cast<double>(arg), 1.0 / 3.0)));
#endif
        }

        /// Hypotenuse implementation.
        /// \param x first argument
        /// \param y second argument
        /// \return function value stored in single-preicision
        static expr hypot(float x, float y)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::hypot(x, y));
#else
            return expr((builtin_isinf(x) || builtin_isinf(y)) ? std::numeric_limits<float>::infinity() : static_cast<float>(std::sqrt(static_cast<double>(x) * x + static_cast<double>(y) * y)));
#endif
        }

        /// Power implementation.
        /// \param base value to exponentiate
        /// \param exp power to expontiate to
        /// \return function value stored in single-preicision
        static expr pow(float base, float exp) { return expr(std::pow(base, exp)); }

        /// Sine implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr sin(float arg) { return expr(std::sin(arg)); }

        /// Cosine implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr cos(float arg) { return expr(std::cos(arg)); }

        /// Tan implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr tan(float arg) { return expr(std::tan(arg)); }

        /// Arc sine implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr asin(float arg) { return expr(std::asin(arg)); }

        /// Arc cosine implementation.
        /// \param y second argument
        /// \return function value stored in single-preicision
        static expr hypot(float x, float y)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::hypot(x, y));
#else
            return expr((builtin_isinf(x) || builtin_isinf(y)) ? std::numeric_limits<float>::infinity() : static_cast<float>(std::sqrt(static_cast<double>(x) * x + static_cast<double>(y) * y)));
#endif
        }

        /// Power implementation.
        /// \param base value to exponentiate
        /// \param exp power to expontiate to
        /// \return function value stored in single-preicision
        static expr pow(float base, float exp) { return expr(std::pow(base, exp)); }

        /// Sine implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr sin(float arg) { return expr(std::sin(arg)); }

        /// Cosine implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr cos(float arg) { return expr(std::cos(arg)); }

        /// Tan implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr tan(float arg) { return expr(std::tan(arg)); }

        /// Arc sine implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr asin(float arg) { return expr(std::asin(arg)); }

        /// Arc cosine implementation.
        /// Hyperbolic area tangent implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr atanh(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::atanh(arg));
#else
            return expr(static_cast<float>(0.5 * std::log((1.0 + arg) / (1.0 - arg))));
#endif
        }

        /// Error function implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr erf(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::erf(arg));
#else
            return expr(static_cast<float>(erf(static_cast<double>(arg))));
#endif
        }

        /// Complementary implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr erfc(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::erfc(arg));
#else
            return expr(static_cast<float>(1.0 - erf(static_cast<double>(arg))));
#endif
        }

        /// Gamma logarithm implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr lgamma(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::lgamma(arg));
#else
            if (builtin_isinf(arg))
                return expr(std::numeric_limits<float>::infinity());
            if (arg < 0.0f)
            {
                float i, f = std::modf(-arg, &i);
                if (f == 0.0f)
                    return expr(std::numeric_limits<float>::infinity());
                return expr(static_cast<float>(1.1447298858494001741434273513531 -
                                               std::log(std::abs(std::sin(3.1415926535897932384626433832795 * f))) - lgamma(1.0 - arg)));
            }
            return expr(static_cast<float>(lgamma(static_cast<double>(arg))));
#endif
        }

        /// Gamma implementation.
        /// \param arg function argument
        /// \return function value stored in single-preicision
        static expr tgamma(float arg)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::tgamma(arg));
#else
            if (arg == 0.0f)
                return builtin_signbit(arg) ? expr(-std::numeric_limits<float>::infinity()) : expr(std::numeric_limits<float>::infinity());
            if (arg < 0.0f)
            {
                float i, f = std::modf(-arg, &i);
                if (f == 0.0f)
                    return expr(std::numeric_limits<float>::quiet_NaN());
                double value = 3.1415926535897932384626433832795 / (std::sin(3.1415926535897932384626433832795 * f) * std::exp(lgamma(1.0 - arg)));
                return expr(static_cast<float>((std::fmod(i, 2.0f) == 0.0f) ? -value : value));
            }
            if (builtin_isinf(arg))
                return expr(arg);
            return expr(static_cast<float>(std::exp(lgamma(static_cast<double>(arg)))));
#endif
        }

        /// Floor implementation.
        /// \param arg value to round
        /// \return rounded value
        static half floor(half arg) { return half(binary, round_half<std::round_toward_neg_infinity>(arg.data_)); }

        /// Ceiling implementation.
        /// \param arg value to round
        /// \return rounded value
        static half ceil(half arg) { return half(binary, round_half<std::round_toward_infinity>(arg.data_)); }

        /// Truncation implementation.
        /// \param arg value to round
        /// \return rounded value
        static half trunc(half arg) { return half(binary, round_half<std::round_toward_zero>(arg.data_)); }

        /// Nearest integer implementation.
        /// \param arg value to round
        /// \return rounded value
        static half round(half arg) { return half(binary, round_half_up(arg.data_)); }

        /// Nearest integer implementation.
        /// \param arg value to round
        /// \return rounded value
        static long lround(half arg) { return detail::half2int_up<long>(arg.data_); }

        /// Nearest integer implementation.
        /// \param arg value to round
        /// \return rounded value
        static half rint(half arg) { return half(binary, round_half<half::round_style>(arg.data_)); }

        /// Nearest integer implementation.
        /// \param arg value to round
        /// \return rounded value
        static long lrint(half arg) { return detail::half2int<half::round_style, long>(arg.data_); }

#if HALF_ENABLE_CPP11_LONG_LONG
        /// Nearest integer implementation.
        /// \param arg value to round
        /// \return rounded value
        static long long llround(half arg) { return detail::half2int_up<long long>(arg.data_); }

        /// Nearest integer implementation.
        /// \param arg value to round
        /// \return rounded value
        static long long llrint(half arg) { return detail::half2int<half::round_style, long long>(arg.data_); }
#endif

        /// Decompression implementation.
        /// \param arg number to decompress
        /// \param exp address to store exponent at
        /// \return normalized significant
        static half frexp(half arg, int *exp)
        {
            int m = arg.data_ & 0x7FFF, e = -14;
            if (m >= 0x7C00 || !m)
                return *exp = 0, arg;
            for (; m < 0x400; m <<= 1, --e)
                ;
            return *exp = e + (m >> 10), half(binary, (arg.data_ & 0x8000) | 0x3800 | (m & 0x3FF));
        }

        /// Decompression implementation.
        /// \param arg number to decompress
        /// \param iptr address to store integer part at
        /// \return fractional part
        static half modf(half arg, half *iptr)
        {
            unsigned int e = arg.data_ & 0x7FFF;
            if (e >= 0x6400)
                return *iptr = arg, half(binary, arg.data_ & (0x8000U | -(e > 0x7C00)));
            if (e < 0x3C00)
                return iptr->data_ = arg.data_ & 0x8000, arg;
            e >>= 10;
            unsigned int mask = (1 << (25 - e)) - 1, m = arg.data_ & mask;
            iptr->data_ = arg.data_ & ~mask;
            if (!m)
                return half(binary, arg.data_ & 0x8000);
            for (; m < 0x400; m <<= 1, --e)
                ;
            return half(binary, static_cast<uint16>((arg.data_ & 0x8000) | (e << 10) | (m & 0x3FF)));
        }

        /// Scaling implementation.
        /// \param arg number to scale
        /// \param exp power of two to scale by
        /// \return scaled number
        static half scalbln(half arg, long exp)
        {
            unsigned int m = arg.data_ & 0x7FFF;
            if (m >= 0x7C00 || !m)
                return arg;
            for (; m < 0x400; m <<= 1, --exp)
                ;
            exp += m >> 10;
            uint16 value = arg.data_ & 0x8000;
            if (exp > 30)
            {
                if (half::round_style == std::round_toward_zero)
                    value |= 0x7BFF;
                else if (half::round_style == std::round_toward_infinity)
                    value |= 0x7C00 - (value >> 15);
                else if (half::round_style == std::round_toward_neg_infinity)
                    value |= 0x7BFF + (value >> 15);
                else
                    value |= 0x7C00;
            }
            else if (exp > 0)
                value |= (exp << 10) | (m & 0x3FF);
            else if (exp > -11)
            {
                m = (m & 0x3FF) | 0x400;
                if (half::round_style == std::round_to_nearest)
                {
                    m += 1 << -exp;
#if HALF_ROUND_TIES_TO_EVEN
                    m -= (m >> (1 - exp)) & 1;
#endif
                }
                else if (half::round_style == std::round_toward_infinity)
                    m += ((value >> 15) - 1) & ((1 << (1 - exp)) - 1U);
                else if (half::round_style == std::round_toward_neg_infinity)
                    m += -(value >> 15) & ((1 << (1 - exp)) - 1U);
                value |= m >> (1 - exp);
            }
            else if (half::round_style == std::round_toward_infinity)
                value -= (value >> 15) - 1;
            else if (half::round_style == std::round_toward_neg_infinity)
                value += value >> 15;
            return half(binary, value);
        }

        /// Exponent implementation.
        /// \param arg number to query
        /// \return floating point exponent
        static int ilogb(half arg)
        {
            int abs = arg.data_ & 0x7FFF;
            if (!abs)
                return FP_ILOGB0;
            if (abs < 0x7C00)
            {
                int exp = (abs >> 10) - 15;
                if (abs < 0x400)
                    for (; abs < 0x200; abs <<= 1, --exp)
                        ;
                return exp;
            }
            if (abs > 0x7C00)
                return FP_ILOGBNAN;
            return INT_MAX;
        }

        /// Exponent implementation.
        /// \param arg number to query
        /// \return floating point exponent
        static half logb(half arg)
        {
            int abs = arg.data_ & 0x7FFF;
            if (!abs)
                return half(binary, 0xFC00);
            if (abs < 0x7C00)
            {
                int exp = (abs >> 10) - 15;
                if (abs < 0x400)
                    for (; abs < 0x200; abs <<= 1, --exp)
                        ;
                uint16 bits = (exp < 0) << 15;
                if (exp)
                {
                    unsigned int m = std::abs(exp) << 6, e = 18;
                    for (; m < 0x400; m <<= 1, --e)
                        ;
                    bits |= (e << 10) + m;
                }
                return half(binary, bits);
            }
            if (abs > 0x7C00)
                return arg;
            return half(binary, 0x7C00);
        }

        /// Enumeration implementation.
        /// \param from number to increase/decrease
        /// \param to direction to enumerate into
        /// \return next representable number
        static half nextafter(half from, half to)
        {
            uint16 fabs = from.data_ & 0x7FFF, tabs = to.data_ & 0x7FFF;
            if (fabs > 0x7C00)
                return from;
            if (tabs > 0x7C00 || from.data_ == to.data_ || !(fabs | tabs))
                return to;
            if (!fabs)
                return half(binary, (to.data_ & 0x8000) + 1);
            bool lt = ((fabs == from.data_) ? static_cast<int>(fabs) : -static_cast<int>(fabs)) <
                      ((tabs == to.data_) ? static_cast<int>(tabs) : -static_cast<int>(tabs));
            return half(binary, from.data_ + (((from.data_ >> 15) ^ static_cast<unsigned>(lt)) << 1) - 1);
        }

        /// Enumeration implementation.
        /// \param from number to increase/decrease
        /// \param to direction to enumerate into
        /// \return next representable number
        static half nexttoward(half from, long double to)
        {
            if (isnan(from))
                return from;
            long double lfrom = static_cast<long double>(from);
            if (builtin_isnan(to) || lfrom == to)
                return half(static_cast<float>(to));
            if (!(from.data_ & 0x7FFF))
                return half(binary, (static_cast<detail::uint16>(builtin_signbit(to)) << 15) + 1);
            return half(binary, from.data_ + (((from.data_ >> 15) ^ static_cast<unsigned>(lfrom < to)) << 1) - 1);
        }

        /// Sign implementation
        /// \param x first operand
        /// \param y second operand
        /// \return composed value
        static half copysign(half x, half y) { return half(binary, x.data_ ^ ((x.data_ ^ y.data_) & 0x8000)); }

        /// Classification implementation.
        /// \param arg value to classify
        /// \retval true if infinite number
        /// \retval false else
        static int fpclassify(half arg)
        {
            unsigned int abs = arg.data_ & 0x7FFF;
            return abs ? ((abs > 0x3FF) ? ((abs >= 0x7C00) ? ((abs > 0x7C00) ? FP_NAN : FP_INFINITE) : FP_NORMAL) : FP_SUBNORMAL) : FP_ZERO;
        }

        /// Classification implementation.
        /// \param arg value to classify
        /// \retval true if finite number
        /// \retval false else
        static bool isfinite(half arg) { return (arg.data_ & 0x7C00) != 0x7C00; }

            if (isnan(from))
                return from;
            long double lfrom = static_cast<long double>(from);
            if (builtin_isnan(to) || lfrom == to)
                return half(static_cast<float>(to));
            if (!(from.data_ & 0x7FFF))
                return half(binary, (static_cast<detail::uint16>(builtin_signbit(to)) << 15) + 1);
            return half(binary, from.data_ + (((from.data_ >> 15) ^ static_cast<unsigned>(lfrom < to)) << 1) - 1);
        }

        /// Sign implementation
        /// \param x first operand
        /// \param y second operand
        /// \return composed value
        static half copysign(half x, half y) { return half(binary, x.data_ ^ ((x.data_ ^ y.data_) & 0x8000)); }

        /// Classification implementation.
        /// \param arg value to classify
        /// \retval true if infinite number
        /// \retval false else
        static int fpclassify(half arg)
        {
            unsigned int abs = arg.data_ & 0x7FFF;
            return abs ? ((abs > 0x3FF) ? ((abs >= 0x7C00) ? ((abs > 0x7C00) ? FP_NAN : FP_INFINITE) : FP_NORMAL) : FP_SUBNORMAL) : FP_ZERO;
        }

        /// Classification implementation.
        /// \param arg value to classify
        /// \retval true if finite number
        /// \retval false else
        static bool isfinite(half arg) { return (arg.data_ & 0x7C00) != 0x7C00; }


        /// Comparison implementation.
        /// \param x first operand
        /// \param y second operand
        /// \retval true if \a x >= \a y
        /// \retval false else
        static bool isgreaterequal(half x, half y)
        {
            int xabs = x.data_ & 0x7FFF, yabs = y.data_ & 0x7FFF;
            return xabs <= 0x7C00 && yabs <= 0x7C00 && (((xabs == x.data_) ? xabs : -xabs) >= ((yabs == y.data_) ? yabs : -yabs));
        }

        /// Comparison implementation.
        /// \param x first operand
        /// \param y second operand
        /// \retval true if \a x < \a y
        /// \retval false else
        static bool isless(half x, half y)
        {
            int xabs = x.data_ & 0x7FFF, yabs = y.data_ & 0x7FFF;
            return xabs <= 0x7C00 && yabs <= 0x7C00 && (((xabs == x.data_) ? xabs : -xabs) < ((yabs == y.data_) ? yabs : -yabs));
        }

        /// Comparison implementation.
        /// \param x first operand
        /// \param y second operand
        /// \retval true if \a x <= \a y
        /// \retval false else
        static bool islessequal(half x, half y)
        {
            int xabs = x.data_ & 0x7FFF, yabs = y.data_ & 0x7FFF;
            return xabs <= 0x7C00 && yabs <= 0x7C00 && (((xabs == x.data_) ? xabs : -xabs) <= ((yabs == y.data_) ? yabs : -yabs));
        }

        /// Comparison implementation.
        /// \param x first operand
        /// \param y second operand
        /// \retval true if either \a x > \a y nor \a x < \a y
        /// \retval false else
        static bool islessgreater(half x, half y)
        {
            int xabs = x.data_ & 0x7FFF, yabs = y.data_ & 0x7FFF;
            if (xabs > 0x7C00 || yabs > 0x7C00)
                return false;
            int a = (xabs == x.data_) ? xabs : -xabs, b = (yabs == y.data_) ? yabs : -yabs;
            return a < b || a > b;
        }

        /// Comparison implementation.
        /// \param x first operand
        /// \param y second operand
        /// \retval true if operand unordered
        /// \retval false else
        static bool isunordered(half x, half y) { return isnan(x) || isnan(y); }

    private:
        static double erf(double arg)
        {
            if (builtin_isinf(arg))
                return (arg < 0.0) ? -1.0 : 1.0;
            double x2 = arg * arg, ax2 = 0.147 * x2, value = std::sqrt(1.0 - std::exp(-x2 * (1.2732395447351626861510701069801 + ax2) / (1.0 + ax2)));
            return builtin_signbit(arg) ? -value : value;
        }

        static double lgamma(double arg)
        {
            double v = 1.0;
            for (; arg < 8.0; ++arg)
                v *= arg;
            double w = 1.0 / (arg * arg);
            return (((((((-0.02955065359477124183006535947712 * w + 0.00641025641025641025641025641026) * w +
                         -0.00191752691752691752691752691753) *
                            w +
                        8.4175084175084175084175084175084e-4) *
                           w +
                       -5.952380952380952380952380952381e-4) *
                          w +
                      7.9365079365079365079365079365079e-4) *
                         w +
                     -0.00277777777777777777777777777778) *
                        w +
                    0.08333333333333333333333333333333) /
                       arg +
                   0.91893853320467274178032973640562 - std::log(v) - arg + (arg - 0.5) * std::log(arg);
        }
    };

    /// Wrapper for unary half-precision functions needing specialization for individual argument types.
    /// \tparam T argument type
    template <typename T>
    struct unary_specialized
    {
        /// Negation implementation.
        /// \param arg value to negate
        /// \return negated value
        static HALF_CONSTEXPR half negate(half arg) { return half(binary, arg.data_ ^ 0x8000); }

        /// Absolute value implementation.
        /// \param arg function argument
        /// \return absolute value
        static half fabs(half arg) { return half(binary, arg.data_ & 0x7FFF); }
    };
    template <>
    struct unary_specialized<expr>
    {
        static HALF_CONSTEXPR expr negate(float arg) { return expr(-arg); }
        static expr fabs(float arg) { return expr(std::fabs(arg)); }
    };

    /// Wrapper for binary half-precision functions needing specialization for individual argument types.
    /// \tparam T first argument type
    /// \tparam U first argument type
    template <typename T, typename U>
    struct binary_specialized
    {
        /// Minimum implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return minimum value
        static expr fmin(float x, float y)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::fmin(x, y));
#else
            if (builtin_isnan(x))
                return expr(y);
            if (builtin_isnan(y))
                return expr(x);
            return expr(std::min(x, y));
#endif
        }

        /// Maximum implementation.
        /// \param x first operand
        /// \param y second operand
        /// \return maximum value
        static expr fmax(float x, float y)
        {
#if HALF_ENABLE_CPP11_CMATH
            return expr(std::fmax(x, y));
#else
            if (builtin_isnan(x))
                return expr(y);
            if (builtin_isnan(y))
                return expr(x);
            return expr(std::max(x, y));
#endif
        }
    };
    template <>
    struct binary_specialized<half, half>
    {
        static half fmin(half x, half y)
        {
            int xabs = x.data_ & 0x7FFF, yabs = y.data_ & 0x7FFF;
            if (xabs > 0x7C00)
                return y;
            if (yabs > 0x7C00)
                return x;
            return (((xabs == x.data_) ? xabs : -xabs) > ((yabs == y.data_) ? yabs : -yabs)) ? y : x;
        }
        static half fmax(half x, half y)
        {
            int xabs = x.data_ & 0x7FFF, yabs = y.data_ & 0x7FFF;
            if (xabs > 0x7C00)
                return y;
            if (yabs > 0x7C00)
                return x;
            return (((xabs == x.data_) ? xabs : -xabs) < ((yabs == y.data_) ? yabs : -yabs)) ? y : x;
        }
    };

    /// Helper class for half casts.
    /// This class template has to be specialized for all valid cast argument to define an appropriate static `cast` member
    /// function and a corresponding `type` member denoting its return type.
    /// \tparam T destination type
    /// \tparam U source type
    /// \tparam R rounding mode to use
    template <typename T, typename U, std::float_round_style R = (std::float_round_style)(HALF_ROUND_STYLE)>
    struct half_caster
    {
    };
    template <typename U, std::float_round_style R>
    struct half_caster<half, U, R>
    {
#if HALF_ENABLE_CPP11_STATIC_ASSERT && HALF_ENABLE_CPP11_TYPE_TRAITS
        static_assert(std::is_arithmetic<U>::value, "half_cast from non-arithmetic type unsupported");
#endif

        static half cast(U arg)
        {
            return cast_impl(arg, is_float<U>());
        }

    private:
        static half cast_impl(U arg, true_type) { return half(binary, float2half<R>(arg)); }
        static half cast_impl(U arg, false_type) { return half(binary, int2half<R>(arg)); }
    };
    template <typename T, std::float_round_style R>
    struct half_caster<T, half, R>
    {
#if HALF_ENABLE_CPP11_STATIC_ASSERT && HALF_ENABLE_CPP11_TYPE_TRAITS
        static_assert(std::is_arithmetic<T>::value, "half_cast to non-arithmetic type unsupported");
#endif

        static T cast(half arg)
        {
            return cast_impl(arg, is_float<T>());
        }

    private:
        static T cast_impl(half arg, true_type) { return half2float<T>(arg.data_); }
        static T cast_impl(half arg, false_type) { return half2int<R, T>(arg.data_); }
    };
    template <typename T, std::float_round_style R>
    struct half_caster<T, expr, R>
    {
#if HALF_ENABLE_CPP11_STATIC_ASSERT && HALF_ENABLE_CPP11_TYPE_TRAITS
        static_assert(std::is_arithmetic<T>::value, "half_cast to non-arithmetic type unsupported");
#endif

        static T cast(expr arg)
        {
            return cast_impl(arg, is_float<T>());
        }

    private:
        static T cast_impl(float arg, true_type) { return static_cast<T>(arg); }
        static T cast_impl(half arg, false_type) { return half2int<R, T>(arg.data_); }
    };
    template <std::float_round_style R>
    struct half_caster<half, half, R>
    {
        static half cast(half arg) { return arg; }
    };

    template <std::float_round_style R>
    struct half_caster<half, expr, R> : half_caster<half, half, R>
    {
    };

    /// \name Comparison operators
    /// \{

    /// Comparison for equality.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if operands equal
    /// \retval false else
    template <typename T, typename U>
    typename enable<bool, T, U>::type operator==(T x, U y) { return functions::isequal(x, y); }

    /// Comparison for inequality.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if operands not equal
    /// \retval false else
    template <typename T, typename U>
    typename enable<bool, T, U>::type operator!=(T x, U y) { return functions::isnotequal(x, y); }

    /// Comparison for less than.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x less than \a y
    /// \retval false else
    template <typename T, typename U>
    typename enable<bool, T, U>::type operator<(T x, U y) { return functions::isless(x, y); }

    /// Comparison for greater than.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x greater than \a y
    /// \retval false else
    template <typename T, typename U>
    typename enable<bool, T, U>::type operator>(T x, U y) { return functions::isgreater(x, y); }

    /// Comparison for less equal.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x less equal \a y
    /// \retval false else
    template <typename T, typename U>
    typename enable<bool, T, U>::type operator<=(T x, U y) { return functions::islessequal(x, y); }

    /// Comparison for greater equal.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x greater equal \a y
    /// \retval false else
    template <typename T, typename U>
    typename enable<bool, T, U>::type operator>=(T x, U y) { return functions::isgreaterequal(x, y); }

    /// \}
    /// \name Arithmetic operators
    /// \{

    /// Add halfs.
    /// \param x left operand
    /// \param y right operand
    /// \return sum of half expressions
    template <typename T, typename U>
    typename enable<expr, T, U>::type operator+(T x, U y) { return functions::plus(x, y); }

    /// Subtract halfs.
    /// \param x left operand
    /// \param y right operand
    /// \return difference of half expressions
    template <typename T, typename U>
    typename enable<expr, T, U>::type operator-(T x, U y) { return functions::minus(x, y); }

    /// Multiply halfs.
    /// \param x left operand
    /// \param y right operand
    /// \return product of half expressions
    template <typename T, typename U>
    typename enable<expr, T, U>::type operator*(T x, U y) { return functions::multiplies(x, y); }

    /// Divide halfs.
    /// \param x left operand
    /// \param y right operand
    /// \return quotient of half expressions
    template <typename T, typename U>
    typename enable<expr, T, U>::type operator/(T x, U y) { return functions::divides(x, y); }

    /// Identity.
    /// \param arg operand
    /// \return uncahnged operand
    template <typename T>
    HALF_CONSTEXPR typename enable<T, T>::type operator+(T arg) { return arg; }

    /// Negation.
    /// \param arg operand
    /// \return negated operand
    template <typename T>
    HALF_CONSTEXPR typename enable<T, T>::type operator-(T arg) { return unary_specialized<T>::negate(arg); }

    /// \}
    /// \name Input and output
    /// \{

    /// Output operator.
    /// \param out output stream to write into
    /// \param arg half expression to write
    /// \return reference to output stream
    template <typename T, typename charT, typename traits>
    typename enable<std::basic_ostream<charT, traits> &, T>::type
    operator<<(std::basic_ostream<charT, traits> &out, T arg) { return functions::write(out, arg); }

    /// Input operator.
    /// \param in input stream to read from
    /// \param arg half to read into
    /// \return reference to input stream
    template <typename charT, typename traits>
    std::basic_istream<charT, traits> &
    operator>>(std::basic_istream<charT, traits> &in, half &arg) { return functions::read(in, arg); }

    /// \}
    /// \name Basic mathematical operations
    /// \{

    /// Absolute value.
    /// \param arg operand
    /// \return absolute value of \a arg
    //		template<typename T> typename enable<T,T>::type abs(T arg) { return unary_specialized<T>::fabs(arg); }
    inline half abs(half arg) { return unary_specialized<half>::fabs(arg); }
    inline expr abs(expr arg) { return unary_specialized<expr>::fabs(arg); }

    /// Absolute value.
    /// \param arg operand
    /// \return absolute value of \a arg
    //		template<typename T> typename enable<T,T>::type fabs(T arg) { return unary_specialized<T>::fabs(arg); }
    inline half fabs(half arg) { return unary_specialized<half>::fabs(arg); }
    inline expr fabs(expr arg) { return unary_specialized<expr>::fabs(arg); }

    /// Remainder of division.
    /// \param x first operand
    /// \param y second operand
    /// \return remainder of floating point division.
    //		template<typename T,typename U> typename enable<expr,T,U>::type fmod(T x, U y) { return functions::fmod(x, y); }
    inline expr fmod(half x, half y) { return functions::fmod(x, y); }
    inline expr fmod(half x, expr y) { return functions::fmod(x, y); }
    inline expr fmod(expr x, half y) { return functions::fmod(x, y); }
    inline expr fmod(expr x, expr y) { return functions::fmod(x, y); }

    /// Remainder of division.
    /// \param x first operand
    /// \param y second operand
    /// \return remainder of floating point division.
    //		template<typename T,typename U> typename enable<expr,T,U>::type remainder(T x, U y) { return functions::remainder(x, y); }
    inline expr remainder(half x, half y) { return functions::remainder(x, y); }
    inline expr remainder(half x, expr y) { return functions::remainder(x, y); }
    inline expr remainder(expr x, half y) { return functions::remainder(x, y); }
    inline expr remainder(expr x, expr y) { return functions::remainder(x, y); }

    /// Remainder of division.
    /// \param x first operand
    /// \param y second operand
    /// \param quo address to store some bits of quotient at
    /// \return remainder of floating point division.
    //		template<typename T,typename U> typename enable<expr,T,U>::type remquo(T x, U y, int *quo) { return functions::remquo(x, y, quo); }
    inline expr remquo(half x, half y, int *quo) { return functions::remquo(x, y, quo); }
    inline expr remquo(half x, expr y, int *quo) { return functions::remquo(x, y, quo); }
    inline expr remquo(expr x, half y, int *quo) { return functions::remquo(x, y, quo); }
    inline expr remquo(expr x, expr y, int *quo) { return functions::remquo(x, y, quo); }

    /// Fused multiply add.
    /// \param x first operand
    /// \param y second operand
    /// \param z third operand
    /// \return ( \a x * \a y ) + \a z rounded as one operation.
    //		template<typename T,typename U,typename V> typename enable<expr,T,U,V>::type fma(T x, U y, V z) { return functions::fma(x, y, z); }
    inline expr fma(half x, half y, half z) { return functions::fma(x, y, z); }
    inline expr fma(half x, half y, expr z) { return functions::fma(x, y, z); }
    inline expr fma(half x, expr y, half z) { return functions::fma(x, y, z); }
    inline expr fma(half x, expr y, expr z) { return functions::fma(x, y, z); }
    inline expr fma(expr x, half y, half z) { return functions::fma(x, y, z); }
    inline expr fma(expr x, half y, expr z) { return functions::fma(x, y, z); }
    inline expr fma(expr x, expr y, half z) { return functions::fma(x, y, z); }
    inline expr fma(expr x, expr y, expr z) { return functions::fma(x, y, z); }

    /// Maximum of half expressions.
    /// \param x first operand
    /// \param y second operand
    /// \return maximum of operands
    //		template<typename T,typename U> typename result<T,U>::type fmax(T x, U y) { return binary_specialized<T,U>::fmax(x, y); }
    inline half fmax(half x, half y) { return binary_specialized<half, half>::fmax(x, y); }
    inline expr fmax(half x, expr y) { return binary_specialized<half, expr>::fmax(x, y); }
    inline expr fmax(expr x, half y) { return binary_specialized<expr, half>::fmax(x, y); }
    inline expr fmax(expr x, expr y) { return binary_specialized<expr, expr>::fmax(x, y); }

    /// Minimum of half expressions.
    /// \param x first operand
    /// \param y second operand
    /// \return minimum of operands
    //		template<typename T,typename U> typename result<T,U>::type fmin(T x, U y) { return binary_specialized<T,U>::fmin(x, y); }
    inline half fmin(half x, half y) { return binary_specialized<half, half>::fmin(x, y); }
    inline expr fmin(half x, expr y) { return binary_specialized<half, expr>::fmin(x, y); }
    inline expr fmin(expr x, half y) { return binary_specialized<expr, half>::fmin(x, y); }
    inline expr fmin(expr x, expr y) { return binary_specialized<expr, expr>::fmin(x, y); }

    /// Positive difference.
    /// \param x first operand
    /// \param y second operand
    /// \return \a x - \a y or 0 if difference negative
    //		template<typename T,typename U> typename enable<expr,T,U>::type fdim(T x, U y) { return functions::fdim(x, y); }
    inline expr fdim(half x, half y) { return functions::fdim(x, y); }
    inline expr fdim(half x, expr y) { return functions::fdim(x, y); }
    inline expr fdim(expr x, half y) { return functions::fdim(x, y); }
    inline expr fdim(expr x, expr y) { return functions::fdim(x, y); }

    /// Get NaN value.
    /// \return quiet NaN
    inline half nanh(const char *) { return functions::nanh(); }

    /// \}
    /// \name Exponential functions
    /// \{

    /// Exponential function.
    /// \param arg function argument
    /// \return e raised to \a arg
    //		template<typename T> typename enable<expr,T>::type exp(T arg) { return functions::exp(arg); }
    inline expr exp(half arg) { return functions::exp(arg); }
    inline expr exp(expr arg) { return functions::exp(arg); }

    /// Exponential minus one.
    /// \param arg function argument
    /// \return e raised to \a arg subtracted by 1
    //		template<typename T> typename enable<expr,T>::type expm1(T arg) { return functions::expm1(arg); }
    inline expr expm1(half arg) { return functions::expm1(arg); }
    inline expr expm1(expr arg) { return functions::expm1(arg); }

    /// Binary exponential.
    /// \param arg function argument
    /// \return 2 raised to \a arg
    //		template<typename T> typename enable<expr,T>::type exp2(T arg) { return functions::exp2(arg); }
    inline expr exp2(half arg) { return functions::exp2(arg); }
    inline expr exp2(expr arg) { return functions::exp2(arg); }

    /// Natural logorithm.
    /// \param arg function argument
    /// \return logarithm of \a arg to base e
    //		template<typename T> typename enable<expr,T>::type log(T arg) { return functions::log(arg); }
    inline expr log(half arg) { return functions::log(arg); }
    inline expr log(expr arg) { return functions::log(arg); }

    /// Common logorithm.
    /// \param arg function argument
    /// \return logarithm of \a arg to base 10
    //		template<typename T> typename enable<expr,T>::type log10(T arg) { return functions::log10(arg); }
    inline expr log10(half arg) { return functions::log10(arg); }
    inline expr log10(expr arg) { return functions::log10(arg); }

    /// Natural logorithm.
    /// \param arg function argument
    /// \return logarithm of \a arg plus 1 to base e
    //		template<typename T> typename enable<expr,T>::type log1p(T arg) { return functions::log1p(arg); }
    inline expr log1p(half arg) { return functions::log1p(arg); }
    inline expr log1p(expr arg) { return functions::log1p(arg); }


    /// Binary logorithm.
    /// \param arg function argument
    /// \return logarithm of \a arg to base 2
    //		template<typename T> typename enable<expr,T>::type log2(T arg) { return functions::log2(arg); }
    inline expr log2(half arg) { return functions::log2(arg); }
    inline expr log2(expr arg) { return functions::log2(arg); }

    /// \}
    /// \name Power functions
    /// \{

    /// Square root.
    /// \param arg function argument
    /// \return square root of \a arg
    //		template<typename T> typename enable<expr,T>::type sqrt(T arg) { return functions::sqrt(arg); }
    inline expr sqrt(half arg) { return functions::sqrt(arg); }
    inline expr sqrt(expr arg) { return functions::sqrt(arg); }

    /// Cubic root.
    /// \param arg function argument
    /// \return cubic root of \a arg
    //		template<typename T> typename enable<expr,T>::type cbrt(T arg) { return functions::cbrt(arg); }
    inline expr cbrt(half arg) { return functions::cbrt(arg); }
    inline expr cbrt(expr arg) { return functions::cbrt(arg); }

    /// Hypotenuse function.
    /// \param x first argument
    /// \param y second argument
    /// \return square root of sum of squares without internal over- or underflows
    //		template<typename T,typename U> typename enable<expr,T,U>::type hypot(T x, U y) { return functions::hypot(x, y); }
    inline expr hypot(half x, half y) { return functions::hypot(x, y); }
    inline expr hypot(half x, expr y) { return functions::hypot(x, y); }
    inline expr hypot(expr x, half y) { return functions::hypot(x, y); }
    inline expr hypot(expr x, expr y) { return functions::hypot(x, y); }

    /// Power function.
    /// \param base first argument
    /// \param exp second argument
    /// \return \a base raised to \a exp
    //		template<typename T,typename U> typename enable<expr,T,U>::type pow(T base, U exp) { return functions::pow(base, exp); }
    inline expr pow(half base, half exp) { return functions::pow(base, exp); }
    inline expr pow(half base, expr exp) { return functions::pow(base, exp); }
    inline expr pow(expr base, half exp) { return functions::pow(base, exp); }
    inline expr pow(expr base, expr exp) { return functions::pow(base, exp); }

    /// \}
    /// \name Trigonometric functions
    /// \{

    /// Sine function.
    /// \param arg function argument
    /// \return sine value of \a arg
    //		template<typename T> typename enable<expr,T>::type sin(T arg) { return functions::sin(arg); }
    inline expr sin(half arg) { return functions::sin(arg); }
    inline expr sin(expr arg) { return functions::sin(arg); }

    /// Cosine function.
    /// \param arg function argument
    /// \return cosine value of \a arg
    //		template<typename T> typename enable<expr,T>::type cos(T arg) { return functions::cos(arg); }
    inline expr cos(half arg) { return functions::cos(arg); }
    inline expr cos(expr arg) { return functions::cos(arg); }

    /// Tangent function.
    /// \param arg function argument
    /// \return tangent value of \a arg
    //		template<typename T> typename enable<expr,T>::type tan(T arg) { return functions::tan(arg); }
    inline expr tan(half arg) { return functions::tan(arg); }
    inline expr tan(expr arg) { return functions::tan(arg); }

    /// Arc sine.
    /// \param arg function argument
    /// \return arc sine value of \a arg
    //		template<typename T> typename enable<expr,T>::type asin(T arg) { return functions::asin(arg); }
    inline expr asin(half arg) { return functions::asin(arg); }
    inline expr asin(expr arg) { return functions::asin(arg); }

    /// Arc cosine function.
    /// \param arg function argument
    /// \return arc cosine value of \a arg
    //		template<typename T> typename enable<expr,T>::type acos(T arg) { return functions::acos(arg); }
    inline expr acos(half arg) { return functions::acos(arg); }
    inline expr acos(expr arg) { return functions::acos(arg); }

    /// Arc tangent function.
    /// \param arg function argument
    /// \return arc tangent value of \a arg
    //		template<typename T> typename enable<expr,T>::type atan(T arg) { return functions::atan(arg); }
    inline expr atan(half arg) { return functions::atan(arg); }
    inline expr atan(expr arg) { return functions::atan(arg); }

    /// Arc tangent function.
    /// \param x first argument
    /// \param y second argument
    /// \return arc tangent value
    //		template<typename T,typename U> typename enable<expr,T,U>::type atan2(T x, U y) { return functions::atan2(x, y); }
    inline expr atan2(half x, half y) { return functions::atan2(x, y); }
    inline expr atan2(half x, expr y) { return functions::atan2(x, y); }
    inline expr atan2(expr x, half y) { return functions::atan2(x, y); }
    inline expr atan2(expr x, expr y) { return functions::atan2(x, y); }

    /// \}
    /// \name Hyperbolic functions
    /// \{

    /// Hyperbolic sine.
    /// \param arg function argument
    /// \return hyperbolic sine value of \a arg
    //		template<typename T> typename enable<expr,T>::type sinh(T arg) { return functions::sinh(arg); }
    inline expr sinh(half arg) { return functions::sinh(arg); }
    inline expr sinh(expr arg) { return functions::sinh(arg); }

    /// Hyperbolic cosine.
    /// \param arg function argument
    /// \return hyperbolic cosine value of \a arg
    //		template<typename T> typename enable<expr,T>::type cosh(T arg) { return functions::cosh(arg); }
    inline expr cosh(half arg) { return functions::cosh(arg); }
    inline expr cosh(expr arg) { return functions::cosh(arg); }

    /// Hyperbolic tangent.
    /// \param arg function argument
    /// \return hyperbolic tangent value of \a arg
    //		template<typename T> typename enable<expr,T>::type tanh(T arg) { return functions::tanh(arg); }
    inline expr tanh(half arg) { return functions::tanh(arg); }
    inline expr tanh(expr arg) { return functions::tanh(arg); }

    /// Hyperbolic area sine.
    /// \param arg function argument
    /// \return area sine value of \a arg
    //		template<typename T> typename enable<expr,T>::type asinh(T arg) { return functions::asinh(arg); }
    inline expr asinh(half arg) { return functions::asinh(arg); }
    inline expr asinh(expr arg) { return functions::asinh(arg); }

    /// Hyperbolic area cosine.
    /// \param arg function argument
    /// \return area cosine value of \a arg
    //		template<typename T> typename enable<expr,T>::type acosh(T arg) { return functions::acosh(arg); }
    inline expr acosh(half arg) { return functions::acosh(arg); }
    inline expr acosh(expr arg) { return functions::acosh(arg); }

    /// Hyperbolic area tangent.
    /// \param arg function argument
    /// \return area tangent value of \a arg
    //		template<typename T> typename enable<expr,T>::type atanh(T arg) { return functions::atanh(arg); }
    inline expr atanh(half arg) { return functions::atanh(arg); }
    inline expr atanh(expr arg) { return functions::atanh(arg); }

    /// \}
    /// \name Error and gamma functions
    /// \{

    /// Error function.
    /// \param arg function argument
    /// \return error function value of \a arg
    //		template<typename T> typename enable<expr,T>::type erf(T arg) { return functions::erf(arg); }
    inline expr erf(half arg) { return functions::erf(arg); }
    inline expr erf(expr arg) { return functions::erf(arg); }

    /// Complementary error function.
    /// \param arg function argument
    /// \return 1 minus error function value of \a arg
    //		template<typename T> typename enable<expr,T>::type erfc(T arg) { return functions::erfc(arg); }
    inline expr erfc(half arg) { return functions::erfc(arg); }
    inline expr erfc(expr arg) { return functions::erfc(arg); }

    /// Natural logarithm of gamma function.
    /// \param arg function argument
    /// \return natural logarith of gamma function for \a arg
    //		template<typename T> typename enable<expr,T>::type lgamma(T arg) { return functions::lgamma(arg); }
    inline expr lgamma(half arg) { return functions::lgamma(arg); }
    inline expr lgamma(expr arg) { return functions::lgamma(arg); }

    /// Gamma function.
    /// \param arg function argument
    /// \return gamma function value of \a arg
    //		template<typename T> typename enable<expr,T>::type tgamma(T arg) { return functions::tgamma(arg); }
    inline expr tgamma(half arg) { return functions::tgamma(arg); }
    inline expr tgamma(expr arg) { return functions::tgamma(arg); }

    /// \}
    /// \name Rounding
    /// \{

    /// Nearest integer not less than half value.
    /// \param arg half to round
    /// \return nearest integer not less than \a arg
    //		template<typename T> typename enable<half,T>::type ceil(T arg) { return functions::ceil(arg); }
    inline half ceil(half arg) { return functions::ceil(arg); }
    inline half ceil(expr arg) { return functions::ceil(arg); }

    /// Nearest integer not greater than half value.
    /// \param arg half to round
    /// \return nearest integer not greater than \a arg
    //		template<typename T> typename enable<half,T>::type floor(T arg) { return functions::floor(arg); }
    inline half floor(half arg) { return functions::floor(arg); }
    inline half floor(expr arg) { return functions::floor(arg); }

    /// Nearest integer not greater in magnitude than half value.
    /// \param arg half to round
    /// \return nearest integer not greater in magnitude than \a arg
    //		template<typename T> typename enable<half,T>::type trunc(T arg) { return functions::trunc(arg); }
    inline half trunc(half arg) { return functions::trunc(arg); }
    inline half trunc(expr arg) { return functions::trunc(arg); }

    /// Nearest integer.
    /// \param arg half to round
    /// \return nearest integer, rounded away from zero in half-way cases
    //		template<typename T> typename enable<half,T>::type round(T arg) { return functions::round(arg); }
    inline half round(half arg) { return functions::round(arg); }
    inline half round(expr arg) { return functions::round(arg); }

    /// Nearest integer.
    /// \param arg half to round
    /// \return nearest integer, rounded away from zero in half-way cases
    //		template<typename T> typename enable<long,T>::type lround(T arg) { return functions::lround(arg); }
    inline long lround(half arg) { return functions::lround(arg); }
    inline long lround(expr arg) { return functions::lround(arg); }

    /// Nearest integer using half's internal rounding mode.
    /// \param arg half expression to round
    /// \return nearest integer using default rounding mode
    //		template<typename T> typename enable<half,T>::type nearbyint(T arg) { return functions::nearbyint(arg); }
    inline half nearbyint(half arg) { return functions::rint(arg); }
    inline half nearbyint(expr arg) { return functions::rint(arg); }

    /// Nearest integer using half's internal rounding mode.
    /// \param arg half expression to round
    /// \return nearest integer using default rounding mode
    //		template<typename T> typename enable<half,T>::type rint(T arg) { return functions::rint(arg); }
    inline half rint(half arg) { return functions::rint(arg); }
    inline half rint(expr arg) { return functions::rint(arg); }

    /// Nearest integer using half's internal rounding mode.
    /// \param arg half expression to round
    /// \return nearest integer using default rounding mode
    //		template<typename T> typename enable<long,T>::type lrint(T arg) { return functions::lrint(arg); }
    inline long lrint(half arg) { return functions::lrint(arg); }
    inline long lrint(expr arg) { return functions::lrint(arg); }
#if HALF_ENABLE_CPP11_LONG_LONG
    /// Nearest integer.
    /// \param arg half to round
    /// \return nearest integer, rounded away from zero in half-way cases
    //		template<typename T> typename enable<long long,T>::type llround(T arg) { return functions::llround(arg); }
    inline long long llround(half arg) { return functions::llround(arg); }
    inline long long llround(expr arg) { return functions::llround(arg); }

    /// Nearest integer using half's internal rounding mode.
    /// \param arg half expression to round
    /// \return nearest integer using default rounding mode
    //		template<typename T> typename enable<long long,T>::type llrint(T arg) { return functions::llrint(arg); }
    inline long long llrint(half arg) { return functions::llrint(arg); }
    inline long long llrint(expr arg) { return functions::llrint(arg); }
#endif

    /// \}
    /// \name Floating point manipulation
    /// \{

    /// Decompress floating point number.
    /// \param arg number to decompress
    /// \param exp address to store exponent at
    /// \return significant in range [0.5, 1)
    //		template<typename T> typename enable<half,T>::type frexp(T arg, int *exp) { return functions::frexp(arg, exp); }
    inline half frexp(half arg, int *exp) { return functions::frexp(arg, exp); }
    inline half frexp(expr arg, int *exp) { return functions::frexp(arg, exp); }

    /// Multiply by power of two.
    /// \param arg number to modify
    /// \param exp power of two to multiply with
    /// \return \a arg multplied by 2 raised to \a exp
    //		template<typename T> typename enable<half,T>::type ldexp(T arg, int exp) { return functions::scalbln(arg, exp); }
    inline half ldexp(half arg, int exp) { return functions::scalbln(arg, exp); }
    inline half ldexp(expr arg, int exp) { return functions::scalbln(arg, exp); }

    /// Extract integer and fractional parts.
    /// \param arg number to decompress
    /// \param iptr address to store integer part at
    /// \return fractional part
    //		template<typename T> typename enable<half,T>::type modf(T arg, half *iptr) { return functions::modf(arg, iptr); }
    inline half modf(half arg, half *iptr) { return functions::modf(arg, iptr); }
    inline half modf(expr arg, half *iptr) { return functions::modf(arg, iptr); }

    /// Multiply by power of two.
    /// \param arg number to modify
    /// \param exp power of two to multiply with
    /// \return \a arg multplied by 2 raised to \a exp
    //		template<typename T> typename enable<half,T>::type scalbn(T arg, int exp) { return functions::scalbln(arg, exp); }
    inline half scalbn(half arg, int exp) { return functions::scalbln(arg, exp); }
    inline half scalbn(expr arg, int exp) { return functions::scalbln(arg, exp); }

    /// Multiply by power of two.
    /// \param arg number to modify
    /// \param exp power of two to multiply with
    /// \return \a arg multplied by 2 raised to \a exp
    //		template<typename T> typename enable<half,T>::type scalbln(T arg, long exp) { return functions::scalbln(arg, exp); }
    inline half scalbln(half arg, long exp) { return functions::scalbln(arg, exp); }
    inline half scalbln(expr arg, long exp) { return functions::scalbln(arg, exp); }

    /// Extract exponent.
    /// \param arg number to query
    /// \return floating point exponent
    /// \retval FP_ILOGB0 for zero
    /// \retval FP_ILOGBNAN for NaN
    /// \retval MAX_INT for infinity
    //		template<typename T> typename enable<int,T>::type ilogb(T arg) { return functions::ilogb(arg); }
    inline int ilogb(half arg) { return functions::ilogb(arg); }
    inline int ilogb(expr arg) { return functions::ilogb(arg); }

    /// Extract exponent.
    /// \param arg number to query
    /// \return floating point exponent
    //		template<typename T> typename enable<half,T>::type logb(T arg) { return functions::logb(arg); }
    inline half logb(half arg) { return functions::logb(arg); }
    inline half logb(expr arg) { return functions::logb(arg); }

    /// Next representable value.
    /// \param from value to compute next representable value for
    /// \param to direction towards which to compute next value
    /// \return next representable value after \a from in direction towards \a to
    //		template<typename T,typename U> typename enable<half,T,U>::type nextafter(T from, U to) { return functions::nextafter(from, to); }
    inline half nextafter(half from, half to) { return functions::nextafter(from, to); }
    inline half nextafter(half from, expr to) { return functions::nextafter(from, to); }
    inline half nextafter(expr from, half to) { return functions::nextafter(from, to); }
    inline half nextafter(expr from, expr to) { return functions::nextafter(from, to); }

    /// Next representable value.
    /// \param from value to compute next representable value for
    /// \param to direction towards which to compute next value
    /// \return next representable value after \a from in direction towards \a to
    //		template<typename T> typename enable<half,T>::type nexttoward(T from, long double to) { return functions::nexttoward(from, to); }
    inline half nexttoward(half from, long double to) { return functions::nexttoward(from, to); }
    inline half nexttoward(expr from, long double to) { return functions::nexttoward(from, to); }

    /// Take sign.
    /// \param x value to change sign for
    /// \param y value to take sign from
    /// \return value equal to \a x in magnitude and to \a y in sign
    //		template<typename T,typename U> typename enable<half,T,U>::type copysign(T x, U y) { return functions::copysign(x, y); }
    inline half copysign(half x, half y) { return functions::copysign(x, y); }
    inline half copysign(half x, expr y) { return functions::copysign(x, y); }
    inline half copysign(expr x, half y) { return functions::copysign(x, y); }
    inline half copysign(expr x, expr y) { return functions::copysign(x, y); }

    /// \}
    /// \name Floating point classification
    /// \{

    /// Classify floating point value.
    /// \param arg number to classify
    /// \retval FP_ZERO for positive and negative zero
    /// \retval FP_SUBNORMAL for subnormal numbers
    /// \retval FP_INFINITY for positive and negative infinity
    /// \retval FP_NAN for NaNs
    /// \retval FP_NORMAL for all other (normal) values
    //		template<typename T> typename enable<int,T>::type fpclassify(T arg) { return functions::fpclassify(arg); }
    inline int fpclassify(half arg) { return functions::fpclassify(arg); }
    inline int fpclassify(expr arg) { return functions::fpclassify(arg); }

    inline half nextafter(expr from, expr to) { return functions::nextafter(from, to); }

    /// Next representable value.
    /// \param from value to compute next representable value for
    /// \param to direction towards which to compute next value
    /// \return next representable value after \a from in direction towards \a to
    //		template<typename T> typename enable<half,T>::type nexttoward(T from, long double to) { return functions::nexttoward(from, to); }
    inline half nexttoward(half from, long double to) { return functions::nexttoward(from, to); }
    inline half nexttoward(expr from, long double to) { return functions::nexttoward(from, to); }

    /// Take sign.
    /// \param x value to change sign for
    /// \param y value to take sign from
    /// \return value equal to \a x in magnitude and to \a y in sign
    //		template<typename T,typename U> typename enable<half,T,U>::type copysign(T x, U y) { return functions::copysign(x, y); }
    inline half copysign(half x, half y) { return functions::copysign(x, y); }
    inline half copysign(half x, expr y) { return functions::copysign(x, y); }
    inline half copysign(expr x, half y) { return functions::copysign(x, y); }
    inline half copysign(expr x, expr y) { return functions::copysign(x, y); }

    /// \}
    /// \name Floating point classification
    /// \{

    /// Classify floating point value.
    /// \param arg number to classify
    /// \retval FP_ZERO for positive and negative zero
    /// \retval FP_SUBNORMAL for subnormal numbers
    /// \retval FP_INFINITY for positive and negative infinity
    /// \retval FP_NAN for NaNs
    /// \retval FP_NORMAL for all other (normal) values
    //		template<typename T> typename enable<int,T>::type fpclassify(T arg) { return functions::fpclassify(arg); }
    inline int fpclassify(half arg) { return functions::fpclassify(arg); }
    inline int fpclassify(expr arg) { return functions::fpclassify(arg); }

    inline bool isgreater(expr x, half y) { return functions::isgreater(x, y); }
    inline bool isgreater(expr x, expr y) { return functions::isgreater(x, y); }

    /// Comparison for greater equal.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x greater equal \a y
    /// \retval false else
    //		template<typename T,typename U> typename enable<bool,T,U>::type isgreaterequal(T x, U y) { return functions::isgreaterequal(x, y); }
    inline bool isgreaterequal(half x, half y) { return functions::isgreaterequal(x, y); }
    inline bool isgreaterequal(half x, expr y) { return functions::isgreaterequal(x, y); }
    inline bool isgreaterequal(expr x, half y) { return functions::isgreaterequal(x, y); }
    inline bool isgreaterequal(expr x, expr y) { return functions::isgreaterequal(x, y); }

    /// Comparison for less than.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x less than \a y
    /// \retval false else
    //		template<typename T,typename U> typename enable<bool,T,U>::type isless(T x, U y) { return functions::isless(x, y); }
    inline bool isless(half x, half y) { return functions::isless(x, y); }
    inline bool isless(half x, expr y) { return functions::isless(x, y); }
    inline bool isless(expr x, half y) { return functions::isless(x, y); }
    inline bool isless(expr x, expr y) { return functions::isless(x, y); }

    /// Comparison for less equal.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if \a x less equal \a y
    /// \retval false else
    //		template<typename T,typename U> typename enable<bool,T,U>::type islessequal(T x, U y) { return functions::islessequal(x, y); }
    inline bool islessequal(half x, half y) { return functions::islessequal(x, y); }
    inline bool islessequal(half x, expr y) { return functions::islessequal(x, y); }
    inline bool islessequal(expr x, half y) { return functions::islessequal(x, y); }
    inline bool islessequal(expr x, expr y) { return functions::islessequal(x, y); }

    /// Comarison for less or greater.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if either less or greater
    /// \retval false else
    //		template<typename T,typename U> typename enable<bool,T,U>::type islessgreater(T x, U y) { return functions::islessgreater(x, y); }
    inline bool islessgreater(half x, half y) { return functions::islessgreater(x, y); }
    inline bool islessgreater(half x, expr y) { return functions::islessgreater(x, y); }
    inline bool islessgreater(expr x, half y) { return functions::islessgreater(x, y); }
    inline bool islessgreater(expr x, expr y) { return functions::islessgreater(x, y); }

    /// Check if unordered.
    /// \param x first operand
    /// \param y second operand
    /// \retval true if unordered (one or two NaN operands)
    /// \retval false else
    //		template<typename T,typename U> typename enable<bool,T,U>::type isunordered(T x, U y) { return functions::isunordered(x, y); }
    inline bool isunordered(half x, half y) { return functions::isunordered(x, y); }
    inline bool isunordered(half x, expr y) { return functions::isunordered(x, y); }
    inline bool isunordered(expr x, half y) { return functions::isunordered(x, y); }
    inline bool isunordered(expr x, expr y) { return functions::isunordered(x, y); }

    /// \name Casting
    /// \{


    /// Cast to or from half-precision floating point number.
    /// This casts between [half](\ref half_float::half) and any built-in arithmetic type. The values are converted
    /// directly using the given rounding mode, without any roundtrip over `float` that a `static_cast` would otherwise do.
    /// It uses the default rounding mode.
    ///
    /// Using this cast with neither of the two types being a [half](\ref half_float::half) or with any of the two types
    /// not being a built-in arithmetic type (apart from [half](\ref half_float::half), of course) results in a compiler
    /// error and casting between [half](\ref half_float::half)s is just a no-op.
    /// \tparam T destination type (half or built-in arithmetic type)
    /// \tparam U source type (half or built-in arithmetic type)
    /// \param arg value to cast
    /// \return \a arg converted to destination type
    template <typename T, typename U>
    T half_cast(U arg) { return half_caster<T, U>::cast(arg); }

    /// Cast to or from half-precision floating point number.
    /// This casts between [half](\ref half_float::half) and any built-in arithmetic type. The values are converted
    /// directly using the given rounding mode, without any roundtrip over `float` that a `static_cast` would otherwise do.
    ///
    /// Using this cast with neither of the two types being a [half](\ref half_float::half) or with any of the two types
    /// not being a built-in arithmetic type (apart from [half](\ref half_float::half), of course) results in a compiler
    /// error and casting between [half](\ref half_float::half)s is just a no-op.
    /// \tparam T destination type (half or built-in arithmetic type)
    /// \tparam R rounding mode to use.
    /// \tparam U source type (half or built-in arithmetic type)
    /// \param arg value to cast
    /// \return \a arg converted to destination type
    template <typename T, std::float_round_style R, typename U>
    T half_cast(U arg) { return half_caster<T, U, R>::cast(arg); }
    /// \}
} // namespace detail

using detail::operator==;
using detail::operator!=;
using detail::operator<;
using detail::operator>;
using detail::operator<=;
using detail::operator>=;
using detail::operator+;
using detail::operator-;
using detail::operator*;
using detail::operator/;
using detail::operator<<;
using detail::operator>>;

using detail::abs;
using detail::acos;
using detail::acosh;
using detail::asin;
using detail::asinh;
using detail::atan;
using detail::atan2;
using detail::atanh;
using detail::cbrt;
using detail::ceil;
using detail::cos;
using detail::cosh;
using detail::erf;
using detail::erfc;
using detail::exp;
using detail::exp2;
using detail::expm1;
using detail::fabs;
using detail::fdim;
using detail::floor;
using detail::fma;
using detail::fmax;
using detail::fmin;
using detail::fmod;
using detail::hypot;
using detail::lgamma;
using detail::log;
using detail::log10;
using detail::log1p;
using detail::log2;
using detail::lrint;
using detail::lround;
using detail::nanh;
using detail::nearbyint;
using detail::pow;
using detail::remainder;
using detail::remquo;
using detail::rint;
using detail::round;
using detail::sin;
using detail::sinh;
using detail::sqrt;
using detail::tan;
using detail::tanh;
using detail::tgamma;
using detail::trunc;
#if HALF_ENABLE_CPP11_LONG_LONG
using detail::llrint;
using detail::llround;
#endif
using detail::copysign;
using detail::fpclassify;
using detail::frexp;
using detail::ilogb;
using detail::isfinite;
using detail::isgreater;
using detail::isgreaterequal;
using detail::isinf;
using detail::isless;
using detail::islessequal;
using detail::islessgreater;
using detail::isnan;
using detail::isnormal;
using detail::isunordered;
using detail::ldexp;
using detail::logb;
using detail::modf;
using detail::nextafter;
using detail::nexttoward;
using detail::scalbln;
using detail::scalbn;
using detail::signbit;

using detail::half_cast;
} // namespace half_float

/// Extensions to the C++ standard library.
namespace std
{
/// Numeric limits for half-precision floats.
/// Because of the underlying single-precision implementation of many operations, it inherits some properties from
/// `std::numeric_limits<float>`.
template <>
class numeric_limits<half_float::half> : public numeric_limits<float>
{
public:
    /// Supports signed values.
    static HALF_CONSTEXPR_CONST bool is_signed = true;

    /// Is not exact.
    static HALF_CONSTEXPR_CONST bool is_exact = false;

    /// Doesn't provide modulo arithmetic.
    static HALF_CONSTEXPR_CONST bool is_modulo = false;

    /// IEEE conformant.
    static HALF_CONSTEXPR_CONST bool is_iec559 = true;

    /// Supports infinity.
    static HALF_CONSTEXPR_CONST bool has_infinity = true;

    /// Supports quiet NaNs.
    static HALF_CONSTEXPR_CONST bool has_quiet_NaN = true;

    /// Supports subnormal values.
    static HALF_CONSTEXPR_CONST float_denorm_style has_denorm = denorm_present;

    /// Rounding mode.
    /// Due to the mix of internal single-precision computations (using the rounding mode of the underlying
    /// single-precision implementation) with the rounding mode of the single-to-half conversions, the actual rounding
    /// mode might be `std::round_indeterminate` if the default half-precision rounding mode doesn't match the
    /// single-precision rounding mode.
    static HALF_CONSTEXPR_CONST float_round_style round_style = (std::numeric_limits<float>::round_style ==
                                                                 half_float::half::round_style)
                                                                    ? half_float::half::round_style
                                                                    : round_indeterminate;

    /// Significant digits.
    static HALF_CONSTEXPR_CONST int digits = 11;

    /// Significant decimal digits.
    static HALF_CONSTEXPR_CONST int digits10 = 3;

    /// Required decimal digits to represent all possible values.
    static HALF_CONSTEXPR_CONST int max_digits10 = 5;

    /// Number base.
    static HALF_CONSTEXPR_CONST int radix = 2;

    /// One more than smallest exponent.
    static HALF_CONSTEXPR_CONST int min_exponent = -13;

    /// Smallest normalized representable power of 10.
    static HALF_CONSTEXPR_CONST int min_exponent10 = -4;

    /// One more than largest exponent
    static HALF_CONSTEXPR_CONST int max_exponent = 16;

    /// Largest finitely representable power of 10.
    static HALF_CONSTEXPR_CONST int max_exponent10 = 4;

    /// Smallest positive normal value.
    static HALF_CONSTEXPR half_float::half min() HALF_NOTHROW { return half_float::half(half_float::detail::binary, 0x0400); }

    /// Smallest finite value.
    static HALF_CONSTEXPR half_float::half lowest() HALF_NOTHROW { return half_float::half(half_float::detail::binary, 0xFBFF); }

    /// Largest finite value.
    static HALF_CONSTEXPR half_float::half max() HALF_NOTHROW { return half_float::half(half_float::detail::binary, 0x7BFF); }

    /// Difference between one and next representable value.
    static HALF_CONSTEXPR half_float::half epsilon() HALF_NOTHROW { return half_float::half(half_float::detail::binary, 0x1400); }

    /// Maximum rounding error.
    static HALF_CONSTEXPR half_float::half round_error() HALF_NOTHROW
    {
        return half_float::half(half_float::detail::binary, (round_style == std::round_to_nearest) ? 0x3800 : 0x3C00);
    }

    /// Positive infinity.
    static HALF_CONSTEXPR half_float::half infinity() HALF_NOTHROW { return half_float::half(half_float::detail::binary, 0x7C00); }

    /// Quiet NaN.
    static HALF_CONSTEXPR half_float::half quiet_NaN() HALF_NOTHROW { return half_float::half(half_float::detail::binary, 0x7FFF); }

    /// Signalling NaN.
    static HALF_CONSTEXPR half_float::half signaling_NaN() HALF_NOTHROW { return half_float::half(half_float::detail::binary, 0x7DFF); }

    /// Smallest positive subnormal value.
    static HALF_CONSTEXPR half_float::half denorm_min() HALF_NOTHROW { return half_float::half(half_float::detail::binary, 0x0001); }
};

#if HALF_ENABLE_CPP11_HASH
/// Hash function for half-precision floats.
/// This is only defined if C++11 `std::hash` is supported and enabled.
template <>
struct hash<half_float::half> //: unary_function<half_float::half,size_t>
{
    /// Type of function argument.
    typedef half_float::half argument_type;

    /// Function return type.
    typedef size_t result_type;

    /// Compute hash function.
    /// \param arg half to hash
    /// \return hash value
    result_type operator()(argument_type arg) const
    {
        return hash<half_float::detail::uint16>()(static_cast<unsigned>(arg.data_) & -(arg.data_ != 0x8000));
    }
};
#endif
} // namespace std

#undef HALF_CONSTEXPR
#undef HALF_CONSTEXPR_CONST
#undef HALF_NOEXCEPT
#undef HALF_NOTHROW
#ifdef HALF_POP_WARNINGS
#pragma warning(pop)
#undef HALF_POP_WARNINGS
#endif

#endif
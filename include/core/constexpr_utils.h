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
 * @Date: 2026-01-06 15:46:41
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-01-07 10:08:48
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
namespace cexpr
{
constexpr int64_t char_to_int(char c)
{
    if (c >= '0' && c <= '9')
    {
        return c - '0';
    }
    else if (c >= 'a' && c <= 'f')
    {
        return c - 'a' + 10;
    }
    else if (c >= 'A' && c <= 'F')
    {
        return c - 'A' + 10;
    }
    else
    {
        return 0;
    }
}
constexpr int64_t str_to_int(const char *str, size_t len)
{
    int64_t result = 0;
    int64_t sign = 1;
    uint32_t i = 0;
    int64_t base = 10;
    if (str[i] == '-')
    {
        sign = -1;
        i++;
    }
    else if (str[i] == '+')
    {
        i++;
    }
    if (str[i] == '0' && str[i + 1] == 'x')
    {
        i += 2;
        base = 16;
    }
    while (i < len && str[i] != '\0')
    {
        result = result * base + char_to_int(str[i]);
        i++;
    }

    return result * sign;
}
constexpr size_t str_len(const char *str)
{
    size_t len = 0;
    while (str[len] != '\0')
    {
        ++len;
    }
    return len;
}
constexpr bool compare_strings(const char *str1, const char *str2, size_t length)
{
    uint32_t i = 0;
    while (i < length - 1 && str1[i] && (str1[i] == str2[i]))
    {
        i++;
    }
    return str1[i] == str2[i];
}
constexpr const char *strip(const char *str)
{
    while (*str == ' ')
    {
        ++str;
    }
    return str;
}
constexpr size_t len_to_splitter(const char *str, const char splitter)
{
    size_t i = 0;
    while (str[i] && str[i] != splitter)
    {
        i++;
    }
    return i;
}

// 编译时多类型值容器（C++17实现）
class ConstexprValue
{
public:
    enum class Type
    {
        BOOL,
        INT,
        UINT,
        FLOAT,
        DOUBLE,
        STRING
    };

    // constexpr构造函数 - 必须通过成员初始化列表初始化所有成员
    constexpr ConstexprValue(bool b) noexcept
        : type_(Type::BOOL), bool_value_(b), int_value_(0), uint_value_(0), float_value_(0.0f), double_value_(0.), str_value_(nullptr) {}

    constexpr ConstexprValue(unsigned long long i) noexcept
        : type_(Type::UINT), bool_value_(false), int_value_(0), uint_value_(i), float_value_(0.0f), double_value_(0.), str_value_(nullptr) {}

    constexpr ConstexprValue(long long i) noexcept
        : type_(Type::INT), bool_value_(false), int_value_(i), uint_value_(0), float_value_(0.0f), double_value_(0.), str_value_(nullptr) {}

    constexpr ConstexprValue(float f) noexcept
        : type_(Type::FLOAT), bool_value_(false), int_value_(0), uint_value_(0), float_value_(f), double_value_(0.), str_value_(nullptr) {}

    constexpr ConstexprValue(double f) noexcept
        : type_(Type::DOUBLE), bool_value_(false), int_value_(0), uint_value_(0), float_value_(0.0f), double_value_(f), str_value_(nullptr) {}

    constexpr ConstexprValue(const char *s) noexcept
        : type_(Type::STRING), bool_value_(false), int_value_(0), uint_value_(0), float_value_(0.0f), double_value_(0.), str_value_(s) {}

    constexpr ConstexprValue(std::string_view s) noexcept
        : type_(Type::STRING), bool_value_(false), int_value_(0), uint_value_(0), float_value_(0.0f), double_value_(0.), str_value_(s.data()) {}

    // 默认构造函数
    constexpr ConstexprValue() noexcept
        : type_(Type::INT), bool_value_(false), int_value_(0), uint_value_(0), float_value_(0.0f), double_value_(0.), str_value_(nullptr) {}

    // 获取类型
    constexpr Type type() const noexcept { return type_; }

    // 类型安全的取值方法
    constexpr bool as_bool() const noexcept
    {
        return type_ == Type::BOOL ? bool_value_ : false;
    }

    constexpr int as_int() const noexcept
    {
        return type_ == Type::INT ? int_value_ : 0;
    }

    constexpr int as_uint() const noexcept
    {
        return type_ == Type::UINT ? uint_value_ : 0;
    }

    constexpr float as_float() const noexcept
    {
        return type_ == Type::FLOAT ? float_value_ : 0.0f;
    }

    constexpr float as_double() const noexcept
    {
        return type_ == Type::DOUBLE ? double_value_ : 0.;
    }

    constexpr std::string_view as_string() const noexcept
    {
        return type_ == Type::STRING ? (str_value_ ? std::string_view(str_value_) : "") : "";
    }

    // 编译时类型检查
    template <typename T>
    constexpr bool is_type() const noexcept
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            return type_ == Type::BOOL;
        }
        else if constexpr (std::is_same_v<T, int> ||
                           std::is_same_v<T, long> ||
                           std::is_same_v<T, long long>)
        {
            return type_ == Type::INT;
        }
        else if constexpr (std::is_same_v<T, unsigned int> ||
                           std::is_same_v<T, unsigned long> ||
                           std::is_same_v<T, unsigned long long>)
        {
            return type_ == Type::UINT;
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return type_ == Type::FLOAT;
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return type_ == Type::DOUBLE;
        }
        else if constexpr (std::is_same_v<T, const char *> ||
                           std::is_same_v<T, std::string_view>)
        {
            return type_ == Type::STRING;
        }
        return false;
    }

    // 编译时取值（类型安全）
    template <typename T>
    constexpr auto get() const noexcept
    {
        static_assert(std::is_same_v<T, bool> || std::is_same_v<T, int> || std::is_same_v<T, long> ||
                          std::is_same_v<T, long long> || std::is_same_v<T, unsigned int> ||
                          std::is_same_v<T, unsigned long> || std::is_same_v<T, unsigned long long> ||
                          std::is_same_v<T, float> || std::is_same_v<T, double> ||
                          std::is_same_v<T, const char *> || std::is_same_v<T, std::string_view>,
                      "Unsupported type");

        if constexpr (std::is_same_v<T, bool>)
        {
            return as_bool();
        }
        else if constexpr (std::is_same_v<T, int> ||
                           std::is_same_v<T, long> ||
                           std::is_same_v<T, long long>)
        {
            return as_int();
        }
        else if constexpr (std::is_same_v<T, unsigned int> ||
                           std::is_same_v<T, unsigned long> ||
                           std::is_same_v<T, unsigned long long>)
        {
            return as_uint();
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return as_float();
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            return as_double();
        }
        else if constexpr (std::is_same_v<T, const char *>)
        {
            return str_value_;
        }
        else
        {
            return as_string();
        }
    }

private:
    Type type_;
    bool bool_value_;
    long long int_value_;
    unsigned long long uint_value_;
    float float_value_;
    double double_value_;
    const char *str_value_;
};

// 编译时多类型字典（C++17实现）
template <size_t N>
class ConstexprMultiDict
{
public:
    struct KeyValue
    {
        std::string_view key;
        ConstexprValue value;

        // constexpr构造函数 - 直接初始化所有成员
        constexpr KeyValue(const char *k, bool b) noexcept
            : key(k), value(b) {}

        constexpr KeyValue(const std::pair<const char *, bool> p) noexcept
            : key(p.first), value(p.second) {}

        constexpr KeyValue(const char *k, int i) noexcept
            : key(k), value((long long)i) {}

        constexpr KeyValue(const std::pair<const char *, int> p) noexcept
            : key(p.first), value((long long)p.second) {}

        constexpr KeyValue(const char *k, long i) noexcept
            : key(k), value((long long)i) {}

        constexpr KeyValue(const std::pair<const char *, long> p) noexcept
            : key(p.first), value((long long)p.second) {}

        constexpr KeyValue(const char *k, long long i) noexcept
            : key(k), value(i) {}

        constexpr KeyValue(const std::pair<const char *, long long> p) noexcept
            : key(p.first), value(p.second) {}

        constexpr KeyValue(const char *k, unsigned int i) noexcept
            : key(k), value((unsigned long long)i) {}

        constexpr KeyValue(const std::pair<const char *, unsigned int> p) noexcept
            : key(p.first), value((unsigned long long)p.second) {}

        constexpr KeyValue(const char *k, unsigned long i) noexcept
            : key(k), value((unsigned long long)i) {}

        constexpr KeyValue(const std::pair<const char *, unsigned long> p) noexcept
            : key(p.first), value((unsigned long long)p.second) {}

        constexpr KeyValue(const char *k, unsigned long long i) noexcept
            : key(k), value(i) {}

        constexpr KeyValue(const std::pair<const char *, unsigned long long> p) noexcept
            : key(p.first), value(p.second) {}

        constexpr KeyValue(const char *k, float f) noexcept
            : key(k), value(f) {}

        constexpr KeyValue(const std::pair<const char *, float> p) noexcept
            : key(p.first), value(p.second) {}

        constexpr KeyValue(const char *k, double f) noexcept
            : key(k), value(f) {}

        constexpr KeyValue(const std::pair<const char *, double> p) noexcept
            : key(p.first), value(p.second) {}

        constexpr KeyValue(const char *k, const char *s) noexcept
            : key(k), value(s) {}
        constexpr KeyValue(const std::pair<const char *, const char *> p) noexcept
            : key(p.first), value(p.second) {}
    };

    // constexpr构造函数
    constexpr ConstexprMultiDict(const std::array<KeyValue, N> &pairs) noexcept
        : data_(pairs) {}

    // 查找键对应的值
    constexpr ConstexprValue at(std::string_view key) const noexcept
    {
        for (const auto &pair : data_)
        {
            if (pair.key == key)
            {
                return pair.value;
            }
        }
        return ConstexprValue{}; // 返回默认构造的值
    }

    // // 类型安全的查找方法
    // template<typename T>
    // constexpr auto get(std::string_view key, T default_val = T{}) const noexcept {
    //     auto value = at(key);
    //     if (value.is_type<T>()) {
    //         return value.get<T>();
    //     }
    //     return default_val;
    // }

    // 检查键是否存在
    constexpr bool contains(std::string_view key) const noexcept
    {
        for (const auto &pair : data_)
        {
            if (pair.key == key)
            {
                return true;
            }
        }
        return false;
    }

    // // 检查键是否存在且为指定类型
    // template<typename T>
    // constexpr bool contains_as(std::string_view key) const noexcept {
    //     auto value = at(key);
    //     return value.is_type<T>();
    // }

    // 获取字典大小
    constexpr size_t size() const noexcept
    {
        return N;
    }

    // 通过索引访问
    constexpr const KeyValue &operator[](size_t index) const noexcept
    {
        return data_[index];
    }

    // 迭代器支持
    constexpr auto begin() const noexcept { return data_.begin(); }
    constexpr auto end() const noexcept { return data_.end(); }

    // C++17 constexpr if 支持的类型安全遍历
    template <typename F>
    constexpr void for_each(F &&func) const
    {
        for (const auto &pair : data_)
        {
            if constexpr (std::is_invocable_v<F, std::string_view, bool>)
            {
                if (pair.value.type() == ConstexprValue::Type::BOOL)
                {
                    func(pair.key, pair.value.as_bool());
                }
            }
            else if constexpr (std::is_invocable_v<F, std::string_view, int>)
            {
                if (pair.value.type() == ConstexprValue::Type::INT)
                {
                    func(pair.key, pair.value.as_int());
                }
            }
            else if constexpr (std::is_invocable_v<F, std::string_view, float>)
            {
                if (pair.value.type() == ConstexprValue::Type::FLOAT)
                {
                    func(pair.key, pair.value.as_float());
                }
            }
            else if constexpr (std::is_invocable_v<F, std::string_view, float>)
            {
                if (pair.value.type() == ConstexprValue::Type::DOUBLE)
                {
                    func(pair.key, pair.value.as_double());
                }
            }
            else if constexpr (std::is_invocable_v<F, std::string_view, std::string_view>)
            {
                if (pair.value.type() == ConstexprValue::Type::STRING)
                {
                    func(pair.key, pair.value.as_string());
                }
            }
            else
            {
                func(pair.key, pair.value);
            }
        }
    }

private:
    std::array<KeyValue, N> data_;
};

// C++17 辅助函数 - 使用折叠表达式创建字典
template <typename... Args>
constexpr auto make_constexpr_dict(Args &&... args)
{
    constexpr size_t size = sizeof...(Args);
    std::array<typename ConstexprMultiDict<size>::KeyValue, size> pairs = {
        typename ConstexprMultiDict<size>::KeyValue(std::forward<Args>(args))...};
    return ConstexprMultiDict<size>(pairs);
}

} // namespace cexpr
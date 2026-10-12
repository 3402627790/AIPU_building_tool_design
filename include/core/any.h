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
 * @Date: 2020-02-09 12:15:03
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-24 16:31:04
 */
#ifndef CORE_ANY_H
#define CORE_ANY_H

#include <assert.h>
#include <map>
#include <ostream>
#include <string>

namespace aipubt
{

class Any;
#define ENABLE_ANY_TYPEID 1
#if __cplusplus < 201402L //need C++14 or higher
#error C++ standard prior to C++14 are not supported
#endif
struct TypeIDStorage
{
};

#ifdef ENABLE_ANY_TYPEID

/**
 * @brief a help class for compiling-time string operation
 * 
 */
struct string_view
{
    char const *data;
    std::size_t size;
};

template <class T>
constexpr string_view get_name()
{
    char const *p = __PRETTY_FUNCTION__;
    while (*p++ != '=')
        ;
    for (; *p == ' '; ++p)
        ;
    char const *p2 = p;
    int count = 1;
    for (;; ++p2)
    {
        switch (*p2)
        {
        case '[':
            ++count;
            break;
        case ']':
            --count;
            if (!count)
                return {p, std::size_t(p2 - p)};
        }
    }
    return {};
}

template <class T>
constexpr string_view get_name_wo_ns()
{
    char const *p = __PRETTY_FUNCTION__;
    while (*p++ != '=')
        ;
    for (; *p == ' '; ++p)
        ;
    int ns = 0;
    int i = 0;
    for (; *(p + i) != ']'; i++)
    {
        if (*(p + i) == ':' && *(p + i + 1) == ':')
        {
            ns = i + 2;
        }
    }
    p = p + ns;
    char const *p2 = p;
    int count = 1;
    for (;; ++p2)
    {
        switch (*p2)
        {
        case '[':
            ++count;
            break;
        case ']':
            --count;
            if (!count)
                return {p, std::size_t(p2 - p)};
        }
    }
    return {};
}

std::map<TypeIDStorage *, std::string> &get_any_type_name_dict();

template <class T>
TypeIDStorage *get_any_type_id()
{
    static TypeIDStorage id;
    return &id;
}

template <typename T>
TypeIDStorage *GET_ANY_TYPEID()
{
    auto tid = get_any_type_id<T>();
    static int run_once = 0;
    if (run_once == 0)
    {
        string_view sv = get_name<T>();
        get_any_type_name_dict()[tid] = std::string(sv.data, sv.size);
        run_once = 1;
    }
    return tid;
}

#endif //ENABLE_ANY_TYPEID
/**
 * @brief a homemade Any class
 * 
 */
class Any
{
public:
    class BaseData
    {
    public:
        TypeIDStorage *m_type = 0;
        virtual BaseData *clone() = 0;
        virtual ~BaseData() {}
    };
    template <typename T>
    class __attribute__((visibility("hidden"))) MData : public BaseData
    {
    public:
        MData(const T &data) : m_data(data) // this->m_data=data
        {
#ifdef ENABLE_ANY_TYPEID
            m_type = GET_ANY_TYPEID<T>();
#endif //ENABLE_ANY_TYPEID
        }
        ~MData() override {}
        T &get_value()
        {
            return m_data;
        }

        BaseData *clone() override
        {
            return new MData(m_data);
        }

    private:
        T m_data;
    };

    inline Any() = default;
    Any(const Any &other)
    {
        if (m_value)
        {
            delete m_value;
            m_value = nullptr;
        }
        if (other.m_value)
            m_value = other.m_value->clone();
    }

    Any(const char *x)
    {
        m_value = new MData<std::string>(std::string(x));
    }

    template <typename T>
    Any(const T &x)
    {
        m_value = new MData<T>(x);
    }
    ~Any()
    {
        if (m_value)
        {
            delete m_value;
            m_value = nullptr;
        }
    }

    inline BaseData *value() const
    {
        return m_value;
    }

    Any &operator=(const Any &other)
    {
        if (m_value)
        {
            delete m_value;
        }
        if (other.m_value)
            m_value = other.m_value->clone();
        return *this;
    }

    //using  string instead of char*
    Any &operator=(const char *other)
    {
        if (m_value)
            delete m_value;
        m_value = new MData<std::string>(std::string(other));
        return *this;
    }

    template <typename T>
    Any &operator=(const T &other)
    {
        if (m_value)
            delete m_value;
        m_value = new MData<T>(other);
        return *this;
    }

    template <typename T>
    operator T &()
    {
        return cast<T>();
    }

    template <typename T>
    T &cast()
    {
        if (!m_value)
        {
            m_value = new MData<T>(T());
        }
        return static_cast<Any::MData<T> *>(m_value)->get_value();
    }

    template <typename T>
    const T &cast() const
    {
        if (!m_value)
        {
            static T a;
            return a;
        }
        return static_cast<Any::MData<T> *>(m_value)->get_value();
    }

    static const Any& none()
    {
        static const Any _none;
        return _none;
    }

    bool is_none() const
    {
        return this == &(Any::none());
    }

private:
    BaseData *m_value = nullptr;
#ifdef ENABLE_ANY_TYPEID
public:
    static void add_type_id_name(TypeIDStorage *id, const std::string &name)
    {
        get_any_type_name_dict()[id] = name;
    }
    static const std::string get_type_name_by_id(TypeIDStorage *id)
    {
        if (get_any_type_name_dict().count(id) == 0)
        {
            return "Any";
        }
        return get_any_type_name_dict()[id];
    }
    TypeIDStorage *get_type_id() const
    {
        if (!m_value)
            return 0;
        return m_value->m_type;
    }
    const std::string get_type_name() const
    {
        return get_type_name_by_id(get_type_id());
    }
    std::ostream &print(std::ostream &oss) const;

#endif //ENABLE_ANY_TYPEID
};

template <typename T>
inline T &cast_any(const Any &x)
{
    if (!x.value())
    {
        return ((Any &)(x)).operator T &();
    }
    return static_cast<Any::MData<T> *>(x.value())->get_value();
}

};     //namespace aipubt
#endif //!CORE_ANY_H
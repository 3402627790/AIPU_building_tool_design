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
 * @Date: 2020-05-09 11:43:27 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-02-22 16:13:03
 */
#ifndef _UTILS_JSON_H
#define _UTILS_JSON_H

#include <assert.h>
#include <core/any.h>
#include <initializer_list>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace utils
{
class JsonObject
{
public:
    enum JSONType
    {
        _NULL = 0,
        BOOL = 1,
        INT = 2,
        FLOAT = 3,
        OBJ = 4,
        STR = 5,
        ARR = 6,
    };

    JSONType t;
    aipubt::Any v;

    JsonObject() : t(_NULL), v(nullptr) {}
    JsonObject(const std::string &s) : t(STR), v((s)) {}
    JsonObject(const char *s) : t(STR), v((s)) {}

    template <typename T, typename std::enable_if<std::is_integral<T>::value>::type * = nullptr>
    JsonObject(T int_value)
    {
        t = INT;
        v = (int64_t)int_value;
    }

    template <typename T, typename std::enable_if<std::is_floating_point<T>::value>::type * = nullptr>
    JsonObject(T float_value)
    {
        t = FLOAT;
        v = (double)float_value;
    }

    JsonObject(bool b)
    {
        t = BOOL;
        v = b;
    }

    JsonObject(const JsonObject &other)
    {
        v = other.v;
        t = other.t;
    }

    template <typename T>
    JsonObject(std::initializer_list<T> l)
    {
        for (auto &i : l)
        {
            push_back(i);
        }
    }

    JsonObject(const std::vector<JsonObject> &l)
    {
        for (auto &i : l)
        {
            push_back(i);
        }
    }

    JsonObject(std::initializer_list<std::pair<const std::string &, JsonObject>> l)
    {
        for (auto &kv : l)
        {
            (*this)[kv.first] = kv.second;
        }
    }

    JsonObject(const std::vector<std::pair<std::string, JsonObject>> &l)
    {
        for (auto &kv : l)
        {
            (*this)[kv.first] = kv.second;
        }
    }

    JsonObject(JSONType type)
    {
        t = type;
        switch (type)
        {
        case BOOL:
            v = false;
            break;
        case INT:
            v = (int64_t)0;
            break;
        case FLOAT:
            v = (double)0;
            break;
        case OBJ:
            v = std::vector<std::pair<std::string, JsonObject>>();
            break;
        case STR:
            v = std::string();
            break;
        case ARR:
            v = std::vector<JsonObject>();
            break;
        default:
            v = nullptr;
            break;
        }
    }

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type * = nullptr>
    operator T() const
    {
        if (t == FLOAT)
        {
            return static_cast<T>(aipubt::cast_any<double>(v));
        }
        else if (t == INT)
        {
            return static_cast<T>(aipubt::cast_any<int64_t>(v));
        }
        else
        {
            assert(t == FLOAT || t == INT);
            return static_cast<T>(aipubt::cast_any<int64_t>(v));
        }
    }

    operator bool() const
    {
        assert(t == BOOL);
        return aipubt::cast_any<bool>(v);
    }

    operator std::string &()
    {
        assert(t == STR);
        return aipubt::cast_any<std::string>(v);
    }

    operator std::string() const
    {
        assert(t == STR);
        return aipubt::cast_any<std::string>(v);
    }

    JsonObject &operator[](int i)
    {
        return array()[i];
    }

    JsonObject &operator[](const std::string &key)
    {
        if (t != OBJ)
        {
            t = OBJ;
            v = std::vector<std::pair<std::string, JsonObject>>();
        }
        auto &dict_map = aipubt::cast_any<std::vector<std::pair<std::string, JsonObject>>>(v);
        for (auto &kv : dict_map)
        {
            if (kv.first == key)
            {
                return kv.second;
            }
        }
        dict_map.push_back({key, JsonObject()});
        return dict_map[dict_map.size() - 1].second;
    }

    JsonObject &operator[](const char *key)
    {
        return (*this)[std::string(key)];
    }

    JsonObject &operator=(std::vector<JsonObject> &arr)
    {
        if (t != ARR)
        {
            t = ARR;
        }
        v = arr;
        return *this;
    }

    std::vector<JsonObject> &array()
    {
        assert(t == ARR);
        return aipubt::cast_any<std::vector<JsonObject>>(v);
    }

    std::vector<JsonObject> array() const
    {
        assert(t == ARR);
        return aipubt::cast_any<std::vector<JsonObject>>(v);
    }

    std::vector<std::pair<std::string, JsonObject>> &obj()
    {
        assert(t == OBJ);
        return aipubt::cast_any<std::vector<std::pair<std::string, JsonObject>>>(v);
    }

    std::vector<std::pair<std::string, JsonObject>> obj() const
    {
        assert(t == OBJ);
        return aipubt::cast_any<std::vector<std::pair<std::string, JsonObject>>>(v);
    }

    void push_back(const JsonObject &obj)
    {
        if (t != ARR)
        {
            t = ARR;
            v = std::vector<JsonObject>();
        }
        array().push_back(obj);
    }

    bool exist(const std::string &key)
    {
        if (t != OBJ)
        {
            return false;
        }
        auto &obj_map = aipubt::cast_any<std::vector<std::pair<std::string, JsonObject>>>(v);
        for (auto &it : obj_map)
        {
            if (it.first == key)
            {
                return true;
            }
        }
        return false;
    }

    bool is_null() { return t == _NULL; }
    bool is_bool() { return t == BOOL; }
    bool is_int() { return t == INT; }
    bool is_float() { return t == FLOAT; }
    bool is_object() { return t == OBJ; }
    bool is_string() { return t == STR; }
    bool is_array() { return t == ARR; }

    friend std::ostream &operator<<(std::ostream &os, const JsonObject &obj);
    std::string to_string() const;
    static JsonObject parse(const std::string &src);
};

inline bool operator==(JsonObject &o, const std::string &s)
{
    return o.operator std::string() == s;
}
inline bool operator==(const std::string &s, JsonObject &o)
{
    return o == s;
}
inline bool operator!=(JsonObject &o, const std::string &s)
{
    return o.operator std::string() != s;
}
inline bool operator!=(const std::string &s, JsonObject &o)
{
    return o != s;
}

inline bool operator==(JsonObject &o, const char *s)
{
    return o.operator std::string() == std::string(s);
}
inline bool operator==(const char *s, JsonObject &o)
{
    return o == s;
}
inline bool operator!=(JsonObject &o, const char *s)
{
    return o.operator std::string() != std::string(s);
}
inline bool operator!=(const char *s, JsonObject &o)
{
    return o != s;
}

inline bool operator==(JsonObject &o, int v)
{
    return o.operator int() == v;
}
inline bool operator==(int v, JsonObject &o)
{
    return o == v;
}
inline bool operator!=(JsonObject &o, int v)
{
    return o.operator int() != v;
}
inline bool operator!=(int v, JsonObject &o)
{
    return o != v;
}

inline bool operator==(JsonObject &o, uint32_t v)
{
    return o.operator uint32_t() == v;
}
inline bool operator==(uint32_t v, JsonObject &o)
{
    return o == v;
}
inline bool operator!=(JsonObject &o, uint32_t v)
{
    return o.operator uint32_t() != v;
}
inline bool operator!=(uint32_t v, JsonObject &o)
{
    return o != v;
}

inline bool operator==(JsonObject &o, float v)
{
    return o.operator float() == v;
}
inline bool operator==(float v, JsonObject &o)
{
    return o == v;
}
inline bool operator!=(JsonObject &o, float v)
{
    return o.operator float() != v;
}
inline bool operator!=(float v, JsonObject &o)
{
    return o != v;
}
}; // namespace utils

#endif //!_UTILS_JSON_H
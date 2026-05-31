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
 * @Date: 2020-05-09 11:43:31
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-02-24 11:03:10
 */
#include "utils/json.h"
#include "utils/string_utils.h"
#include <iomanip>
// #include <json/json.h>
#include <sstream>
#include <stack>

// using value_t = nlohmann::detail::value_t;
// using json = nlohmann::json;

namespace utils
{
static std::string escape_json(const std::string &s)
{
    std::ostringstream o;
    for (auto c = s.cbegin(); c != s.cend(); c++)
    {
        switch (*c)
        {
        case '"':
            o << "\\\"";
            break;
        case '\\':
            o << "\\\\";
            break;
        case '\b':
            o << "\\b";
            break;
        case '\f':
            o << "\\f";
            break;
        case '\n':
            o << "\\n";
            break;
        case '\r':
            o << "\\r";
            break;
        case '\t':
            o << "\\t";
            break;
        default:
            o << *c;
        }
    }
    return o.str();
}

static std::string unescape_json(const char *ptr, size_t length)
{
    std::ostringstream o;
    auto end = ptr + length;
    for (auto c = ptr; c != end; c++)
    {
        if (*c == '\\' && c + 1 != end)
        {
            auto d = c + 1;
            switch (*d)
            {
            case '"':
                o << '\"';
                c++;
                break;
            case '\\':
                o << '\\';
                c++;
                break;
            case 'b':
                o << '\b';
                c++;
                break;
            case 'f':
                o << '\f';
                c++;
                break;
            case 'n':
                o << '\n';
                c++;
                break;
            case 'r':
                o << '\r';
                c++;
                break;
            case 't':
                o << '\t';
                c++;
                break;
            default:
                o << *c;
            }
        }
        else
        {
            o << *c;
        }
    }
    return o.str();
}

// static std::string unescape_json(const std::string &s)
// {
//     return unescape_json(s.c_str(),s.size());
// }
// inline bool has_only_digits(const std::string &s)
// {
//     return s.find_first_not_of("+-0123456789") == std::string::npos;
// }
inline bool has_only_digits(const char *ptr, size_t length)
{
    for (size_t i = 0; i < length; ++i)
    {
        if (ptr[i] > '9' || ptr[i] < '0')
        {
            if (ptr[i] != '+' && ptr[i] != '-')
            {
                return false;
            }
        }
    }
    return true;
}

inline bool isnumber(const char *ptr, size_t length)
{
    unsigned counter = 0;
    for (size_t i = 0; i < length; ++i)
    {
        if (ptr[i] > '9' || ptr[i] < '0')
        {
            if (ptr[i] == '+' && counter * 0x1 == 0)
            {
                counter = counter | 0x1;
            }
            else if (ptr[i] == '-' && counter * 0b10 == 0)
            {
                counter = counter | 0b10;
            }
            else if (ptr[i] == '.' && counter * 0b100 == 0)
            {
                counter = counter | 0b100;
            }
            else if (ptr[i] == 'e' && counter * 0b1000 == 0)
            {
                counter = counter | 0b1000;
            }
            else
            {
                return false;
            }
        }
    }
    return true;
}

std::ostream &operator<<(std::ostream &oss, const JsonObject &obj)
{
    auto tabs = [](uint32_t num) -> std::string {
        std::ostringstream str;
        for (size_t i = 0; i < num; ++i)
        {
            str << "\t";
        }
        return str.str();
    };
    static uint32_t tab_num = 0;
    if (obj.t == JsonObject::ARR)
    {
        oss << "[";
        auto arr = obj.array();
        for (size_t i = 0; i < arr.size(); ++i)
        {
            if (i != 0)
            {
                oss << ",";
            }
            oss << arr[i];
        }
        oss << "]";
    }
    else if (obj.t == JsonObject::STR)
    {
        oss << "\"" << escape_json((std::string)obj) << "\"";
    }
    else if (obj.t == JsonObject::OBJ)
    {
        oss << "\n"
            << tabs(tab_num) << "{\n";
        tab_num++;
        bool _b = false;
        for (auto kv : obj.obj())
        {
            if (_b)
            {
                oss << ",\n";
            }
            else
            {
                _b = true;
            }
            oss << tabs(tab_num) << "\"" << kv.first << "\":" << kv.second;
        }
        tab_num--;
        oss << "\n"
            << tabs(tab_num) << "}";
    }
    else if (obj.t == JsonObject::_NULL)
    {
        oss << "null";
    }
    else
    {
        switch (obj.t)
        {
        case JsonObject::INT:
        {
            oss << (int64_t)obj;
            break;
        }
        case JsonObject::BOOL:
        {
            bool value = obj;
            if (value)
            {
                oss << "true";
            }
            else
            {
                oss << "false";
            }
            break;
        }
        case JsonObject::FLOAT:
        {
            oss << (double)obj;
            break;
        }
        default:
            break;
        }
    }
    return oss;
}

// static void convert_json_object(const JsonObject &src, json &dst)
// {
//     switch (src.t)
//     {
//     case JsonObject::BOOL:
//     {
//         dst = src.operator bool();
//         break;
//     }
//     case JsonObject::INT:
//     {
//         dst = src.operator int64_t();
//         break;
//     }
//     case JsonObject::FLOAT:
//     {
//         dst = src.operator double();
//         break;
//     }
//     case JsonObject::OBJ:
//     {
//         dst = json::object();
//         for (auto &kv : src.obj())
//         {
//             json obj;
//             convert_json_object(kv.second, obj);
//             dst[kv.first] = obj;
//         }
//         break;
//     }
//     case JsonObject::STR:
//     {
//         dst = src.operator std::string();
//         break;
//     }
//     case JsonObject::ARR:
//     {
//         dst = json::array();
//         for (auto &v : src.array())
//         {
//             json obj;
//             convert_json_object(v, obj);
//             dst.push_back(obj);
//         }
//         break;
//     }
//     default:
//         break;
//     }
// }

// std::ostream &operator<<(std::ostream &oss, const JsonObject &obj)
// {
//     json res;
//     convert_json_object(obj, res);
//     oss << res.dump(4);
//     return oss;
// }

std::string JsonObject::to_string() const
{
    std::ostringstream oss;
    oss << *this;
    return oss.str();
}

enum TOKENType
{
    _NO_TOKEN = 0,
    IDENTIFIER,
    STR_IDENTIFIER,
    COMMA,           // ,
    COLON,           // :
    LSQUARE_BRACKET, // [
    RSQUARE_BRACKET, // ]
    LCURLY_BRACKET,  // {
    RCURLY_BRACKET,  // }
    DOUBLE_QUOTE,    // "
};

struct Token
{
    TOKENType type = _NO_TOKEN;
    size_t pos = 0;
    size_t length = 0;
};

inline void skip_space(const char *ptr, size_t &pos)
{
    while (ptr[pos] != '\0' && isspace(ptr[pos]))
    {
        pos++;
    }
}

inline void skip_comments(const char *ptr, size_t &pos)
{
    while (ptr[pos] == '/' && ptr[pos + 1] == '/')
    {
        pos += 2;
        while (ptr[pos] != '\n' && ptr[pos] != '\0')
        {
            pos++;
        }
        skip_space(ptr, pos);
    }
    while (ptr[pos] == '/' && ptr[pos + 1] == '*')
    {
        pos += 2;
        while (ptr[pos] != '*' && ptr[pos + 1] != '/')
        {
            pos++;
        }
    }
}

inline bool is_token(const char c)
{
    return (c == '"' || c == ',' || c == ':' || c == '[' || c == ']' || c == '{' || c == '}');
}

inline bool str_cmp_(const char *ptr, size_t length, const char *to_comp)
{
    for (size_t i = 0; i < length; ++i)
    {
        if (to_comp[i] == '\0')
        {
            return false;
        }
        if (ptr[i] != to_comp[i])
        {
            return false;
        }
    }
    return true;
}
static void tokenize(const char *ptr, size_t &pos, Token &t)
{
    t.pos = pos;
    t.length = 1;
    if (ptr[pos] == '"')
    {
        t.type = STR_IDENTIFIER;
        pos++;
        t.pos = pos;
        while (ptr[pos] != '"')
        {
            if (ptr[pos] == '\\')
            {
                pos++;
            }
            pos++;
        }
        t.length = pos - t.pos;
    }
    else if (ptr[pos] == ',')
    {
        t.type = COMMA;
    }
    else if (ptr[pos] == ':')
    {
        t.type = COLON;
    }
    else if (ptr[pos] == '{')
    {
        t.type = LCURLY_BRACKET;
    }
    else if (ptr[pos] == '}')
    {
        t.type = RCURLY_BRACKET;
    }
    else if (ptr[pos] == '[')
    {
        t.type = LSQUARE_BRACKET;
    }
    else if (ptr[pos] == ']')
    {
        t.type = RSQUARE_BRACKET;
    }
    else
    {
        t.type = IDENTIFIER;
        pos++;
        while (!is_token(ptr[pos]) && !isspace(ptr[pos]) && ptr[pos] != '\0')
        {
            pos++;
        }
        t.length = pos - t.pos;
        return;
    }
    pos++;
}

static inline JsonObject identifier_to_json(Token &t, const char *ptr)
{
    if (t.type == STR_IDENTIFIER)
    {
        return unescape_json(ptr + t.pos, t.length);
    }
    else if (has_only_digits(ptr + t.pos, t.length))
    {
        int64_t i;
        std::string s(ptr, t.pos, t.length);
        std::istringstream iss(s);
        iss >> std::noskipws >> i;
        return i;
    }
    else if (str_cmp_(ptr + t.pos, t.length, "true"))
    {
        return true;
    }
    else if (str_cmp_(ptr + t.pos, t.length, "false"))
    {
        return false;
    }
    else if (!str_cmp_(ptr + t.pos, t.length, "null"))
    {

        if (isnumber(ptr + t.pos, t.length))
        {
            double f;
            std::string s(ptr, t.pos, t.length);
            std::istringstream iss(s);
            iss >> std::noskipws >> f;
            return f;
        }
        else
        {
            return unescape_json(ptr + t.pos, t.length);
        }
    }
    return JsonObject(); // JsonObject::_NULL
}

JsonObject JsonObject::parse(const std::string &src)
{
    JsonObject current_obj;
    size_t pos = 0;
    auto ptr = src.c_str();
    std::stack<Token> tokens; //for quote/bracket
    std::stack<JsonObject> objs;
    std::stack<std::string> keys;
    // bool is_first_object=true;
    size_t num_objs = 0;
    while (ptr[pos] != '\0')
    {
        skip_space(ptr, pos);
        skip_comments(ptr, pos);
        if (ptr[pos] == '\0')
        {
            break;
        }
        Token tk;
        tokenize(ptr, pos, tk);
        if (tk.type == COMMA)
        {
            continue;
        }
        else if (tk.type == IDENTIFIER || tk.type == STR_IDENTIFIER)
        {
            if (current_obj.t == ARR)
            {
                current_obj.push_back(identifier_to_json(tk, ptr));
            }
            else if (keys.size() && keys.size() == num_objs)
            {
                current_obj[keys.top()] = identifier_to_json(tk, ptr);
                keys.pop();
            }
            else
            {
                objs.push(current_obj);
                current_obj = identifier_to_json(tk, ptr);
            }
        }
        else if (tk.type == LCURLY_BRACKET)
        {
            if (current_obj.t != _NULL)
            {
                objs.push(current_obj);
            }
            current_obj = JsonObject(OBJ);
            tokens.push(tk);
            num_objs++;
        }
        else if (tk.type == LSQUARE_BRACKET)
        {
            objs.push(current_obj);
            current_obj = JsonObject(ARR);
            tokens.push(tk);
        }
        else if (tk.type == RSQUARE_BRACKET || tk.type == RCURLY_BRACKET)
        {
            tokens.pop();
            auto c = std::move(current_obj);
            if (objs.size())
            {
                current_obj = std::move(objs.top());
                if (current_obj.t == ARR)
                {
                    current_obj.push_back(std::move(c));
                }
                else if (current_obj.t == OBJ)
                {
                    if (keys.empty())
                    {
                        //failed
                        return JsonObject();
                    }
                    current_obj[keys.top()] = c;
                    keys.pop();
                }
                objs.pop();
            }
            if (tk.type == RCURLY_BRACKET)
            {
                num_objs--;
            }
        }
        else if (tk.type == COLON)
        {
            std::string key;
            if (current_obj.t == STR)
            {
                key = current_obj.operator std::string &();
            }
            else
            {
                key = current_obj.to_string();
            }
            keys.push(key);
            current_obj = objs.top();
            objs.pop();
        }
    }
    return current_obj;
}

// static void convert_json_object(const json &src, JsonObject &dst)
// {
//     switch (src.type())
//     {
//     case value_t::object:
//     {
//         dst = JsonObject(JsonObject::OBJ);
//         for (auto &kv : src.items())
//         {
//             JsonObject obj;
//             convert_json_object(kv.value(), obj);
//             dst[kv.key()] = obj;
//         }
//         break;
//     }
//     case value_t::array:
//     {
//         dst = JsonObject(JsonObject::ARR);
//         for (auto &v : src)
//         {
//             JsonObject obj;
//             convert_json_object(v, obj);
//             dst.push_back(obj);
//         }
//         break;
//     }
//     case value_t::string:
//     {
//         dst = src.get<std::string>();
//         break;
//     }
//     case value_t::boolean:
//     {
//         dst = src.get<bool>();
//         break;
//     }
//     case value_t::number_integer:
//     {
//         dst = src.get<int64_t>();
//         break;
//     }
//     case value_t::number_unsigned:
//     {
//         dst = src.get<uint64_t>();
//         break;
//     }
//     case value_t::number_float:
//     {
//         dst = src.get<double>();
//         break;
//     }
//     default:
//         break;
//     }
// }

// JsonObject JsonObject::parse(const std::string &src)
// {
//     JsonObject ret;
//     convert_json_object(json::parse(src), ret);
//     return ret;
// }
}; // namespace utils

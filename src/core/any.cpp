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
 * @Date: 2021-02-05 13:52:58 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 14:36:35
 */
#include "core/any.h"
#include <cstdint>
#include <functional>
namespace aipubt
{
#ifdef ENABLE_ANY_TYPEID
using print_func = std::function<void(std::ostream &, const Any &)>;

std::map<TypeIDStorage *, std::string> &get_any_type_name_dict()
{
    static std::map<TypeIDStorage *, std::string> m_dict;
    return m_dict;
};

#define AddPrintFunc(T)                                      \
    {                                                        \
        get_any_type_id<T>(),                                \
            [](std::ostream &oss, const Any &v) {            \
                if (v.get_type_id() == get_any_type_id<T>()) \
                {                                            \
                    oss << v.cast<T>();                      \
                }                                            \
            }                                                \
    }

static std::map<TypeIDStorage *, print_func> &get_any_print_func()
{
    static std::map<TypeIDStorage *, print_func> m_dict =
        {
            AddPrintFunc(std::string),
            AddPrintFunc(int),
            AddPrintFunc(long),
            AddPrintFunc(uint32_t),
            AddPrintFunc(uint64_t),
            AddPrintFunc(float),
            AddPrintFunc(double),
            {get_any_type_id<bool>(),
             [](std::ostream &oss, const Any &v) {
                 if (v.get_type_id() == get_any_type_id<bool>())
                 {
                     if (v.cast<bool>())
                     {
                         oss << "true";
                     }
                     else
                     {
                         oss << "false";
                     }
                 }
             }},
        };
    return m_dict;
};

void add_any_print_function(TypeIDStorage *tid, print_func f)
{
    get_any_print_func()[tid] = f;
}
std::ostream &Any::print(std::ostream &oss) const
{
    if (get_any_print_func().count(get_type_id()))
    {
        get_any_print_func().at(get_type_id())(oss, *this);
    }
    else
    {
        oss << "[Any:" << get_type_name() << " not support print]";
    }
    return oss;
}

#endif //ENABLE_ANY_TYPEID
};     //namespace aipubt

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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-01-29 14:23:23
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-05-28 08:50:18
 */
#ifndef BUILDER_TEXT_GENERATOR_H
#define BUILDER_TEXT_GENERATOR_H
#include "builder/ar/ar_reader.h"
#include "builder/bldtypes.h"
#include "core/node.h"
#include "core/optype.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace aipubt
{
// text and its size
typedef std::function<BuilderText(NodePtr)> text_generator_func;
typedef std::function<const std::string(NodePtr)> code_generator_func;
typedef std::function<const std::string(NodePtr)> kernel_generator_func;

text_generator_func get_text_generator(const OpType t);
code_generator_func get_code_generator(const OpType t);
kernel_generator_func get_kernel_generator(const OpType t);

std::map<std::string, BuilderText> &get_text_cache_map();
const std::string get_kernel_func_name(const std::string &libname);
const std::string get_lib_full_filename(const std::string &libname);
std::vector<std::string> get_all_lib_paths();
const std::vector<ARFileReader> &get_all_afiles();

class KernelGeneratorRegister
{
public:
    KernelGeneratorRegister(OpType t, kernel_generator_func func);
    static std::map<OpType, kernel_generator_func> &get_kernel_generator_map()
    {
        static std::map<OpType, kernel_generator_func> KERNEL_GENERATORS_MAP;
        return KERNEL_GENERATORS_MAP;
    }
};

class TEXTGeneratorRegister
{
public:
    TEXTGeneratorRegister(OpType t, text_generator_func func, code_generator_func code = nullptr);
    static std::map<OpType, code_generator_func> &get_code_generator_map()
    {
        static std::map<OpType, code_generator_func> CODE_GENERATORS_MAP;
        return CODE_GENERATORS_MAP;
    }
    static std::map<OpType, text_generator_func> &get_text_generator_map()
    {
        static std::map<OpType, text_generator_func> TEXT_GENERATORS_MAP;
        return TEXT_GENERATORS_MAP;
    }
};

template <class... Args>
std::function<BuilderText(Args...)>
TEXT_GENERATOR_DECORATOR(const std::string (*f)(Args...))
{
    return
        [f](Args &&... args) { return get_text_content_by_filename(f(args...)); };
}

#define TEXT_GENERATOR(optype) \
    const std::string code_generator_func_##optype(NodePtr node)

#define TEXT_GENERATOR_REGISTER(optype)                                 \
    static TEXTGeneratorRegister _create_text_generator_##optype(       \
        optype, TEXT_GENERATOR_DECORATOR(code_generator_func_##optype), \
        code_generator_func_##optype);

#define TEXT_GENERATOR_END(optype) TEXT_GENERATOR_REGISTER(optype);

#define TEXT_GENERATOR_REGISTER_SHARED_WITH(optype, shared_optype)             \
    const std::string code_generator_func_##shared_optype(NodePtr node);       \
    static TEXTGeneratorRegister _create_text_generator_##optype(              \
        optype, TEXT_GENERATOR_DECORATOR(code_generator_func_##shared_optype), \
        code_generator_func_##shared_optype);

#define KERNEL_GENERATOR(optype) \
    const std::string kernel_generator_func_##optype(NodePtr node)

#define KERNEL_GENERATOR_REGISTER(optype)                             \
    static KernelGeneratorRegister _create_kernel_generator_##optype( \
        optype, kernel_generator_func_##optype);

#define KERNEL_GENERATOR_END(optype) KERNEL_GENERATOR_REGISTER(optype);

inline bool is_valid_code(const std::string &code)
{
    return !code.empty();
}

inline bool is_valid_kernel(const std::string &kernel)
{
    return !kernel.empty();
}
} // namespace aipubt
#endif //! BUILDER_TEXT_GENERATOR_H
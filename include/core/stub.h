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
 * @Date: 2021-09-08 10:16:43 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-09-08 15:12:47
 */
#ifndef CORE_STUB_H
#define CORE_STUB_H

#include "core/any.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace aipubt
{
class Graph;
using GraphPtr = std::shared_ptr<Graph>;
class StubFunction
{
public:
    StubFunction() : m_func(nullptr) {}
    StubFunction(void (*function)()) : m_func(function) {}
    template <typename R, typename... Args>
    operator std::function<R(Args...)>()
    {
        return (R(*)(Args...))(m_func);
    }

protected:
    void (*m_func)();
};
class StubRegister
{
public:
    StubRegister(const std::string &name, void (*function)()) : StubRegister(name, StubFunction(function)){};
    StubRegister(const std::string &name, StubFunction function);
};

void load_dynamic_lib(const std::string &path);

StubFunction get_a_stub_function(const std::string &name);
const std::vector<std::function<void(GraphPtr)>> &get_aipurun_hooks();
void add_aipurun_hooks(std::function<void(GraphPtr)> h);
void clear_aipurun_hooks();
#define ADD_STUB(func) \
    static StubRegister _##func_stub(#func, (void (*)())func);
} // namespace aipubt

#endif //!CORE_STUB_H
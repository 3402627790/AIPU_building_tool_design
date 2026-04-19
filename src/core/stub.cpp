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
 * @Date: 2021-09-08 10:10:40 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2021-09-15 15:29:16
 */

/**
 * @brief this file is for some C++ stub, all c++ stub function are store in a map.
 * 
 */
#include "core/stub.h"
#include "core/any.h"
#include "core/logger.h"
#include <dlfcn.h>
#include <vector>

namespace aipubt
{

static std::map<std::string, StubFunction> &get_the_stub_map()
{
    static std::map<std::string, StubFunction> smap;
    return smap;
}

StubRegister::StubRegister(const std::string &name, StubFunction function)
{
    get_the_stub_map()[name] = function;
}

StubFunction get_a_stub_function(const std::string &name)
{
    if (get_the_stub_map().count(name))
    {
        return get_the_stub_map().at(name);
    }
    return nullptr;
}

class StubLibManager
{
public:
    ~StubLibManager();
    std::map<std::string, void *> m_handles;
};
StubLibManager::~StubLibManager()
{
    for (auto &kv : m_handles)
    {
        if (kv.second)
        {
            dlclose(kv.second);
        }
    }
}

void load_dynamic_lib(const std::string &filename)
{
    static StubLibManager m_instance;
    if (m_instance.m_handles.count(filename))
    {
        return;
    }
    void *handle = nullptr;
    handle = dlopen(filename.c_str(), RTLD_LAZY);
    if (handle == NULL)
    {
        WARN() << "Loading dynamic lib error: " << dlerror();
    }
    else
    {
        m_instance.m_handles[filename] = handle;
    }
}

static std::vector<std::function<void(GraphPtr)>> &_get_aipurun_hooks()
{
    static std::vector<std::function<void(GraphPtr)>> hooks;
    return hooks;
}
const std::vector<std::function<void(GraphPtr)>> &get_aipurun_hooks()
{
    return _get_aipurun_hooks();
}
void add_aipurun_hooks(std::function<void(GraphPtr)> h)
{
    _get_aipurun_hooks().push_back(h);
}
void clear_aipurun_hooks()
{
    _get_aipurun_hooks().clear();
}
} // namespace aipubt
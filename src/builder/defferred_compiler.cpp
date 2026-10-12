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
 * @Author: Kyle Huang
 * @Date: 2026-08-26 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-08-26 14:27:33
 */

#include "builder/defferred_compiler.h"
#include "builder/llir_compiler.h"
#include "core/base_logger.h"

namespace aipubt
{

bool DefferredCompiler::set_graph(GraphPtr g)
{
    m_graph = g;
    if (!g)
    {
        return false;
    }
    return true;
}

void DefferredCompiler::config(const std::map<std::string, std::string> &device_lib_names)
{
    m_device_lib_names = device_lib_names;
    return;
}

std::map<std::string, std::string> DefferredCompiler::compile_to_obj()
{
    std::map<std::string, std::string> device_obj_names;
    for (auto pair : m_device_lib_names)
    {
        std::string bc_lib = pair.second;
        std::string o_lib = pair.second + ".o";
        aipubt::LLIRCompileOptions opts;
        int rc = aipubt::compileLLIRToObject(bc_lib, o_lib, opts);

        if (rc != 0)
        {
            ERROR() << "compile error";
        }
        device_obj_names[pair.first] = o_lib;
    }
    return device_obj_names;
}

} // namespace aipubt
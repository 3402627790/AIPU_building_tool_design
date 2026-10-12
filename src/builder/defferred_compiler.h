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
 * @Date: 2026-08-26 14:27:29
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-08-26 14:27:29
 * */
#ifndef BUILDER_DEFFERRED_COMPILER_H
#define BUILDER_DEFFERRED_COMPILER_H

#include "core/graph.h"

namespace aipubt
{

class DefferredCompiler
{
public:
    bool set_graph(GraphPtr g);
    void config(const std::map<std::string, std::string> &device_lib_names);
    std::map<std::string, std::string> compile_to_obj();

private:
    GraphPtr m_graph;
    std::map<std::string, std::string> m_device_lib_names;
};

} // namespace aipubt
#endif //!BUILDER_DEFFERRED_COMPILER_H

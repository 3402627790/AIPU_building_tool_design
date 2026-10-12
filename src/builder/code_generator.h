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
 * @Date: 2026-04-24 14:27:29
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:29
 * */
#ifndef BUILDER_CODE_GENERATROR_H
#define BUILDER_CODE_GENERATROR_H

#include "builder/cps_elf.h"
#include "builder/gop_plugin.h"
#include "core/graph.h"
#include "core/logger.h"
#include "core/node.h"
#include "core/subgraph.h"

namespace aipubt
{
class CodeGenerator
{
public:
    void set_graph(GraphPtr g);
    void set_device_lib(const std::map<std::string, std::string> &kernel_lib);
    void set_host_lib(const std::map<std::string, std::string> &kernel_lib);
    void gen_device_code(CpsElfPtr elf);
    void gen_host_code(HybridELFPtr elf);

private:
    const std::string get_cpshost_code();
    void gen_host_funcs();
    void gen_host_task_func(std::stringstream &oss, const HostTaskInfo &info);

    GraphPtr m_graph;
    std::map<std::string, std::string> m_device_lib;
    std::map<std::string, std::string> m_host_lib;
    std::stringstream m_host_code;
    HostTaskInfos m_host_tasks;
};
} // namespace aipubt
#endif //! BUILDER_CODE_GENERATROR_H
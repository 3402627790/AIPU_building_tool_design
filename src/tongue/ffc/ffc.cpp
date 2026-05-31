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
 * @Date: 2024-11-15 15:29:54 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-11-15 15:31:13
 */
#include "ffc.h"

namespace ffc
{

FFCompiler::FFCompiler(const std::string &arch, const std::vector<Tensor> &tasks)
{
    // TODO: reorg the tasks
    // here we suppose the tasks are already well broken down and well organized
    m_arch = arch;
    init(tasks);
}

void FFCompiler::init(const std::vector<Tensor> &tasks)
{
    // extract all tasks
    if (tasks.empty())
    {
        return;
    }
    std::vector<Tensor> ts;
    std::function<void(Tensor)> dfs;
    dfs = [&dfs, &ts](Tensor t) {
        if (t->type > uOp::Argument && !contains(ts, t))
        {
            ts.push_back(t);
        }
        for (auto p : t->inputs)
        {
            dfs(p);
        }
    };
    dfs(tasks.back());
    m_tensors = std::vector<Tensor>(ts.rbegin(), ts.rend());
}

bool FFCompiler::compile()
{
    Pipeline p(m_arch);
    for (auto t : m_tensors)
    {
        if (p.feed(t))
        {
            continue;
        }
        else
        {
            m_pipelines.push_back(p);
            p = Pipeline(m_arch);
            if (!p.feed(t))
            {
                return false;
            }
        }
    }
    m_pipelines.push_back(p);
    for (auto &p : m_pipelines)
    {
        p.build();
    }
    return true;
}

std::vector<Code> FFCompiler::gen_code()
{
    std::vector<Code> codes;
    for (auto &p : m_pipelines)
    {
        // generate code for each pipeline
        std::string auto_func_name = "auto_pipeline_" + std::to_string(get_uid());
        Code c = {p.desc_pkg(auto_func_name), {}}; // TODO: generate the actual code
        codes.emplace_back(c);
    }
    return codes;
}

} // namespace ffc
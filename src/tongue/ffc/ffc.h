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
 * @Last Modified time: 2024-11-20 10:44:17
 */
#pragma once
#include "hal/desc.h"
#include "hal/resource.h"
#include "pipeline.h"
#include "tensor.h"
#include "tongue/sbe/ssa.h"

namespace ffc
{
using namespace HAL;
using namespace tongue;

// code for ffc include two parts:
//  1. a control code-> tpc instructions
//  2. a ff descriptor
//  a code is an issue of AIFF

struct Code
{
    using TPCCode = std::vector<sbe::Node>;
    DescriptorPkg desc;
    TPCCode code;
};

class FFCompiler
{
public:
    FFCompiler(const std::string &arch, const std::vector<Tensor> &tasks);

    bool compile();
    std::vector<Code> gen_code();

    std::string arch() const { return m_arch; }
    const std::vector<Pipeline> &pipelines() const { return m_pipelines; }
    const std::vector<Tensor> &tensors() const { return m_tensors; }

private:
    void init(const std::vector<Tensor> &tasks);
    static uint32_t get_uid()
    {
        static uint32_t m_func_id = 0;
        return ++m_func_id;
    }

    std::string m_arch;
    std::vector<Pipeline> m_pipelines;
    std::vector<Tensor> m_tensors;
};

} // namespace ffc
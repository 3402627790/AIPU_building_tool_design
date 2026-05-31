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
 * @Author: Samuel Han
 * @Date: 2022-07-06 10:27:09
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-07-06 15:44:14
 */

#include "quantization/dequantizer.h"
#include "quantization_transform/adaptor.h"
#include "quantization_transform/passes/passes.h"
#include "quantization_transform/run_mode.h"

namespace qtlibn
{
Dequantizer::Dequantizer(aipubt::GraphPtr graph) : m_graph(graph) {}
Dequantizer::Dequantizer(aipubt::GraphPtr graph, std::string run_mode)
    : m_graph(graph), m_run_mode(run_mode) {}

Dequantizer::~Dequantizer()
{
    if (m_graph)
    {
        m_graph = nullptr;
    }
}

void Dequantizer::dequantize(void)
{
    RunMode run_mode = string_to_runmode(m_run_mode);
    if (run_mode == RunMode::RunModeSize)
    {
        run_mode = RunMode::FAsym;
        INFO() << "when dequantize the " << m_graph << ", the graph doesnot set run_mode, we use the fasym run mode to dequantize";
    }
    Adaptor adaptor(run_mode);
    for (auto &node : *m_graph)
    {
        // transform quantization info to float_min/float_max
        adaptor.calculate_max_min(node);
        // transform quantized constant data to float values
        adaptor.dequantize_constant(node);
        // transform quantized param to float param
        adaptor.dequantize_param(node);
    }
}

} // namespace qtlibn

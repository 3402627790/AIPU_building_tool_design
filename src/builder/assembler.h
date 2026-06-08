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
 * @Author: Kyle Huang
 * @Date: 2026-04-24 14:27:29
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:29
 * */
#ifndef BUILDER_ASSEMBLER_H
#define BUILDER_ASSEMBLER_H

#include "builder/cps_elf.h"
#include "builder/gop_plugin.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/node.h"
#include "core/proxy_tensor.h"
#include "core/subgraph.h"
#include "core/surface.h"

namespace aipubt
{
struct param_info_t
{
    uint16_t size;
    uint16_t kind;
};

struct launch_info_t
{
    uint32_t ib_size;
    uint16_t static_sm_bytes;
    uint16_t param_size;
    uint8_t sreg_param_size;
    uint8_t sreg_cnt;
    uint8_t vreg_cnt;
    uint8_t head_wg_vreg_cnt;
    uint8_t isa_set;
    uint8_t subnormal_en;
    uint8_t saturation_en;
    uint8_t padding;
};

class Assembler
{
public:
    bool set_graph(GraphPtr g);
    void set_elf(const CpsElfPtr &elf);
    void assembly();
    std::vector<std::vector<task_t>> get_streams();
    const BuilderParams &get_params() const;
    std::map<uint32_t, uint32_t> get_param_offset_table();
    std::map<TensorPtr, uint32_t> get_tensor_index_map();
    std::map<ConstTensorPtr, uint32_t> get_const_index_map();

private:
    GraphPtr m_graph;
    std::vector<std::vector<task_t>> m_streams;
    BuilderParams m_params;
    std::map<uint32_t, uint32_t> m_param_offset_table;
    std::map<TensorPtr, uint32_t> m_tensor_index_map;
    std::map<ConstTensorPtr, uint32_t> m_const_index_map;
    CpsElfPtr m_elf;
    uint32_t m_tensor_index = 1;
    uint32_t m_const_index = 1;
    uint32_t m_task_id = 0;
    uint32_t get_tensor_index(const TensorPtr t);
    uint32_t get_const_index(const ConstTensorPtr t);
    uint32_t get_task_id();
};

BuilderParams get_kernel_params(GraphPtr g, const KernelInfo &info);
} // namespace aipubt
#endif //!BUILDER_ASSEMBLER_H
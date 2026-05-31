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
#ifndef BUILDER_OBJECT_GENERATOR_H
#define BUILDER_OBJECT_GENERATOR_H

#include "builder/cps_elf.h"
#include "builder/gop_plugin.h"
#include "core/const_tensor.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/node.h"
#include "core/subgraph.h"
#include <vector>

namespace aipubt
{
class ObjectGenerator
{
    enum TensorType
    {
        input = 0,
        output = 1,
        stream = 2,
        param = 3,
    };

public:
    void set_streams(const std::vector<std::vector<task_t>> &streams);
    void set_param_offset_table(const std::map<uint32_t, uint32_t> &param_offset_table);
    void set_params(const std::vector<uint32_t> &params);
    void set_tensor_index_map(const std::map<TensorPtr, uint32_t> &tensor_index_map);
    void set_const_index_map(const std::map<ConstTensorPtr, uint32_t> &const_index_map);
    void set_slots(const std::vector<MemoryObjectPtr> &slots);
    void set_graph(const GraphPtr g);
    void set_constant(const std::set<ConstTensorPtr> &constant);
    bool gen(CpsElfPtr &elf) const;

private:
    std::vector<stream_t> m_streams;
    std::vector<uint32_t> m_params;
    std::map<uint32_t, uint32_t> m_param_offset_table;
    std::map<TensorPtr, uint32_t> m_tensor_index_map;
    std::map<ConstTensorPtr, uint32_t> m_const_index_map;
    std::map<std::string, std::string> m_kernel_lib;
    std::vector<MemoryObjectPtr> m_slots;
    GraphPtr m_graph;
    std::set<ConstTensorPtr> m_constant;

    tensor_info_t gen_info_from_global_tensor(const std::string &tensor_name) const;
};
};     // namespace aipubt
#endif //!BUILDER_OBJECT_GENERATOR_H
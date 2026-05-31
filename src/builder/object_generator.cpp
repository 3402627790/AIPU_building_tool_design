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
 * @Date: 2026-04-24 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:33
 */
#include "builder/object_generator.h"

namespace aipubt
{

void ObjectGenerator::set_streams(const std::vector<std::vector<task_t>> &streams)
{
    for (size_t i = 0; i < streams.size(); ++i)
    {
        stream_t stream;
        stream.id = i;
        stream.tasks = streams[i];
        m_streams.push_back(stream);
    }
}
void ObjectGenerator::set_slots(const std::vector<MemoryObjectPtr> &slots)
{
    m_slots = slots;
}

void ObjectGenerator::set_constant(const std::set<ConstTensorPtr> &constant)
{
    m_constant = constant;
}

void ObjectGenerator::set_param_offset_table(const std::map<uint32_t, uint32_t> &param_offset_table)
{
    m_param_offset_table = param_offset_table;
}
void ObjectGenerator::set_params(const std::vector<uint32_t> &params)
{
    m_params = params;
}
void ObjectGenerator::set_tensor_index_map(const std::map<TensorPtr, uint32_t> &tensor_index_map)
{
    m_tensor_index_map = tensor_index_map;
}

void ObjectGenerator::set_const_index_map(const std::map<ConstTensorPtr, uint32_t> &const_index_map)
{
    m_const_index_map = const_index_map;
}

void ObjectGenerator::set_graph(const GraphPtr g)
{
    m_graph = g;
}

tensor_info_t ObjectGenerator::gen_info_from_global_tensor(const std::string &tensor_name) const
{
    auto tensor = m_graph->global_tensors()[tensor_name];
    tensor_info_t info;
    info.id = m_tensor_index_map.at(tensor);
    info.dtype = tensor->type();
    info.shape = tensor->shape();
    return info;
};

bool ObjectGenerator::gen(CpsElfPtr &elf) const
{
    memory_plan_t mp;
    std::map<MemoryObjectPtr, uint32_t> slot_index;
    for (auto s : m_slots)
    {
        slot_index[s] = mp.slots.size();
        mp.slots.emplace_back(s->size, s->align);
    }
    std::map<uint32_t, TensorPtr> index_tensor_map;
    for (auto &pair : m_tensor_index_map)
    {
        auto tensor = pair.first;
        auto tensor_idx = pair.second;
        auto slot = tensor->memory_offset()->get_base_object();
        auto offset = tensor->offset();

        if (!slot_index.count(slot))
        {
            continue;
            WARN() << "tensor " << tensor->name() << " is not assigned";
        }

        DEBUG() << "tensor: " << tensor_idx << " " << tensor->name() << " is assigned slot " << slot->buffer_id << " offset " << offset;
        mp.tso[tensor_idx] = std::make_pair(slot_index[slot], offset);
        index_tensor_map[tensor_idx] = tensor;
    }

    elf->set_memory_plan(mp);

    for (auto pair : m_param_offset_table)
    {
        DEBUG() << "param offset: " << pair.first << " is tensor: " << pair.second << " " << index_tensor_map[pair.second];
    }

    elf->set_param_tab(m_param_offset_table);

    elf->set_graph(m_streams);

    if (m_params.empty())
    {
        ERROR() << "params are empty!";
    }

    size_t params_size = m_params.size() * sizeof(uint32_t);
    char *params_buf = new char[params_size];
    std::memcpy(params_buf, m_params.data(), params_size);
    elf->set_param(params_buf, params_size);
    if (params_buf)
    {
        delete[] params_buf;
        params_buf = nullptr;
    }

    std::multimap<uint8_t, tensor_info_t> tt;
    for (auto t : m_graph->get_input_tensors())
    {
        if (!m_tensor_index_map.count(t))
        {
            ERROR() << t->name() << "not exist in index map";
            continue;
        }
        tensor_info_t info;
        info.id = m_tensor_index_map.at(t);
        info.dtype = t->type();
        info.shape = t->shape();
        tt.insert({TensorType::input, info});
    }
    for (auto t : m_graph->get_output_tensors())
    {
        if (!m_tensor_index_map.count(t))
        {
            ERROR() << t->name() << "not exist in index map";
            continue;
        }
        tensor_info_t info;
        info.id = m_tensor_index_map.at(t);
        info.dtype = t->type();
        info.shape = t->shape();
        tt.insert({TensorType::output, info});
    }

    auto param_info = gen_info_from_global_tensor("param");
    tt.insert({TensorType::param, param_info});

    auto stream0_info = gen_info_from_global_tensor("stream0");
    tt.insert({TensorType::stream, stream0_info});
    auto stream1_info = gen_info_from_global_tensor("stream1");
    tt.insert({TensorType::stream, stream1_info});
    auto stream2_info = gen_info_from_global_tensor("stream2");
    tt.insert({TensorType::stream, stream2_info});
    auto stream3_info = gen_info_from_global_tensor("stream3");
    tt.insert({TensorType::stream, stream3_info});

    elf->set_tensor_tab(tt);

    if (m_constant.size())
    {
        uint64_t size = 0;
        std::map<uint32_t, uint64_t> wt_tab;
        std::map<MemoryObjectPtr, uint64_t> assigned;

        for (auto t : m_constant)
        {
            auto ptr = t->memory_offset()->get_base_object();
            auto index = m_const_index_map.at(t);
            if (!assigned.count(ptr))
            {
                assigned[ptr] = size;
                size += ptr->size;
            }
            wt_tab[index] = assigned[ptr];
        }

        char *wt;
        wt = new char[size];
        memset(wt, 0, sizeof(char) * size);
        for (auto t : m_constant)
        {
            auto index = m_const_index_map.at(t);
            auto offset = wt_tab[index];
            memcpy(wt + offset, t->data<char>(), t->mem_size());
        }

        elf->set_data(wt, size);
        elf->set_weight_tab(wt_tab);

        if (wt)
        {
            delete[] wt;
            wt = nullptr;
        }
    }

    return true;
}
}; // namespace aipubt
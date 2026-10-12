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
 * @Date: 2026-04-24 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2026-04-24 14:27:33
 */
#include "builder/object_generator.h"
#include "core/config.h"
namespace aipubt
{

inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}
void ObjectGenerator::set_streams(const std::vector<std::vector<task_t>> &streams)
{
    for (size_t i = 0; i < streams.size(); ++i)
    {
        if (streams[i].empty())
        {
            continue;
        }
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

void ObjectGenerator::set_const_slots(const std::vector<MemoryObjectPtr> &slots)
{
    m_const_slots = slots;
}

void ObjectGenerator::set_constant(const std::vector<ConstTensorPtr> &constant)
{
    m_constant = constant;
}

void ObjectGenerator::set_param_offset_table(const std::map<uint32_t, uint32_t> &param_offset_table)
{
    m_param_offset_table = param_offset_table;
}

void ObjectGenerator::set_params(const BuilderParams &params)
{
    m_params.clear();
    m_params.reserve(params.size());
    for (auto &p : params)
    {
        m_params.push_back(p);
    }
}

void ObjectGenerator::set_tensor_index_map(const std::map<TensorPtr, uint32_t> &tensor_index_map)
{
    m_tensor_index_map = tensor_index_map;
}

void ObjectGenerator::set_graph(const GraphPtr g)
{
    m_graph = g;
}

bool ObjectGenerator::gen(CpsElfPtr elf)
{
    std::map<MemoryObjectPtr, buf_loc_t> buf_map;
    std::map<uint32_t, buf_loc_t> tensor_offset_map;
    for (size_t i = 0; i < m_slots.size(); ++i)
    {
        const auto &s = m_slots[i];
        elf->set_buffer(BUF_TYPE_SLOT, s->buffer_id, s->size, s->align, s->get_mem_id());
        DEBUG() << "set slot as buf " << (int)BUF_TYPE_SLOT << " index " << s->buffer_id << " size: 0x" << std::hex << s->size << " align: 0x" << s->align;
        buf_map[s] = buf_loc_t{BUF_TYPE_SLOT, (uint32_t)s->buffer_id, 0};
    }

    auto param_tensor = m_graph->global_tensors()["param"];
    auto param_ptr = param_tensor->memory_offset()->get_base_object();
    buf_map[param_ptr] = buf_loc_t{BUF_TYPE_PARAM, 0, 0};

    auto stream0_tensor = m_graph->global_tensors()["stream0"];
    auto stream0_ptr = stream0_tensor->memory_offset()->get_base_object();
    buf_map[stream0_ptr] = buf_loc_t{BUF_TYPE_STREAM, 0, 0};

    auto stream1_tensor = m_graph->global_tensors()["stream1"];
    auto stream1_ptr = stream1_tensor->memory_offset()->get_base_object();
    buf_map[stream1_ptr] = buf_loc_t{BUF_TYPE_STREAM, 1, 0};

    auto stream2_tensor = m_graph->global_tensors()["stream2"];
    auto stream2_ptr = stream2_tensor->memory_offset()->get_base_object();
    buf_map[stream2_ptr] = buf_loc_t{BUF_TYPE_STREAM, 2, 0};

    auto stream3_tensor = m_graph->global_tensors()["stream3"];
    auto stream3_ptr = stream3_tensor->memory_offset()->get_base_object();
    buf_map[stream3_ptr] = buf_loc_t{BUF_TYPE_STREAM, 3, 0};

    auto print_tensor = m_graph->global_tensors()["print"];
    auto print_ptr = print_tensor->memory_offset()->get_base_object();
    buf_map[print_ptr] = buf_loc_t{BUF_TYPE_PRINT, 0, 0};

    elf->set_param_tab(m_param_offset_table);

    elf->set_graph(m_streams);

    if (m_params.empty())
    {
        ERROR() << "params are empty!";
    }

    size_t params_size = m_params.size() * sizeof(uint32_t);
    char *params_buf = new char[params_size];
    std::memcpy(params_buf, m_params.data(), params_size);
    elf->set_buffer(BUF_TYPE_PARAM, 0, params_size, 1, -1, params_buf);
    DEBUG() << "set param as buf " << (int)BUF_TYPE_PARAM << " index 0 size: 0x" << std::hex << params_size;

    elf->set_buffer(BUF_TYPE_PRINT, 0, print_tensor->mem_size(), 8, -1);
    DEBUG() << "set print as buf " << (int)BUF_TYPE_PRINT << " index 0 size: 0x" << std::hex << print_tensor->mem_size();

    std::multimap<uint8_t, tensor_info_t> tt;
    for (auto t : m_graph->get_input_tensors())
    {
        auto ptr = t->memory_offset()->get_base_object();
        if (contains(m_const_slots, ptr))
        {
            continue;
        }

        if (!m_tensor_index_map.count(t))
        {
            WARN() << t->name() << " not exist in index map";
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
            WARN() << t->name() << " not exist in index map";
            continue;
        }
        tensor_info_t info;
        info.id = m_tensor_index_map.at(t);
        info.dtype = t->type();
        info.shape = t->shape();
        tt.insert({TensorType::output, info});
    }

    bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    if (enable_ds)
    {
        TensorPtr t = m_graph->global_tensors()["output_shape"];
        if (!m_tensor_index_map.count(t))
        {
            WARN() << t->name() << " not exist in index map";
        }
        tensor_info_t info;
        info.id = m_tensor_index_map.at(t);
        info.dtype = t->type();
        info.shape = t->shape();
        tt.insert({TensorType::output_shape, info});
    }

    elf->set_tensor_tab(tt);

    std::map<MemoryObjectPtr, char *> const_buf;
    for (auto s : m_const_slots)
    {
        char *wt;
        wt = new char[s->size];
        memset(wt, 0, sizeof(char) * s->size);
        const_buf[s] = wt;
    }

    for (auto t : m_constant)
    {
        auto s = t->memory_offset()->get_base_object();
        auto offset = t->offset();
        if (!const_buf.count(s))
        {
            ERROR();
        }
        auto wt = const_buf[s];
        if (offset + t->mem_size() > s->size)
        {
            ERROR();
        }
        memcpy(wt + offset, t->data<char>(), t->mem_size());
        // DEBUG() << "weight: " << t->name() << " in " << offset << " size 0x" << std::hex << t->mem_size();
    }

    for (auto buf : const_buf)
    {
        auto s = buf.first;
        auto wt = buf.second;
        elf->set_buffer(BUF_TYPE_WEIGHT, (uint32_t)s->buffer_id, s->size, s->align, s->get_mem_id(), wt);
        buf_map[s] = buf_loc_t{BUF_TYPE_WEIGHT, (uint32_t)s->buffer_id, s->size};
        DEBUG() << "set weight as buf " << (int)BUF_TYPE_WEIGHT << " index " << (uint32_t)s->buffer_id << " size: 0x" << std::hex << s->size;
    }

    for (auto &pair : m_tensor_index_map)
    {
        auto tensor = pair.first;
        auto tensor_idx = pair.second;
        auto base_ptr = tensor->memory_offset()->get_base_object();
        auto offset = tensor->offset();

        if (!buf_map.count(base_ptr))
        {
            ERROR() << "tensor " << tensor->name() << " is not assigned";
            continue;
        }

        auto buf_info = buf_map[base_ptr];

        DEBUG() << "tensor: " << tensor_idx << " " << tensor->name() << " is assigned buf " << (int)buf_info.type << " index " << buf_info.index << " offset 0x" << std::hex << offset << " size: 0x" << tensor->mem_size();
        tensor_offset_map[tensor_idx] = buf_loc_t{buf_info.type, buf_info.index, offset};
    }

    elf->set_tensor_offsets(tensor_offset_map);

    // std::map<int, std::string> totype = {{0, "param"}, {1, "weight"}, {2, "slot"}, {3, "stream"}};
    // for (auto pair : m_param_offset_table)
    // {
    //     auto param_offset = pair.first;
    //     auto tensor_idx = pair.second;
    //     TensorPtr tensor;
    //     for (auto &pair1 : m_tensor_index_map)
    //     {
    //         if (pair1.second == tensor_idx)
    //         {
    //             tensor = pair1.first;
    //             break;
    //         }
    //     }
    //     ConstTensorPtr const_tensor;
    //     for (auto &pair2 : m_const_index_map)
    //     {
    //         if (pair2.second == tensor_idx)
    //         {
    //             const_tensor = pair2.first;
    //             break;
    //         }
    //     }
    //     if (tensor)
    //     {
    //         auto buf = tensor_offset_map[tensor_idx];
    //         DEBUG() << "param offset: 0x" << std::hex << param_offset << " index: " << std::dec << tensor_idx << " tensor: " << tensor->name() << " buf: " << totype[buf.type] << buf.index << " buf offset: 0x" << std::hex << buf.offset;
    //     }
    //     else if (const_tensor)
    //     {
    //         auto buf = tensor_offset_map[tensor_idx];
    //         DEBUG() << "param offset: 0x" << std::hex << param_offset << " index: " << std::dec << tensor_idx << " tensor: " << const_tensor->name() << " buf: " << totype[buf.type] << buf.index << " buf offset: 0x" << std::hex << buf.offset;
    //     }
    //     else
    //     {
    //         ERROR();
    //     }
    // }

    elf->set_event_tab(m_streams);

    return true;
}
} // namespace aipubt
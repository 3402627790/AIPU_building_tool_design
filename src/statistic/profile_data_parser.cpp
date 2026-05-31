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
 * @Author: Lang Hu
 * @Date: 2023-07-05 15:36:34
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2023-09-13 14:15:07
 */
#include "profile_data_parser.h"
#include "builder/profiler.h"
#include "builder/target.h"
#include <cstdint>
#include <fstream>
#include <stack>

namespace aipubt
{

static std::string read_file(const std::string &file_path)
{
    std::ifstream ifs(file_path);
    if (!ifs)
    {
        DEBUG() << "Failed to read file: " << file_path;
        return std::string();
    }
    std::stringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

template <typename T>
static T read(const char *ptr, uint64_t offset = 0)
{
    return *reinterpret_cast<const T *>(ptr + offset);
}

ProfileDataParser::ProfileDataParser(const std::string &graph_json_path, const std::string &binary_path)
{
    auto json_content = read_file(graph_json_path);
    m_graph_json = json::parse(json_content);
    m_binary = read_file(binary_path);
}

static void get_actual_nodes(json &node, json &res)
{
    if (node["op_type"].get<std::string>() == "SubGraph")
    {
        if (node.contains("plugin") && node["plugin"].get<std::string>().size())
        {
            res.push_back(node);
        }
        else
        {
            for (auto &n : node["internal_nodes"])
            {
                get_actual_nodes(n, res);
            }
        }
    }
    else
    {
        res.push_back(node);
    }
}

bool ProfileDataParser::fetch_from_binary_x2(json &subgraphs)
{
    auto &auxiliary_info = m_graph_json["profile"]["auxiliary info"];
    auto subgraph_sect_offset = auxiliary_info["subgraph_sect_offset"].get<uint32_t>();
    auto tec_sect_offset = auxiliary_info["tec_sect_offset"].get<uint32_t>();
    auto aiff_sect_offset = auxiliary_info["aiff_sect_offset"].get<uint32_t>();
    auto dma_sect_offset = auxiliary_info["dma_sect_offset"].get<uint32_t>();
    auto custom_sect_offset = auxiliary_info["custom_sect_offset"].get<uint32_t>();
    auto subgraph_record_size = auxiliary_info["subgraph_record_size"].get<uint32_t>();
    auto aiff_dump_bytes = auxiliary_info["aiff_dump_bytes"].get<uint32_t>();
    auto tec_dump_bytes = auxiliary_info["tec_dump_bytes"].get<uint32_t>();
    auto dma_dump_bytes = auxiliary_info["dma_dump_bytes"].get<uint32_t>();
    auto node_dump_bytes = auxiliary_info["node_dump_bytes"].get<uint32_t>();
    auto custom_dump_bytes = auxiliary_info["custom_dump_bytes"].get<uint32_t>();
    auto aiff_area_size = auxiliary_info["aiff_area_size"].get<uint32_t>();
    auto tec_area_size = auxiliary_info["tec_area_size"].get<uint32_t>();
    auto dma_area_size = auxiliary_info["dma_area_size"].get<uint32_t>();
    auto custom_area_size = auxiliary_info["custom_area_size"].get<uint32_t>();
    auto core_num = auxiliary_info["core_num"].get<uint32_t>();
    auto tec_num = auxiliary_info["tec_num"].get<uint32_t>();

    auto &subgraph_order = auxiliary_info["subgraph_order"];
    auto &subgraph_levels = auxiliary_info["subgraph_levels"];
    if (!auxiliary_info.contains("tec_events"))
    {
        auxiliary_info["tec_events"] = json::array();
    }
    auto &tec_events = auxiliary_info["tec_events"];
    if (!auxiliary_info.contains("aiff_events"))
    {
        auxiliary_info["aiff_events"] = json::array();
    }
    auto &aiff_events = auxiliary_info["aiff_events"];

    auto version = auxiliary_info["graph_json_version"].get<std::string>();

    bool enable_fast_perf = false;
    if (auxiliary_info.contains("enable_fast_perf"))
    {
        enable_fast_perf = auxiliary_info["enable_fast_perf"].get<bool>();
    }
    if (auxiliary_info.contains("disable_dma_perf") && auxiliary_info["disable_dma_perf"].get<bool>())
    {
        DEBUG() << "GBuilder has disabled DMA perf as it may cost time and memory, "
                << "all data about DMA in the report would be zero. "
                << "If need DMA information, add --enable_dma_perf when using GBuilder.";
    }

    for (auto &n : m_graph_json["nodes"])
    {
        if (n.contains("perf_node") && n["perf_node"].get<bool>())
        {
            json nodes, perf_nodes;
            get_actual_nodes(n, nodes);
            n["actual_nodes"] = nodes;
            std::copy_if(nodes.begin(), nodes.end(), std::back_inserter(perf_nodes), [](json &o) -> bool {
                return o.contains("perf_node") && o["perf_node"].get<bool>();
            });
            n["actual_perf_nodes"] = perf_nodes;
            subgraphs.push_back(n);
        }
    }
    if (subgraphs.size() != subgraph_order.size())
    {
        DEBUG() << "Illegal graph json: len(subgraph) != len(subgraph_order)";
        return false;
    }
    std::sort(subgraphs.begin(), subgraphs.end(), [&subgraph_order](json &a, json &b) -> bool {
        auto get_order_index = [](json &node, json &order) -> int {
            int index = -1;
            for (uint32_t i = 0; i < order.size(); ++i)
            {
                if (order[i].get<std::string>() == node["layer node name"].get<std::string>())
                {
                    index = i;
                    break;
                }
            }
            if (index == -1)
            {
                DEBUG() << "Order index not found: " << node["layer node name"].get<std::string>();
            }
            return index;
        };
        return get_order_index(a, subgraph_order) < get_order_index(b, subgraph_order);
    });

    bool is_buffer_full = false;
    uint32_t max_perf_tec_count = 0;
    uint32_t max_perf_aiff_count = 0;
    uint32_t max_perf_dma_count = 0;
    uint32_t max_perf_custom_count = 0;
    std::vector<std::vector<json>> dma_data_list(core_num);
    for (auto &dma_data : dma_data_list)
    {
        dma_data = std::vector<json>(tec_num, json::array());
    }
    auto bin_buff = m_binary.c_str();
    for (uint32_t i = 0; i < subgraphs.size(); ++i)
    {
        auto &subgraph = subgraphs[i];
        subgraph["level"] = subgraph_levels[i];
        auto sg_buff = bin_buff + i * subgraph_record_size;
        auto sg_record_buff = sg_buff + subgraph_sect_offset;
        uint32_t core_id = read<uint32_t>(sg_record_buff, 0);
        uint64_t tick_start = read<uint64_t>(sg_record_buff, 4);
        uint32_t node_count = read<uint32_t>(sg_record_buff, 12);

        auto &actual_perf_nodes = subgraph["actual_perf_nodes"];
        if (actual_perf_nodes.size() != node_count)
        {
            DEBUG() << "Incompatiable perf node number in graph json and binary";
            return false;
        }
        json tick_end_list, read_end_list, write_end_list;
        json profile_data, tec_event_list;
        profile_data["tick_start"] = tick_start;
        auto profiler = AIPUBuilder::get_profiler();
        profiler->unregister_all();
        for (uint32_t j = 0; j < actual_perf_nodes.size(); ++j)
        {
            auto sg_data_buff = sg_record_buff + 16 + j * node_dump_bytes;
            uint64_t tick_end = read<uint64_t>(sg_data_buff, 0);
            uint32_t read_end = read<uint32_t>(sg_data_buff, 8);
            uint32_t write_end = read<uint32_t>(sg_data_buff, 12);
            int event_offset = 16;
            for (uint32_t tec_id = 0; tec_id < tec_num; ++tec_id)
            {
                for (auto &event : tec_events)
                {
                    auto event_id = event.get<uint32_t>();
                    auto key = "Tec" + std::to_string(tec_id) + " " + profiler->event_description(event_id);
                    if (!tec_event_list.contains(key))
                    {
                        tec_event_list[key] = json::array();
                    }
                    auto value = read<uint32_t>(sg_data_buff, event_offset);
                    tec_event_list[key].push_back(value);
                    event_offset += 4;
                }
            }
            tick_end_list.push_back(tick_end);
            read_end_list.push_back(read_end);
            write_end_list.push_back(write_end);
        }
        profile_data["core_id"] = core_id;
        profile_data["tick_end"] = tick_end_list;
        profile_data["read_end"] = read_end_list;
        profile_data["write_end"] = write_end_list;
        profile_data["tec_event_end"] = tec_event_list;

        json tec_list;
        for (uint32_t j = 0; j < tec_num; ++j)
        {
            tec_list.push_back(json::object());
        }
        auto tec_buff = sg_buff + tec_sect_offset;
        for (uint32_t tec_id = 0; tec_id < tec_num; ++tec_id)
        {
            auto tec_record_buff = tec_buff + tec_id * tec_area_size;
            uint32_t buffer_status = read<uint32_t>(tec_record_buff, 0);
            uint32_t tec_record_num = read<uint32_t>(tec_record_buff, 4);
            if (enable_fast_perf)
            {
                tec_record_num = 0;
            }
            if (buffer_status)
            {
                is_buffer_full = true;
                max_perf_tec_count = std::max(max_perf_tec_count, tec_record_num);
                continue;
            }
            tec_list[tec_id]["tec_record_num"] = tec_record_num;
            tec_list[tec_id]["tec_full"] = buffer_status;

            json record_list;
            for (uint32_t exec_id = 0; exec_id < tec_record_num; ++exec_id)
            {
                json tec_exec;
                auto tec_data_buff = tec_record_buff + 8 + exec_id * tec_dump_bytes;
                uint64_t tec_stamp = read<uint64_t>(tec_data_buff, 0);
                for (uint32_t j = 0; j < tec_events.size(); ++j)
                {
                    uint32_t id = tec_events[i];
                    auto value = read<uint32_t>(tec_data_buff, 8 + j * 4);
                    tec_exec[std::to_string(id)] = value;
                }
                tec_exec["stamp"] = tec_stamp;
                record_list.push_back(tec_exec);
            }
            tec_list[tec_id]["tec_record"] = record_list;
        }

        auto aiff_buff = sg_buff + aiff_sect_offset;
        for (uint32_t tec_id = 0; tec_id < tec_num; ++tec_id)
        {
            auto aiff_record_buff = aiff_buff + tec_id * aiff_area_size;
            uint32_t buffer_status = read<uint32_t>(aiff_record_buff, 0);
            uint32_t aiff_record_num = read<uint32_t>(aiff_record_buff, 4);
            if (enable_fast_perf)
            {
                aiff_record_num = 0;
            }
            if (buffer_status)
            {
                is_buffer_full = true;
                max_perf_aiff_count = std::max(max_perf_aiff_count, aiff_record_num);
                continue;
            }
            tec_list[tec_id]["aiff_record_num"] = aiff_record_num;
            tec_list[tec_id]["aiff_full"] = buffer_status;

            json record_list;
            for (uint32_t exec_id = 0; exec_id < aiff_record_num; ++exec_id)
            {
                json aiff_exec;
                auto aiff_data_buff = aiff_record_buff + 64 + exec_id * aiff_dump_bytes;

                auto aiff_start = read<uint64_t>(aiff_data_buff, 0);
                auto busy_cycle = read<uint32_t>(aiff_data_buff, 8);
                auto mtp0_busy_cycle = read<uint32_t>(aiff_data_buff, 12);
                auto mtp1_busy_cycle = read<uint32_t>(aiff_data_buff, 16);
                auto itp_busy_cycle = read<uint32_t>(aiff_data_buff, 20);
                auto ptp_busy_cycle = read<uint32_t>(aiff_data_buff, 24);
                auto wrb_busy_cycle = read<uint32_t>(aiff_data_buff, 28);
                auto rdma_weight_rd_cnt = read<uint32_t>(aiff_data_buff, 32);
                auto rdma_act_rd_cnt = read<uint32_t>(aiff_data_buff, 36);
                auto udma_param_rd_cnt = read<uint32_t>(aiff_data_buff, 40);
                auto udma_act_wr_cnt = read<uint32_t>(aiff_data_buff, 44);
                auto wdma_act_wr_cnt = read<uint32_t>(aiff_data_buff, 48);
                auto bif_rd_cnt = read<uint32_t>(aiff_data_buff, 52);
                auto bif_wr_cnt = read<uint32_t>(aiff_data_buff, 56);

                aiff_exec["start"] = aiff_start;
                aiff_exec["duration"] = busy_cycle;
                aiff_exec["mtp0_busy_cycle"] = mtp0_busy_cycle;
                aiff_exec["mtp1_busy_cycle"] = mtp1_busy_cycle;
                aiff_exec["itp_busy_cycle"] = itp_busy_cycle;
                aiff_exec["ptp_busy_cycle"] = ptp_busy_cycle;
                aiff_exec["wrb_busy_cycle"] = wrb_busy_cycle;
                aiff_exec["rdma_weight_read_count"] = rdma_weight_rd_cnt;
                aiff_exec["rdma_act_read_count"] = rdma_act_rd_cnt;
                aiff_exec["udma_param_read_count"] = udma_param_rd_cnt;
                aiff_exec["udma_act_write_count"] = udma_act_wr_cnt;
                aiff_exec["wdma_act_write_count"] = wdma_act_wr_cnt;
                aiff_exec["bif_read_count"] = bif_rd_cnt;
                aiff_exec["bif_write_count"] = bif_wr_cnt;
                if (aiff_events.size())
                {
                    aiff_exec["com0"] = read<uint32_t>(aiff_data_buff, 60);
                }
                if (aiff_events.size() > 1)
                {
                    aiff_exec["com1"] = read<uint32_t>(aiff_data_buff, 64);
                }
                record_list.push_back(aiff_exec);
            }
            tec_list[tec_id]["aiff_record"] = record_list;
        }

        auto dma_buff = sg_buff + dma_sect_offset;
        if (version == "1.0.0")
        {
            uint32_t axi_bandwidth = get_target()->axi_bandwidth();
            for (uint32_t tec_id = 0; tec_id < tec_num; ++tec_id)
            {
                auto dma_record_buff = dma_buff + tec_id * dma_area_size;
                uint32_t buffer_status = read<uint32_t>(dma_record_buff, 0);
                uint32_t dma_record_num = read<uint32_t>(dma_record_buff, 4);
                if (enable_fast_perf)
                {
                    dma_record_num = 0;
                }
                if (buffer_status)
                {
                    is_buffer_full = true;
                    max_perf_dma_count = std::max(max_perf_dma_count, dma_record_num);
                    continue;
                }
                tec_list[tec_id]["dma_record_num"] = dma_record_num;
                tec_list[tec_id]["dma_full"] = buffer_status;

                json record_list;
                for (uint32_t exec_id = 0; exec_id < dma_record_num; ++exec_id)
                {
                    json dma_exec;
                    auto dma_data_buff = dma_record_buff + 64 + exec_id * dma_dump_bytes;

                    auto dma_start = read<uint64_t>(dma_data_buff, 0);
                    auto busy_cycle = read<uint32_t>(dma_data_buff, 8);
                    auto read_num = read<uint32_t>(dma_data_buff, 16);
                    auto write_num = read<uint32_t>(dma_data_buff, 24);

                    dma_exec["start"] = dma_start;
                    dma_exec["duration"] = busy_cycle;
                    dma_exec["read_bytes"] = read_num * axi_bandwidth;
                    dma_exec["write_bytes"] = write_num * axi_bandwidth;
                    record_list.push_back(dma_exec);
                }
                tec_list[tec_id]["dma_record"] = record_list;
            }
        }
        else
        {
            DEBUG() << "Unsupported graph json version: " << version;
            return false;
        }

        auto custom_buff = sg_buff + custom_sect_offset;
        for (uint32_t tec_id = 0; tec_id < tec_num; ++tec_id)
        {
            auto custom_record_buff = custom_buff + tec_id * custom_area_size;
            uint32_t buffer_status = read<uint32_t>(custom_record_buff, 0);
            uint32_t custom_record_num = read<uint32_t>(custom_record_buff, 4);
            if (enable_fast_perf)
            {
                custom_record_num = 0;
            }
            if (buffer_status)
            {
                is_buffer_full = true;
                max_perf_custom_count = std::max(max_perf_custom_count, custom_record_num);
                continue;
            }
            tec_list[tec_id]["custom_record_num"] = custom_record_num;
            tec_list[tec_id]["custom_full"] = buffer_status;

            json custom_list;
            for (uint32_t exec_id = 0; exec_id < custom_record_num; ++exec_id)
            {
                auto custom_data_buff = custom_record_buff + 8 + exec_id * custom_dump_bytes;
                auto custom_event_id = read<uint32_t>(custom_data_buff, 0);
                auto custom_tick = read<uint64_t>(custom_data_buff, 4);
                auto key = std::to_string(custom_event_id / 2);
                if (!custom_list.contains(key))
                {
                    custom_list[key] = json::array();
                }
                json value;
                if (custom_event_id % 2 == 0)
                {
                    value["begin"] = custom_tick;
                }
                else
                {
                    value["end"] = custom_tick;
                }
                custom_list[key].push_back(value);
            }

            json custom_exec;
            for (auto &kv : custom_list.items())
            {
                custom_exec[kv.key()] = json::array();
                std::stack<json> begin_stack;
                for (auto &event : kv.value())
                {
                    if (event.contains("begin"))
                    {
                        begin_stack.push(event["begin"]);
                    }
                    else if (begin_stack.size())
                    {
                        auto tick_begin = begin_stack.top().get<uint64_t>();
                        begin_stack.pop();
                        auto tick_end = event["end"].get<uint64_t>();
                        uint32_t duration = tick_end - tick_begin;
                        json record;
                        record["start"] = tick_begin;
                        record["duration"] = duration;
                        custom_exec[kv.key()].push_back(record);
                    }
                }
            }
            tec_list[tec_id]["custom_record"] = custom_exec;
        }
        profile_data["tec_info"] = tec_list;
        subgraph["profile_data"] = profile_data;
    }

    if (is_buffer_full)
    {
        std::ostringstream oss;
        if (max_perf_aiff_count)
        {
            oss << "set max_aiff_sample_point_num to " << max_perf_aiff_count << ", ";
        }
        if (max_perf_dma_count)
        {
            oss << "set max_dma_sample_point_num to " << max_perf_dma_count << ", ";
        }
        if (max_perf_custom_count)
        {
            oss << "set max_custom_sample_point_num to " << max_perf_custom_count << ", ";
        }
        DEBUG() << "The default profile buffer is full, please " << oss.str() << "and run buildtool again";
        return false;
    }
    return true;
}

static uint64_t get_total_cycles(json &periods)
{
    if (!periods.size())
    {
        return 0;
    }
    std::sort(periods.begin(), periods.end(), [](json &a, json &b) -> bool {
        return a["start"].get<uint64_t>() < b["start"].get<uint64_t>();
    });
    auto start = periods[0]["start"].get<uint64_t>();
    auto duration = periods[0]["duration"].get<uint32_t>();
    uint64_t end = start + duration;
    uint64_t cycles = 0;
    for (uint32_t i = 1; i < periods.size(); ++i)
    {
        auto &period = periods[i];
        auto p_start = period["start"].get<uint64_t>();
        auto p_duration = period["duration"].get<uint32_t>();
        if (p_start <= end && p_start + p_duration >= end)
        {
            end = p_start + p_duration;
        }
        else if (p_start > end)
        {
            cycles += end - start;
            start = p_start;
            end = p_start + p_duration;
        }
    }
    cycles += end - start;
    return cycles;
}

static json get_tensor_shapes(json &graph_json, json &tensors)
{
    auto &tensor_info = graph_json["tensors"];
    json shapes;
    for (auto &name : tensors)
    {
        if (tensor_info.contains(name))
        {
            shapes.push_back(tensor_info[name]["shape"]);
        }
    }
    return shapes;
}

void ProfileDataParser::build_graph_info_x2(json &subgraphs, json &res)
{
    uint32_t tcb_switch_cycle = 0;
    for (uint32_t i = 1; i < subgraphs.size(); ++i)
    {
        auto &tick_end_list = subgraphs[i - 1]["profile_data"]["tick_end"];
        auto pre_end = tick_end_list[tick_end_list.size() - 1].get<uint64_t>();
        auto cur_start = subgraphs[i]["profile_data"]["tick_start"].get<uint64_t>();
        tcb_switch_cycle += cur_start - pre_end;
    }

    auto frequency = m_graph_json["profile"]["frequency"].get<uint32_t>();
    auto &auxiliary_info = m_graph_json["profile"]["auxiliary info"];
    auto tec_num = auxiliary_info["tec_num"].get<uint32_t>();
    auto core_num = auxiliary_info["core_num"].get<uint32_t>();
    uint32_t axi_bandwidth = get_target()->axi_bandwidth() / 8;
    if (!auxiliary_info.contains("aiff_events"))
    {
        auxiliary_info["aiff_events"] = json::array();
    }
    auto &aiff_events = auxiliary_info["aiff_events"];

    auto profiler = AIPUBuilder::get_profiler();
    std::vector<std::string> aiff_event_names;
    for (auto &event : aiff_events)
    {
        auto event_id = event.get<uint32_t>();
        aiff_event_names.push_back(profiler->event_description(event_id));
    }
    json total_sg_nodes;
    std::set<std::string> tec_event_names;
    for (auto &subgraph : subgraphs)
    {
        auto &perf_nodes = subgraph["actual_perf_nodes"];
        auto &profile_data = subgraph["profile_data"];
        auto tick_start = profile_data["tick_start"].get<uint64_t>();
        auto &tick_end_list = profile_data["tick_end"];
        auto &read_end_list = profile_data["read_end"];
        auto &write_end_list = profile_data["write_end"];
        auto &tec_event_end = profile_data["tec_event_end"];

        std::vector<uint64_t> tick_list(1, tick_start);
        for (auto &tick : tick_end_list)
        {
            tick_list.push_back(tick.get<uint64_t>());
        }
        std::vector<uint32_t> read_list(1, 0), write_list(1, 0);
        for (auto &count : read_end_list)
        {
            read_list.push_back(count.get<uint32_t>());
        }
        for (auto &count : write_end_list)
        {
            write_list.push_back(count.get<uint32_t>());
        }
        for (auto &kv : tec_event_end.items())
        {
            tec_event_names.insert(kv.key());
            kv.value().insert(kv.value().begin(), 0);
        }
        for (uint32_t i = 0; i < perf_nodes.size(); ++i)
        {
            auto &node = perf_nodes[i];
            node["actual_tick_start"] = tick_list[i];
            node["actual_tick_end"] = tick_list[i + 1];
            node["read_bytes"] = (read_list[i + 1] - read_list[i]) * axi_bandwidth;
            node["write_bytes"] = (write_list[i + 1] - write_list[i]) * axi_bandwidth;
            for (auto &kv : tec_event_end.items())
            {
                node[kv.key()] = kv.value()[i + 1].get<uint32_t>() - kv.value()[i].get<uint32_t>();
            }
        }

        json sg_node;
        sg_node["core_id"] = profile_data["core_id"];
        sg_node["level"] = subgraph["level"];
        for (auto &node : perf_nodes)
        {
            json tec_data_list;
            auto node_start = node["actual_tick_start"].get<uint64_t>();
            auto node_end = node["actual_tick_end"].get<uint64_t>();
            for (uint32_t tec_id = 0; tec_id < tec_num; ++tec_id)
            {
                json tec_data;
                auto &tec_item = profile_data["tec_info"][tec_id];
                auto &tec_record = tec_item["tec_record"];
                auto &aiff_record = tec_item["aiff_record"];
                auto &dma_record = tec_item["dma_record"];
                auto &custom_record = tec_item["custom_record"];

                json record_list;
                for (auto &record : tec_record)
                {
                    auto stamp = record["stamp"].get<uint64_t>();
                    if (stamp > node_start && stamp < node_end)
                    {
                        record_list.push_back(record);
                    }
                    else if (stamp > node_end)
                    {
                        break;
                    }
                }
                tec_data["tec"] = record_list;

                record_list = json::array();
                for (auto &record : aiff_record)
                {
                    auto start = record["start"].get<uint64_t>();
                    if (start > node_start && start < node_end)
                    {
                        record_list.push_back(record);
                    }
                    else if (start > node_end)
                    {
                        break;
                    }
                }
                tec_data["aiff"] = record_list;

                record_list = json::array();
                for (auto &record : dma_record)
                {
                    auto start = record["start"].get<uint64_t>();
                    if (start > node_start && start < node_end)
                    {
                        record_list.push_back(record);
                    }
                    else if (start > node_end)
                    {
                        break;
                    }
                }
                tec_data["dma"] = record_list;

                json custom_data;
                for (auto &kv : custom_record.items())
                {
                    custom_data[kv.key()] = json::array();
                    for (auto &record : kv.value())
                    {
                        auto start = record["start"].get<uint64_t>();
                        if (start > node_start && start < node_end)
                        {
                            custom_data[kv.key()].push_back(record);
                        }
                        else if (start > node_end)
                        {
                            break;
                        }
                    }
                }
                tec_data["custom"] = custom_data;
                tec_data_list.push_back(tec_data);
            }
            node["runtime_info"] = tec_data_list;
        }
        sg_node["nodes"] = perf_nodes;
        total_sg_nodes.push_back(sg_node);
    }

    uint64_t tick_start = UINT64_MAX;
    uint64_t tick_end = 0;
    uint64_t total_dma_cycles = 0;
    uint64_t total_aiff_cycles = 0;
    uint32_t total_read = 0;
    uint32_t total_write = 0;
    uint32_t total_dma_read = 0;
    uint32_t total_dma_write = 0;
    uint32_t node_num = 0;
    double total_ops = 0;
    json sg_info_list;
    for (auto &sg_node : total_sg_nodes)
    {
        json sg_info, node_info_list;
        sg_info["level"] = sg_node["level"];
        auto core_id = sg_node["core_id"].get<uint32_t>();
        for (auto &node : sg_node["nodes"])
        {
            json node_info;
            for (auto &event_name : tec_event_names)
            {
                if (node.contains(event_name))
                {
                    node_info[event_name] = node[event_name];
                }
            }
            auto node_start = node["actual_tick_start"].get<uint64_t>();
            auto node_end = node["actual_tick_end"].get<uint64_t>();
            uint64_t cycles = node_end - node_start;
            node_info["start"] = node_start;
            node_info["end"] = node_end;
            node_info["cycles"] = cycles;
            node_info["cycle_ratio"] = 0;
            tick_start = std::min(node_start, tick_start);
            tick_end = std::max(node_end, tick_end);

            uint32_t mtp0_busy_cycle = 0, mtp1_busy_cycle = 0;
            uint32_t itp_busy_cycle = 0, ptp_busy_cycle = 0;
            uint32_t com0 = 0, com1 = 0;
            json periods;
            auto tec_data_list = node["runtime_info"];
            for (auto &tec_data : tec_data_list)
            {
                for (auto &record : tec_data["aiff"])
                {
                    mtp0_busy_cycle += record["mtp0_busy_cycle"].get<uint32_t>();
                    mtp1_busy_cycle += record["mtp1_busy_cycle"].get<uint32_t>();
                    itp_busy_cycle += record["itp_busy_cycle"].get<uint32_t>();
                    ptp_busy_cycle += record["ptp_busy_cycle"].get<uint32_t>();
                    if (record.contains("com0"))
                    {
                        com0 += record["com0"].get<uint32_t>();
                    }
                    if (record.contains("com1"))
                    {
                        com1 += record["com1"].get<uint32_t>();
                    }
                    periods.push_back(record);
                }
            }
            uint64_t aiff_cycles = get_total_cycles(periods);
            node_info["aiff_busy_cycles"] = aiff_cycles;
            node_info["aiff_utilization"] = (double)aiff_cycles / (double)cycles * 100;
            node_info["mtp0_busy_cycles"] = mtp0_busy_cycle;
            node_info["mtp0_utilization"] = (double)mtp0_busy_cycle / (double)cycles * 100;
            node_info["mtp1_busy_cycles"] = mtp1_busy_cycle;
            node_info["mtp1_utilization"] = (double)mtp1_busy_cycle / (double)cycles * 100;
            node_info["itp_busy_cycles"] = itp_busy_cycle;
            node_info["itp_utilization"] = (double)itp_busy_cycle / (double)cycles * 100;
            node_info["ptp_busy_cycles"] = ptp_busy_cycle;
            node_info["ptp_utilization"] = (double)ptp_busy_cycle / (double)cycles * 100;
            total_aiff_cycles += aiff_cycles;

            periods = json::array();
            for (auto &tec_data : tec_data_list)
            {
                for (auto &record : tec_data["dma"])
                {
                    periods.push_back(record);
                }
            }
            uint64_t dma_cycles = get_total_cycles(periods);
            uint32_t dma_read_bytes = 0;
            uint32_t dma_write_bytes = 0;
            for (auto &period : periods)
            {
                dma_read_bytes += period["read_bytes"].get<uint32_t>();
                dma_write_bytes += period["write_bytes"].get<uint32_t>();
            }
            node_info["dma_busy_cycles"] = dma_cycles;
            node_info["dma_utilization"] = (double)dma_cycles / (double)cycles * 100;
            node_info["dma_read_bytes"] = dma_read_bytes;
            node_info["dma_write_bytes"] = dma_write_bytes;
            total_dma_cycles += dma_cycles;
            total_dma_read += dma_read_bytes;
            total_dma_write += dma_write_bytes;

            node_info["layer_name"] = node["layer node name"];
            node_info["op_type"] = node["op_type"];
            if (node["plugin"].get<std::string>().empty())
            {
                node_info["plugin"] = node["op_type"];
            }
            else
            {
                node_info["plugin"] = node["plugin"];
            }
            node_info["input_shapes"] = get_tensor_shapes(m_graph_json, node["inputs"]);
            node_info["output_shapes"] = get_tensor_shapes(m_graph_json, node["outputs"]);
            if (!node.contains("internal_nodes"))
            {
                node["internal_nodes"] = json::array();
            }
            node_info["internal_nodes"] = node["internal_nodes"];
            node_info["core_id"] = core_id;
            node_info["runtime_info"] = tec_data_list;

            auto read_bytes = node["read_bytes"].get<uint32_t>();
            auto write_bytes = node["write_bytes"].get<uint32_t>();
            node_info["aipu_read_bytes"] = read_bytes;
            node_info["aipu_write_bytes"] = write_bytes;
            node_info["BW(GB/s)"] = (read_bytes + write_bytes) * (double)frequency / (cycles * 1000);
            total_read += read_bytes;
            total_write += write_bytes;

            double ops = 0;
            if (node.contains("ops_count"))
            {
                ops += node["ops_count"].get<uint32_t>();
            }
            node_info["algorithm_ops(million)"] = ops / 1000000;
            total_ops += ops;

            if (aiff_event_names.size())
            {
                node_info[aiff_event_names[0]] = com0;
            }
            if (aiff_event_names.size() > 1)
            {
                node_info[aiff_event_names[1]] = com1;
            }
            node_info_list.push_back(node_info);
            node_num += 1;
        }
        sg_info["nodes"] = node_info_list;
        sg_info_list.push_back(sg_info);
    }
    uint64_t total_cycles = tick_end - tick_start;
    for (auto &sg_info : sg_info_list)
    {
        for (auto &node_info : sg_info["nodes"])
        {
            node_info["cycle_ratio"] = node_info["cycles"].get<uint64_t>() / (double)total_cycles * 100;
        }
    }
    double cost_time = total_cycles / (double)(1000 * frequency);
    res["target"] = m_graph_json["platform target"];
    res["tec_num"] = tec_num;
    res["core_num"] = core_num;
    res["subgraphs"] = sg_info_list;
    res["subgraph_switch_cycles"] = tcb_switch_cycle;
    res["total_cycles"] = total_cycles;
    res["total_time"] = cost_time;
    res["fps"] = 1000 / cost_time;
    res["total_dma_busy_cycles"] = total_dma_cycles;
    res["total_aiff_busy_cycles"] = total_aiff_cycles;
    res["aipu_read_bytes"] = total_read;
    res["aipu_write_bytes"] = total_write;
    res["dma_read_bytes"] = total_dma_read;
    res["dma_write_bytes"] = total_dma_write;
    res["subgraph_num"] = sg_info_list.size();
    res["layer_num"] = node_num;
    res["total_ops(million)"] = total_ops / 1000000;
}

void ProfileDataParser::parse(json &res)
{
    res = json();
    if (!m_binary.size())
    {
        return;
    }
    auto target = m_graph_json["platform target"].get<std::string>();
    if (target.rfind("X2", 0) != 0)
    {
        DEBUG() << "Unsupported target: " << target;
        return;
    }
    json subgraphs;
    if (!fetch_from_binary_x2(subgraphs))
    {
        DEBUG() << "Failed to fetch from binary file";
        return;
    }
    build_graph_info_x2(subgraphs, res);
}

} // namespace aipubt

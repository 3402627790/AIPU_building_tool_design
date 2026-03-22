//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2021 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Yuchou Gan
 * @Date: 2023-04-10 18:17:36
 * @Last Modified by: Yuchou Gan
 * @Last Modified time: 2023-04-10 18:17:36
 */

#include "json.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
using namespace nlohmann;

namespace perfdata {
static std::map<std::string, std::vector<uint32_t>> x1_default_events = {
    {"default", {0, 64, 82, 108, 110, 112, 120, 123, 256, 257, 258, 259, 265, 266, 267}},
    {"TPC", {0, 64, 82, 108, 110, 112, 120, 123, 256, 257, 258, 259, 265, 266, 267}},
    {"AIFF", {0, 64, 82, 108, 110, 112, 120, 123, 256, 257, 258, 259, 265, 266, 267}},
    {"DMA", {0, 64, 82, 108, 110, 112, 120, 123, 256, 257, 258, 259, 265, 266, 267}},
};


template<typename T>
T get_value(const char* binary, ssize_t offset)
{
    return *reinterpret_cast<const T*>((binary + offset));
}

std::string get_file_content(const std::string& path)
{
    std::ifstream ss(path);
    if (!ss)
    {
        std::cerr << "Cannot Open file: " << path << std::endl;
        return "";
    }
    std::string file_str((std::istreambuf_iterator<char>(ss)),
                 std::istreambuf_iterator<char>());
    return file_str;
}


std::vector<json> get_op_nodes(const json& node) {
    std::vector<json> ret;
    std::function<void(const json&)> internal_get;
    internal_get = [&] (const json& internal_node) {
        if (internal_node["op_type"] == "SubGraph") {
            if (internal_node.contains("plugin") && 
                internal_node["plugin"] != "" &&
                internal_node["plugin"] != 0) {
                    ret.push_back(internal_node);
            }
            else
            {
                for (auto& n : internal_node["internal_nodes"]) {
                    internal_get(n);
                }
            }
        } else {
            ret.push_back(internal_node);
        }
    };
    internal_get(node);
    return ret;
};

int fetch_info_from_binary(const json& graph_json, const std::string& binary, json& ret)
{
    json info = graph_json["profile"];
    std::string p_events = info["profile events"];
    if (p_events != "default")
    {
        std::cerr << "Only support default mode" << std::endl;
        return -1;
    }
    std::string unit = info["prof_unit"];
    std::vector<uint32_t>& events = x1_default_events[unit];
    std::vector<uint32_t> normal_events, special_events;
    std::copy_if(events.begin(), events.end(), std::back_inserter(normal_events), [](uint32_t i) {
        return i < 256;
    });
    std::copy_if(events.begin(), events.end(), std::back_inserter(special_events), [](uint32_t i) {
        return i >= 256;
    });

    std::vector<json> perf_nodes;
    for (auto& node : graph_json["nodes"])
    {
        if (node.contains("perf_node") && node["perf_node"] == true)
        {
            auto actual_node = get_op_nodes(node);
            perf_nodes.insert(perf_nodes.end(), actual_node.begin(), actual_node.end());
        }
    }

    uint32_t perf_nodes_num = perf_nodes.size();
    json& ret_nodes = ret["profile_data"];

    const int head_size = 24;

    const char* bin = binary.c_str();
    uint32_t current_offset = get_value<uint32_t>(bin, 4);
    uint32_t node_bin_size = 8 * 8 + special_events.size() * 4;
    if ((current_offset - head_size) % node_bin_size != 0)
    {
        std::cerr << "The binary not match graph.json content, maybe it's broken;" << std::endl;
        return -1;
    }
    uint32_t runtime_perf_num = (current_offset - head_size) / node_bin_size;
    if (runtime_perf_num != perf_nodes_num)
    {
        std::cerr << "The binary record " << runtime_perf_num 
            << " nodes whereas the graph.json perfs " 
            << perf_nodes_num << " nodes, they are not match." 
            << std::endl;
        return -1;
    }

    const uint32_t header_size = 24;
    for (uint32_t id = 0; id < perf_nodes_num; id++)
    {
        json ret_dict;
        const json& node = perf_nodes[id];
        ret_dict["name"] = node["layer node name"];
        json& info = ret_dict["runtime_information"];

        const char* node_bin = bin + id * node_bin_size + header_size;
        for (uint32_t idx = 0; idx < normal_events.size(); idx++)
        {
            uint64_t val = get_value<uint64_t>(node_bin, sizeof(uint64_t) * idx);
            info[std::to_string(normal_events[idx])] = val;
        }
        const char* node_bin2 = node_bin + 8 * sizeof(uint64_t);
        for (uint32_t idx = 0; idx < special_events.size(); idx++)
        {
            uint32_t val = get_value<uint32_t>(node_bin2, sizeof(uint32_t) * idx);
            info[std::to_string(special_events[idx])] = val;
        }
        ret_nodes.push_back(ret_dict);
    }

    for (int32_t node_id = perf_nodes_num - 1; node_id > 0; node_id--)
    {
        for (uint32_t eid = 0; eid < normal_events.size(); eid++)
        {
            std::string eid_str = std::to_string(normal_events[eid]);
            uint64_t val = ret_nodes[node_id]["runtime_information"][eid_str];
            uint64_t val_pre = ret_nodes[node_id - 1]["runtime_information"][eid_str];
            ret_nodes[node_id]["runtime_information"][eid_str] = val - val_pre;
        }
    }

    // finetune the result
    for (uint32_t id = 0; id < perf_nodes_num; id++)
    {
        json& node = ret_nodes[id]["runtime_information"];
        if (node.count("0"))
        {
            uint64_t cycle_count = node["0"];
            node.erase("0");
            node["count_cycle"] = cycle_count;
        }

        if (node.count("64"))
        {
            uint64_t TPC_credit_cycle = node["64"];
            node.erase("64");
            node["TPC_credit_cycle"] = TPC_credit_cycle;
        }

        if (node.count("82"))
        {
            uint64_t AIFF_busy_cycle = node["82"];
            node.erase("82");
            node["AIFF_busy_cycle"] = AIFF_busy_cycle;
        }

        if (node.count("108"))
        {
            uint64_t DMA_busy_cycle = node["108"];
            node.erase("108");
            node["DMA_busy_cycle"] = DMA_busy_cycle;
        }

        const uint64_t axi_width = 256 / 8;
        if (node.count("110"))
        {
            uint64_t DMA_read_bytes = (uint64_t)node["110"] * axi_width;
            node.erase("110");
            node["DMA_read_bytes"] = DMA_read_bytes;
        }

        if (node.count("112"))
        {
            uint64_t DMA_write_bytes = (uint64_t)node["112"] * axi_width;
            node.erase("112");
            node["DMA_write_bytes"] = DMA_write_bytes;
        }

        if (node.count("120"))
        {
            uint64_t AIPU_read_bytes = (uint64_t)node["120"] * axi_width;
            node.erase("120");
            node["AIPU_read_bytes"] = AIPU_read_bytes;
        }

        if (node.count("123"))
        {
            uint64_t AIPU_write_bytes = (uint64_t)node["123"] * axi_width;
            node.erase("123");
            node["AIPU_write_bytes"] = AIPU_write_bytes;
        }

        if (node.count("256"))
        {
            uint64_t MTP0_busy_cycle = (uint64_t)node["256"];
            node.erase("256");
            node["MTP0_busy_cycle"] = MTP0_busy_cycle;
        }

        if (node.count("257"))
        {
            uint64_t MTP1_busy_cycle = (uint64_t)node["257"];
            node.erase("257");
            node["MTP1_busy_cycle"] = MTP1_busy_cycle;
        }

        if (node.count("258"))
        {
            uint64_t ITP_busy_cycle = (uint64_t)node["258"];
            node.erase("258");
            node["ITP_busy_cycle"] = ITP_busy_cycle;
        }

        if (node.count("259"))
        {
            uint64_t PTP_busy_cycle = (uint64_t)node["259"];
            node.erase("259");
            node["PTP_busy_cycle"] = PTP_busy_cycle;
        }

        if (node.count("265"))
        {
            uint64_t RDMA_WT_RD_CNT = (uint64_t)node["265"];
            node.erase("265");
            node["RDMA_WT_RD_CNT"] = RDMA_WT_RD_CNT;
        }

        if (node.count("266"))
        {
            uint64_t RDMA_ACT_RD_CNT = (uint64_t)node["266"];
            node.erase("266");
            node["RDMA_ACT_RD_CNT"] = RDMA_ACT_RD_CNT;
        }

        if (node.count("267"))
        {
            uint64_t WDMA_ACT_WR_CNT = (uint64_t)node["267"];
            node.erase("267");
            node["WDMA_ACT_WR_CNT"] = WDMA_ACT_WR_CNT;
        }
    }
    return 0;
}


void buildup_graph_info(json& graph_json, json& ret)
{
    ret["graph_name"] = graph_json["graph_name"];
    ret["frequency"] = graph_json["profile"]["frequency"];
    std::vector<json> all_nodes;
    for (auto& n : graph_json["nodes"])
    {
        std::vector<json> nodes = get_op_nodes(n);
        all_nodes.insert(all_nodes.end(), nodes.begin(), nodes.end());
    }
    json& graph = ret["graph"];
    json& graph_nodes = graph["nodes"];
    for(auto& n : all_nodes)
    {
        std::string name = n["layer node name"];
        graph_nodes[name] = n;
    }
    graph["tensors"] = graph_json["tensors"];
}

json get_performance_data(std::string graph_json_path, std::string binary_path)
{
    auto json_str = get_file_content(graph_json_path);
    auto bin_str = get_file_content(binary_path);
    if (json_str.size() == 0 || bin_str.size() == 0)
    {
        return json();
    }

    json graph_json = json::parse(json_str);
    std::string target = graph_json["platform target"];
    if (target.rfind("X1", 0) != 0)
    {
        std::cerr << "Only support X1 target" << std::endl;
        return json();
    }
    json out;
    int ret = fetch_info_from_binary(graph_json, bin_str, out);
    if (ret != 0)
    {
        return json();
    }
    buildup_graph_info(graph_json, out);
    return out;
}
}
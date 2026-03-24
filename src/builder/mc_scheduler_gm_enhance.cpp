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
 * @Date: 2023-07-13 14:27:33 
 * @Last Modified by: Kyle Huang 
 * @Last Modified time: 2023-07-13 14:27:33 
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "mc_scheduler.h"
#include "utils/utils.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>
namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}
static bool kernel_empty(NodePtr node)
{
    if (node->attrs().count(Constants::plugin_attrs_name))
    {
        return false;
    }
    return true;
}
static int get_load_from_model(NodePtr node)
{
    //should use performance model to get load
    int load = -1;
    if (node->attrs().count("count_cycles"))
    {
        load = node->get_attr("count_cycles");
        // INFO() << node->name() << "load: " << load;
    }

    if (load == -1)
    {
        WARN() << "no load for " << node->name() << " type: " << node->type();
        load = 1;
    }

    return load;
}

static int get_dma_load(uint32_t size)
{
    int load = 0;
    load = 450 + 0.13 * size;
    return load;
}

static int get_total_constant(NodePtr node)
{
    int total_constant_size = 0;
    for (auto cts : node->constants())
    {
        auto t = std::make_shared<Tensor>(*cts.second);
        total_constant_size += t->memory_offset()->size;
        total_constant_size = aligned(total_constant_size, t->memory_offset()->get_base_object()->align);
    }
    return total_constant_size;
}

static int get_cal_intense(NodePtr node)
{
    int cal_intense = 0;
    int total_constant_size = get_total_constant(node);
    int ops = node->get_attr<Expr>("ops_count")->value();
    cal_intense = total_constant_size ? ops / total_constant_size : 0;
    return cal_intense;
}

static bool is_memory_bound(NodePtr node, int parallel_cal_num)
{
    uint32_t cal_intense = get_cal_intense(node);
    uint32_t max_cal = Target::get_the_target()->ff_mac_num() * 1.25;
    uint32_t max_bw = Target::get_the_target()->axi_bandwidth() / 8 / parallel_cal_num;
    DEBUG() << "cal intense for " << node << " " << cal_intense << " max_cal " << max_cal << " max_bw " << max_bw;
    if (cal_intense > (max_cal / max_bw))
    {
        return false;
    }
    return true;
}

static bool connect_same_w_node(std::vector<NodePtr> nodes)
{
    std::map<std::string, NodePtr> hex_map;
    bool all_same_w = true;
    for (auto n : nodes)
    {
        std::string total_hex;
        for (auto kv : n->constants())
        {
            // avoid nullptr
            if (!kv.second || !kv.second->data<void>())
            {
                continue;
            }
            auto hash = utils::sha256_hex(kv.second->data<void>(), kv.second->mem_size());
            total_hex = total_hex + hash;
        }
        auto it = hex_map.find(total_hex);
        if (it != hex_map.end())
        {
            // auto first_node = it->second;
            // if (first_node->attrs().count(Constants::same_w))
            // {
            //     auto next_node = first_node->get_attr<NodeRef>(Constants::same_w).lock();
            //     n->set_attr(Constants::same_w, NodeRef(next_node));
            // }
            // else
            // {
            //     n->set_attr(Constants::same_w, NodeRef(first_node));
            // }
            // first_node->set_attr(Constants::same_w, n);
            // INFO() << "same node: " << n;
            continue;
        }
        else
        {
            if (!hex_map.empty())
            {
                all_same_w = false;
            }
            hex_map.insert(std::make_pair(total_hex, n));
        }
    }
    return all_same_w;
}

static std::vector<MemoryObjectPtr> get_available_segment(const std::vector<MemoryObjectPtr> &taken_segments, MemoryObjectPtr gm_slot)
{
    std::vector<MemoryObjectPtr> ordered_segments;

    std::copy_if(taken_segments.begin(), taken_segments.end(), std::back_inserter(ordered_segments), [](MemoryObjectPtr v) { return (v->size != 0); });

    std::stable_sort(ordered_segments.begin(), ordered_segments.end(), [](const MemoryObjectPtr &a, const MemoryObjectPtr &b) {
        return a->relative_offset < b->relative_offset;
    });

    std::vector<MemoryObjectPtr> available_segments;

    auto segment = std::make_shared<MemoryObject>(0);
    segment->set_base_offset(gm_slot);
    segment->relative_offset = 0;
    segment->size = 0;

    for (auto seg : ordered_segments)
    {
        if (segment->relative_offset < seg->relative_offset)
        {
            segment->size = seg->relative_offset - segment->relative_offset;
            available_segments.push_back(segment);
            auto new_segment = std::make_shared<MemoryObject>(0);
            segment = new_segment;
            segment->set_base_offset(gm_slot);
            segment->relative_offset = seg->relative_offset + seg->size;
            segment->size = 0;
        }
        else
        {
            segment->relative_offset = seg->relative_offset + seg->size;
        }
    }

    if (segment->relative_offset < gm_slot->size)
    {
        segment->size = gm_slot->size - segment->relative_offset;
        available_segments.push_back(segment);
    }

    return available_segments;
}

static bool check_segments(const std::vector<MemoryObjectPtr> &segments)
{
    if (segments.size() == 0)
    {
        return true;
    }

    std::vector<MemoryObjectPtr> ordered_segments;

    std::copy_if(segments.begin(), segments.end(), std::back_inserter(ordered_segments), [](MemoryObjectPtr v) { return (v->size != 0); });

    std::stable_sort(ordered_segments.begin(), ordered_segments.end(), [](const MemoryObjectPtr &a, const MemoryObjectPtr &b) {
        return a->relative_offset < b->relative_offset;
    });

    auto tmp_seg = std::make_shared<MemoryObject>(0);
    tmp_seg->set_base_offset(segments[0]);
    tmp_seg->relative_offset = 0;
    tmp_seg->size = 0;
    for (auto seg : ordered_segments)
    {
        if (seg->offset() >= tmp_seg->relative_offset + tmp_seg->size)
        {
            tmp_seg->size = seg->offset() + seg->size;
        }
        else
        {
            WARN() << "segments are overlapped!";
        }
    }
    return true;
}
// static bool remove_segment(std::vector<MemoryObjectPtr> &segments, MemoryObjectPtr removed_seg)
// {
//     if (removed_seg->size == 0)
//     {
//         return false;
//     }
//     auto seg = std::make_shared<MemoryObject>(0);
//     seg->set_base_offset(removed_seg->get_base_object());
//     seg->relative_offset = removed_seg->offset();
//     seg->size = removed_seg->size;

//     for (int i = 0; i < (int)segments.size(); ++i)
//     {
//         auto s = segments[i];
//         if (seg->relative_offset >= s->relative_offset && seg->relative_offset < s->relative_offset + s->size)
//         {
//             if (seg->relative_offset + seg->size == s->relative_offset + s->size)
//             {
//                 s->size = s->size - seg->size;
//             }
//             else if (s->relative_offset + s->size > seg->relative_offset + seg->size)
//             {
//                 s->size = seg->relative_offset - s->relative_offset;
//                 auto new_seg = std::make_shared<MemoryObject>(0);
//                 new_seg->set_base_offset(s->get_base_object());
//                 new_seg->relative_offset = seg->relative_offset + seg->size;
//                 new_seg->size = s->relative_offset + s->size - (seg->relative_offset + seg->size);
//                 segments.push_back(new_seg);
//             }
//             else
//             {
//                 s->size = seg->relative_offset - s->relative_offset;
//             }

//             if (s->size == 0)
//             {
//                 segments.erase(segments.begin() + i);
//             }

//             break;
//         }
//     }
//     return true;
// }
static bool add_segment(std::vector<MemoryObjectPtr> &segments, MemoryObjectPtr added_seg)
{
    if (added_seg->size == 0)
    {
        return false;
    }
    auto seg = std::make_shared<MemoryObject>(0);
    seg->set_base_offset(added_seg->get_base_object());
    seg->relative_offset = added_seg->offset();
    seg->size = added_seg->size;

    for (int i = 0; i < (int)segments.size(); ++i)
    {
        bool overlapped = false;
        auto s = segments[i];
        if (s->relative_offset == seg->relative_offset)
        {
            uint32_t size = std::max(s->size, seg->size);
            seg->size = size;
            overlapped = true;
        }
        else if (s->relative_offset > seg->relative_offset)
        {
            if (seg->relative_offset + seg->size >= s->relative_offset + s->size)
            {
                overlapped = true;
            }
            else if (seg->relative_offset + seg->size >= s->relative_offset)
            {
                seg->size = s->relative_offset - seg->relative_offset + s->size;
                overlapped = true;
            }
        }
        else //s->offset() < seg->offset()
        {
            if (s->relative_offset + s->size >= seg->relative_offset + seg->size)
            {
                seg->relative_offset = s->relative_offset;
                seg->size = s->size;
                overlapped = true;
            }
            else if (s->relative_offset + s->size >= seg->relative_offset)
            {
                seg->size = seg->relative_offset - s->relative_offset + seg->size;
                seg->relative_offset = s->relative_offset;
                overlapped = true;
            }
        }
        if (overlapped)
        {
            segments.erase(segments.begin() + i);
            i--;
        }
    }

    segments.push_back(seg);

    return true;
}
static std::vector<std::vector<MemoryObjectPtr>> generate_liveness_mem(std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> liveness)
{
    uint32_t max_clock = 100000; //clock shouldn't larger than 10000;
    std::vector<std::vector<MemoryObjectPtr>> liveness_mem;
    std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>>::iterator it;
    uint32_t infi_clock = 0;
    for (it = liveness.begin(); it != liveness.end(); ++it)
    {
        auto value = it->second;
        if (value.first > infi_clock)
            infi_clock = value.first;
    }
    infi_clock++;
    for (it = liveness.begin(); it != liveness.end(); ++it)
    {
        auto value = it->second;
        if (value.second < max_clock && value.second > infi_clock)
            infi_clock = value.second;
    }
    infi_clock++;
    for (it = liveness.begin(); it != liveness.end(); ++it)
    {
        auto key = it->first;
        auto value = it->second;
        // DEBUG() << "TEST 2 mem: " << key << " live: " << value.first << " " << value.second;
        if (value.second > max_clock)
            value.second = infi_clock;
        if (liveness_mem.size() < value.second + 1)
            liveness_mem.resize(value.second + 1);
        for (size_t j = value.first; j < value.second + 1; ++j)
        {
            liveness_mem[j].push_back(it->first);
        }
    }
    return liveness_mem;
}

static bool generate_gm_available_segments(GraphPtr g, const std::vector<std::vector<NodePtr>> &levels, MemoryObjectPtr gm_slot, const std::map<GraphPtr, std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>>> &m_liveness_map, std::map<NodePtr, std::vector<MemoryObjectPtr>> &gm_available_segments, std::vector<std::vector<MemoryObjectPtr>> &gm_level_share_segments, std::map<NodePtr, uint32_t> &node_to_level)
{
    std::vector<MemoryObjectPtr> core_segments;
    std::vector<std::vector<MemoryObjectPtr>> level_segments;
    std::vector<MemoryObjectPtr> gm_taken_segments;

    std::map<MemoryObjectPtr, TensorPtr> tensor_map;
    for (auto level : levels)
    {
        for (auto node : level)
        {
            auto sgn = SubGraphNode::from_node(node);
            for (auto sg_node : sgn->subgraph()->nodes())
            {
                for (auto t : sg_node->inputs())
                {
                    auto ptr = t->memory_offset();
                    tensor_map[ptr] = t;
                }
                for (auto t : sg_node->outputs())
                {
                    auto ptr = t->memory_offset();
                    tensor_map[ptr] = t;
                }
            }
        }
    }

    INFO() << "gm offset: " << gm_slot->offset() << " gm size: " << gm_slot->size;
    auto target = Target::get_the_target();

    auto top_it = m_liveness_map.find(g);
    std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> top_liveness;
    if (top_it != m_liveness_map.end())
    {
        top_liveness = top_it->second;
    }
    else
    {
        WARN() << "no top liveness";
    }
    auto top_liveness_mem = generate_liveness_mem(top_liveness);
    INFO() << "top liveness mem:";
    for (size_t z = 0; z < top_liveness_mem.size(); ++z)
    {
        auto mem_vec = top_liveness_mem[z];
        INFO() << "clock: " << z;
        for (auto mem : mem_vec)
        {
            INFO() << "mem in clock: " << mem;
            if (tensor_map.find(mem) != tensor_map.end())
            {
                INFO() << "tensor in clock: " << tensor_map[mem];
            }
            else
            {
                INFO() << "no tensor";
            }
        }
    }
    INFO() << "top liveness mem done";

    for (size_t z = 0; z < levels.size(); ++z)
    {
        INFO() << "level " << z;
        auto level = levels[z];
        if (level.size() > target->core_num())
        {
            WARN() << z << " sg num(" << level.size() << ") larger than core num(" << target->core_num() << "), this may cause gm data overlap!";
        }
        gm_taken_segments.clear();
        level_segments.clear();
        uint32_t top_mem_clock = 0;
        for (size_t i = 0; i < level.size(); ++i) // collect available segment for this core
        {
            core_segments.clear();
            auto node = level[i];

            auto sgn = SubGraphNode::from_node(node);
            for (auto sg_node : sgn->subgraph()->nodes())
            {
                for (auto t : sg_node->outputs())
                {
                    auto ptr = t->memory_offset();
                    // DEBUG() << "live time " << j << ": " << mem;
                    if (ptr->get_base_object() == gm_slot->get_base_object())
                    {
                        // INFO() << t->name() << " top is in gm! relative offset: " << ptr->relative_offset << " offset(): " << ptr->offset();
                        INFO() << sg_node << " in gm offset: " << ptr->offset() << " size: " << ptr->size;
                        add_segment(core_segments, ptr);     //gm taken by this core
                        add_segment(gm_taken_segments, ptr); //gm taken by all cores
                    }
                }
            }

            level_segments.push_back(core_segments);
            if (node->attrs().count("mem_clock"))
            {
                top_mem_clock = node->get_attr<uint32_t>(Constants::mem_clock);
            }
            INFO() << node << " mem clock: " << top_mem_clock;
        }

        if (top_mem_clock < top_liveness_mem.size())
        {
            auto mem_vec = top_liveness_mem[top_mem_clock];
            for (auto mem : mem_vec)
            {
                INFO() << "mem in clock: " << mem;
                if (tensor_map.find(mem) != tensor_map.end())
                {
                    INFO() << "tensor in clock: " << tensor_map[mem];
                }
                else
                {
                    INFO() << "no tensor";
                }
                if (mem->get_base_object() == gm_slot->get_base_object())
                {
                    // INFO() << "mem in gm: " << mem << " offset: 0x" << std::hex << mem->offset() << " real: 0x" << mem->offset() + 45629440 << " size: 0x" << mem->size;
                    auto seg = std::make_shared<MemoryObject>(0);
                    seg->set_base_offset(gm_slot->get_base_object());
                    seg->relative_offset = mem->offset();
                    seg->size = mem->size;
                    INFO() << "memory in clock offset: " << seg->relative_offset << " size: " << seg->size;
                    add_segment(gm_taken_segments, seg);
                }
            }
        }

        for (auto s : gm_taken_segments) //for output time clock
        {
            INFO() << "gm taken segment: " << s->relative_offset << " " << s->size;
        }

        auto share_segments = get_available_segment(gm_taken_segments, gm_slot);
        gm_level_share_segments.push_back(share_segments);

        for (auto s : share_segments) //for output time clock
        {
            INFO() << "gm share segment: " << s->relative_offset << " " << s->size;
        }

        for (size_t i = 0; i < level.size(); ++i)
        {
            auto node = level[i];

            std::vector<MemoryObjectPtr> keep_ptrs;
            std::vector<MemoryObjectPtr> alive_ptrs;

            auto sgn = SubGraphNode::from_node(node);
            auto sg_it = m_liveness_map.find(sgn->subgraph());
            std::map<MemoryObjectPtr, std::pair<uint32_t, uint32_t>> sg_liveness;
            if (sg_it != m_liveness_map.end())
            {
                sg_liveness = sg_it->second;
            }
            else
            {
                WARN() << "no sg liveness";
            }
            auto sg_liveness_mem = generate_liveness_mem(sg_liveness);

            for (auto sg_node : sgn->subgraph()->nodes())
            {
                // INFO() << "cal liveness for " << sg_node->name();
                uint32_t mem_clock = 0;
                if (sg_node->attrs().count("mem_clock"))
                {
                    mem_clock = sg_node->get_attr<uint32_t>(Constants::mem_clock);
                }

                node_to_level.insert(std::make_pair(sg_node, z));

                if (sg_liveness_mem.size() == 0)
                {
                    continue; //can only use gm share segments
                }
                std::vector<MemoryObjectPtr> taken_segments;
                alive_ptrs.clear();

                for (auto t : sg_node->outputs())
                {
                    auto ptr = t->memory_offset();
                    if (t->is_io_tensor() && ptr->get_base_object() == gm_slot->get_base_object())
                    {
                        // INFO() << "io tensor is in gm";
                        auto seg = std::make_shared<MemoryObject>(0);
                        seg->set_base_offset(gm_slot->get_base_object());
                        seg->relative_offset = ptr->offset();
                        seg->size = ptr->size;
                        keep_ptrs.push_back(seg);
                        // INFO() << "add seg " << seg->relative_offset << " " << seg->size;
                    }
                }
                if (mem_clock < sg_liveness_mem.size())
                {
                    auto mem_vec = sg_liveness_mem[mem_clock];
                    for (auto mem : mem_vec)
                    {
                        if (mem->get_base_object() == gm_slot->get_base_object())
                        {
                            auto seg = std::make_shared<MemoryObject>(0);
                            seg->set_base_offset(gm_slot->get_base_object());
                            seg->relative_offset = mem->offset();
                            seg->size = mem->size;
                            // INFO() << "add seg " << seg->relative_offset << " " << seg->size;
                            alive_ptrs.push_back(seg);
                        }
                    }
                }

                for (auto s : alive_ptrs) //for output time clock
                {
                    add_segment(taken_segments, s);
                    // INFO() << "alive segments: " << s->relative_offset << " " << s->size;
                }

                for (auto s : keep_ptrs) //for io tensor
                {
                    add_segment(taken_segments, s);
                    // INFO() << "io segments: " << s->relative_offset << " " << s->size;
                }

                for (size_t j = 0; j < level_segments.size(); ++j)
                {
                    if (i != j)
                    {
                        for (auto seg : level_segments[j])
                        {
                            add_segment(taken_segments, seg);
                            // INFO() << "gm share segments: " << seg->relative_offset << " " << seg->size;
                        }
                    }
                }
                for (auto s : share_segments)
                {
                    add_segment(taken_segments, s);
                }
                auto available_segments = get_available_segment(taken_segments, gm_slot);
                check_segments(available_segments);
                INFO() << "available segments for " << sg_node->name();
                for (auto seg : available_segments)
                {
                    INFO() << "offset: " << seg->relative_offset << " size: " << seg->size;
                }
                gm_available_segments.insert(std::make_pair(sg_node, available_segments));

                INFO() << "share segments for " << sg_node->name();
                for (auto seg : gm_level_share_segments[z])
                {
                    INFO() << "offset: " << seg->relative_offset << " size: " << seg->size;
                }
            }
        }
    }
    return true;
}
static std::vector<MemoryObjectPtr> get_mutual_segments(const std::vector<MemoryObjectPtr> &seg1_vec, const std::vector<MemoryObjectPtr> &seg2_vec, uint32_t min_mutual_size, uint32_t align = 256)
{
    std::vector<MemoryObjectPtr> mutual_segments;
    for (auto seg1 : seg1_vec)
    {
        for (auto seg2 : seg2_vec)
        {
            if ((seg2->offset() >= seg1->offset() && seg2->offset() < seg1->offset() + seg1->size) ||
                (seg1->offset() >= seg2->offset() && seg1->offset() < seg2->offset() + seg2->size))
            {
                auto seg = std::make_shared<MemoryObject>(0);
                seg->set_base_offset(seg1->get_base_object());
                seg->relative_offset = std::max(seg1->offset(), seg2->offset());
                seg->relative_offset = aligned(seg->relative_offset, align);
                seg->size = std::min(seg1->offset() + seg1->size, seg2->offset() + seg2->size) - seg->relative_offset;
                if (seg->size >= min_mutual_size)
                    mutual_segments.push_back(seg);
            }
        }
    }
    return mutual_segments;
}
static bool update_segments(std::vector<MemoryObjectPtr> &segments, MemoryObjectPtr seg)
{
    bool updated = false;
    for (size_t i = 0; i < segments.size(); ++i)
    {
        auto s = segments[i];
        if (seg->offset() == s->offset())
        {
            updated = true;
            if (s->size == seg->size)
            {
                segments.erase(segments.begin() + i);
            }
            else if (s->size < seg->size)
            {
                segments.erase(segments.begin() + i);
            }
            else
            {
                segments[i]->relative_offset = seg->offset() + seg->size;
                segments[i]->size = segments[i]->size - seg->size;
            }
            break;
        }
        else if (seg->offset() > s->offset())
        {
            if (seg->offset() < s->offset() + s->size)
            {
                updated = true;
                if (s->offset() + s->size == seg->offset() + seg->size)
                {
                    segments[i]->size = seg->offset() - s->offset();
                }
                else if (s->offset() + s->size < seg->offset() + seg->size)
                {
                    segments[i]->size = seg->offset() - s->offset();
                }
                else
                {
                    auto new_seg = std::make_shared<MemoryObject>(0);
                    new_seg->set_base_offset(s->get_base_object());
                    new_seg->relative_offset = seg->offset() + seg->size;
                    new_seg->size = s->offset() + s->size - (seg->offset() + seg->size);
                    segments[i]->size = seg->offset() - s->offset();
                    segments.insert(segments.begin() + i, new_seg);
                }
                break;
            }
        }
        else if (seg->offset() < s->offset())
        {
            if (seg->offset() + seg->size > s->offset())
            {
                updated = true;
                if (s->offset() + s->size == seg->offset() + seg->size)
                {
                    segments.erase(segments.begin() + i);
                }
                else if (s->offset() + s->size < seg->offset() + seg->size)
                {
                    segments.erase(segments.begin() + i);
                }
                else
                {
                    segments[i]->relative_offset = seg->offset() + seg->size;
                    segments[i]->size = s->offset() + s->size - (seg->offset() + seg->size);
                }
                break;
            }
        }
    }
    return updated;
}
static bool is_candidate(NodePtr node, uint32_t total_constant_size, uint32_t parallel_cal_num)
{
    if (!total_constant_size)
    {
        return false;
    }
    if (node->attrs().count(Constants::plugin_attrs_name))
    {
        auto plugin_info = node->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
        if (plugin_info->name == "SoftmaxOptPlugin")
        {
            return false;
        }
    }
    if (!is_memory_bound(node, parallel_cal_num))
    {
        DEBUG() << "will not put weight to gm due to " << node << " is not memory bound";
        return false;
    }
    return true;
}

static bool find_seg_for_weight(std::map<NodePtr, std::vector<MemoryObjectPtr>> &gm_segments, std::vector<std::vector<MemoryObjectPtr>> &gm_level_share_segments, std::map<NodePtr, uint32_t> &node_to_level, uint32_t total_constant_size, NodePtr m_node, int dma_load, std::vector<NodePtr> &exe_nodes, NodePtr &end_node, MemoryObjectPtr &seg)
{
    std::vector<NodePtr> last_nodes;
    std::vector<MemoryObjectPtr> mutual_segments;
    std::vector<MemoryObjectPtr> tmp_mutual_segments;

    // for (auto seg : gm_segments[m_node])
    // {
    //     DEBUG() << "[DEBUG] m_node offset: " << seg->relative_offset << " size: " << seg->size;
    // }

    auto share_segments = gm_level_share_segments[node_to_level[m_node]];
    if (m_node->attrs().count(Constants::same_w_same_level))
    {
        tmp_mutual_segments = share_segments;
    }
    else
    {
        tmp_mutual_segments = gm_segments[m_node];
        for (auto seg : share_segments)
        {
            add_segment(tmp_mutual_segments, seg);
        }
    }

    NodePtr last_exe_node = m_node->get_attr<NodeRef>("last_exe_node").lock();
    while (last_exe_node)
    {
        auto last_share_segments = gm_level_share_segments[node_to_level[last_exe_node]];
        auto last_available_segment = gm_segments[last_exe_node];
        for (auto seg : last_share_segments)
        {
            add_segment(last_available_segment, seg);
        }
        mutual_segments = get_mutual_segments(last_available_segment, tmp_mutual_segments, total_constant_size);
        if (mutual_segments.size())
        {
            last_nodes.push_back(last_exe_node);
            tmp_mutual_segments = mutual_segments;
            // INFO() << "mutual_segments size: " << mutual_segments.size() << " last_nodes size: " << last_nodes.size();
            last_exe_node = last_exe_node->get_attr<NodeRef>("last_exe_node").lock();
        }
        else
        {
            break;
        }
    }

    int total_last_load = 0;
    GraphPtr sg;
    for (size_t i = 0; i < last_nodes.size(); ++i)
    {
        auto n = last_nodes[i];
        if (kernel_empty(n) && exe_nodes.size() == 0)
        {
            continue;
        }
        if (total_last_load >= dma_load)
        {
            break;
        }
        if (n->attrs().count(Constants::mov_w_to_gm) || n->attrs().count(Constants::mov_w_to_gm_on) || n->attrs().count(Constants::mov_w_to_gm_wait))
        {
            break;
        }
        if (exe_nodes.size() == 0)
        {
            sg = n->graph();
        }
        else if (sg != n->graph())
        {
            INFO() << "all exe nodes must in the same graph!";
            break;
        }
        total_last_load += get_load_from_model(n);
        exe_nodes.insert(exe_nodes.begin(), n);
    }

    if (total_last_load < dma_load)
    {
        DEBUG() << "[DEBUG] cannot move weight for " << m_node->name() << " due to dma load too large!";
        DEBUG() << "[DEBUG] total last load: " << total_last_load << " dma load:" << dma_load;
        DEBUG() << "[DEBUG] total constant size: " << total_constant_size;
        return false;
    }

    for (int i = exe_nodes.size() - 1; i >= 0; --i)
    {
        auto n = exe_nodes[i];
        if ((end_node == nullptr) && !kernel_empty(n))
        {
            end_node = n;
            break;
        }
    }

    bool ret = false;
    for (auto m_seg : tmp_mutual_segments)
    {
        if (m_seg->size >= total_constant_size)
        {
            seg = m_seg;
            ret = true;
            break;
        }
    }
    if (!ret)
        DEBUG() << "[DEBUG] cannot move weight for " << m_node->name() << " due to no available gm segment!";

    return ret;
}

static std::vector<NodePtr> generate_ordered_nodes(const std::vector<NodePtr> &candidates)
{
    std::vector<NodePtr> ordered;
    ordered = candidates;

    std::stable_sort(ordered.begin(), ordered.end(), [](const NodePtr &a, const NodePtr &b) {
        return (get_cal_intense(a) > get_cal_intense(b)) ||
               (get_cal_intense(a) == get_cal_intense(b) && get_total_constant(a) < get_total_constant(b));
    });

    return ordered;
}
static void update_gm_for_weight(const std::vector<std::vector<NodePtr>> &levels, std::map<NodePtr, std::vector<MemoryObjectPtr>> &gm_segments, std::vector<std::vector<MemoryObjectPtr>> &gm_level_share_segments, std::map<NodePtr, uint32_t> &node_to_level)
{
    std::vector<NodePtr> gm_candidates;
    bool load_from_cfg = BuilderConfig::get<std::string>(Constants::load_gm_enhance_cfg) == "" ? false : true;
    utils::JsonObject cfg_graph_info;
    std::map<std::string, utils::JsonObject> cfg_gm_candidates;
    if (load_from_cfg)
    {
        std::stringstream buffer;
        std::ifstream ifile(BuilderConfig::get<std::string>(Constants::load_gm_enhance_cfg));
        buffer << ifile.rdbuf();
        cfg_graph_info = utils::JsonObject::parse(buffer.str());
        auto obj = cfg_graph_info.obj();
        ifile.close();

        for (auto cfg_node : obj)
        {
            // INFO() << "cfg node: " << cfg_node.first;
            auto cfg_info = cfg_node.second;
            bool cfg_gm_candidate = cfg_info["gm_candidate"];
            if (cfg_gm_candidate)
            {
                INFO() << "cfg node: " << cfg_node.first << " is in gm candidate";
                cfg_gm_candidates[cfg_node.first] = cfg_node.second;
            }
        }
    }

    auto gm_enhance_strategy = BuilderConfig::get<uint32_t>(Constants::gm_enhance_strategy);
    auto gm_size = BuilderConfig::get<uint32_t>(Constants::gm_size);
    for (size_t i = 0; i < levels.size(); ++i)
    {
        auto last_level = levels[0];
        if (i > 0)
        {
            last_level = levels[i - 1];
        }
        auto level = levels[i];
        auto parallel_cal_num = level.size();
        bool same_w_same_level = false;
        std::vector<NodePtr> all_nodes;
        for (auto node : level)
        {
            SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
            for (auto m_node : *sgnode->subgraph())
            {
                if (m_node->type() == OpType_Input || m_node->type() == OpType_SubGraphInput || m_node->type() == OpType_InternalConstant)
                {
                    continue;
                }
                all_nodes.push_back(m_node);
            }
        }
        if (connect_same_w_node(all_nodes) && level.size() > 1 && gm_enhance_strategy != 2)
        {
            same_w_same_level = true;
        }

        NodePtr root_node;
        bool no_root_node = true;
        for (size_t j = 0; j < level.size(); ++j)
        {
            auto node = level[j];

            SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
            std::vector<MemoryObjectPtr> last_available_segments;
            for (auto m_node : *sgnode->subgraph())
            {
                if (m_node->type() == OpType_Input || m_node->type() == OpType_SubGraphInput || m_node->type() == OpType_InternalConstant)
                {
                    continue;
                }

                if (same_w_same_level)
                {
                    if (no_root_node)
                    {
                        no_root_node = false;
                        root_node = m_node;
                    }
                    else
                    {
                        if (!root_node)
                        {
                            WARN() << "no root node";
                        }
                    }
                    m_node->set_attr(Constants::same_w_same_level, NodeRef(root_node));
                    INFO() << "same w same level: " << m_node << " root node: " << root_node;
                }

                uint32_t total_constant_size = 0;
                for (auto cts : m_node->constants())
                {
                    auto t = std::make_shared<Tensor>(*cts.second);
                    total_constant_size += t->memory_offset()->size;
                    total_constant_size = aligned(total_constant_size, t->memory_offset()->get_base_object()->align);
                }
                if (load_from_cfg)
                {
                    auto it = cfg_gm_candidates.find(m_node->name());
                    if (it != cfg_gm_candidates.end())
                    {
                        auto cfg = cfg_gm_candidates[m_node->name()];
                        std::string share_weight = cfg["share_weight"];
                        if (share_weight == "" || (share_weight == m_node->name()))
                        {
                            total_constant_size = cfg["constant_size"];
                            gm_candidates.push_back(m_node);
                        }
                    }
                }
                else if (!BuilderConfig::get<bool>(Constants::dump_gm_enhance_cfg))
                {
                    if (gm_enhance_strategy == 1 || gm_enhance_strategy == 2)
                    {
                        // auto gm_debug_num = aipubt::BuilderConfig::get<uint32_t>("number_for_debug");
                        if (total_constant_size > 1024 && total_constant_size <= gm_size)
                        {
                            gm_candidates.push_back(m_node);
                        }
                    }
                    else if (is_candidate(m_node, total_constant_size, parallel_cal_num))
                    {
                        // auto gm_debug_num = aipubt::BuilderConfig::get<uint32_t>("number_for_debug");
                        if (total_constant_size > 1024 && total_constant_size <= gm_size)
                        {
                            gm_candidates.push_back(m_node);
                        }
                    }
                }
            }
        }
    }

    INFO() << "gm candidates:";
    for (auto n : gm_candidates)
    {
        INFO() << "Node: " << n << " " << get_cal_intense(n);
    }

    if (BuilderConfig::get<bool>(Constants::dump_gm_enhance_cfg))
    {
        utils::JsonObject node_info;
        utils::JsonObject graph_info;
        for (size_t i = 0; i < levels.size(); ++i)
        {
            auto level = levels[i];
            for (size_t j = 0; j < level.size(); ++j)
            {
                auto node = level[j];
                SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
                for (auto m_node : *sgnode->subgraph())
                {
                    // node_info["node_name"] = m_node->name();
                    if (contains(gm_candidates, m_node))
                    {
                        node_info["gm_candidate"] = true;
                    }
                    else
                    {
                        node_info["gm_candidate"] = false;
                    }
                    node_info["high_priority"] = false;
                    node_info["constant_size"] = get_total_constant(m_node);
                    node_info["dma_cycles"] = get_dma_load(get_total_constant(m_node));
                    if (m_node->attrs().count(Constants::same_w_same_level))
                    {
                        auto root_node = m_node->get_attr<NodeRef>(Constants::same_w_same_level).lock();
                        node_info["share_weight"] = root_node->name();
                    }
                    else
                    {
                        node_info["share_weight"] = "";
                    }
                    graph_info[m_node->name()] = node_info;
                }
            }
        }
        std::ofstream os("gm_enhance.json");
        if (os.is_open())
        {
            os << graph_info;
            os << std::endl;
            os.close();
        }
    }

    std::vector<NodePtr> gm_ordered;
    if (load_from_cfg)
    {
        int high_prio_cnt = 0;
        for (auto candidate : gm_candidates)
        {
            auto cfg = cfg_gm_candidates[candidate->name()];
            bool is_high_priority_cfg = cfg["high_priority"];
            if (is_high_priority_cfg == true)
            {
                gm_ordered.insert(gm_ordered.begin() + high_prio_cnt, candidate);
                high_prio_cnt++;
            }
            else
            {
                gm_ordered.push_back(candidate);
            }
        }
    }
    else if (gm_enhance_strategy == 1)
    {
        gm_ordered = gm_candidates;
    }
    else
    {
        gm_ordered = generate_ordered_nodes(gm_candidates);
    }

    // if(BuilderConfig::get<bool>("load_gm_enhance_cfg"))

    INFO() << "ordered:";
    for (auto node : gm_ordered)
    {
        INFO() << "Node: " << node << " " << get_cal_intense(node) << " " << get_total_constant(node);
    }

    std::vector<NodePtr> exe_nodes;
    NodePtr end_node;
    MemoryObjectPtr seg;

    for (auto m_node : gm_ordered)
    {
        if (m_node->attrs().count(Constants::same_w_same_level))
        {
            auto root_node = m_node->get_attr<NodeRef>(Constants::same_w_same_level).lock();
            if (root_node != m_node)
            {
                continue;
            }
        }
        uint32_t total_constant_size = 0;
        for (auto cts : m_node->constants())
        {
            auto t = std::make_shared<Tensor>(*cts.second);
            total_constant_size += t->memory_offset()->size;
            total_constant_size = aligned(total_constant_size, t->memory_offset()->get_base_object()->align);
        }
        exe_nodes.clear();
        end_node = nullptr;
        int dma_load = 0;
        if (load_from_cfg)
        {
            auto cfg = cfg_gm_candidates[m_node->name()];
            dma_load = cfg["dma_cycles"];
            total_constant_size = cfg["constant_size"];
        }
        else if (gm_enhance_strategy == 1 || gm_enhance_strategy == 2)
        {
            dma_load = 0;
            if (m_node->attrs().count(Constants::same_w_same_level))
            {
                dma_load = 1;
            }
            total_constant_size = total_constant_size * 1.1;
        }
        else
        {
            dma_load = get_dma_load(total_constant_size);
        }
        // int dma_load = get_dma_load(total_constant_size);
        if (find_seg_for_weight(gm_segments, gm_level_share_segments, node_to_level, total_constant_size, m_node, dma_load, exe_nodes, end_node, seg))
        {
            INFO() << "can put constant into gm! " << m_node->name();
            auto constant_seg = std::make_shared<MemoryObject>(0);
            constant_seg->set_base_offset(seg->get_base_object());
            constant_seg->relative_offset = seg->offset();
            constant_seg->size = total_constant_size;

            NodePtr start_node;
            bool wait_in_front = true;
            if (exe_nodes.size())
            {
                start_node = exe_nodes[0];
                wait_in_front = false;
            }
            else
            {
                if (m_node->attrs().count(Constants::same_w_same_level))
                {
                    DEBUG() << "cannot move w due to last node cannot move";
                    start_node = nullptr;
                    end_node = nullptr;
                }
                else
                {
                    start_node = m_node;
                    end_node = m_node;
                    wait_in_front = true;
                }
            }

            if (start_node && end_node)
            {
                start_node->set_attr(Constants::mov_w_to_gm, NodeRef(end_node));
                start_node->set_attr(Constants::mov_w_gm_offset, constant_seg->offset());
                start_node->set_attr(Constants::mov_w_gm_size, total_constant_size);
                end_node->set_attr(Constants::mov_w_to_gm_wait, NodeRef(m_node));
                end_node->set_attr(Constants::wait_w_in_front, wait_in_front);
                m_node->set_attr(Constants::ld_w_from_gm, NodeRef(start_node));
                m_node->set_attr(Constants::ld_w_gm_offset, constant_seg->offset());
                bool mov_is_on = false;
                for (auto n : exe_nodes)
                {
                    if (!update_segments(gm_segments[n], constant_seg) && !update_segments(gm_level_share_segments[node_to_level[n]], constant_seg))
                    {
                        WARN() << "no segment match!";
                    }
                    if (n == end_node)
                    {
                        mov_is_on = false;
                    }
                    if (mov_is_on)
                    {
                        n->set_attr(Constants::mov_w_to_gm_on, NodeRef(end_node));
                    }
                    if (n == start_node)
                    {
                        mov_is_on = true;
                    }
                }

                if (!update_segments(gm_segments[m_node], constant_seg) && !update_segments(gm_level_share_segments[node_to_level[m_node]], constant_seg))
                {
                    WARN() << "no segment match!";
                }
                else
                {
                    INFO();
                }
            }
        }
    }
    //opt DMA
}

static bool try_seg_for_data(std::map<NodePtr, std::vector<MemoryObjectPtr>> &gm_segments, std::vector<std::vector<MemoryObjectPtr>> &gm_level_share_segments, std::map<NodePtr, uint32_t> &node_to_level, TensorPtr m_tensor, MemoryObjectPtr &seg)
{
    uint32_t output_size;
    auto m_node = m_tensor->node();
    auto share_segments = gm_level_share_segments[node_to_level[m_node]];
    auto available_segment = gm_segments[m_node];
    bool found = false;

    auto ptr = m_tensor->memory_offset();
    output_size = m_tensor->size();

    for (auto s : available_segment)
    {
        if (s->size >= output_size)
        {
            seg = s;
            found = true;
            break;
        }
    }
    if (!found)
    {
        for (auto s : share_segments)
        {
            if (s->size >= output_size)
            {
                seg = s;
                found = true;
                break;
            }
        }
    }

    return found;
}

static bool get_exe_node(NodePtr node, uint32_t data_size, std::vector<MemoryObjectPtr> &share_segments, std::map<NodePtr, std::vector<MemoryObjectPtr>> &gm_segments, std::vector<NodePtr> &next_nodes, std::vector<MemoryObjectPtr> &tmp_mutual_segments)
{
    NodePtr next_exe_node = node->get_attr<NodeRef>("next_exe_node").lock();
    bool found_node = false;
    std::vector<MemoryObjectPtr> mutual_segments;

    while (next_exe_node)
    {
        // if (n->attrs().count(Constants::mov_d_to_ddr) || n->attrs().count(Constants::mov_d_to_ddr_on) || n->attrs().count(Constants::mov_d_to_ddr_wait))
        // {
        //     ;
        // }
        auto next_available_segment = gm_segments[next_exe_node];
        for (auto seg : share_segments)
        {
            next_available_segment.push_back(seg);
        }
        mutual_segments = get_mutual_segments(next_available_segment, tmp_mutual_segments, data_size);

        if (mutual_segments.size())
        {
            if (next_exe_node->attrs().count(Constants::mov_d_to_ddr) || next_exe_node->attrs().count(Constants::mov_d_to_ddr_on) || next_exe_node->attrs().count(Constants::mov_d_to_ddr_wait))
            {
                next_nodes.clear();
            }
            else
            {
                next_nodes.push_back(next_exe_node);
            }
            tmp_mutual_segments = mutual_segments;
        }
        else
        {
            found_node = true;
            break;
        }

        next_exe_node = next_exe_node->get_attr<NodeRef>("next_exe_node").lock();
    }

    return found_node;
}

static int get_dma_load(const std::vector<NodePtr> &nodes)
{
    int total_load = 0;
    for (auto n : nodes)
    {
        total_load += get_load_from_model(n);
    }
    return total_load;
}

static bool find_seg_for_data(std::map<NodePtr, std::vector<MemoryObjectPtr>> &gm_segments, std::vector<std::vector<MemoryObjectPtr>> &gm_level_share_segments, std::map<NodePtr, uint32_t> &node_to_level, const std::vector<std::vector<NodePtr>> &levels, TensorPtr m_tensor, int dma_load, std::vector<NodePtr> &exe_nodes, std::vector<NodePtr> &dma_nodes, MemoryObjectPtr &seg)
{

    std::vector<MemoryObjectPtr> mutual_segments;
    std::vector<MemoryObjectPtr> tmp_mutual_segments;

    uint32_t output_size;
    auto m_node = m_tensor->node();
    int current_level = node_to_level[m_node];
    bool found_node = false;

    auto share_segments = gm_level_share_segments[current_level];

    auto ptr = m_tensor->memory_offset();
    output_size = m_tensor->size();

    std::vector<std::vector<NodePtr>> exe_node_candidates;
    std::vector<NodePtr> next_nodes;
    tmp_mutual_segments = gm_segments[m_node];
    for (auto seg : share_segments)
    {
        tmp_mutual_segments.push_back(seg);
    }
    found_node = get_exe_node(m_node, output_size, share_segments, gm_segments, next_nodes, tmp_mutual_segments);
    exe_node_candidates.push_back(next_nodes);

    for (size_t i = current_level + 1; i < levels.size(); ++i)
    {
        if (found_node)
        {
            break;
        }

        auto level = levels[i];
        share_segments = gm_level_share_segments[i];
        std::vector<std::vector<NodePtr>> tmp_node_cadidates;
        int index;
        int cnt = 0;
        for (size_t j = 0; j < level.size(); ++j)
        {
            auto node = level[j];
            SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
            next_nodes.clear();

            bool get_node = get_exe_node(sgnode->subgraph()->nodes()[0], output_size, share_segments, gm_segments, next_nodes, tmp_mutual_segments);
            if (get_node)
            {
                cnt++;
                index = j;
            }
            tmp_node_cadidates.push_back(next_nodes);
        }

        if (cnt == 0) // doesn't find the node then we use the largest dma load
        {
            int max_index = 0;
            int max_load = 0;
            for (size_t j = 0; j < tmp_node_cadidates.size(); ++j)
            {
                auto nodes = tmp_node_cadidates[j];
                if (get_dma_load(nodes) > max_load)
                {
                    max_index = j;
                }
            }
            auto nodes = tmp_node_cadidates[max_index];
            exe_node_candidates.push_back(nodes);
        }
        else if (cnt == 1) // find the node
        {
            auto nodes = tmp_node_cadidates[index];
            exe_node_candidates.push_back(nodes);
            found_node = true;
        }
    }

    // INFO() << "found exe candidates";
    size_t max_dma_load_idx = 0;
    int max_dma_load = 0;
    for (size_t i = 0; i < exe_node_candidates.size(); ++i)
    {
        auto nodes = exe_node_candidates[i];
        auto dma_load = get_dma_load(nodes);
        if (dma_load > max_dma_load)
        {
            max_dma_load_idx = i;
            max_dma_load = dma_load;
        }
    }
    next_nodes = exe_node_candidates[max_dma_load_idx];
    int total_next_load = 0;
    for (size_t i = 0; i < next_nodes.size(); ++i)
    {
        auto n = next_nodes[i];
        if (kernel_empty(n))
        {
            continue;
        }

        total_next_load += get_load_from_model(n);
        dma_nodes.push_back(n);
    }

    if (total_next_load < dma_load)
    {
        DEBUG() << "[DEBUG] cannot move data for " << m_node->name() << " due to dma load too large!";
        DEBUG() << "[DEBUG] total next load: " << total_next_load << " dma load:" << dma_load;
        DEBUG() << "[DEBUG] total data size: " << output_size;
        return false;
    }

    for (auto nodes : exe_node_candidates)
    {
        for (auto n : nodes)
        {
            exe_nodes.push_back(n);
        }
    }

    // auto total_nodes = exe_nodes;
    // total_nodes.push_back(m_node);
    // auto s = get_mutual_seg(total_nodes, output_size, share_segments, gm_segments);

    if (!tmp_mutual_segments.size())
    {
        DEBUG() << "[DEBUG] cannot move weight for " << m_node->name() << " due to no available gm segment!";
        return false;
    }
    seg = tmp_mutual_segments[0];
    return true;
}

static void update_gm_for_featuremap(const std::vector<std::vector<NodePtr>> &levels, std::map<NodePtr, std::vector<MemoryObjectPtr>> &gm_segments, std::vector<std::vector<MemoryObjectPtr>> &gm_level_share_segments, std::map<NodePtr, uint32_t> &node_to_level, MemoryObjectPtr gm)
{
    MemoryObjectPtr seg;
    // std::map<NodePtr, NodePtr> gm_candidates;
    std::vector<NodePtr> gm_candidates;
    // std::map<NodePtr, MemoryObjectPtr> node_seg_map;
    for (size_t i = 0; i < levels.size(); ++i)
    {
        auto level = levels[i];
        for (auto node : level)
        {
            SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
            for (auto m_node : *sgnode->subgraph())
            {
                if (m_node->type() == OpType_Input || m_node->type() == OpType_SubGraphInput || m_node->type() == OpType_InternalConstant)
                {
                    continue;
                }
                if (kernel_empty(m_node))
                {
                    continue;
                }
                if (m_node->attrs().count("fixed_mem"))
                {
                    continue;
                }
                for (auto o : m_node->outputs())
                {
                    auto ptr = o->memory_offset();
                    bool skip = false;
                    if (ptr->get_base_object() == gm->get_base_object() || ptr->get_base_object()->get_base_memory() >= 0xf8000000)
                    {
                        continue;
                    }
                    if (!skip && try_seg_for_data(gm_segments, gm_level_share_segments, node_to_level, o, seg))
                    {
                        // INFO() << "seg for node: " << m_node << " offset: " << seg->relative_offset;
                        gm_candidates.push_back(m_node);
                    }
                }
            }
        }
    }

    std::vector<NodePtr> gm_ordered;
    INFO() << "gm candidates:";
    for (auto node : gm_candidates)
    {
        INFO() << node;
        gm_ordered.push_back(node);
    }

    std::vector<NodePtr> exe_nodes;
    std::vector<NodePtr> dma_nodes;
    for (auto m_node : gm_ordered)
    {
        exe_nodes.clear();
        dma_nodes.clear();
        for (size_t i = 0; i < m_node->outputs().size(); ++i)
        {
            auto o = m_node->outputs()[i];
            int dma_load = get_dma_load(o->size());
            if (find_seg_for_data(gm_segments, gm_level_share_segments, node_to_level, levels, o, dma_load, exe_nodes, dma_nodes, seg))
            {
                INFO() << "can put output into gm! " << m_node->name() << " offset: " << seg->offset();
                auto out_seg = std::make_shared<MemoryObject>(0);
                out_seg->set_base_offset(seg->get_base_object());
                out_seg->relative_offset = seg->offset();
                out_seg->size = o->size();

                auto start_node = dma_nodes[0];
                auto end_node = dma_nodes[dma_nodes.size() - 1];
                for (size_t i = 0; i < dma_nodes.size(); ++i)
                {
                    auto n = dma_nodes[i];
                    n->set_attr(Constants::mov_d_to_ddr_on, NodeRef(m_node));
                    INFO() << "dma node: " << n;
                }

                m_node->set_attr(Constants::wr_out_to_gm, seg->offset());
                m_node->set_attr(Constants::wr_out_to_gm_idx, i);
                start_node->set_attr(Constants::mov_d_to_ddr, NodeRef(end_node));
                end_node->set_attr(Constants::mov_d_to_ddr_wait, NodeRef(m_node));
                INFO() << "start_node " << start_node << " end_node " << end_node;
                bool mov_is_on = false;
                std::vector<uint32_t> updated_level;
                for (auto n : exe_nodes)
                {
                    INFO() << "exe node: " << n;
                    if (contains(updated_level, node_to_level[n]))
                    {
                        continue;
                    }
                    if (update_segments(gm_level_share_segments[node_to_level[n]], out_seg))
                    {
                        updated_level.push_back(node_to_level[n]);
                    }
                    else if (!update_segments(gm_segments[n], out_seg))
                    {
                        WARN() << "no segment match!";
                    }
                    if (n == end_node)
                    {
                        mov_is_on = false;
                    }
                    if (mov_is_on)
                    {
                        n->set_attr(Constants::mov_w_to_gm_on, NodeRef(end_node));
                    }
                    if (n == start_node)
                    {
                        mov_is_on = true;
                    }
                }
                if (!contains(updated_level, node_to_level[m_node]))
                {
                    if (!update_segments(gm_segments[m_node], out_seg) && !update_segments(gm_level_share_segments[node_to_level[m_node]], out_seg))
                    {
                        WARN() << "no segment match!";
                    }
                }
            }
        }
    }
}

static void set_exe_node(const std::vector<std::vector<NodePtr>> &levels)
{
    NodePtr last_exe_node = nullptr;
    for (size_t i = 0; i < levels.size(); ++i)
    {
        auto level = levels[i];

        for (size_t j = 0; j < level.size(); ++j)
        {
            auto node = level[j];
            auto sgn = SubGraphNode::from_node(node);
            if (sgn)
            {
                last_exe_node = nullptr;
                for (auto m_node : *sgn->subgraph())
                {
                    if (last_exe_node)
                    {
                        m_node->set_attr("last_exe_node", NodeRef(last_exe_node));
                        last_exe_node->set_attr("next_exe_node", NodeRef(m_node));
                    }
                    last_exe_node = m_node;
                }
            }
        }
    }
}

void MCScheduler::print_gm_avg_usage()
{
    auto gm_slot = m_gm_region_slots[0];
    for (auto level : m_levels)
    {
        for (auto node : level)
        {
            uint32_t total_assigned_gm_size = 0;
            uint32_t node_gm_size = 0, gm_size = 0;
            SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
            float total_avg_usage = 0;
            int node_cnt = 0;
            if (sgnode)
            {
                for (auto n : *sgnode->subgraph())
                {
                    if (kernel_empty(n))
                    {
                        continue;
                    }
                    node_gm_size = 0;
                    std::vector<MemoryObjectPtr> allocated;
                    for (auto o : n->outputs())
                    {
                        auto ptr = o->memory_offset();
                        if (ptr->get_base_object() == gm_slot->get_base_object() && !contains(allocated, ptr))
                        {
                            node_gm_size += ptr->size;
                            allocated.push_back(ptr);
                        }
                    }
                    for (auto o : n->inputs())
                    {
                        auto ptr = o->memory_offset();
                        if (ptr->get_base_object() == gm_slot->get_base_object() && !contains(allocated, ptr))
                        {
                            node_gm_size += ptr->size;
                            allocated.push_back(ptr);
                        }
                    }
                    gm_size = gm_size < node_gm_size ? node_gm_size : gm_size;
                }
                total_assigned_gm_size = gm_size;
                // INFO() << "total assigned gm size: " << total_assigned_gm_size;
                for (auto n : *sgnode->subgraph())
                {
                    if (kernel_empty(n))
                    {
                        continue;
                    }
                    node_gm_size = 0;
                    std::vector<MemoryObjectPtr> allocated;
                    for (auto o : n->outputs())
                    {
                        auto ptr = o->memory_offset();
                        if (ptr->get_base_object() == gm_slot->get_base_object() && !contains(allocated, ptr))
                        {
                            node_gm_size += ptr->size;
                            allocated.push_back(ptr);
                            // INFO() << "add output: " << o << " size: " << ptr->size << " total: " << node_gm_size;
                        }
                    }
                    for (auto o : n->inputs())
                    {
                        auto ptr = o->memory_offset();
                        if (ptr->get_base_object() == gm_slot->get_base_object() && !contains(allocated, ptr))
                        {
                            allocated.push_back(ptr);
                            node_gm_size += ptr->size;
                            // INFO() << "add input: " << o << " size: " << ptr->size << " total: " << node_gm_size;
                        }
                    }
                    float avg_usage = (float)node_gm_size / (float)total_assigned_gm_size;
                    INFO() << "average gm usage for " << n << " is " << avg_usage;
                    total_avg_usage += avg_usage;
                    node_cnt++;
                }
            }
            INFO() << "top average gm usage for " << node << " is " << total_avg_usage / node_cnt;
        }
    }
}

bool MCScheduler::gm_enhance()
{
    std::map<NodePtr, std::vector<MemoryObjectPtr>> gm_available_segments;
    std::vector<std::vector<MemoryObjectPtr>> gm_level_share_segments;
    std::map<NodePtr, uint32_t> node_to_level;
    generate_gm_available_segments(m_graph, m_levels, m_gm_region_slots[0], m_liveness_map, gm_available_segments, gm_level_share_segments, node_to_level);
    int z = 0;
    for (auto level : m_levels)
    {
        INFO() << "share gm segment for level " << z;
        for (auto seg : gm_level_share_segments[z])
        {
            INFO() << "offset: " << seg->offset() << " size: " << seg->size;
        }
        z++;
        for (auto n : level)
        {
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                for (auto sg_node : sgn->subgraph()->nodes())
                {
                    INFO() << "available gm segment for " << sg_node->name();
                    for (auto seg : gm_available_segments[sg_node])
                    {
                        INFO() << "offset: " << seg->offset() << " size: " << seg->size;
                    }
                }
            }
        }
    }
    set_exe_node(m_levels);
    if (BuilderConfig::get<bool>(Constants::enable_dynamic_gm))
    {
        update_gm_for_featuremap(m_levels, gm_available_segments, gm_level_share_segments, node_to_level, m_gm_region_slots[0]);
    }
    if (BuilderConfig::get<bool>(Constants::enable_gm_enhance))
    {
        update_gm_for_weight(m_levels, gm_available_segments, gm_level_share_segments, node_to_level);
    }
    // auto w_seg = std::make_shared<MemoryObject>(0);
    // w_seg->set_base_offset(m_gm_region_slots[0]);
    // w_seg->relative_offset = 0;
    // w_seg->size = 0;
    // auto share_seg = std::make_shared<MemoryObject>(0);
    // share_seg->set_base_offset(m_gm_region_slots[0]);
    // share_seg->relative_offset = 0;
    // share_seg->size = 0;
    // bool clear_w_gm = false;
    // std::vector<MemoryObjectPtr> segments;
    // z = 0;
    // bool display = false;
    // bool share = false;
    // for (auto level : m_levels)
    // {
    //     z++;
    //     for (size_t i = 0; i < level.size(); ++i) // collect available segment for this core
    //     {
    //         if (share)
    //         {
    //             w_seg->relative_offset = share_seg->relative_offset;
    //             w_seg->size = share_seg->size;
    //         }
    //         auto node = level[i];
    //         auto sgn = SubGraphNode::from_node(node);
    //         for (auto sg_node : sgn->subgraph()->nodes())
    //         {
    //             // DEBUG() << "[DEBUG] level " << z - 1 << " node: " << sg_node;
    //             segments.clear();
    //             if (clear_w_gm)
    //             {
    //                 clear_w_gm = false;
    //                 w_seg->relative_offset = 0;
    //                 w_seg->size = 0;
    //             }

    //             if (sg_node->attrs().count(Constants::mov_w_to_gm))
    //             {
    //                 auto next_node = sg_node->get_attr<NodeRef>(Constants::mov_w_to_gm).lock();
    //                 auto end_node = next_node->get_attr<NodeRef>(Constants::mov_w_to_gm_wait).lock();
    //                 uint32_t offset = sg_node->get_attr<uint32_t>(Constants::mov_w_gm_offset);
    //                 uint32_t size = sg_node->get_attr<uint32_t>(Constants::mov_w_gm_size);
    //                 DEBUG() << "[DEBUG] "
    //                         << "mov w to gm " << end_node->name()
    //                         << " offset: 0x" << std::hex << std::setw(8) << std::setfill('0') << offset
    //                         << " size: 0x" << std::hex << std::setw(8) << std::setfill('0') << size;
    //                 w_seg->relative_offset = offset;
    //                 w_seg->size = size;
    //                 if (end_node->attrs().count(Constants::same_w_same_level))
    //                 {
    //                     share = true;
    //                 }
    //                 else
    //                 {
    //                     share = false;
    //                 }
    //             }

    //             if (display)
    //             {
    //                 DEBUG() << " gm weight: " << w_seg->relative_offset << " size: " << w_seg->size;
    //             }

    //             if (sg_node->attrs().count(Constants::mov_w_to_gm_on))
    //             {
    //                 auto next_node = sg_node->get_attr<NodeRef>(Constants::mov_w_to_gm_on).lock();
    //                 auto end_node = next_node->get_attr<NodeRef>(Constants::mov_w_to_gm_wait).lock();
    //                 DEBUG() << "[DEBUG] "
    //                         << "mov w to gm on " << end_node->name();
    //             }
    //             if (sg_node->attrs().count(Constants::mov_w_to_gm_wait))
    //             {
    //                 auto next_node = sg_node->get_attr<NodeRef>(Constants::mov_w_to_gm_wait).lock();
    //                 DEBUG() << "[DEBUG] "
    //                         << "mov w to gm wait " << next_node->name();
    //             }
    //             if (sg_node->attrs().count(Constants::ld_w_from_gm))
    //             {
    //                 uint32_t offset = sg_node->get_attr<uint32_t>(Constants::ld_w_gm_offset);
    //                 DEBUG() << "[DEBUG] "
    //                         << "ld w from gm "
    //                         << " offset: 0x" << std::hex << std::setw(8) << std::setfill('0') << offset;
    //                 clear_w_gm = true;
    //             }
    //             else if (sg_node->attrs().count(Constants::same_w_same_level))
    //             {
    //                 auto root_node = sg_node->get_attr<NodeRef>(Constants::same_w_same_level).lock();
    //                 if (root_node->attrs().count(Constants::ld_w_from_gm))
    //                 {
    //                     uint32_t offset = root_node->get_attr<uint32_t>(Constants::ld_w_gm_offset);
    //                     DEBUG() << "[DEBUG] "
    //                             << "ld w from gm "
    //                             << " offset: 0x" << std::hex << std::setw(8) << std::setfill('0') << offset;
    //                     clear_w_gm = true;
    //                 }
    //             }
    //             for (auto t : sg_node->inputs())
    //             {
    //                 auto ptr = t->memory_offset();
    //                 if (ptr->get_base_object() == m_gm_region_slots[0]->get_base_object())
    //                 {
    //                     auto seg = std::make_shared<MemoryObject>(0);
    //                     seg->set_base_offset(m_gm_region_slots[0]);
    //                     seg->relative_offset = ptr->offset();
    //                     seg->size = ptr->size;
    //                     add_segment(segments, seg);
    //                     if (display)
    //                     {
    //                         DEBUG() << " input: " << seg->relative_offset << " size: " << seg->size;
    //                     }
    //                 }
    //             }
    //             for (auto t : sg_node->outputs())
    //             {
    //                 auto ptr = t->memory_offset();
    //                 if (ptr->get_base_object() == m_gm_region_slots[0]->get_base_object())
    //                 {
    //                     auto seg = std::make_shared<MemoryObject>(0);
    //                     seg->set_base_offset(m_gm_region_slots[0]);
    //                     seg->relative_offset = ptr->offset();
    //                     seg->size = ptr->size;
    //                     add_segment(segments, seg);
    //                     if (display)
    //                     {
    //                         DEBUG() << " output: " << seg->relative_offset << " size: " << seg->size;
    //                     }
    //                 }
    //             }
    //             if (w_seg->size)
    //             {
    //                 segments.push_back(w_seg);
    //             }
    //             check_segments(segments);
    //             DEBUG() << "gm segments for " << sg_node;
    //             for (auto seg : segments)
    //             {
    //                 DEBUG() << "offset: 0x" << std::hex << seg->relative_offset << " size: 0x" << seg->size;
    //             }
    //         }
    //         if (share)
    //         {
    //             share_seg->relative_offset = w_seg->relative_offset;
    //             share_seg->size = w_seg->size;
    //         }
    //     }
    // }

    std::vector<NodePtr> ordered_nodes;
    for (auto level : m_levels)
    {
        for (auto n : level)
        {
            ordered_nodes.push_back(n);
        }
    }

    m_graph->sort_by_ordered(ordered_nodes);
    return true;
}

}; // namespace aipubt

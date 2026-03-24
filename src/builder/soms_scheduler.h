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
 * @Date: 2024-07-15 14:27:29 
 * @Last Modified by: Kyle Huang 
 * @Last Modified time: 2024-07-15 11:16:43
 */
#ifndef BUILDER_SOMS_SCHEDULER_H
#define BUILDER_SOMS_SCHEDULER_H

#include "scheduler.h"
#include <set>
namespace aipubt
{

class SOMSScheduler : public Scheduler
{
public:
    using Scheduler::Scheduler;
    virtual std::vector<MemoryObjectPtr> mem_allocate(const std::set<TensorPtr> &not_reuse_tensor = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &inputs = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &keeps_tensors = std::set<TensorPtr>(),
                                                      const std::set<TensorPtr> &skip_tensors = {}) override;

    void disable_sm();

private:
    bool m_enable_sm = true;
    std::map<MemoryObjectPtr, int32_t> m_mem_core_map;
    std::vector<MemoryObjectPtr> m_slots;
    std::map<MemoryObjectPtr, std::vector<NodePtr>> m_p_consumers;
    std::map<MemoryObjectPtr, std::vector<NodePtr>> m_p_generators;
    std::map<NodePtr, NodePtr> m_streams;
    std::set<MemoryObjectPtr> m_input_mem;
    std::set<MemoryObjectPtr> m_keep_mem;
    std::map<MemoryObjectPtr, TensorPtr> m_tensor_map;
    std::map<std::pair<MemoryObjectPtr, MemoryObjectPtr>, uint32_t> m_conflict_reason;
    std::map<std::pair<NodePtr, uint32_t>, std::set<NodePtr>> m_anc_streams_cache;
    std::map<MemoryObjectPtr, std::vector<NodePtr>> m_src_streams_cache;
    std::map<MemoryObjectPtr, std::vector<NodePtr>> m_dest_streams_cache;
    std::map<std::pair<NodePtr, MemoryObjectPtr>, uint32_t> m_start_time_cache;
    std::map<std::pair<NodePtr, MemoryObjectPtr>, uint32_t> m_end_time_cache;
    std::map<MemoryObjectPtr, uint32_t> m_start_level_cache;
    std::map<MemoryObjectPtr, uint32_t> m_end_level_cache;

    std::set<MemoryObjectPtr> allocate_sm(std::set<MemoryObjectPtr> &skip_mem);
    std::set<MemoryObjectPtr> mem_live_init(const std::set<MemoryObjectPtr> &skip_mem);
    std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>> gen_mem_conflict(const std::set<MemoryObjectPtr> &all_mem);
    std::vector<MemoryObjectPtr> offset_assignment(MemoryObjectPtr gm_s, const std::map<MemoryObjectPtr, std::vector<MemoryObjectPtr>> &conflict_map, const std::set<MemoryObjectPtr> &all_mems, const std::set<MemoryObjectPtr> &not_reuse_mem);
    std::set<NodePtr> anc_streams(NodePtr n, uint32_t level);

    bool anc_stream_reuse(MemoryObjectPtr m1, MemoryObjectPtr m2);
    bool same_stream_reuse(MemoryObjectPtr m1, MemoryObjectPtr m2);
    // uint32_t max_parent_id(MemoryObjectPtr m, NodePtr s);
    uint32_t start_time(MemoryObjectPtr m, NodePtr s);
    uint32_t end_time(MemoryObjectPtr m, NodePtr s);
    std::vector<NodePtr> src_stream(MemoryObjectPtr m);
    std::vector<NodePtr> dest_stream(MemoryObjectPtr m);
    // uint32_t start_level(TensorPtr t);
    bool is_output_tensor(MemoryObjectPtr m);
    // void find_anc_nodes(NodePtr n, std::set<NodePtr> &ancestors);
    void find_anc_streams(NodePtr n, std::set<NodePtr> &ancestors, const uint32_t &level);
    int32_t get_start_level(MemoryObjectPtr m);
    int32_t get_end_level(MemoryObjectPtr m);
    NodePtr update_gen_consume(GraphPtr g, int &node_stamp, std::set<MemoryObjectPtr> &all_mems, const std::set<MemoryObjectPtr> &skip_mem, const int core_id);
    void update_stream(GraphPtr g, const std::set<MemoryObjectPtr> &skip_mem, NodePtr stream);
    void update_sm(GraphPtr g, std::set<MemoryObjectPtr> &sm_mem);
    void sm_skip_tensors(GraphPtr g, std::set<TensorPtr> &skip_tensors, const std::set<MemoryObjectPtr> &skip_mem);
    bool same_stream(MemoryObjectPtr m1, MemoryObjectPtr m2);
    void check_all_mems(GraphPtr g, const std::set<MemoryObjectPtr> &all_mems, const std::set<MemoryObjectPtr> &skip_mem);
};
};     // namespace aipubt
#endif //!BUILDER_SOMS_SCHEDULER_H

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
 * @Author: Yuchou.Gan
 * @Date: 2020-03-11 17:58:05
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-07-26 16:18:14
 */

#ifndef AIPU_PROFILER
#define AIPU_PROFILER
#include "builder/readonly_param.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/node.h"
#include <memory>
#include <string>
#include <vector>

namespace aipubt
{
namespace AIPUBuilder
{
#define UNUSED(expr)  \
    do                \
    {                 \
        (void)(expr); \
    } while (0)
    class Profiler;
    class TargetDependentProfilerInferface;
    using InterfacePtr = std::shared_ptr<TargetDependentProfilerInferface>;
    using AnyFunction = std::function<Any(Any)>;
    class TargetDependentProfilerInferface
    {
    public:
        virtual ~TargetDependentProfilerInferface() {}
        virtual TensorPtr allocate_buffer(GraphPtr g) = 0;
        virtual AnyFunction get_function(std::string) { return nullptr; }
        virtual bool register_event(uint32_t) { return false; }
        virtual bool register_default_events(std::string &prof_unit)
        {
            UNUSED(prof_unit);
            return false;
        }
        virtual bool unregister_all() { return true; }
        virtual std::vector<uint32_t> registered_events() { return {}; }
        virtual std::vector<uint32_t> all_legal_events() { return {}; }
        virtual std::string event_description(uint32_t id)
        {
            UNUSED(id);
            return "";
        }
        virtual std::vector<uint32_t> get_default_events(std::string &prof_unit)
        {
            UNUSED(prof_unit);
            return {};
        }
        static InterfacePtr get_or_register_interface(std::string, InterfacePtr);
    };

    Profiler *get_profiler();
    class Profiler
    {
    public:
        bool register_event(uint32_t);
        bool register_events(std::string events);
        bool register_default_events(std::string &prof_unit);
        bool unregister_all();
        std::vector<uint32_t> registered_events();
        std::string show_event_description();
        std::string event_description(uint32_t id);
        bool &enable_profile();
        InterfacePtr interface() { return this->_interface; }
        TensorPtr &tensor();
        bool generate_graph_json(GraphPtr g, const std::string &file_name = "graph.json");
        std::string json_from_graph(GraphPtr);
        std::string json_from_node(Node *);
        friend Profiler *get_profiler();
        /* Get value from config; if it's not set in config, return the default value(the argument of the function). */
        uint64_t get_max_perf_sample_num(uint64_t max_perf_count = 0) const;
        uint64_t get_max_aiff_perf_sample_num(uint64_t max_perf_aiff_count = 2048) const;
        uint64_t get_max_dma_perf_sample_num(uint64_t max_perf_dma_count = 65536) const;
        uint64_t get_max_custom_perf_sample_num(uint64_t max_perf_custom_count = 256) const;
        static inline bool disable_dma_perf(bool disable_dma = true)
        {
            if (BuilderConfig::exist(Constants::enable_dma_perf))
            {
                disable_dma = !BuilderConfig::get<bool>(Constants::enable_dma_perf);
            }
            return disable_dma;
        }
        static inline bool enable_fast_perf(bool fast_perf = false)
        {
            if (BuilderConfig::exist(Constants::fast_perf))
            {
                fast_perf = BuilderConfig::get<bool>(Constants::fast_perf);
            }
            return fast_perf;
        }
        void set_mem_info(MemoryObjectPtr gm) {
            m_gm = gm;
        }

    private:
        Profiler();
        bool enable = false;
        MemoryObjectPtr m_gm;
        std::shared_ptr<Tensor> profile_tensor;
        InterfacePtr _interface;
    };

    // to remove virtual subgraph nodes and get real operation nodes
    void get_operation_nodes(GraphPtr graph, std::vector<NodePtr> &ret);
    // to remove virtual subgraph nodes and get max depth
    void get_operation_depth(GraphPtr graph, uint32_t &depth, std::vector<NodePtr> &ret);
} // namespace AIPUBuilder
} // namespace aipubt
#endif

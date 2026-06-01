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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Yuchou.Gan
 * @Date: 2020-05-08 13:21:03
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-02-21 15:51:06
 */
#include "../src/core/ifnode.h"
#include "../src/core/loopnode.h"
#include "builder/linker/ZY.h"
#include "builder/lowering/lowering.h"
#include "builder/ops/ops.h"
#include "builder/plugin.h"
#include "builder/text_generator.h"
#include "core/plugin_loader.h"
#include <algorithm>
#include <builder/profiler.h>
#include <core/buildtool_version.h>
#include <core/config.h>
#include <core/subgraph.h>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utils/string_utils.h>
#include <utils/utils.h>

namespace aipubt
{
namespace AIPUBuilder
{
    std::mutex instance_lock;

    InterfacePtr TargetDependentProfilerInferface::get_or_register_interface(std::string name, InterfacePtr interface)
    {
        static std::map<std::string, InterfacePtr> registered_interface;
        if (interface == nullptr)
        {
            return registered_interface.count(name) ? registered_interface[name] : nullptr;
        }
        registered_interface[name] = interface;
        return interface;
    }

    Profiler::Profiler()
    {
        auto target = aipubt::get_target();
        if (target->isainfo().type <= ISA_ZYv3)
        {
            _interface = TargetDependentProfilerInferface::get_or_register_interface("x1", nullptr);
        }
        else if (target->isainfo().type == ISA_ZYv5)
        {
            _interface = TargetDependentProfilerInferface::get_or_register_interface("x2", nullptr);
        }
        else if (target->isainfo().type >= ISA_ZYv6)
        {
            _interface = TargetDependentProfilerInferface::get_or_register_interface("x3", nullptr);
        }
    }

    bool Profiler::register_events(std::string events)
    {
        unregister_all();
        auto event_ids = utils::split_string(events, ',');
        for (auto &&id : event_ids)
        {
            id = utils::trim(id);
            if (id.find_first_not_of("0123456789") != std::string::npos)
            {
                ERROR() << "invalidate events string";
                return false;
            }
        }

        bool ret = true;
        std::vector<int> vec;
        for (auto &&id : event_ids)
        {
            std::istringstream iss(id);
            int id_num;
            iss >> id_num;
            vec.push_back(id_num);
            //ret &= register_event(id_num);
        }
        sort(vec.begin(), vec.end());
        for (auto id : vec)
        {
            ret &= register_event(id);
        }
        return ret;
    }

    bool Profiler::register_event(uint32_t event)
    {
        return _interface->register_event(event);
    }

    bool Profiler::unregister_all()
    {
        return _interface->unregister_all();
    }

    std::vector<uint32_t> Profiler::registered_events()
    {
        return _interface->registered_events();
    }

    std::string Profiler::show_event_description()
    {
        std::stringstream os;
        os << std::left << std::setw(5) << "id"
           << "event descriptor" << std::endl;
        auto eids = _interface->all_legal_events();
        for (auto &it : eids)
        {
            os << std::left << std::setw(5) << it << this->event_description(it)
               << std::endl;
        }
        return os.str();
    }

    std::string Profiler::event_description(const uint32_t id)
    {
        return _interface->event_description(id);
    }

    bool &Profiler::enable_profile() { return enable; }

    TensorPtr &Profiler::tensor() { return profile_tensor; }

    static std::map<aipubt::ISAType, uint32_t> __default_frequency__ = {
        {aipubt::ISA_ZY, 800},
        {aipubt::ISA_ZYv2, 1000},
        {aipubt::ISA_ZYv3, 1000},
        {aipubt::ISA_ZYv5, 1000},
        {aipubt::ISA_ZYv6, 1000},
    };

    bool Profiler::register_default_events(std::string &prof_unit)
    {
        _interface->unregister_all();
        return _interface->register_default_events(prof_unit);
    }

    static utils::JsonObject jsonobj_from_node(Node *node)
    {
        utils::JsonObject json;
        json["layer node name"] = node->name();
        json["op_type"] = optype_to_string(node->type());
        auto attrs = node->attrs();
        if (attrs.count(Constants::plugin_attrs_name))
        {
            auto plugin_info = node->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
            json["plugin"] = plugin_info->name;
        }
        else
        {
            json["plugin"] = "";
        }

        if (attrs.count(Constants::node_perf) && node->get_attr<bool>(Constants::node_perf))
        {
            json["perf_node"] = true;
            int layer_id = node->get_layer_id();
            json["perf_node_id"] = layer_id;

            if (attrs.count("_kernel")) {
                json["kernel"] = node->get_attr<std::string>("_kernel");
            }
        }

        if (attrs.count("ops_count"))
        {
            if (BuilderConfig::get<bool>(Constants::enable_ds))
            {
                json["ops_count"] = node->get_attr<Expr>("ops_count")->to_string();
            }
            else
            {
                auto ops_count_str = node->get_attr<Expr>("ops_count")->to_string();
                if (!utils::is_number(ops_count_str))
                {
                    ERROR() << "meets invalid ops_count(" << ops_count_str << ") in generating graph.json.";
                }
                json["ops_count"] = std::stoll(ops_count_str);
            }
        }
        else
        {
            json["ops_count"] = 0;
        }

        int total_constant_size = 0;
        for (auto cts : node->constants())
        {
            if (cts.second)
            {
                auto t = std::make_shared<Tensor>(*cts.second);
                if(t->memory_offset()->get_base_object()->buffer_id == -1)
                    continue;
                total_constant_size += t->memory_offset()->size;
                auto aligned = [](ssize_t s, ssize_t align) { return (s + align - 1) & ~(align - 1); };
                total_constant_size = aligned(total_constant_size, t->memory_offset()->get_base_object()->align);
            }
        }
        json["constant_size"] = total_constant_size;

        if (node->attrs().count("weight_reload") && node->attrs().count("weight_size"))
        {
            int64_t weight_reload = node->get_attr<uint32_t>("weight_reload");
            int64_t weight_size = node->get_attr<uint32_t>("weight_reload");
            json["weight_reload"] = weight_reload;
            json["weight_size"] = weight_size;
        }

        if (attrs.count(Constants::operator_id))
        {
            std::vector<uint32_t> operator_ids = node->get_attr<std::vector<uint32_t>>(Constants::operator_id);
            auto &ids = json["operator_id"];
            for (uint32_t id : operator_ids)
            {
                ids.push_back(id);
            }
        }

        int32_t task_id = node->get_attr_deep<uint32_t>(Constants::schedule_gid);
        json["task_id"] = task_id;

        json["inputs"] = utils::JsonObject();
        json["outputs"] = utils::JsonObject();
        auto inputs = node->inputs();
        auto outputs = node->outputs();
        for (auto inp : inputs)
        {
            json["inputs"].push_back(inp->name());
        }
        for (auto out : outputs)
        {
            json["outputs"].push_back(out->name());
        }

        // MAC util need knows weight types
        json["constants"] = utils::JsonObject();
        for (auto &constant : node->constants())
        {
            auto key = constant.first; // e.g. "weights"
            auto val = constant.second;
            json["constants"][key] = utils::JsonObject();
            json["constants"][key]["dtype"] = Tensor::dtype_to_string(val->type());
            json["constants"][key]["shape"] = val->shape().symbol().to_string();
            json["constants"][key]["size"] = val->shape().size();
            if(val->memory_offset()->get_base_object()->buffer_id != -1) {
                json["constants"][key]["used"] = true;
            }
            Tensor::CompressInfo compress_info = val->get_compress_info();
            // Perf Model need WDC info
            if (compress_info.ori_size > 0)
            {
                json["constants"][key]["ori_size"] = compress_info.ori_size;
            }
        }

        if (SubGraphNode::is_subgraph_node(node->shared_from_this()))
        {
            auto &internal_nodes = json["internal_nodes"];
            SubGraphNode *m_sg = static_cast<SubGraphNode *>(node);
            auto nodes = m_sg->subgraph()->nodes();
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                internal_nodes.push_back(jsonobj_from_node(nodes[i].get()));
            }
        }
        else
        {
            json["params"] = utils::JsonObject();
            auto m_params = node->params();
            for (auto params : m_params)
            {
                std::stringstream oss;
                oss << params.second;
                json["params"][params.first] = oss.str();
            }
        }
        return json;
    }

    std::string Profiler::json_from_node(Node *node)
    {
        return jsonobj_from_node(node).to_string();
    }

    static std::vector<NodePtr> get_non_virtual_nodes(GraphPtr graph)
    {
        // remove virtual subgraph
        std::vector<NodePtr> ret;
        std::function<void(NodePtr)> get_operation_nodes_func = [&ret, &get_operation_nodes_func](NodePtr n) {
            if (SubGraphNode::is_subgraph_node(n))
            {
                SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(n);
                if (sgnode->attrs().count(Constants::plugin_attrs_name))
                {
                    ret.push_back(n);
                }
                else
                {
                    for (auto node : *sgnode->subgraph())
                    {
                        get_operation_nodes_func(node);
                    }
                }
            }
            else
            {
                ret.push_back(n);
            }
        };

        for (auto node : *graph)
        {
            get_operation_nodes_func(node);
        }
        return ret;
    }

    /**
     * @brief get_mem_info
     * fork from add_mem_info_to_dot.cpp
     * It removed adderess of gm info because it would be change after dumper
     * @param out
     * @param gm (from builder)
     * @return
     */
    static std::string get_memory(TensorPtr out, MemoryObjectPtr gm)
    {
        auto addr_to_sram_id = [](uint32_t addr, std::vector<uint32_t> &sram_bases, uint32_t mask) -> int32_t {
            auto it = std::find(sram_bases.begin(), sram_bases.end(), addr & mask);
            if (it != sram_bases.end())
            {
                return std::distance(sram_bases.begin(), it);
            }
            else
            {
                return -1;
            }
        };

        uint32_t addr = out->memory_offset()->offset();

        std::vector<uint32_t> gsram_bases = {
            0xF8000000,
            0xF8800000,
        };
        std::vector<uint32_t> lsram_bases = {
            0xF0000000,
            0xF0080000,
            0xF0100000,
            0xF0180000,
            0xF0200000,
            0xF0280000,
            0xF0300000,
            0xF0380000,
        };
        std::string memory_str = "ddr";
        auto gsram_id = addr_to_sram_id(addr, gsram_bases, 0xFF800000);
        auto lsram_id = addr_to_sram_id(addr, lsram_bases, 0xFFF80000);
        if (gsram_id >= 0)
        {
            memory_str = ("gsram_" + std::to_string(gsram_id) + "");
        }
        else if (lsram_id >= 0)
        {
            memory_str = ("lsram_" + std::to_string(lsram_id) + "");
        }
        else if (BuilderConfig::get<bool>(Constants::enable_tcm))
        {
            uint32_t dtcm_base = 0xD0000000;
            uint32_t dtcm_size = BuilderConfig::get<uint32_t>(Constants::tcm_size);
            if (addr >= dtcm_base && addr < (dtcm_base + dtcm_size))
            {
                memory_str = "dtcm";
            }
        }
        else if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
        {
            uint32_t ocm_addr = BuilderConfig::get<uint64_t>("ocm_addr");
            uint32_t ocm_size = BuilderConfig::get<uint64_t>("ocm_size");

            if (addr >= ocm_addr && addr < ocm_addr + ocm_size)
            {
                memory_str = "ocm";
            }
            else if (!BuilderConfig::get<bool>(Constants::disable_gm))
            {
                if (out->memory_offset()->get_base_object() == gm)
                {
                    memory_str = "gm0";
                }
            }
        }

        return memory_str;
    }

    static utils::JsonObject jsonobj_from_graph(GraphPtr g, MemoryObjectPtr gm)
    {
        utils::JsonObject json;
        auto m_nodes = g->nodes();
        json["graph_name"] = utils::verify_name(g->name());
        std::string target_name = aipubt::get_target()->name();
        if (aipubt::get_target()->isainfo().type >= ISA_ZYv5)
        {
            auto core_num = aipubt::get_target()->core_num();
            if (core_num > 1)
            {
                target_name = target_name + "MP" + std::to_string(core_num);
            }
        }
        json["platform target"] = target_name;
        json["total nodes number"] = m_nodes.size();
        json["perf nodes number"] = g->get_perf_nodes().size();
        utils::NullBuf nullbuf;
        std::ostream ws(&nullbuf);
        std::ostringstream oss;
        g->serialize(oss, ws);
        json["IR"] = oss.str();
        oss.clear();

        if (g->symbols().size())
        {
            auto &symbols = json["symbols"];
            for (auto &kv : g->symbols())
            {
                auto &sym = symbols[kv.first];
                sym["type"] = kv.second.type;
                for (auto d : kv.second.dims)
                {
                    sym["dims"].push_back(d);
                }
            }
        }

        auto &nodes = json["nodes"];
        for (uint32_t i = 0; i < m_nodes.size(); i++)
        {
            nodes.push_back(jsonobj_from_node(m_nodes[i].get()));
        }

        auto &tensors = json["tensors"];

        auto real_nodes = get_non_virtual_nodes(g);
        for (auto &node : real_nodes)
        {
            auto inputs = node->inputs();
            auto outputs = node->outputs();

            for (auto inp : inputs)
            {
                std::string name = inp->name();
                if (!tensors.exist(name))
                {
                    tensors[name] = utils::JsonObject();
                    tensors[name]["dtype"] = Tensor::dtype_to_string(inp->type());
                    tensors[name]["shape"] = inp->shape().symbol().to_string();
                    tensors[name]["memory"] = get_memory(inp, gm);
                    if(inp->is_io_tensor()) {
                        tensors[name]["is_io_tensor"] = true;
                    }
                }
            }
            for (auto out : outputs)
            {
                std::string name = out->name();
                if (!tensors.exist(name))
                {
                    tensors[name] = utils::JsonObject();
                    tensors[name]["dtype"] = Tensor::dtype_to_string(out->type());
                    tensors[name]["shape"] = out->shape().symbol().to_string();
                    tensors[name]["memory"] = get_memory(out, gm);
                    if(out->is_io_tensor()) {
                        tensors[name]["is_io_tensor"] = true;
                    }
                }
            }

            // search the tensors of internal nodes
            if (SubGraphNode::is_subgraph_node(node))
            {
                auto m_sg = SubGraphNode::from_node(node);
                for (auto internal_node : m_sg->subgraph()->nodes())
                {
                    for (auto inp : internal_node->inputs())
                    {
                        std::string name = inp->name();
                        if (!tensors.exist(name))
                        {
                            tensors[name] = utils::JsonObject();
                            tensors[name]["dtype"] = Tensor::dtype_to_string(inp->type());
                            tensors[name]["shape"] = inp->shape().symbol().to_string();
                        }
                    }
                    for (auto out : internal_node->outputs())
                    {
                        std::string name = out->name();
                        if (!tensors.exist(name))
                        {
                            tensors[name] = utils::JsonObject();
                            tensors[name]["dtype"] = Tensor::dtype_to_string(out->type());
                            tensors[name]["shape"] = out->shape().symbol().to_string();
                        }
                    }
                }
            }
        }

        return json;
    }

    std::string Profiler::json_from_graph(GraphPtr g)
    {
        auto json = jsonobj_from_graph(g, m_gm);
        auto &profile_out = json["profile"];
        auto &profile_event = profile_out["profile events"];
        auto events = registered_events();
        std::string prof_unit = aipubt::BuilderConfig::get<std::string>("prof_unit");

        auto target = aipubt::get_target();
        auto default_events = _interface->get_default_events(prof_unit);
        if (default_events == events)
        {
            profile_event = "default";
        }
        else
        {
            for (auto it : events)
            {
                profile_event.push_back(it);
            }
        }
        if (profile_tensor)
        {
            profile_out["output size"] = profile_tensor->mem_size();
        }
        profile_out["frequency"] = __default_frequency__[target->isainfo().type];
        profile_out["prof_unit"] = prof_unit;
        auto fn = interface()->get_function("auxiliary");
        if (fn)
        {
            utils::JsonObject val = cast_any<utils::JsonObject>(fn(0));
            profile_out["auxiliary info"] = val;
        }
#ifdef INTERNAL_MAGIC_NUMBER
        json["magic number"] = STR(INTERNAL_MAGIC_NUMBER);
#endif
        return json.to_string();
    }

    bool Profiler::generate_graph_json(GraphPtr g, const std::string &file_name)
    {
        auto json = json_from_graph(g);
        std::ofstream os(file_name);
        if (os.is_open())
        {
            os << json;
            os << std::endl;
            os.close();
            return true;
        }
        return false;
    }

    Profiler *get_profiler()
    {
        std::lock_guard<std::mutex> lck(instance_lock);
        static std::shared_ptr<Profiler> profiler = nullptr;
        if (profiler == nullptr)
        {
            Profiler *_profiler = new Profiler();
            profiler = std::shared_ptr<Profiler>(_profiler);
        }
        return profiler.get();
    }

    void get_operation_depth(GraphPtr graph, uint32_t &depth, std::vector<NodePtr> &ret)
    {
        // remove virtual subgraph
        std::function<uint32_t(NodePtr, std::vector<NodePtr> &)> get_operation_depth_func = [&get_operation_depth_func](NodePtr n, std::vector<NodePtr> &list) {
            uint32_t node_depth = 0;

            if ((SubGraphNode::is_subgraph_node(n) && n->attrs().count(Constants::plugin_attrs_name)) ||
                (n->type() == OpType_LayoutConvertor && !is_a_remapping_convertor(n, false)))
            {
                node_depth++;
                list.push_back(n);
            }
            else if (SubGraphNode::is_subgraph_node(n))
            {
                // no Constants::plugin_attrs_name
                SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(n);
                if (n->type() == OpType_If)
                {
                    std::unordered_map<NodePtr, size_t> index_map;
                    std::vector<NodePtr> then_nodes;
                    std::vector<NodePtr> else_nodes;
                    std::vector<NodePtr> then_branch_nodes;
                    std::vector<NodePtr> else_branch_nodes;
                    uint32_t if_depth = 0;
                    uint32_t head_depth = 0;
                    uint32_t sep_depth = 0;
                    uint32_t tail_depth = 0;
                    uint32_t then_depth = 0;
                    uint32_t else_depth = 0;
                    IfNodePtr ifnode = std::static_pointer_cast<IfNode>(n);
                    NodePtr head = ifnode->head();
                    // NodePtr sep = ifnode->sep();
                    NodePtr tail = ifnode->tail();
                    auto graphs = *ifnode->subgraph();
                    auto nodes = graphs.nodes();

                    for (size_t i = 0; i < nodes.size(); i++)
                    {
                        index_map[nodes[i]] = i;
                    }

                    ifnode->get_branch_nodes(then_nodes, else_nodes);
                    std::sort(then_nodes.begin(), then_nodes.end(), [&index_map](NodePtr a, NodePtr b) {
                        return index_map[a] < index_map[b];
                    });
                    std::sort(else_nodes.begin(), else_nodes.end(), [&index_map](NodePtr a, NodePtr b) {
                        return index_map[a] < index_map[b];
                    });

                    head_depth = get_operation_depth_func(head, list);

                    for (auto node : then_nodes)
                    {
                        then_depth += get_operation_depth_func(node, then_branch_nodes);
                    }
                    for (auto node : else_nodes)
                    {
                        else_depth += get_operation_depth_func(node, else_branch_nodes);
                    }
                    if (then_depth > else_depth)
                    {
                        if_depth = then_depth;
                        list.insert(list.end(), then_branch_nodes.begin(), then_branch_nodes.end());
                    }
                    else
                    {
                        if_depth = else_depth;
                        list.insert(list.end(), else_branch_nodes.begin(), else_branch_nodes.end());
                    }

                    // sep has no RO
                    // sep_depth = get_operation_depth_func(sep, list);
                    tail_depth = get_operation_depth_func(tail, list);

                    node_depth = head_depth + sep_depth + tail_depth + if_depth;
                }
                else if (n->type() == OpType_Loop)
                {
                    uint32_t loop_head_depth = 0;
                    uint32_t loop_body_depth = 0;
                    uint32_t loop_tail_depth = 0;
                    uint32_t loop_count = 0;
                    LoopNodePtr sgnode = std::static_pointer_cast<LoopNode>(n);
                    NodePtr head = sgnode->head();
                    NodePtr tail = sgnode->tail();
                    auto params = sgnode->params();
                    auto graphs = *sgnode->subgraph();
                    auto nodes = graphs.nodes();
                    size_t nodes_count = nodes.size();

                    if (params.count("loop_default_count"))
                    {
                        loop_count = params["loop_default_count"];
                    }
                    else
                    {
                        if (params.count("scan_outputs_num") && params["scan_outputs_num"] != 0)
                        {
                            auto tail_inputs = tail->inputs();
                            auto tail_outputs = tail->outputs();
                            auto input_rit = tail_inputs.crbegin();
                            auto output_rit = tail_outputs.crbegin();
                            if ((*output_rit)->shape().size() > (*input_rit)->shape().size())
                            {
                                loop_count = (*output_rit)->shape().size() / (*input_rit)->shape().size();
                            }
                        }
                    }
                    if (loop_count == 0)
                    {
                        loop_count = 1;
                        WARN() << "The loop_default_count is not found, so the loop count defaults to 1. This may cause the profiler to lose some node data.";
                    }

                    loop_head_depth = get_operation_depth_func(head, list);

                    for (uint32_t i = 1; i < nodes_count - 1; i++)
                    {
                        loop_body_depth += get_operation_depth_func(nodes[i], list);
                    }
                    loop_tail_depth = get_operation_depth_func(tail, list);

                    node_depth = loop_head_depth + (loop_body_depth + loop_tail_depth) * loop_count;
                }
                else
                {
                    /* It is a virtual subgraph */
                    for (auto node : *sgnode->subgraph())
                    {
                        node_depth += get_operation_depth_func(node, list);
                    }
                }
            }
            return node_depth;
        };

        for (auto node : *graph)
        {
            depth += get_operation_depth_func(node, ret);
        }
    }

    void get_operation_nodes(GraphPtr graph, std::vector<NodePtr> &ret)
    {
        // remove virtual subgraph
        std::function<void(NodePtr)> get_operation_nodes_func = [&ret, &get_operation_nodes_func](NodePtr n) {
            if (SubGraphNode::is_subgraph_node(n))
            {
                SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(n);
                if (sgnode->attrs().count(Constants::plugin_attrs_name))
                {
                    ret.push_back(n);
                }
                else
                {
                    /* It is a virtual subgraph */
                    for (auto node : *sgnode->subgraph())
                    {
                        get_operation_nodes_func(node);
                    }
                }
            }
            else
            {
                ret.push_back(n);
            }
        };

        for (auto node : *graph)
        {
            get_operation_nodes_func(node);
        }
    }

    uint64_t Profiler::get_max_perf_sample_num(uint64_t max_perf_count) const
    {
        if (BuilderConfig::exist(Constants::max_perf_sample_num))
        {
            uint64_t val = BuilderConfig::get<uint64_t>(Constants::max_perf_sample_num);
            if (val > 0)
            {
                max_perf_count = val;
            }
        }
        return max_perf_count;
    }

    uint64_t Profiler::get_max_aiff_perf_sample_num(uint64_t max_perf_aiff_count) const
    {
        if (BuilderConfig::exist(Constants::max_aiff_perf_sample_num))
        {
            uint64_t val = BuilderConfig::get<uint64_t>(Constants::max_aiff_perf_sample_num);
            if (val > 0)
            {
                max_perf_aiff_count = val;
            }
        }
        return max_perf_aiff_count;
    }

    uint64_t Profiler::get_max_dma_perf_sample_num(uint64_t max_perf_dma_count) const
    {
        if (BuilderConfig::exist(Constants::max_dma_perf_sample_num))
        {
            uint64_t val = BuilderConfig::get<uint64_t>(Constants::max_dma_perf_sample_num);
            if (val > 0)
            {
                max_perf_dma_count = val;
            }
        }
        return max_perf_dma_count;
    }

    uint64_t Profiler::get_max_custom_perf_sample_num(uint64_t max_perf_custom_count) const
    {
        if (BuilderConfig::exist(Constants::max_custom_perf_sample_num))
        {
            uint64_t val = BuilderConfig::get<uint64_t>(Constants::max_custom_perf_sample_num);
            if (val > 0)
            {
                max_perf_custom_count = val;
            }
        }
        return max_perf_custom_count;
    }

} // namespace AIPUBuilder
} // namespace aipubt

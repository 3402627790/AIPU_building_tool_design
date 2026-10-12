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
 * @Author: Neo Wang 
 * @Date: 2020-01-29 14:25:56 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-04 17:47:28
 */
#include "builder/builder.h"
#include "builder/plugin.h"
#include "builder/profiler.h"
#include "builder/ro_generator.h"
#include "builder/target.h"
#include "builder/text_generator.h"
#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/core.h"
#include "core/graph.h"
#include "core/logger.h"
#include "core/node.h"
#include "core/plugin_loader.h"
#include "utils/utils.h"

#define SM_OFFSET 0xf8000000
#define SM_SIZE 0x40000 // 256K

static inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

namespace aipubt
{
/*
 * Description of the RO reserved for GBuilder in each layer:
 * |-------------------------------------------------------|
 *    ro[0]
 * |-------------------------------------------------------|
 *    ...
 * |-------------------------------------------------------|
 *    ro[51]: descriptor dumptensor base
 * |-------------------------------------------------------|
 *    ro[52]: layer id for profiler/debugger/simulator
 * |-------------------------------------------------------|
 *    ro[53]: output tensor shape base for dynamic shape
 * |-------------------------------------------------------|
 *    ro[54]: intermediate ro base for dynamic shape
 * |-------------------------------------------------------|
 *    ro[55]: descriptor base
 * |-------------------------------------------------------|
 *    ro[56]: descriptor param base
 * |-------------------------------------------------------|
 *    ro[57]: descriptor act base
 * |-------------------------------------------------------|
 *    ro[58]: for weight to sm
 * |-------------------------------------------------------|
 *    ro[59]: for weight to sm
 * |-------------------------------------------------------|
 *    ro[60]: for weight to sm
 * |-------------------------------------------------------|
 *    ro[61]: for weight to gm
 * |-------------------------------------------------------|
 *    ro[62]: for weight to gm
 * |-------------------------------------------------------|
 *    ro[63]: for weight to gm
 * |-------------------------------------------------------|
 */
RO_GENERATOR(OpType_SubGraph)
{
    SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
    auto upper_ro_size = node->get_attr<size_t>(Constants::ro_offset) / 4;
    BuilderParams params;

    if (!node->attrs().count(Constants::plugin_attrs_name))
    {
        //get parambase for the virtual subgraph
        auto upper_pb = node->get_attr<TensorPtr>(Constants::parambase);
        auto tp_base = node->get_attr<TensorPtr>(Constants::temp_param_base);
        auto ts_base = node->get_attr<TensorPtr>(Constants::tensor_shape_base);
        for (auto m_node : *sgnode->subgraph())
        {
            auto ro_generator = get_ro_generator(m_node->type());
            if (!m_node->attrs().count(Constants::parambase))
            {
                auto _pbt = std::make_shared<Tensor>(m_node->name() + "/pb");
                m_node->set_attr(Constants::parambase, _pbt);
                m_node->placeholders().push_back(_pbt);
            }
            auto pb = m_node->get_attr<TensorPtr>(Constants::parambase);
            pb->memory_offset()->set_base_offset(upper_pb->memory_offset());
            pb->memory_offset()->relative_offset = sizeof(uint32_t) * params.size();
            m_node->set_attr(Constants::temp_param_base, tp_base);
            m_node->set_attr(Constants::tensor_shape_base, ts_base);
            if (ro_generator)
            {
                size_t ro_start = upper_ro_size + params.size();
                m_node->set_attr(Constants::ro_offset, sizeof(uint32_t) * ro_start);
                ro_generator(m_node, params);
                size_t ro_size = upper_ro_size + params.size() - ro_start;
                m_node->set_attr(Constants::real_ro_size, sizeof(uint32_t) * ro_size);
                if (int res = params.size() % DEFAULT_RO_NUMBER_PER_LAYER)
                {
                    for (int n = 64; n > res; --n)
                    {
                        params.push_back(0);
                    }
                }
                ro_size = upper_ro_size + params.size() - ro_start;
                m_node->set_attr(Constants::ro_size, sizeof(uint32_t) * ro_size);
                m_node->set_attr(Constants::ro_end, upper_ro_size + params.size());
                // INFO() << "ro end for " << m_node << " " << upper_ro_size + params.size();
                bool load_gm_w = m_node->attrs().count(Constants::ld_w_from_gm);
                bool load_gm_w_from_other = false;
                NodePtr root_node = m_node;
                if (!m_node->attrs().count(Constants::ld_w_from_gm) && m_node->attrs().count(Constants::same_w_same_level))
                {
                    root_node = m_node->get_attr<NodeRef>(Constants::same_w_same_level).lock();
                    load_gm_w_from_other = root_node->attrs().count(Constants::ld_w_from_gm);
                }
                auto switch_for_debug = aipubt::BuilderConfig::get<bool>("switch_for_debug");
                // load_gm_w_from_other = false;
                if (load_gm_w || load_gm_w_from_other)
                {
                    INFO() << "ld w from gm " << m_node;
                    auto start_node = root_node->get_attr<NodeRef>(Constants::ld_w_from_gm).lock();
                    auto end_node = start_node->get_attr<NodeRef>(Constants::mov_w_to_gm).lock();
                    if (start_node)
                    {
                        uint32_t gm_base = 0xf0000000;
                        uint32_t gm_offset = start_node->get_attr<uint32_t>(Constants::mov_w_gm_offset) + gm_base;
                        uint32_t const_size = start_node->get_attr<uint32_t>(Constants::mov_w_gm_size);
                        uint32_t total_constant_size = 0;
                        for (auto cts : m_node->constants())
                        {
                            total_constant_size += cts.second->mem_size();
                            total_constant_size = aligned(total_constant_size, cts.second->memory_offset()->get_base_object()->align);
                        }
                        int gm_size = BuilderConfig::get<int>(Constants::gm_size);
                        auto gm_memory = std::make_shared<Tensor>(TensorShape({gm_size}));
                        gm_memory->memory_offset()->set_base_offset(gm_offset);
                        uint32_t ws = 0;
                        TensorPtr new_w;
                        if (total_constant_size > const_size || switch_for_debug)
                        {
                            WARN() << "cannot move " << root_node->name() << " weight to gm due to weight size is larger("
                                   << total_constant_size << ") than original(" << const_size << ") after ro generator";
                            if (m_node->attrs().count(Constants::ld_w_from_gm))
                            {
                                m_node->erase_attr(Constants::ld_w_from_gm);
                                m_node->erase_attr(Constants::ld_w_gm_offset);
                            }
                            new_w = ComputeTensor::zeros({(int32_t)total_constant_size}, Tensor::Dtype::UINT8);
                            for (auto it = m_node->constants().begin(); it != m_node->constants().end(); ++it)
                            {
                                memcpy(new_w->data() + ws, it->second->data(), it->second->mem_size());
                                it->second->memory_offset()->set_base_offset(new_w->memory_offset()->get_base_object());
                                it->second->memory_offset()->relative_offset = ws;
                                ws += it->second->mem_size();
                                ws = aligned(ws, it->second->memory_offset()->get_base_object()->align);
                            }
                            m_node->constants().clear();
                            m_node->constants()[m_node->name() + "_merge_constants"] = new_w;
                        }
                        else
                        {
                            new_w = ComputeTensor::zeros({(int32_t)const_size}, Tensor::Dtype::UINT8);
                            for (auto c : m_node->constants())
                            {
                                memcpy(new_w->data() + ws, c.second->data(), c.second->mem_size());
                                c.second->memory_offset()->set_base_offset(gm_memory->memory_offset());
                                c.second->memory_offset()->relative_offset = ws;
                                ws += c.second->mem_size();
                                ws = aligned(ws, c.second->memory_offset()->get_base_object()->align);
                            }

                            m_node->constants().clear();
                            m_node->constants()[m_node->name() + "_merge_constants"] = new_w;
                        }

                        if (load_gm_w)
                        {
                            uint32_t ro_end = start_node->get_attr<uint32_t>(Constants::ro_end);
                            if (ro_end > ro.size())
                            {
                                params[ro_end - ro.size() - 3] = new_w;
                                params[ro_end - ro.size() - 2] = gm_memory;
                                params[ro_end - ro.size() - 1] = new_w->mem_size();
                            }
                            else
                            {
                                ro[ro_end - 3] = new_w;
                                ro[ro_end - 2] = gm_memory;
                                ro[ro_end - 1] = new_w->mem_size();
                            }
                        }
                    }
                }
                if (m_node->attrs().count(Constants::movwtosm))
                {
                    auto next_node = m_node->get_attr<NodeRef>(Constants::movwtosm).lock();
                    if (next_node)
                    {
                        // we memger all constants to one
                        int ws = 0;
                        int preload = m_node->get_attr<int>("_preload_sm_size");
                        for (auto c : next_node->constants())
                        {
                            ws += c.second->mem_size();
                            ws = aligned(ws, c.second->memory_offset()->get_base_object()->align);
                        }
                        if (ws < preload && ws > 0)
                        {
                            int sm = 0;
                            if (m_node->attrs().count("_use_sm") && m_node->get_attr<int>("_use_sm") == 0)
                            {
                                sm = 1;
                            }
                            next_node->set_attr("_use_sm", sm);
                            auto sm_memory = std::make_shared<Tensor>(TensorShape({preload}));
                            sm_memory->memory_offset()->set_base_offset(SM_OFFSET + (SM_SIZE / 2) * sm);
                            // we memger all constants to one
                            auto new_w = ComputeTensor::zeros({(int32_t)ws}, Tensor::Dtype::UINT8);
                            ws = 0;
                            for (auto c : next_node->constants())
                            {
                                memcpy(new_w->data() + ws, c.second->data(), c.second->mem_size());
                                c.second->memory_offset()->set_base_offset(sm_memory->memory_offset());
                                c.second->memory_offset()->relative_offset = ws;
                                ws += c.second->mem_size();
                                ws = aligned(ws, c.second->memory_offset()->get_base_object()->align);
                            }
                            next_node->constants().clear();
                            next_node->constants()["_merge_constants"] = new_w;
                            // INFO() << "MOVWtOSM........................."
                            //        << pb->memory_offset()->relative_offset << " and size=" << new_w->mem_size();
                            // params[params.size() - 6] = new_w;
                            // params[params.size() - 5] = sm_memory;
                            // params[params.size() - 4] = new_w->mem_size();
                            uint32_t ro_end = m_node->get_attr<uint32_t>(Constants::ro_end);
                            if (ro_end > ro.size())
                            {
                                params[ro_end - ro.size() - 6] = new_w;
                                params[ro_end - ro.size() - 5] = sm_memory;
                                params[ro_end - ro.size() - 4] = new_w->mem_size();
                            }
                            else
                            {
                                ro[ro_end - 6] = new_w;
                                ro[ro_end - 5] = sm_memory;
                                ro[ro_end - 4] = new_w->mem_size();
                            }
                        }
                    }
                }
                // always set bases for dynamic shape and max-batch
                bool is_control_flow_node = false;
                if (SubGraphNode::is_subgraph_node(m_node) &&
                    !m_node->attrs().count(Constants::plugin_attrs_name))
                {
                    is_control_flow_node = true;
                }
                if (!is_control_flow_node)
                {
                    auto ro_size = m_node->get_attr<uint32_t>(Constants::ro_size) / 4;
                    TensorPtr desc_base = nullptr;
                    if (m_node->attrs().count(Constants::descriptorbase))
                    {
                        desc_base = m_node->get_attr<TensorPtr>(Constants::descriptorbase);
                    }
                    TensorPtr desc_param_base = nullptr;
                    if (m_node->attrs().count(Constants::descriptor_param_base))
                    {
                        desc_param_base = m_node->get_attr<TensorPtr>(Constants::descriptor_param_base);
                    }
                    TensorPtr desc_act_base = nullptr;
                    if (m_node->attrs().count(Constants::descriptor_act_base))
                    {
                        desc_act_base = m_node->get_attr<TensorPtr>(Constants::descriptor_act_base);
                    }
                    TensorPtr desc_dt_base = nullptr;
                    if (m_node->attrs().count(Constants::descriptor_dt_base))
                    {
                        desc_dt_base = m_node->get_attr<TensorPtr>(Constants::descriptor_dt_base);
                    }
                    uint32_t desc_size = 0;
                    if (m_node->attrs().count(Constants::desc_size))
                    {
                        desc_size = m_node->get_attr<uint32_t>(Constants::desc_size) / 4;
                    }
                    if (m_node->attrs().count(Constants::descriptor_param))
                    {
                        desc_size += m_node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_param).size();
                    }
                    if (m_node->attrs().count(Constants::descriptor_act))
                    {
                        desc_size += m_node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_act).size();
                    }
                    if (m_node->attrs().count(Constants::descriptor_dt))
                    {
                        desc_size += m_node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_dt).size();
                    }
                    if (ro_size)
                    {
                        int layer_id = m_node->get_layer_id();
                        // DEBUG() << m_node->name() << " layer_id=" << layer_id;
                        uint32_t ro_index = params.size() - ro_size;
                        for (uint32_t i = 0; i < ro_size / DEFAULT_RO_NUMBER_PER_LAYER; ++i)
                        {
                            uint32_t start = ro_index + i * DEFAULT_RO_NUMBER_PER_LAYER;
                            // layer_id is used by profiler/debugger/simulator to locate plugin node
                            params[start + 52] = layer_id;
                            params[start + 53] = ts_base;
                            params[start + 54] = tp_base;
                            if (desc_size)
                            {
                                ReadOnlyParam empty(0);
                                params[start + 51] = desc_dt_base ? ReadOnlyParam(desc_dt_base) : empty;
                                params[start + 55] = desc_base ? ReadOnlyParam(desc_base) : empty;
                                params[start + 56] = desc_param_base ? ReadOnlyParam(desc_param_base) : empty;
                                params[start + 57] = desc_act_base ? ReadOnlyParam(desc_act_base) : empty;
                            }
                        }
                    }
                }
            }
        }
        auto &constants = node->constants();
        //clean
        constants.clear();
        int idx = 0;
        for (auto sn : sgnode->subgraph()->nodes())
        {
            //rename the weight as node_name/weight_name
            for (auto cts : sn->constants())
            {
                auto t = std::make_shared<Tensor>(*cts.second);
                cts.second->memory_offset()->set_base_offset(t->memory_offset());
                cts.second->memory_offset()->relative_offset = 0;
                constants[std::to_string(idx) + "/" + cts.first] = t;
                idx++;
            }
        }
        if (get_target()->isainfo().type < ISA_ZYv5)
        {
            // fill 64 ro for the subgraph, only for old format
            for (int i = 0; i < 64; ++i)
            {
                params.push_back(0);
            }
        }
    }
    else
    {
        auto ml = node->get_attr<NodePtrList>("_match_list");
        auto plugin_info = node->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
        auto plugin = plugin_info->init_handle();
        BaseNodePtrList bml;
        for (auto p : ml)
        {
            bml.push_back(p);
        }
        // INFO() << "Generating RO for " << sgnode->name() << "...  ro offset:" << upper_ro_size;
        params = plugin->generate_params(sgnode, bml);
        std::string error_msg;
        if (get_target()->isainfo().type >= ISA_ZYv5 &&
            !check_params(params, error_msg))
        {
            ERROR() << "Failed to check params: " << error_msg << "\n"
                    << "Node: " << sgnode->name() << "\n"
                    << "Plugin: " << plugin_info->name;
        }
    }
    for (auto p : params)
    {
        ro.push_back(p);
    }

    int workspace_size = node->get_attr<int>("workspace_size");
    for (auto n : *sgnode->subgraph())
    {
        auto ws = n->get_attr<int>("workspace_size");
        workspace_size = workspace_size > ws ? workspace_size : ws;
    }
    node->set_attr("workspace_size", workspace_size);
}
RO_GENERATOR_END(OpType_SubGraph);

static void gen_rt_tensor_param(NodePtr node)
{
    BuilderParams desc_rt_param;
    if (BuilderConfig::get<bool>(Constants::dump_rt_output_tensors))
    {
        desc_rt_param.push_back((uint32_t)node->outputs().size());
        for (auto o : node->outputs())
        {
            desc_rt_param.push_back(o);
            desc_rt_param.push_back(o->buffer_size());
        }
    }
    else if (BuilderConfig::get<bool>(Constants::dump_rt_input_tensors))
    {
        desc_rt_param.push_back((uint32_t)node->inputs().size());
        for (auto i : node->inputs())
        {
            desc_rt_param.push_back(i);
            desc_rt_param.push_back(i->buffer_size());
        }
    }
    else if (BuilderConfig::get<int>(Constants::dump_rt_tensor_id) != -1)
    {
        int target_id = BuilderConfig::get<int>(Constants::dump_rt_tensor_id);
        NodePtr realtime_n = nullptr;
        auto upper_sg = std::dynamic_pointer_cast<SubGraph>(node->graph());
        for (auto n : upper_sg->nodes())
        {
            int32_t layer_id = n->get_attr<int>("layer_id");
            // INFO() << "layer id " << n->name() << " " << layer_id << " " << target_id;
            if (target_id == layer_id)
            {
                realtime_n = n;
                break;
            }
        }
        if (realtime_n)
        {
            desc_rt_param.push_back((uint32_t)realtime_n->outputs().size());
            for (auto o : realtime_n->outputs())
            {
                desc_rt_param.push_back(o);
                desc_rt_param.push_back(o->buffer_size());
            }
        }
        else
        {
            desc_rt_param.push_back(0);
        }
    }
    node->set_attr(Constants::descriptor_dt, desc_rt_param);
}

DESC_GENERATOR(OpType_SubGraph)
{
    SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
    auto upper_desc_size = node->get_attr<size_t>(Constants::desc_offset) / 4;
    int workspace_size = node->get_attr<int>("workspace_size");
    for (auto n : *sgnode->subgraph())
    {
        auto ws = n->get_attr<int>("workspace_size");
        workspace_size = workspace_size > ws ? workspace_size : ws;
    }
    node->set_attr("workspace_size", workspace_size);
    BuilderParams params;
    bool dump_rt_tensor = BuilderConfig::get<bool>(Constants::dump_rt_output_tensors) || BuilderConfig::get<bool>(Constants::dump_rt_input_tensors) || (BuilderConfig::get<int>(Constants::dump_rt_tensor_id) != -1);

    if (!node->attrs().count(Constants::plugin_attrs_name))
    {
        TensorPtr next_descriptor_length = nullptr;
        TensorPtr next_descriptor_offset = nullptr;
        TensorPtr next_descriptor_eof = nullptr;
        //get descbase for the virtual subgraph
        auto upper_db = node->get_attr<PlaceholderPtr>(Constants::descriptorbase);
        auto act_base = node->get_attr<PlaceholderPtr>(Constants::descriptor_act_base);
        auto param_base = node->get_attr<PlaceholderPtr>(Constants::descriptor_param_base);
        auto dt_base = node->get_attr<PlaceholderPtr>(Constants::descriptor_dt_base);

        std::vector<ReadOnlyParam> desc_act;
        std::vector<ReadOnlyParam> desc_param;
        std::vector<ReadOnlyParam> desc_dt;
        for (auto m_node : *sgnode->subgraph())
        {
            auto desc_generator = get_descriptor_generator(m_node->type());
            auto m_act_base = std::make_shared<Placeholder>(act_base);
            auto m_param_base = std::make_shared<Placeholder>(param_base);
            auto m_dt_base = std::make_shared<Placeholder>(dt_base);
            if (desc_generator)
            {
                if (!m_node->attrs().count(Constants::parambase))
                {
                    auto _pbt = std::make_shared<Tensor>(m_node->name() + "/parambase");
                    m_node->set_attr(Constants::parambase, _pbt);
                }
                if (!m_node->attrs().count(Constants::descriptorbase))
                {
                    auto _pbt = std::make_shared<Tensor>(m_node->name() + "/descriptorbase");
                    m_node->set_attr(Constants::descriptorbase, _pbt);
                }
                m_node->set_attr(Constants::first_descriptor_length, next_descriptor_length);
                m_node->set_attr(Constants::first_descriptor_offset, next_descriptor_offset);
                m_node->set_attr(Constants::previous_descriptor_eof, next_descriptor_eof);
                m_node->set_attr(Constants::descriptor_act_base, m_act_base);
                m_node->set_attr(Constants::descriptor_param_base, m_param_base);
                m_node->set_attr(Constants::descriptor_dt_base, m_dt_base);
                auto db = m_node->get_attr<PlaceholderPtr>(Constants::descriptorbase);
                *db = *upper_db + sizeof(uint32_t) * params.size();
                size_t desc_start = upper_desc_size + params.size();
                m_node->set_attr(Constants::desc_offset, sizeof(uint32_t) * desc_start);
                desc_generator(m_node, params);
                size_t desc_size = upper_desc_size + params.size() - desc_start;
                m_node->set_attr(Constants::real_desc_size, sizeof(uint32_t) * desc_size);
                if (int res = params.size() % DEFAULT_RO_NUMBER_PER_LAYER)
                {
                    for (int n = 64; n > res; --n)
                    {
                        params.push_back(0);
                    }
                }
                desc_size = upper_desc_size + params.size() - desc_start;
                m_node->set_attr(Constants::desc_size, sizeof(uint32_t) * desc_size);
            }
            else if (dump_rt_tensor)
            {
                m_node->set_attr(Constants::descriptor_dt_base, m_dt_base);
                gen_rt_tensor_param(m_node);
            }

            // if debug mode or profiler enable, set next descriptor nullptr to break the chain
            if (BuilderConfig::get<bool>(Constants::enable_gm_enhance) ||
                BuilderConfig::get<bool>(Constants::enable_dynamic_gm) ||
                BuilderConfig::get<bool>(Constants::enable_x2_sm_opt) ||
                BuilderConfig::get<bool>(Constants::break_descriptor_chain))
            {
                next_descriptor_length = nullptr;
                next_descriptor_offset = nullptr;
                next_descriptor_eof = nullptr;
            }
            else
            {
                next_descriptor_length = m_node->get_attr<TensorPtr>(Constants::next_descriptor_length);
                next_descriptor_offset = m_node->get_attr<TensorPtr>(Constants::next_descriptor_offset);
                next_descriptor_eof = m_node->get_attr<TensorPtr>(Constants::next_descriptor_eof);
            }
            *m_param_base = *param_base + (desc_param.size() * sizeof(uint32_t));
            if (m_node->attrs().count(Constants::descriptor_param))
            {
                for (auto k : m_node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_param))
                {
                    desc_param.push_back(k);
                }
            }
            *m_act_base = *act_base + (desc_act.size() * sizeof(uint32_t));
            if (m_node->attrs().count(Constants::descriptor_act))
            {
                for (auto k : m_node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_act))
                {
                    desc_act.push_back(k);
                }
            }
            *m_dt_base = *dt_base + (desc_dt.size() * sizeof(uint32_t));
            if (m_node->attrs().count(Constants::descriptor_dt))
            {
                for (auto k : m_node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_dt))
                {
                    desc_dt.push_back(k);
                }
            }
        }
        if (desc_act.size())
        {
            node->set_attr(Constants::descriptor_act, desc_act);
        }
        if (desc_param.size())
        {
            node->set_attr(Constants::descriptor_param, desc_param);
        }
        if (desc_dt.size())
        {
            node->set_attr(Constants::descriptor_dt, desc_dt);
        }
    }
    else
    {

        auto desc_generator = get_descriptor_generator(node->type());
        if (desc_generator)
        {
            auto ml = node->get_attr<NodePtrList>("_match_list");
            auto plugin_info = node->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
            auto plugin = plugin_info->init_handle();
            BaseNodePtrList bml;
            for (auto p : ml)
            {
                bml.push_back(p);
            }
            std::set<TensorPtr> origin_c;
            if (BuilderConfig::get<bool>(Constants::remove_origin_constant))
            {
                for (auto cts : sgnode->constants())
                {
                    origin_c.insert(cts.second);
                }
            }
            DEBUG() << "Generating descriptor for " << sgnode->name() << "...";
            params = plugin->generate_descriptor(sgnode, bml);

            if (BuilderConfig::get<bool>(Constants::remove_origin_constant))
            {
                std::set<TensorPtr> new_c;
                for (auto cts : sgnode->constants())
                {
                    new_c.insert(cts.second);
                }
                std::set<TensorPtr> remove_c;
                for (auto c : origin_c)
                {
                    if (!contains(new_c, c))
                    {
                        remove_c.insert(c);
                    }
                }
                std::set<void *> remove_d;
                for (auto c : remove_c)
                {
                    auto data = (void *)c->data();
                    remove_d.insert(data);
                }
                for (auto sn : sgnode->subgraph()->nodes())
                {
                    for (auto cts : sn->constants())
                    {
                        auto data = (void *)cts.second->data();
                        if (contains(remove_d, data))
                        {
                            // INFO() << "remove constant " << cts.second->name() << " " << cts.second->get_ref_count();
                            cts.second->set_data(nullptr, 0);
                        }
                    }
                }

                auto upper_sg = std::dynamic_pointer_cast<SubGraph>(node->graph());
                auto upper_node = upper_sg->subgraphnode();
                for (auto cts : upper_node->constants())
                {
                    auto data = (void *)cts.second->data();
                    if (contains(remove_d, data))
                    {
                        // INFO() << "remove constant " << cts.second->name() << " " << cts.second->get_ref_count();
                        cts.second->set_data(nullptr, 0);
                    }
                }
                for (auto c : remove_c)
                {
                    // INFO() << "remove constant " << c->name() << " " << c->get_ref_count();
                    c->set_data(nullptr, 0);
                }
            }
        }

        if (dump_rt_tensor)
        {
            gen_rt_tensor_param(node);
        }
    }
    descriptor.insert(descriptor.end(), params.begin(), params.end());
}
DESC_GENERATOR_END(OpType_SubGraph);

KERNEL_GENERATOR(OpType_SubGraph)
{
    SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
    auto codetree = node->get_attr<CodeTreePtr>(Constants::codetree);
    if (!node->attrs().count(Constants::plugin_attrs_name))
    {
        //use default generator
        std::ostringstream oss;
        SubGraphNodePtr sg = std::static_pointer_cast<SubGraphNode>(node);
        std::string terminator = BuilderConfig::get<std::string>("terminator");
        for (auto m_node : *sg->subgraph())
        {
            if (!m_node->attrs().count(Constants::codetree))
            {
                m_node->set_attr(Constants::codetree, std::make_shared<CodeTree>(m_node->name(), m_node));
            }
            auto m_codetree = m_node->get_attr<CodeTreePtr>(Constants::codetree);
            auto code_generator = get_kernel_generator(m_node->type());
            if (code_generator)
            {
                auto code = code_generator(m_node);
                if (code.empty())
                {
                    //TODO get the kernel function
                }
                m_node->set_attr(Constants::kernel_func, code); //set debug kernel info

                if (m_node->attrs().count(Constants::mov_w_to_gm))
                {
                    oss << "_aipubt_cpy_weight_to_gm\n";
                    codetree->push_back("_aipubt_cpy_weight_to_gm");
                }
                if (m_node->attrs().count(Constants::mov_w_to_gm_wait))
                {
                    bool wait_w_in_front = m_node->get_attr(Constants::wait_w_in_front);
                    if (wait_w_in_front)
                    {
                        oss << "_aipubt_cpy_weight_to_gm_wait\n";
                        oss << "_aipubt_tec_sync\n";
                        codetree->push_back("_aipubt_cpy_weight_to_gm_wait");
                        codetree->push_back("_aipubt_tec_sync");
                    }
                }
                bool mov_w_to_sm = false;
                if (m_node->attrs().count(Constants::movwtosm))
                {
                    auto next_node = m_node->get_attr<NodeRef>(Constants::movwtosm).lock();
                    if (next_node && next_node->attrs().count("_use_sm"))
                    {
                        mov_w_to_sm = true;
                    }
                }
                if (mov_w_to_sm)
                {
                    oss << "_aipubt_cpy_weight_to_sm\n";
                    codetree->push_back("_aipubt_cpy_weight_to_sm");
                    // INFO() << "insert MOVWTOSM kernel";
                }
                if (m_node->attrs().count(Constants::mov_d_to_ddr))
                {
                    oss << "_aipubt_cpy_data_to_ddr\n";
                    codetree->push_back("_aipubt_cpy_data_to_ddr");
                }

                oss << code << "\n";
                if (SubGraphNode::is_subgraph_node(m_node) &&
                    !m_node->attrs().count(Constants::plugin_attrs_name))
                {
                    // increase depth for control-flow nodes
                    codetree->codes.push_back(m_codetree);
                }
                else
                {
                    for (auto c : m_codetree->codes)
                    {
                        codetree->codes.push_back(c);
                    }
                }

                if (m_node->attrs().count(Constants::mov_w_to_gm_wait))
                {
                    bool wait_w_in_front = m_node->get_attr(Constants::wait_w_in_front);
                    if (!wait_w_in_front)
                    {
                        oss << "_aipubt_cpy_weight_to_gm_wait\n";
                        codetree->push_back("_aipubt_cpy_weight_to_gm_wait");
                    }
                }
                if (m_node->attrs().count(Constants::mov_d_to_ddr_wait))
                {
                    oss << "_aipubt_cpy_data_to_ddr_wait\n";
                    codetree->push_back("_aipubt_cpy_data_to_ddr_wait");
                }
                // if (mov_w_to_sm)
                // {
                //     oss << "_aipubt_cpy_weight_to_sm_wait\n";
                //     codetree->push_back("_aipubt_cpy_weight_to_sm_wait");
                // }
            }
            // else
            // {
            //     ERROR() << "Unsupport node type: "
            //             << optype_to_string(m_node->type()) << "\tname:"
            //             << m_node->name() << " in subgraph node:" << node->name();
            // }
        }
        node->set_attr("_kernel", oss.str());
        node->set_attr("default_code", true);
        return node->get_attr<std::string>("_kernel");
    }
    else
    {
        auto ml = node->get_attr<NodePtrList>("_match_list");
        auto plugin_info = node->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
        auto plugin = plugin_info->init_handle();
        BaseNodePtrList bml;
        for (auto p : ml)
        {
            bml.push_back(p);
        }
        auto kernel = plugin->generate_kernel(sgnode, bml);
        if (kernel.empty() && get_target()->isainfo().type >= ISA_ZYv5)
        {
            auto code_name = plugin->generate_code_name(sgnode, bml);
            if (code_name.size())
            {
                WARN() << "empty kernel name, will trying to use the first kernel in elf for node:" << sgnode->name();
                kernel = get_kernel_func_name(code_name);
            }
        }
        if (kernel.empty() && get_target()->isainfo().type >= ISA_ZYv5)
        {
            if (!sgnode->attrs().count(Constants::next_descriptor_offset) &&
                sgnode->get_attr<PlaceholderPtr>(Constants::next_descriptor_offset))
            {
                WARN() << "empty kernel name, please recheck the plugin's generate kernels in node:" << sgnode->name();
            }
        }

        bool enable_perf = aipubt::AIPUBuilder::get_profiler()->enable_profile();
        if (enable_perf && !kernel.empty())
        {
            kernel += "\n";
            kernel += Constants::profiler_kernel;
        }
        std::vector<std::string> valid_kernels;
        for (auto &s : utils::split_string(kernel, "\n"))
        {
            if (is_valid_kernel(s))
            {
                valid_kernels.push_back(s);
            }
        }
        if (valid_kernels.size())
        {
            codetree->codes.emplace_back(std::make_shared<CodeTree>(valid_kernels[0], node));
            for (size_t i = 1; i < valid_kernels.size(); ++i)
            {
                codetree->push_back(valid_kernels[i]);
            }
        }
        return kernel;
    }
}
KERNEL_GENERATOR_END(OpType_SubGraph);

static const std::string code_generator_func_OpType_SubGraph(NodePtr node)
{
    SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
    bool enable_perf = aipubt::AIPUBuilder::get_profiler()->enable_profile();
    if (!node->attrs().count(Constants::plugin_attrs_name))
    {
        //use default generator
        std::ostringstream oss;
        SubGraphNodePtr sg = std::static_pointer_cast<SubGraphNode>(node);
        std::string terminator = BuilderConfig::get<std::string>("terminator");
        // build a new functions
        oss << ".function " << sgnode->name() << "\n";
        uint32_t opid = 0;
        if (node->attrs().count(Constants::opid_counter))
        {
            opid = node->get_attr<uint32_t>(Constants::opid_counter);
        }
        for (auto m_node : *sg->subgraph())
        {
            auto code_generator = get_code_generator(m_node->type());
            if (code_generator)
            {
                m_node->set_attr(Constants::opid_counter, opid);
                auto code = code_generator(m_node);
                opid = m_node->get_attr<uint32_t>(Constants::opid_counter);
                //text size=0
                if (code.empty())
                {
                    if (terminator == m_node->name())
                    {
                        oss << "_exit\n";
                    }
                    //need check if the empty is allowded.
                    continue;
                }
                //TODO complete the code
                oss << code << "\n";
                auto node_codes = utils::split_string(code, '\n');
                if (terminator == m_node->name())
                {
                    oss << "_exit\n";
                }
                std::vector<uint32_t> operator_ids;
                for (auto single_code : node_codes)
                {
                    if (single_code == Constants::profiler_code)
                    {
                        continue;
                    }
                    operator_ids.push_back(opid);
                    opid++;
                }
                m_node->set_attr(Constants::operator_id, operator_ids);
                // if (m_node->attrs().count(Constants::movwtosm))
                // {
                //     auto next_node = m_node->get_attr<NodeRef>(Constants::movwtosm).lock();
                //     if (next_node && next_node->attrs().count("_use_sm"))
                //     {
                //         oss << "aipubt_internal_lib.o\n";
                //     }
                // }
            }
            else
            {
                BuilderConfig::set(Constants::show_location, true);
                ERROR() << "Unsupport node type: "
                        << optype_to_string(m_node->type()) << "\tname:"
                        << m_node->name() << " in subgraph node:" << node->name() << " definition:\n"
                        << m_node->to_string();
                std::vector<std::string> logs;
                std::vector<std::string> warns;
                if (!SubGraphNode::is_subgraph_node(m_node))
                {
                    logs = m_node->get_attr_deep<std::vector<std::string>>(Constants::build_error_log);
                    warns = m_node->get_attr_deep<std::vector<std::string>>(Constants::build_warn_log);
                    m_node->erase_attr_deep(Constants::build_error_log);
                    m_node->erase_attr_deep(Constants::build_warn_log);
                }
                else
                {
                    logs = sgnode->get_attr_deep<std::vector<std::string>>(Constants::build_error_log);
                    sgnode->erase_attr_deep(Constants::build_error_log);
                    warns = sgnode->get_attr_deep<std::vector<std::string>>(Constants::build_warn_log);
                    sgnode->erase_attr_deep(Constants::build_warn_log);
                }

                for (auto msg : logs)
                {
                    std::cerr << msg << std::endl;
                    GCompilerLogger::increase_error_count();
                }
                for (auto msg : warns)
                {
                    std::cout << msg << std::endl;
                    GCompilerLogger::increase_warn_count();
                }
            }
        }
        if (node->attrs().count(Constants::opid_counter))
        {
            node->set_attr(Constants::opid_counter, opid);
        }
        oss << ".end_function " << sgnode->name() << "\n";
        // call this function
        oss << sgnode->name() << "\n";
        node->set_attr("_code", oss.str());
        node->set_attr("default_code", true);
        return node->get_attr<std::string>("_code");
    }
    else
    {
        auto ml = node->get_attr<NodePtrList>("_match_list");
        auto plugin_info = node->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
        auto plugin = plugin_info->init_handle();
        BaseNodePtrList bml;
        for (auto p : ml)
        {
            bml.push_back(p);
        }

        std::string ret = plugin->generate_code_name(sgnode, bml);
        if (enable_perf && !ret.empty())
        {
            ret += "\n";
            ret += Constants::profiler_code;
            node->set_attr(Constants::node_perf, true);
        }
        return ret;
    }
}

// TEXT_GENERATOR(OpType_SubGraph)
BuilderText text_generator_func_OpType_SubGraph(NodePtr node)
{
    SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
    if (!node->attrs().count(Constants::plugin_attrs_name))
    {
        //use default generator
        // INFO() << "Using default subgraph generator.";
        auto connector = get_text_content_by_filename("layer_connector.bin");
        std::vector<BuilderText> text;
        size_t s = 0;
        SubGraphNodePtr sg = std::static_pointer_cast<SubGraphNode>(node);
        for (auto m_node : *sg->subgraph())
        {
            auto text_generator = get_text_generator(m_node->type());
            if (text_generator)
            {
                auto lt = text_generator(m_node);
                //text size=0
                if (lt.second == 0)
                {
                    continue;
                }
                text.push_back(lt);
                s += lt.second;
                s += connector.second;
            }
            else
            {
                ERROR() << "Unsupport node type: "
                        << optype_to_string(m_node->type()) << "\tname:"
                        << m_node->name() << " in subgraph node:" << node->name();
            }
        }
        char *raw = new char[s];
        //free the raw text at builder.cpp:assembly() TODO!!!
        char *ptr = raw;
        // for (auto t : text)
        for (unsigned int i = 0; i < text.size(); ++i)
        {
            auto t = text[i];
            memcpy(ptr, t.first, t.second);
            ptr += t.second;
            memcpy(ptr, connector.first, connector.second);
            ptr += connector.second;
            auto n = sg->subgraph()->nodes()[i];
            // if this is a subgraph type
            if (SubGraphNode::is_subgraph_node(n))
            {
                if (n->attrs().count("default_text"))
                {
                    if (n->get_attr<bool>("default_text"))
                    {
                        delete[] n->get_attr<BuilderText>("text").first;
                    }
                }
            }
            // delete[] t.first;
        }
        BuilderText t = {raw, s};
        node->set_attr("text", t);
        node->set_attr("default_text", true);
        return t;
    }
    else
    {
        auto ml = node->get_attr<NodePtrList>("_match_list");
        auto plugin_info = node->get_attr<BuilderOpPluginInfo *>(Constants::plugin_attrs_name);
        auto plugin = plugin_info->init_handle();
        BaseNodePtrList bml;
        for (auto p : ml)
        {
            bml.push_back(p);
        }
        BuilderText text = get_text_content_by_filename((plugin->generate_code_name(sgnode, bml)));
        return text;
    }
}
static TEXTGeneratorRegister _create_text_generator_OpType_SubGraph(OpType_SubGraph,
                                                                    text_generator_func_OpType_SubGraph,
                                                                    code_generator_func_OpType_SubGraph);
// TEXT_GENERATOR_END(OpType_SubGraph)

RO_GENERATOR(OpType_Loop)
{
    ro_generator_func_OpType_SubGraph(node, ro);
}
RO_GENERATOR_END(OpType_Loop);

DESC_GENERATOR(OpType_Loop)
{
    descriptor_generator_func_OpType_SubGraph(node, descriptor);
}
DESC_GENERATOR_END(OpType_Loop);

KERNEL_GENERATOR(OpType_Loop)
{
    return kernel_generator_func_OpType_SubGraph(node);
}
KERNEL_GENERATOR_END(OpType_Loop);

static TEXTGeneratorRegister _create_text_generator_OpType_Loop(OpType_Loop,
                                                                text_generator_func_OpType_SubGraph,
                                                                code_generator_func_OpType_SubGraph);

RO_GENERATOR(OpType_If)
{
    ro_generator_func_OpType_SubGraph(node, ro);
}
RO_GENERATOR_END(OpType_If);

DESC_GENERATOR(OpType_If)
{
    descriptor_generator_func_OpType_SubGraph(node, descriptor);
}
DESC_GENERATOR_END(OpType_If);

KERNEL_GENERATOR(OpType_If)
{
    return kernel_generator_func_OpType_SubGraph(node);
}
KERNEL_GENERATOR_END(OpType_If);

static TEXTGeneratorRegister _create_text_generator_OpType_If(OpType_If,
                                                              text_generator_func_OpType_SubGraph,
                                                              code_generator_func_OpType_SubGraph);
} // namespace aipubt
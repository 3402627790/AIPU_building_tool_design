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
 * @Author: Neo Wang
 * @Date: 2020-01-29 14:26:28
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2025-10-22 11:40:12
 */
#include "linker/linker_v1.h"
// #include "linker/linker_v2.h"

#include "builder/aipu_binfile.h"
#include "builder/builder.h"
#include "builder/lowering/hash_constant.h"
#include "builder/ops/ops.h"
#include "builder/profiler.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/placeholder.h"
#include "core/quantization.h"
#include "load_balancer.h"
#include "mc_scheduler.h"
#include "reorder_optimizer.h"
#include "ro_generator.h"
#include "scheduler.h"
#include "soms_scheduler.h"
#include "statistic/statistic.h"
#include "text_generator.h"
#include "transform/passes.h"
#include "utils/utils.h"
#include "x2_SM_opt.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <vector>

// #define DUMP_DATA_BASE_ADDRESS 0x20000000
#define DUMP_RO_BASE_ADDRESS 0x10000000
#define MAX(a, b) ((a) > (b) ? (a) : (b))

inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

// #define DUMP_DATA_BASE_ADDRESS 0x0
#define RESERVE_RO_NUMBER 0
// #define DEBUG_BFWRITE
#ifdef DEBUG_BFWRITE
#define BFWRITES(ofs, obj, size)                                                                                  \
    {                                                                                                             \
        ofs.write(reinterpret_cast<char *>((obj)), size);                                                         \
        std::ostringstream _temp_string;                                                                          \
        uint32_t *_m_data = reinterpret_cast<uint32_t *>((obj));                                                  \
        for (unsigned int i = 0; i < size / 4; ++i)                                                               \
        {                                                                                                         \
            _temp_string << std::hex << std::setw(8) << std::uppercase << std::setfill('0') << _m_data[i] << " "; \
        }                                                                                                         \
        DEBUG() << _temp_string.str();                                                                            \
    }
#else
#define BFWRITES(ofs, obj, size) ofs.write(reinterpret_cast<char *>((obj)), size);
#endif
#define BFWRITE(ofs, obj) BFWRITES(ofs, &obj, sizeof(obj))

#define BFREAD(ofs, obj) ofs.read(reinterpret_cast<char *>(&(obj)), sizeof(obj))

#define BFREADS(ofs, obj, size) ofs.read(reinterpret_cast<char *>((obj)), size)
namespace aipubt
{

static std::vector<MemoryObjectPtr> allocate_constant_data(size_t weight_offset, const std::vector<std::vector<NodePtr>> &levels, TensorPtr random_reg, std::vector<TensorPtr> &all_constants, uint32_t &fm_offset, uint32_t bss_id)
{
    auto get_mem_size = [](TensorPtr t) -> uint32_t {
        auto ptr = t->memory_offset()->get_base_object();
        auto size = std::max((uint64_t)ptr->size, (uint64_t)t->mem_size());
        if (Target::get_the_target()->isainfo().type == ISA_ZYv6)
        {
            size = aligned(size, 64);
        }
        return size;
    };
    std::vector<MemoryObjectPtr> constant_slots;
    std::vector<TensorPtr> constant_tensor;
    std::map<void *, MemoryObjectPtr> constant_data;
    std::map<void *, MemoryObjectPtr> zerocpy_constant_data;
    std::vector<TensorPtr> need_copy_to_DTCM;
    for (auto level : levels)
    {
        for (auto node : level)
        {
            if (node->get_attr_deep<bool>(Constants::large_kernel))
            {
                for (auto kv : node->constants())
                {
                    if (utils::endswith(kv.first, "weights"))
                    {
                        need_copy_to_DTCM.push_back(kv.second);
                    }
                }
            }
        }
    }

    for (auto constant : need_copy_to_DTCM)
    {
        constant_tensor.push_back(constant);
    }

    for (auto level : levels)
    {
        for (auto node : level)
        {
            for (auto constant : node->constants())
            {
                // avoid nullptr
                if (!constant.second)
                {
                    continue;
                }
                constant_tensor.push_back(constant.second);
            }
        }
    }

    // add random reg params and data
    if (random_reg)
    {
        constant_tensor.push_back(random_reg);
    }

    std::sort(constant_tensor.begin(), constant_tensor.end(), [get_mem_size](TensorPtr &t1, TensorPtr &t2) {
        return t1->memory_offset()->foot_print / (float)get_mem_size(t1) > t2->memory_offset()->foot_print / (float)get_mem_size(t2);
    });

    int buf_id = 0;
    for (auto constant : constant_tensor)
    {
        auto root_const = constant->memory_offset()->get_base_object();
        if (root_const->m_base)
        {
            continue; //allocate root blob once for shared blobs
        }
        auto size = get_mem_size(constant);
        if (constant->attrs().count(Constants::zerocopy_cst) && constant->get_attr(Constants::zerocopy_cst).cast<bool>())
        {
            if (zerocpy_constant_data.count(constant->data<void>()))
            {
                auto memobject = zerocpy_constant_data[constant->data<void>()];
                if (memobject->size >= size)
                {
                    // allocated
                    root_const->relative_offset = 0;
                    root_const->set_base_offset(memobject);
                    continue;
                }
            }
        }
        else
        {
            if (constant_data.count(constant->data<void>()))
            {
                auto memobject = constant_data[constant->data<void>()];
                if (memobject->size >= size)
                {
                    // allocated
                    root_const->relative_offset = 0;
                    root_const->set_base_offset(memobject);
                    continue;
                }
            }
        }

        weight_offset = aligned(weight_offset, root_const->align);

        MemoryObjectPtr s = std::make_shared<MemoryObject>(0);
        s->m_base = weight_offset;
        s->size = get_mem_size(constant);
        s->align = root_const->align;

        root_const->set_base_offset(s);
        root_const->relative_offset = 0;
        weight_offset += s->size;
        s->bss_id = bss_id;
        s->set_buf_type(MemoryObject::type::constant);
        if (constant->attrs().count(Constants::zerocopy_cst) && constant->get_attr(Constants::zerocopy_cst).cast<bool>())
        {
            s->buffer_id = buf_id;
            s->set_mem_flag(MemoryObject::flag::zero_copy);
            zerocpy_constant_data[constant->data<void>()] = s;
        }
        else
        {
            s->buffer_id = buf_id;
            constant_data[constant->data<void>()] = s;
        }
        buf_id++;
        constant_slots.push_back(s);
        all_constants.push_back(constant);
        // INFO() << constant->name() << " root const: " << root_const << " size: 0x" << std::hex << root_const->size << " buf: " << std::dec << s->buffer_id;
    }
    fm_offset = weight_offset;

    DEBUG() << "constant hash table:";
    for (size_t i = 0; i < constant_slots.size(); ++i)
    {
        auto s = constant_slots[i];
        auto c = all_constants[i]->get_attr<std::string>(Constants::hash);
        DEBUG() << c << " " << s->size;
    }

    return constant_slots;
}

static void dump_tune_strategy(std::vector<std::vector<NodePtr>> &levels, MemoryObjectPtr gm, size_t gm_size)
{
    // utils::JsonObject sg_node_info;
    utils::JsonObject node_info;
    int sg_cnt = 0;
    int in_gm = 0;
    for (size_t i = 0; i < levels.size(); i++)
    {
        auto level = levels[i];
        for (auto n : level)
        {
            INFO() << "[DEBUG] node " << n->name();
            in_gm = 0;
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                for (auto sg_n : sgn->subgraph()->nodes())
                {
                    if (sg_n->type() == OpType_SubGraphInput || sg_n->type() == OpType_Input || sg_n->type() == OpType_Constant || sg_n->type() == OpType_InternalConstant)
                    {
                        continue;
                    }
                    INFO() << "[DEBUG] sg_n " << sg_n;

                    // if (gm && sg_n->outputs()[0]->memory_offset()->base_offset == gm)
                    // {
                    //     in_gm = 1;
                    // }
                    // else
                    // {
                    //     in_gm = 0;
                    // }

                    std::set<MemoryObjectPtr> output_mem;
                    for (auto o : sg_n->outputs())
                    {
                        output_mem.insert(o->memory_offset());
                    }

                    auto sg_sgn = SubGraphNode::from_node(sg_n);
                    if (sg_sgn)
                    {
                        for (auto sg_sg_n : sg_sgn->subgraph()->nodes())
                        {
                            auto mem = sg_sg_n->outputs()[0]->memory_offset();
                            //0: cannot ues gm 1: unuse gm 2: use gm
                            if (!contains(output_mem, mem) || sg_sg_n->outputs()[0]->mem_size() > gm_size || mem->get_base_object()->m_base == 0xf8000000 || sg_sg_n->outputs()[0]->is_io_tensor())
                            {
                                in_gm = 0;
                            }
                            else
                            {
                                if (gm && sg_sg_n->outputs()[0]->memory_offset()->base_offset == gm)
                                {
                                    in_gm = 2;
                                }
                                else
                                {
                                    in_gm = 1;
                                }
                            }
                            INFO() << "[DEBUG] sg node " << sg_sg_n->name() << " " << sg_cnt;
                            utils::JsonObject info;
                            info["sg"] = sg_cnt;
                            info["merge_up"] = 0;
                            info["split"] = 0;
                            info["num"] = 0;
                            info["parallel"] = 0;
                            info["prior_gm"] = 0;
                            info["level"] = i;
                            info["internal"] = 0;
                            info["sg_name"] = sg_n->name();
                            info["tiling"] = sg_sg_n->get_attr<int>("tiling_support_value");
                            info["gm"] = in_gm;
                            info["load_balance"] = 0;
                            node_info[sg_sg_n->name()] = info;
                        }
                    }
                    else
                    {
                        INFO() << "[DEBUG] sg node " << sg_n->name() << " " << sg_cnt;
                        utils::JsonObject info;
                        info["sg"] = sg_cnt;
                        info["merge_up"] = 0;
                        info["split"] = 0;
                        info["num"] = 0;
                        info["parallel"] = 0;
                        info["prior_gm"] = 0;
                        info["level"] = i;
                        info["internal"] = 1;
                        info["sg_name"] = sg_n->name();
                        info["tiling"] = sg_n->get_attr<int>("tiling_support_value");
                        info["gm"] = in_gm;
                        info["load_balance"] = 0;
                        node_info[sg_n->name()] = info;
                    }
                }
            }
            else
            {
                ERROR();
            }
            sg_cnt += 2;
        }
    }
    std::ofstream os("tune_strategy.json");
    if (os.is_open())
    {
        os << node_info;
        os << std::endl;
        os.close();
    }
}

static void set_loop_scan(GraphPtr g)
{
    for (auto node : g->nodes())
    {
        // INFO() << " type " << node->type();
        auto sgn = SubGraphNode::from_node(node);
        if (sgn)
        {
            set_loop_scan(sgn->subgraph());
        }
        if (node->type() == OpType_Loop)
        {
            if (!node->params().count("scan_outputs_num"))
            {
                ERROR();
            }

            int scan_outputs_num = node->params()["scan_outputs_num"];
            int output_size = node->outputs().size();
            for (int i = 0; i < scan_outputs_num; ++i)
            {
                auto t = node->outputs()[output_size - 1 - i];
                auto ptr = t->memory_offset()->get_base_object();
                ptr->set_attr(MemoryObject::attr::loop_scan);
                // INFO() << "set loop scan for " << t->name();
            }

            if (!node->params().count("dependency_inputs_num"))
            {
                ERROR();
            }

            size_t start = node->params()["dependency_inputs_num"] + 2;
            for (size_t i = start; i < node->inputs().size(); ++i)
            {
                auto t = node->inputs()[i];
                auto ptr = t->memory_offset()->get_base_object();
                ptr->set_attr(MemoryObject::attr::loop_scan);
                // INFO() << "set loop scan for " << t->name();
            }
        }
    }
    return;
}

uint32_t GraphBuilder::get_text_size()
{
    if (m_aipubin)
    {
        return m_aipubin->text.bin_size;
    }
    return 0;
}

void GraphBuilder::update_subsec(BssSubSectionDescriptor &subsec, TensorPtr tensor)
{
    subsec.dtype = tensor->type();
    subsec.scale = tensor->quantization().scale();
    subsec.zp = tensor->quantization().offset();
    for (unsigned int k = 0; k < m_tensor_map[tensor].size(); ++k)
    {
        subsec.add_ro_offset(m_tensor_map[tensor][k] * 4);
        // DEBUG() << "BSS_reuse[" << i << "][" << j << "][" << k << "]."
        //         << " offset_in_ro : " << std::dec << subsec.offset_in_ro[k]/4;
    }
}

bool GraphBuilder::desc_add_subsec(const std::vector<TensorPtr> &vec, uint32_t type, BssSubSectionDescriptor &subsec, TensorPtr tensor, BssBufferDescriptor &desc, bool id_is_zero)
{
    bool ret = false;
    auto it = std::find(vec.begin(), vec.end(), tensor);
    if (it != vec.end())
    {
        subsec.type = type;
        subsec.id = id_is_zero ? 0 : std::distance(vec.begin(), it);
        update_subsec(subsec, tensor);
        desc.add_subsection(subsec);
        ret = true;
    }
    else
    {
        ret = false;
    }
    return ret;
}

void GraphBuilder::build_bss_section(BssSection &bss, std::vector<MemoryObjectPtr> &slots,
                                     std::vector<MemoryObjectPtr> &constant_slots,
                                     std::map<MemoryObjectPtr, std::vector<TensorPtr>> &m_buffer_map,
                                     std::map<TensorPtr, std::vector<int>> &m_tensor_map,
                                     std::map<BssSubSectionDescriptor::SubSectionType, std::vector<TensorPtr>> &special_marked_tensor,
                                     uint32_t bss_id)
{
    const std::vector<TensorPtr> &inputs = special_marked_tensor[BssSubSectionDescriptor::input];
    const std::vector<TensorPtr> &outputs = special_marked_tensor[BssSubSectionDescriptor::output];
    const std::vector<TensorPtr> &dumps = special_marked_tensor[BssSubSectionDescriptor::dump];
    const std::vector<TensorPtr> &profile_data = special_marked_tensor[BssSubSectionDescriptor::profile_data];
    const std::vector<TensorPtr> &printf_fifo = special_marked_tensor[BssSubSectionDescriptor::printf_fifo];
    const std::vector<TensorPtr> &op_cnt = special_marked_tensor[BssSubSectionDescriptor::op_cnt];
    const std::vector<TensorPtr> &error_code = special_marked_tensor[BssSubSectionDescriptor::error_code];
    const std::vector<TensorPtr> &cross_submodel = special_marked_tensor[BssSubSectionDescriptor::cross_submodel];
    const std::vector<TensorPtr> &dump_shape = special_marked_tensor[BssSubSectionDescriptor::dump_tensor_shape];
    size_t bss_size = 0;
    for (unsigned int i = 0; i < slots.size(); ++i)
    {
        BssBufferDescriptor desc;
        // workaround for avoid buffer'size == 0
        if (slots[i]->size == 0)
        {
            ERROR() << "Slot size must >0 but got 0.";
        }
        if (slots[i]->size >= 0xe0000000)
        {
            ERROR() << "Slot size must < 3.5G";
        }
        slots[i]->buffer_id = i;
        slots[i]->bss_id = bss_id;
        desc.size = slots[i]->size;
        desc.align_bytes = slots[i]->align;
        for (unsigned int j = 0; j < m_buffer_map[slots[i]->get_base_object()].size(); ++j)
        {
            BssSubSectionDescriptor subsec;
            auto tensor = m_buffer_map[slots[i]->get_base_object()][j];
            subsec.relative_offset = tensor->memory_offset()->offset() - tensor->memory_offset()->get_base_object()->offset();
            subsec.size = tensor->mem_size();
            if (subsec.size > desc.size)
            {
                ERROR() << tensor << "'s size > buffer size:" << subsec.size << ">" << desc.size;
                subsec.size = desc.size;
            }
            aipu_assert(desc.size >= subsec.size && "Tensor size must <= buffer size.");
            subsec.type = (uint32_t)BssSubSectionDescriptor::intermediate;
            subsec.id = 0;
            subsec.addr_mask = ~tensor->memory_offset()->mask;

            bool added = false;

            added = added || desc_add_subsec(inputs, (uint32_t)BssSubSectionDescriptor::input, subsec, tensor, desc);
            if (subsec.size)
            {
                added = desc_add_subsec(outputs, (uint32_t)BssSubSectionDescriptor::output, subsec, tensor, desc) || added;
                added = desc_add_subsec(cross_submodel, (uint32_t)BssSubSectionDescriptor::cross_submodel, subsec, tensor, desc) || added;
                added = desc_add_subsec(dumps, (uint32_t)BssSubSectionDescriptor::dump, subsec, tensor, desc) || added;
                added = desc_add_subsec(profile_data, (uint32_t)BssSubSectionDescriptor::profile_data, subsec, tensor, desc, 1) || added;
                added = desc_add_subsec(printf_fifo, (uint32_t)BssSubSectionDescriptor::printf_fifo, subsec, tensor, desc, 1) || added;
                added = desc_add_subsec(op_cnt, (uint32_t)BssSubSectionDescriptor::op_cnt, subsec, tensor, desc, 1) || added;
                added = desc_add_subsec(error_code, (uint32_t)BssSubSectionDescriptor::error_code, subsec, tensor, desc, 1) || added;
                added = desc_add_subsec(dump_shape, (uint32_t)BssSubSectionDescriptor::dump_tensor_shape, subsec, tensor, desc, 1) || added;
            }

            if (tensor->attrs().count(Constants::output_tensor_id))
            {
                subsec.type = (uint32_t)BssSubSectionDescriptor::output_tensor_shape;
                subsec.id = cast_any<uint32_t>(tensor->get_attr(Constants::output_tensor_id));
                update_subsec(subsec, tensor);
                desc.add_subsection(subsec);
                added = true;
                // DEBUG() << "output shape subsec: id=" << subsec.id << " size=" << subsec.size;
            }

            if (!added)
            {
                update_subsec(subsec, tensor);
                desc.add_subsection(subsec);
            }
        }
        bss.add_buff_desc(desc);
        bss_size += desc.size;
    }
    if (bss_size >= 0xe0000000) // cannot reach to 0xe000_0000 which is reserve memory address
    {
        ERROR() << "BSS buffer exceeded. AIPU only supports maximun 0xe0000000 Bytes, but actual size is " << bss_size << "Bytes";
    }
    for (unsigned int i = 0; i < constant_slots.size(); ++i)
    {
        BssDataDescriptor desc;

        desc.data_offset = constant_slots[i]->get_base_memory() - bss.stack_size;
        if (constant_slots[i]->size == 0 || m_buffer_map[constant_slots[i]->get_base_object()].size() == 0)
        {
            continue;
        }
        desc.size = constant_slots[i]->size;
        desc.align_bytes = constant_slots[i]->align;
        for (unsigned int j = 0; j < m_buffer_map[constant_slots[i]->get_base_object()].size(); ++j)
        {
            BssSubSectionDescriptor subsec;
            auto tensor = m_buffer_map[constant_slots[i]->get_base_object()][j];
            subsec.relative_offset = tensor->memory_offset()->offset() - tensor->memory_offset()->get_base_object()->offset();
            subsec.size = tensor->memory_offset()->get_base_object()->size;
            // if(subsec.size+subsec.relative_offset>desc.size)
            // {
            //     subsec.size=desc.size-subsec.relative_offset;
            // }
            subsec.addr_mask = ~(tensor->memory_offset()->mask);
            subsec.type = (uint32_t)BssSubSectionDescriptor::constant;
            if (tensor->memory_offset()->get_base_object()->get_mem_flag() == MemoryObject::flag::zero_copy)
            {
                subsec.type = (uint32_t)BssSubSectionDescriptor::zerocpy_constant;
            }
            subsec.id = 0;
            subsec.dtype = tensor->type();
            subsec.scale = tensor->quantization().scale();
            subsec.zp = tensor->quantization().offset();

            for (unsigned int k = 0; k < m_tensor_map[tensor].size(); ++k)
            {
                subsec.add_ro_offset(m_tensor_map[tensor][k] * 4);
            }
            desc.add_subsection(subsec);
        }
        bss.add_data_desc(desc);
    }
}

void GraphBuilder::build_ro(BinSection &ro_bin, std::vector<ReadOnlyParam> &ro)
{
    uint32_t *my_ro = new uint32_t[ro.size() + RESERVE_RO_NUMBER];
    ro_bin.bin = (char *)my_ro;
    ro_bin.bin_size = sizeof(uint32_t) * (ro.size() + RESERVE_RO_NUMBER);
    my_ro += RESERVE_RO_NUMBER;
    for (unsigned int i = 0; i < ro.size(); ++i)
    {
        if (ro[i].tensor)
        {
            my_ro[i] = ro[i].tensor->offset();
            // DEBUG() << "RO " << i << " 0x" << std::hex << my_ro[i] << " " << ro[i].tensor->name();
        }
        else
        {
            my_ro[i] = ro[i].value;
        }
    }
    // DEBUG() << ro.size();
}

void GraphBuilder::build_remap(RemapcSection &remap, std::map<RemapcSection::SectionType, TensorPtr> base_tensors,
                               std::vector<ReadOnlyParam> &ro, std::vector<ReadOnlyParam> &decriptor, size_t &remap_size)
{
    std::vector<RemapEntry> remapping_sec;
    auto ro_base = base_tensors[RemapcSection::RO];
    auto desc_base = base_tensors[RemapcSection::DESC];
    auto ro_base_memory_obj = ro_base->memory_offset()->get_base_object();
    auto desc_base_memory_obj = desc_base->memory_offset()->get_base_object();
    for (uint32_t idx = 0; idx < ro.size(); ++idx)
    {
        auto tensor = ro[idx].tensor;
        if (tensor)
        {
            auto root_memory = tensor->memory_offset()->get_base_object();
            if (root_memory == ro_base_memory_obj)
            {
                remapping_sec.push_back({RemapcSection::RO,
                                         idx * 4,
                                         RemapcSection::RO,
                                         static_cast<uint32_t>(tensor->memory_offset()->offset() - root_memory->offset())});
            }
            else if (root_memory == desc_base_memory_obj)
            {
                remapping_sec.push_back({RemapcSection::RO,
                                         idx * 4,
                                         RemapcSection::DESC,
                                         static_cast<uint32_t>(tensor->memory_offset()->offset() - root_memory->offset())});
            }
        }
    }
    for (uint32_t idx = 0; idx < decriptor.size(); ++idx)
    {
        auto tensor = decriptor[idx].tensor;
        if (tensor)
        {
            auto root_memory = tensor->memory_offset()->get_base_object();
            if (root_memory == ro_base_memory_obj)
            {
                remapping_sec.push_back({RemapcSection::DESC,
                                         idx * 4,
                                         RemapcSection::RO,
                                         static_cast<uint32_t>(tensor->memory_offset()->offset() - root_memory->offset())});
            }
            else if (root_memory == desc_base_memory_obj)
            {
                remapping_sec.push_back({RemapcSection::DESC,
                                         idx * 4,
                                         RemapcSection::DESC,
                                         static_cast<uint32_t>(tensor->memory_offset()->offset() - root_memory->offset())});
            }
        }
    }
    remap.table.clear();
    remap.entry_num = remapping_sec.size();
    remap.table = remapping_sec;
    remap_size = sizeof(RemapEntry) * remap.entry_num + sizeof(remap.entry_num);
}
void GraphBuilder::build_text()
{
    std::ostringstream oss;
    LinkerV1 linker;
    linker.set_codes(m_codes);
    uint32_t entry_point = linker.link(oss);
    // LinkerV2 linker2;
    // linker2.link(oss,codes);
    memcpy(m_aipubin->header.extra_data, &entry_point, sizeof(uint32_t));
    m_aipubin->header.type = linker.get_main_entry();
    auto tmp = oss.str();
    m_aipubin->text.bin = new char[tmp.size()];
    m_aipubin->text.bin_size = tmp.size();
    memcpy(m_aipubin->text.bin, tmp.c_str(), tmp.size());
}
void GraphBuilder::build_data(BinSection &data_bin, std::vector<TensorPtr> &data, std::vector<uint64_t> &offsets)
{
    if (data.empty())
    {
        data_bin.bin = 0;
        data_bin.bin_size = 0;
        return;
    }
    size_t base_offset = data[0]->memory_offset()->get_base_object()->offset();
    size_t end_data_buff = base_offset += data[0]->memory_offset()->get_base_object()->size;
    for (auto t : data)
    {
        if (t->data<char>())
        {
            auto ptr = t->memory_offset()->get_base_object();
            if (ptr->offset() < base_offset)
            {
                base_offset = ptr->offset();
            }
            if (ptr->offset() + ptr->size > end_data_buff)
            {
                end_data_buff = ptr->offset() + ptr->size;
            }
        }
    }
    size_t buff_size = end_data_buff - base_offset;
    buff_size = aligned(buff_size, 8);
    if (buff_size > 4ul * 1024 * 1024 * 1024)
    {
        ERROR() << "Constants data exceeded. AIPU only supports maximun 4GB constants data, but actual size is " << buff_size << "Bytes.";
    }
    data_bin.bin = new char[buff_size];
    memset(data_bin.bin, 0, sizeof(char) * buff_size);
    data_bin.bin_size = buff_size;
    char *bin = data_bin.bin;
    for (auto t : data)
    {
        uint64_t offset = 0;
        if (t->data<char>())
        {
            auto ptr = t->memory_offset()->get_base_object();
            offset = ptr->offset() - base_offset;
            memcpy(bin + offset, t->data<char>(), t->mem_size());
        }
        offsets.push_back(offset);
    }
}

GraphBuilder::GraphBuilder(GraphPtr graph) : m_origin_graph(graph)
{
    m_graph = nullptr;
    m_aipubin = new AIPUBin();
}

GraphBuilder::GraphBuilder(std::string file)
{
    m_graph = nullptr;
    m_aipubin = new AIPUBin();
}

AIPUBin GraphBuilder::load(const std::string &filename)
{
    std::ifstream ifs;
    ifs.open(filename, std::ios::in | std::ios::binary);
    AIPUBin bin = load(ifs);
    ifs.close();
    return bin;
}

AIPUBin GraphBuilder::load(std::istream &is)
{
    AIPUBin bin;
    bin.load(is);
    return bin;
}

void GraphBuilder::set_build_args(const std::string key)
{
    // for special
    if (std::string::npos != key.find("profile"))
    {
        aipubt::AIPUBuilder::get_profiler()->enable_profile() = true;
        std::string prof_unit = "default";
        aipubt::AIPUBuilder::get_profiler()->register_default_events(prof_unit);
        aipubt::BuilderConfig::set(Constants::prof_unit, prof_unit);
    }
    else
    {
        aipubt::AIPUBuilder::get_profiler()->enable_profile() = false;
        aipubt::BuilderConfig::set(key, true);
    }
}

bool GraphBuilder::build(const std::string &filename,
                         const std::vector<TensorPtr> &inputs_,
                         const std::vector<TensorPtr> &outputs,
                         const std::vector<int32_t> &dump_ids)
{
    auto ret = build_aipu(inputs_, outputs, dump_ids);
    std::ofstream ofs;
    ofs.open(filename, std::ios::out | std::ios::binary);
    m_aipubin->write(ofs);
    ofs.close();
    // uint32_t lib_loc;
    // memcpy(&lib_loc, m_aipubin->header.extra_data, sizeof(uint32_t));
    // INFO() << "lib_loc[0] = " << lib_loc;

    return ret && !ofs.fail();
}

void GraphBuilder::write(std::ostream &ofs)
{
    m_aipubin->write(ofs);
}
void GraphBuilder::write(const std::string &filename)
{
    std::ofstream ofs;
    ofs.open(filename, std::ios::out | std::ios::binary);
    write(ofs);
    ofs.close();
}

void GraphBuilder::show_section_size(std::string str, uint64_t size, utils::DataUnit unit)
{
    INFO() << str << "0x" << std::hex << std::setfill('0') << std::setw(8) << size << " Bytes"
           << " (" << std::setfill(' ') << std::setw(9) << utils::get_human_readable_size(size, unit) << ")";
}

void GraphBuilder::partition_graph()
{
}

void show_buffer_slots(GraphPtr m_graph, std::vector<MemoryObjectPtr> slots)
{
    for (auto n : m_graph->nodes())
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            auto sgn = SubGraphNode::from_node(n);
            show_buffer_slots(sgn->subgraph(), slots);
        }
        else
        {
            for (auto t : n->outputs())
            {
                for (size_t idx = 0; idx < slots.size(); idx++)
                {
                    auto slot = slots[idx];
                    if (slot->is_shared_by(t->memory_offset()))
                    {
                        DEBUG() << "tensor:" << t->name() << ", reused_buf_idx: " << idx;
                    }
                }
            }
        }
    }
}

static void update_dump_tensors(GraphPtr m_graph, const std::vector<int32_t> &dump_ids, const std::vector<TensorPtr> &inputs, const std::vector<TensorPtr> &outputs, std::set<TensorPtr> &all_dumps)
{
    std::vector<MemoryObjectPtr> internal_const_mem;
    for (auto n : m_graph->nodes())
    {
        if (n->type() == OpType_InternalConstant)
        {
            for (auto out : n->outputs())
            {
                internal_const_mem.push_back(out->memory_offset()->get_base_object());
            }
        }
    }

    if (BuilderConfig::get<bool>("dump_all_tensors"))
    {
        for (auto n : m_graph->nodes())
        {
            for (auto out : n->outputs())
            {
                if (!contains(outputs, out) && !contains(inputs, out) && !contains(internal_const_mem, out->memory_offset()->get_base_object()))
                {
                    all_dumps.insert(out);
                }
            }
        }
    }
    else
    {
        for (auto n : m_graph->nodes())
        {
            int32_t layer_id = n->get_attr<int32_t>("layer_id");
            if (SubGraphNode::is_subgraph_node(n))
            {
                SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(n);
                for (auto real_n : sgn->get_real_node())
                {
                    auto real_id = layer_id + real_n->get_attr<int32_t>("sub_layer_id");
                    if (contains(dump_ids, real_id))
                    {
                        std::vector<std::string> out_names;
                        for (auto t : n->outputs())
                        {
                            out_names.push_back(t->name());
                        }
                        for (auto t : real_n->outputs())
                        {
                            auto iter = std::find(out_names.begin(), out_names.end(), t->name());
                            if (iter == out_names.end())
                            {
                                WARN() << "[dump_tensors]: layer_id=" << real_id << " (skipped as an internal tensor), tensor_name=" << t->name();
                                continue;
                            }
                            auto out = n->outputs()[std::distance(out_names.begin(), iter)];
                            if (contains(inputs, out) || contains(outputs, out) || contains(internal_const_mem, out->memory_offset()->get_base_object()))
                            {
                                WARN() << "[dump_tensors]: layer_id=" << real_id << " (skipped as an in/out tensor),   tensor_name=" << out->name();
                                continue;
                            }
                            all_dumps.insert(out);
                        }
                    }
                }
            }
            else
            {
                if (contains(dump_ids, layer_id))
                {
                    for (auto out : n->outputs())
                    {
                        if (contains(outputs, out) || contains(inputs, out) || contains(internal_const_mem, out->memory_offset()->get_base_object()))
                        {
                            WARN() << "[dump_tensors]: layer_id=" << layer_id << " (skipped as an in/out tensor),   tensor_name=" << out->name();
                            continue;
                        }
                        all_dumps.insert(out);
                    }
                }
            }
        }
    }
}

/**
 * @brief check if all weight has data, which is a must for building
 *
 * @param m_graph
 * @return true
 * @return false
 */
static bool check_graph_weights(GraphPtr m_graph, bool error = true)
{
    for (auto n : m_graph->nodes())
    {
        for (auto kv : n->constants())
        {
            if (!kv.second->data<char>())
            {
                ERROR() << "Node: " << n->name() << " weight:" << kv.first << " missing data.";
                return false;
            }
        }
    }
    return true;
}

// allocate output shape memory for data-aware nodes
void allocate_output_shape(GraphPtr g)
{
    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        uint32_t shape_offset = 0;
        for (auto out : g->get_output_tensors())
        {
            shape_offset += 4 * out->shape().dim();
        }
        g->walk_on_node([&shape_offset](NodePtr n) {
            uint32_t shape_size = 0;
            if (n->type() == OpType_NonZero || n->type() == OpType_CacheUpdate)
            {
                shape_size = 4;
            }
            else if (n->type() == OpType_ConstantOfShape)
            {
                shape_size = 4 * n->inputs()[0]->shape().size();
            }
            else if (n->type() == OpType_Slice && n->inputs().size() == 5)
            {
                shape_size = 4 * n->inputs()[0]->shape().dim();
            }
            else if (n->type() == OpType_ForRangeEnder)
            {
                shape_size = 4 * n->inputs()[2]->shape().size();
            }
            else if (n->type() == OpType_IfEnder)
            {
                for (auto out : n->outputs())
                {
                    shape_size += 4 * out->shape().dim();
                }
            }
            else if (n->type() == OpType_TopK && n->inputs().size() == 2)
            {
                shape_size = 4 * n->inputs()[1]->shape().size();
            }
            if (shape_size > 0)
            {
                n->set_attr(Constants::output_shape_offset, shape_offset);
                n->set_attr(Constants::output_shape_size, shape_size);
                shape_offset += shape_size;
            }
        });
    }
}

std::vector<std::vector<int>> parse_input_shape(GraphPtr g, const std::string &opt)
{
    std::vector<std::vector<int>> input_shapes;
    if (BuilderConfig::exist(opt))
    {
        auto str = BuilderConfig::get<std::string>(opt);
        if (str.find_first_of('=') != std::string::npos)
        {
            std::regex pattern(R"(([a-zA-Z_]\w*)\s*=\s*(\d+))");
            auto begin = std::sregex_iterator(str.begin(), str.end(), pattern);
            auto end = std::sregex_iterator();
            std::map<std::string, int> vals;
            for (auto it = begin; it != end; ++it)
            {
                auto &match = *it;
                vals[match[1].str()] = std::stoi(match[2].str());
            }
            auto input_tensors = g->get_input_tensors();
            for (auto t : input_tensors)
            {
                auto n = t->node();
                auto shape = get_base_shape(t->shape());
                std::vector<int> s = shape;
                std::vector<Expr> ss;
                if (n->params().count("ds_output_shape"))
                {
                    std::vector<std::vector<Expr>> ds_output_shape = n->params()["ds_output_shape"];
                    if (ds_output_shape.size() == 1 && (int)ds_output_shape[0].size() == shape.dim())
                    {
                        ss = ds_output_shape[0];
                    }
                    else
                    {
                        FATAL() << "Unknown ds_output_shape param from Input: " << n->name();
                    }
                }
                else
                {
                    ss = shape.symbol();
                }
                for (int i = 0; i < shape.dim(); ++i)
                {
                    auto expr = ss[i];
                    if (expr->op() == ExprImpl::Op::Constant)
                    {
                        auto name = expr->symbol().name();
                        if (vals.count(name))
                        {
                            s[i] = vals.at(name);
                        }
                    }
                }
                input_shapes.push_back(s);
            }
        }
        else
        {
            for (auto &s : utils::split_string(str, "],", 1))
            {
                input_shapes.push_back(get_base_shape(TensorShape(s)));
            }
        }
    }
    return input_shapes;
}

static void workspace_allocate(GraphPtr g)
{
    for (auto n : g->nodes())
    {
        // INFO() << "workspace for " << n->name();
        if (!n->graph())
        {
            return;
        }
        auto upper_ws = n->graph()->global_tensors()[Constants::workspace];
        auto sgn = SubGraphNode::from_node(n);
        if (sgn)
        {
            n->set_attr(Constants::workspace, upper_ws);
            if (!sgn->subgraph()->global_tensors()[Constants::workspace])
            {
                sgn->subgraph()->global_tensors()[Constants::workspace] = std::make_shared<Tensor>(sgn->subgraph()->name() + "/ws", TensorShape(0));
            }
            auto sgn_ws = sgn->subgraph()->global_tensors()[Constants::workspace];
            sgn_ws->memory_offset()->set_base_offset(upper_ws->memory_offset());
            // INFO() << "ws: " << sgn_ws->memory_offset()->get_base_object();
            workspace_allocate(sgn->subgraph());
        }
        else
        {
            if (n->attrs().count(Constants::workspace))
            {
                TensorPtr ws = n->get_attr(Constants::workspace);
                if (!(ws->memory_offset()->base_offset) ||
                    ((ws != upper_ws) &&
                     (ws->memory_offset() != upper_ws->memory_offset()->base_offset) &&
                     (ws->memory_offset()->base_offset != upper_ws->memory_offset()->base_offset)))
                // !(ws->memory_offset()->is_shared_by(upper_ws->memory_offset())))
                {
                    ws->memory_offset()->set_base_offset(upper_ws->memory_offset());
                }
                // INFO() << "n: " << ws->memory_offset()->get_base_object();
            }
            else
            {
                WARN() << n->name() << " has no workspace!";
            }
        }
    };
}

static void gen_const_layer_map(GraphPtr g, std::map<void *, int> &const_layer_map)
{

    for (auto n : g->nodes())
    {
        INFO() << "for " << n;
        if (SubGraphNode::is_subgraph_node(n) && !n->attrs().count(Constants::plugin_attrs_name))
        {
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                gen_const_layer_map(sgn->subgraph(), const_layer_map);
            }
            else
            {
                ERROR();
            }
        }
        else
        {
            for (auto cts : n->constants())
            {
                auto data = (void *)cts.second->data();
                const_layer_map[data] = n->get_layer_id();
                INFO() << "add data of " << cts.second->name();
            }
        }
    }
}

static void dump_const_info_csv(GraphPtr g, const std::vector<std::vector<std::vector<NodePtr>>> &sub_model, const std::vector<std::vector<MemoryObjectPtr>> &const_slots, std::map<MemoryObjectPtr, std::vector<TensorPtr>> &m_buffer_map)
{
    auto shapeToString = [](const std::vector<int> &shape) -> std::string {
        if (shape.empty())
        {
            return "[]";
        }
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (i > 0)
                oss << ",";
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    };
    std::string filename = "constant_info.csv";
    std::ofstream file(filename);
    if (!file.is_open())
    {
        ERROR() << "cannot open: " << filename;
    }
    else
    {
        std::map<void *, int> const_layer_map;
        gen_const_layer_map(g, const_layer_map);
        std::set<void *> visited;
        file << "name,shape,layerID,size,type,offset,bufferID,bssID\n";
        for (size_t i = 0; i < sub_model.size(); ++i)
        {
            auto const_slot = const_slots[i];
            for (auto s : const_slot)
            {
                auto tensors = m_buffer_map[s->get_base_object()];
                for (auto t : tensors)
                {
                    auto data = (void *)t->data();

                    if (visited.count(data))
                    {
                        continue;
                    }

                    if (!const_layer_map.count(data))
                    {
                        WARN() << "data of " << t->name() << " doesn't in map";
                        continue;
                    }

                    file << "\"" << t->name() << "\","
                         << "\"" << shapeToString(t->shape()) << "\","
                         << "\"" << const_layer_map[data] << "\","
                         << "\"" << t->mem_size() << "\","
                         << "\"" << t->type_s() << "\","
                         << "\"" << s->m_base << "\","
                         << s->buffer_id << ","
                         << i << "\n";

                    visited.insert(data);
                }
            }
        }
        file.close();
        INFO() << "writing " << filename << " done";
    }
}

GraphPtr GraphBuilder::build_graph(const std::vector<TensorPtr> &inputs_,
                                   const std::vector<TensorPtr> &outputs_)
{
    std::vector<TensorPtr> outputs = outputs_;
    if (outputs_.empty())
    {
        outputs = m_origin_graph->get_output_tensors();
    }
    std::vector<TensorPtr> inputs = inputs_;
    if (inputs.empty())
    {
        inputs = m_origin_graph->get_input_tensors();
    }
    if (BuilderConfig::get<bool>(Constants::remove_origin_constant))
    {
        m_graph = m_origin_graph;
    }
    else
    {
        m_graph = m_origin_graph->clone(inputs, outputs, m_graph, true);
    }

    m_graph->topological_sort();
    partition_graph();
    TensorPtr error_code = m_graph->global_tensors()["error_code"];
    if (get_target()->isainfo().type >= ISA_ZYv5)
    {
        error_code->shape()[0] = 0;
    }
    else
    {
        for (auto n : *m_graph)
        {
            n->set_attr("error_code", error_code);
        }
    }
    allocate_output_shape(m_graph);
    BuilderOptimizer *optimizer = new BuilderOptimizer();
    optimizer->optimize(m_graph);
    if (!m_graph->check_valid_name())
    {
        ERROR() << "check valid name failed";
    }
    PerfEstimator perf;
    perf(m_graph);
    INFO() << perf.summary();
    if (get_target()->isainfo().type < ISA_ZYv5)
    {
        TensorPtr workspace = m_graph->global_tensors()["workspace"];
        int workspace_size = workspace->mem_size();

        for (auto n : *m_graph)
        {
            auto ws = n->get_attr<int>("workspace_size");
            workspace_size = workspace_size > ws ? workspace_size : ws;
        }
        // in plugin, sometimes they need workspace, but the size is unknow in setup stage,
        // and they need to set it later in generate_params stage.
        // so gbuilder pre-set the workspace_size to 1 to match plugin
        workspace->shape()[0] = workspace_size > 1 ? workspace_size : 1;
        std::vector<NodePtr> level;
        for (auto n : m_graph->nodes())
        {
            level.push_back(n);
        }
        m_levels.push_back(level);
    }
    else
    {
        LoadBalancer *load_balancer = new LoadBalancer();
        load_balancer->set_graph(m_graph);
        load_balancer->schedule();
        m_levels = load_balancer->get_levels();
        delete load_balancer;
    }

    workspace_allocate(m_graph);

    ReorderOptimizer *reorder_optimizer = new ReorderOptimizer();
    reorder_optimizer->set_graph(m_graph);
    reorder_optimizer->set_levels(m_levels);
    reorder_optimizer->node_reorder(BuilderConfig::get<int32_t>(Constants::branch_reorder_strategy));
    delete reorder_optimizer;

    // update subgraph real nodes
    for (auto n : m_graph->nodes())
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            auto sg = SubGraphNode::from_node(n)->subgraph();
            std::dynamic_pointer_cast<SubGraph>(sg)->update_real_nodes();
        }
    }

    auto ret = m_graph;
    m_graph = nullptr;
    delete optimizer;
    return ret;
}

static std::map<NodePtr, TensorPtr> update_node_output_for_dynamic_gm(GraphPtr g, MemoryObjectPtr gm)
{
    std::map<NodePtr, TensorPtr> node_tensor_map;
    for (auto node : *g)
    {
        SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
        if (sgnode)
        {
            for (auto m_node : *sgnode->subgraph())
            {
                if (m_node->attrs().count(Constants::wr_out_to_gm))
                {
                    uint32_t idx = m_node->get_attr<uint32_t>(Constants::wr_out_to_gm_idx);
                    auto o = m_node->outputs()[idx];
                    auto t = std::make_shared<Tensor>(*o);
                    INFO() << "assembly mov out to gm! " << o->name();

                    for (auto c : o->consumers())
                    {
                        if (c->attrs().count(Constants::mov_d_to_ddr_on))
                        {
                            for (uint32_t j = 0; j < c->inputs().size(); j++)
                            {
                                if (c->inputs()[j] == o)
                                {
                                    c->remove_input(j);
                                    c->add_input(t, j);
                                }
                            }
                        }
                    }

                    m_node->remove_output(idx);
                    m_node->add_output(t, idx);

                    uint32_t offset = m_node->get_attr<uint32_t>(Constants::wr_out_to_gm);
                    int gm_size = BuilderConfig::get<int>(Constants::gm_size);
                    auto gm_memory = std::make_shared<Tensor>(TensorShape({gm_size}));
                    gm_memory->memory_offset()->set_base_offset(gm->get_base_object());
                    // auto segment = std::make_shared<MemoryObject>(0);
                    // segment->set_base_offset(o->memory_offset()->get_base_object());
                    // segment->relative_offset = o->memory_offset()->offset();
                    // tensor_memory_map.insert(std::make_pair(o, segment));
                    t->memory_offset()->set_base_offset(gm_memory->memory_offset());
                    t->memory_offset()->relative_offset = offset;
                    t->memory_offset()->buffer_id = -1;
                    t->set_name("tmp" + o->name());
                    node_tensor_map.insert(make_pair(m_node, o));
                }
            }
        }
    }
    return node_tensor_map;
}

static bool restore_node_output_for_dynamic_gm(GraphPtr g, std::map<NodePtr, TensorPtr> &node_tensor_map, BuilderParams &m_ro)
{
    for (auto node : *g)
    {
        SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
        if (sgnode)
        {
            for (auto m_node : *sgnode->subgraph())
            {
                if (m_node->attrs().count(Constants::mov_d_to_ddr))
                {
                    uint32_t ro_end = m_node->get_attr<uint32_t>(Constants::ro_end);
                    INFO() << "ro end: " << ro_end;
                    auto end_node = m_node->get_attr<NodeRef>(Constants::mov_d_to_ddr).lock();
                    auto root_node = end_node->get_attr<NodeRef>(Constants::mov_d_to_ddr_wait).lock();
                    uint32_t idx = root_node->get_attr<uint32_t>(Constants::wr_out_to_gm_idx);
                    auto o = root_node->outputs()[idx];
                    auto t = node_tensor_map[root_node];
                    auto size = t->size();
                    m_ro[ro_end - 3] = t;
                    m_ro[ro_end - 2] = o;
                    m_ro[ro_end - 1] = size;
                }
            }
        }
    }

    for (auto it : node_tensor_map)
    {
        auto n = it.first;
        auto t = it.second;
        uint32_t idx = n->get_attr<uint32_t>(Constants::wr_out_to_gm_idx);
        auto o = n->outputs()[idx];

        for (auto c : o->consumers())
        {
            for (uint32_t j = 0; j < c->inputs().size(); j++)
            {
                if (c->inputs()[j] == o)
                {
                    c->remove_input(j);
                    c->add_input(t, j);
                }
            }
        }
        n->remove_output(idx);
        n->add_output(t, idx);
    }
    return true;
}

static void check_graph_input_shape(GraphPtr g)
{
    auto in_tensors = g->get_input_tensors();
    for (auto &&opt : {Constants::input_shape, Constants::max_input_shape, Constants::default_input_shape})
    {
        if (!BuilderConfig::exist(opt))
        {
            continue;
        }
        bool check_result = true;
        auto in_shapes = parse_input_shape(g, opt);
        if (in_tensors.size() != in_shapes.size())
        {
            WARN() << "Failed to check option --" << opt << ": "
                   << "number of input tensors is not equal to the given input shape!";
            check_result = false;
        }
        if (check_result)
        {
            for (uint32_t i = 0; i < in_tensors.size(); ++i)
            {
                auto shape = get_base_shape(in_tensors[i]->shape());
                if (shape.dim() != static_cast<int>(in_shapes[i].size()))
                {
                    WARN() << "Failed to check option --" << opt << ": "
                           << "dimension of input tensor " << i << " is not equal to the given input shape!";
                    check_result = false;
                    break;
                }
            }
        }
        if (check_result)
        {
            std::vector<int> max_shape;
            for (uint32_t i = 0; i < in_shapes.size(); ++i)
            {
                if (opt == Constants::max_input_shape)
                {
                    max_shape = in_shapes[i];
                }
                else
                {
                    max_shape = get_base_shape(in_tensors[i]->shape());
                }
                for (uint32_t j = 0; j < in_shapes[i].size(); ++j)
                {
                    if (in_shapes[i][j] < 0 || in_shapes[i][j] > max_shape[j])
                    {
                        WARN() << "Failed to check option --" << opt << ": "
                               << "the given input shape exceeds the valid range of values!";
                        check_result = false;
                        break;
                    }
                }
            }
        }
        if (!check_result)
        {
            BuilderConfig::erase(opt);
        }
    }
}

static void reduce_constant(const std::vector<std::vector<NodePtr>> &levels)
{
    std::map<std::string, TensorPtr> constant_hash;
    uint32_t reduce_size = 0;
    for (auto level : levels)
    {
        for (auto n : level)
        {
            for (auto kv : n->constants())
            {
                if (!kv.second)
                {
                    continue;
                }
                if (kv.second->attrs().count(Constants::hash))
                {
                    std::string hex = kv.second->get_attr(Constants::hash);
                    if (constant_hash.count(hex) &&
                        !kv.second->attrs().count(Constants::lora_node))
                    {
                        // DEBUG()<<"reduce constant:"<<kv.first;
                        reduce_size += kv.second->mem_size();
                        constant_hash[hex]->memory_offset()->foot_print += kv.second->memory_offset()->foot_print;
                        kv.second->memory_offset()->foot_print = constant_hash[hex]->memory_offset()->foot_print;
                        kv.second->share_data_from(*constant_hash[hex]);

                        continue;
                    }
                    constant_hash[hex] = kv.second;
                }
            }
        }
    }
    if (reduce_size > 0)
    {
        INFO() << "Reduce constants memory size: " << utils::get_human_readable_size(reduce_size);
    }
}

// check symbol shape before generate params
static void check_symbol_shape(GraphPtr graph)
{
    // if (get_target()->isainfo().type >= ISA_ZYv5)
    {
        auto &symbols = graph->symbols();
        std::vector<uint32_t> total_input_dims;
        for (auto inp : graph->get_input_tensors())
        {
            auto shape = get_base_shape(inp->shape());
            for (int d = 0; d < shape.dim(); ++d)
            {
                total_input_dims.push_back(shape[d]);
            }
        }
        bool enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
        uint32_t num_failed = 0;
        graph->walk_on_node([&](NodePtr n) {
            if (n->type() == OpType_Input ||
                n->type() == OpType_SubGraphInput ||
                n->type() == OpType_Empty ||
                SubGraphNode::is_subgraph_node(n))
            {
                return;
            }
            // DEBUG();
            // DEBUG() << "node: " << n->name() << " (" << n->type() << "):";
            bool check_result = true;
            // for (auto inp : n->inputs())
            // {
            //     DEBUG() << "input: " << inp->shape().symbol().to_string();
            // }
            for (auto out : n->outputs())
            {
                // DEBUG() << "output: " << out->shape().symbol().to_string();
                for (int d = 0; d < out->shape().dim(); ++d)
                {
                    auto expr = out->shape().symbol()[d];
                    bool skip_check = false;
                    if (expr->op() != ExprImpl::Op::Val)
                    {
                        std::map<Symbol, Symbol::Value> args;
                        auto vars = expr->variables();
                        std::vector<int> dims;
                        for (auto &var : vars)
                        {
                            if (!symbols.count(var.name()))
                            {
                                WARN() << "Unknown symbol " << var.name() << " from expr: " << expr->to_string();
                                skip_check = true;
                                break;
                            }
                            auto &sym = symbols.at(var.name());
                            if (sym.type == 1 || sym.dims.empty())
                            {
                                skip_check = true;
                                break;
                            }
                            dims.push_back(sym.dims[0]);
                        }
                        if (!skip_check)
                        {
                            for (size_t i = 0; i < vars.size(); ++i)
                            {
                                args[vars[i]] = total_input_dims[dims[i]];
                            }
                            if (!expr->eval(args))
                            {
                                if (enable_ds)
                                {
                                    WARN() << "Failed to eval expr: \"" << expr->to_string() << "\"";
                                }
                                check_result = false;
                            }
                        }
                    }
                    if (!skip_check && static_cast<int>(expr->value()) != out->shape()[d])
                    {
                        check_result = false;
                    }
                }
            }
            if (!check_result)
            {
                if (enable_ds)
                {
                    WARN() << "Failed to check symbolic shape of node:\n"
                           << n->to_string();
                }
                num_failed++;
            }
        });
        if (num_failed)
        {
            WARN() << "Check symbolic shape of " << num_failed << " node(s) failed!";
        }
    }
}

static void update_params(GraphPtr graph, BuilderParams &params)
{
    auto &symbols = graph->symbols();
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    std::vector<uint32_t> total_input_dims;
    for (auto inp : graph->get_input_tensors())
    {
        auto shape = get_base_shape(inp->shape());
        for (int i = 0; i < shape.dim(); ++i)
        {
            total_input_dims.push_back(shape[i]);
        }
    }
    auto default_input_shape = BuilderConfig::exist(Constants::default_input_shape);
    std::vector<uint32_t> default_input_dims;
    if (default_input_shape)
    {
        auto default_shape = parse_input_shape(graph, Constants::default_input_shape);
        for (auto &s : default_shape)
        {
            for (auto d : s)
            {
                default_input_dims.push_back(d);
            }
        }
    }
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        Expr expr;
        if (params[i].tensor && params[i].tensor->attrs().count(Constants::expr_offset))
        {
            expr = params[i].tensor->get_attr(Constants::expr_offset);
        }
        else if (params[i].expr)
        {
            expr = params[i].expr;
        }
        if (expr)
        {
            expr = expr->simplify();
            if (!enable_ds || default_input_shape || expr->op() == ExprImpl::Op::Val)
            {
                bool skip_eval = false;
                if (expr->op() != ExprImpl::Op::Val)
                {
                    std::map<Symbol, Symbol::Value> args;
                    auto vars = expr->variables();
                    std::vector<int> dims;
                    for (auto &var : vars)
                    {
                        if (!symbols.count(var.name()))
                        {
                            skip_eval = true;
                            break;
                        }
                        auto &sym = symbols.at(var.name());
                        if (sym.type == 1 || sym.dims.empty())
                        {
                            skip_eval = true;
                            break;
                        }
                        dims.push_back(sym.dims[0]);
                    }
                    if (!skip_eval)
                    {
                        for (size_t j = 0; j < vars.size(); ++j)
                        {
                            if (default_input_shape)
                            {
                                args[vars[j]] = default_input_dims[dims[j]];
                            }
                            else
                            {
                                args[vars[j]] = total_input_dims[dims[j]];
                            }
                        }
                        if (!expr->eval(args))
                        {
                            WARN() << "Failed to eval expr \"" << expr->to_string() << "\"";
                        }
                    }
                }
                if (!skip_eval)
                {
                    if (params[i].expr)
                    {
                        if (default_input_shape && expr->op() != ExprImpl::Op::Val)
                        {
                            auto is_ends_with_itof = [](Expr expr) {
                                if (expr->op() == ExprImpl::Op::Itof)
                                {
                                    return true;
                                }
                                return false;
                            };
                            auto is_ends_with_sqrt_itof = [](Expr expr) {
                                if ((expr->op() == ExprImpl::Op::Sqrt || expr->op() == ExprImpl::Op::Rsqrt) &&
                                    expr->args()[0]->op() == ExprImpl::Op::Itof)
                                {
                                    return true;
                                }
                                return false;
                            };

                            bool store_float_value = false;
                            if (is_ends_with_itof(expr) || is_ends_with_sqrt_itof(expr))
                            {
                                store_float_value = true;
                            }
                            if (expr->op() == ExprImpl::Op::And &&
                                expr->args()[1]->op() == ExprImpl::Op::Val &&
                                (int64_t)expr->args()[1]->value() == 0xffffffff)
                            {
                                auto arg0 = expr->args()[0];
                                if (is_ends_with_itof(arg0) || is_ends_with_sqrt_itof(arg0))
                                {
                                    store_float_value = true;
                                }
                            }

                            uint32_t val = expr->value();
                            if (store_float_value)
                            {
                                float tmp = expr->value();
                                val = *((uint32_t *)&tmp);
                            }
                            params[i].value = val;
                        }
                        else
                        {
                            params[i] = ReadOnlyParam((uint32_t)expr->value());
                        }
                    }
                    else
                    {
                        if (expr->value() < 0)
                        {
                            WARN() << "Found negative expr offset in tensor: " << params[i].tensor->name();
                        }
                        params[i].tensor->memory_offset()->relative_offset += (int32_t)expr->value();
                        if (default_input_shape && expr->op() != ExprImpl::Op::Val)
                        {
                            params[i].tensor->set_attr(Constants::expr_offset, expr - (int32_t)expr->value());
                        }
                        else
                        {
                            params[i].tensor->erase_attr(Constants::expr_offset);
                        }
                    }
                }
                // DEBUG();
                // DEBUG() << "ro[" << i << "] expr: " << expr->to_string();
                // DEBUG() << "value: " << (uint32_t)expr->value();
            }
        }
    }
}

bool GraphBuilder::build_aipu(const std::vector<TensorPtr> &inputs_,
                              const std::vector<TensorPtr> &outputs_,
                              const std::vector<int32_t> &dump_ids)
{
    INFO() << "peak mem " << getPeakMemory() << " MB";

    auto error_cnt = GCompilerLogger::get_error_count();
    bool enable_profile = AIPUBuilder::get_profiler()->enable_profile();

    TensorPtr profile = nullptr;
    TensorPtr printf_fifo = nullptr;
    TensorPtr op_cnt = nullptr;
    TensorPtr error_code = nullptr;
    TensorPtr ts_base = nullptr;
    std::set<TensorPtr> keepnodes;
    std::vector<TensorPtr> outputs = outputs_;
    std::set<TensorPtr> all_dumps;
    auto extsram_infos = get_external_sram();
    bool alloc_output_on_extsram = aipubt::BuilderConfig::get<bool>(Constants::alloc_output_on_extsram);
    if (!extsram_infos.size())
    {
        alloc_output_on_extsram = false;
    }

    bool show_short_name = BuilderConfig::get<bool>(Constants::show_short_name);
    if (show_short_name)
    {
        simplify_name(m_origin_graph);
    }

    if (outputs_.empty())
    {
        outputs = m_origin_graph->get_output_tensors();
    }

    std::vector<TensorPtr> inputs = inputs_;
    if (inputs.empty())
    {
        inputs = m_origin_graph->get_input_tensors();
    }
    if (!check_graph_weights(m_origin_graph))
    {
        ERROR() << "Some weights/bias data missing, please check your weight file is correct or you well set the weight data.";
        return false;
    }

    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        // set default ds_output_shape if some input nodes don't have
        bool has_ds_output_shape = false;
        for (auto t : m_origin_graph->get_input_tensors())
        {
            auto n = t->node();
            if (n->params().count("ds_output_shape"))
            {
                has_ds_output_shape = true;
                break;
            }
        }
        if (has_ds_output_shape)
        {
            for (auto t : m_origin_graph->get_input_tensors())
            {
                auto n = t->node();
                if (!n->params().count("ds_output_shape"))
                {
                    std::vector<Expr> s = t->shape().symbol();
                    n->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{s};
                }
            }
        }
        check_graph_input_shape(m_origin_graph);
        if (!BuilderConfig::exist(Constants::default_input_shape))
        {
            std::vector<std::vector<int>> max_shape;
            if (BuilderConfig::exist(Constants::max_input_shape))
            {
                max_shape = parse_input_shape(m_origin_graph, Constants::max_input_shape);
            }
            else
            {
                for (auto t : m_origin_graph->get_input_tensors())
                {
                    max_shape.push_back(get_base_shape(t->shape()));
                }
            }
            std::ostringstream oss;
            oss << "[";
            for (size_t i = 0; i < max_shape.size(); ++i)
            {
                oss << TensorShape(max_shape[i]);
                if (i != max_shape.size() - 1)
                {
                    oss << ",";
                }
            }
            oss << "]";
            BuilderConfig::set(Constants::default_input_shape, oss.str());
        }
        // all input dimensions are dynamic by default
        // otherwise parse its ds_output_shape
        if (!BuilderConfig::exist(Constants::dynamic_input_dim))
        {
            std::set<std::string> dims;
            uint32_t index = 0;
            for (auto t : m_origin_graph->get_input_tensors())
            {
                auto n = t->node();
                auto shape = get_base_shape(t->shape());
                if (n->params().count("ds_output_shape"))
                {
                    std::vector<std::vector<Expr>> ds_output_shape = n->params()["ds_output_shape"];
                    if (ds_output_shape.size() == 1 && (int)ds_output_shape[0].size() == shape.dim())
                    {
                        for (auto expr : ds_output_shape[0])
                        {
                            if (expr->op() == ExprImpl::Op::Constant)
                            {
                                dims.insert(expr->symbol().name());
                            }
                        }
                    }
                    else
                    {
                        FATAL() << "Unknown ds_output_shape param from Input: " << n->name();
                    }
                    index += shape.dim();
                }
                else
                {
                    for (int i = 0; i < shape.dim(); ++i)
                    {
                        dims.insert("d" + std::to_string(index++));
                    }
                }
            }
            BuilderConfig::set(Constants::dynamic_input_dim, dims);
        }
    }
    m_graph = build_graph(inputs, outputs);
    INFO() << "peak mem (build graph) " << getPeakMemory() << " MB";
    // update to the new graph's tensor
    for (uint32_t i = 0; i < inputs.size(); ++i)
    {
        inputs[i] = m_graph->get_tensor(inputs[i]->name());
    }
    for (uint32_t i = 0; i < outputs.size(); ++i)
    {
        outputs[i] = m_graph->get_tensor(outputs[i]->name());
    }
    update_dump_tensors(m_graph, dump_ids, inputs, outputs, all_dumps);
    std::set_union(outputs.begin(), outputs.end(), all_dumps.begin(), all_dumps.end(), std::inserter(keepnodes, keepnodes.end()));
    std::set<TensorPtr> keeptensors;
    for (auto t : keepnodes)
    {
        if (!t)
        {
            WARN() << " A keeping tensor has been deleted.";
            continue;
        }
        auto nt = m_graph->get_tensor(t->name());
        if (!nt)
        {
            WARN() << " A keeping tensor has been deleted: " << t->name() << ".";
            continue;
        }
        keeptensors.insert(nt);
    }
    keepnodes = std::move(keeptensors);
    auto ro_base = m_graph->global_tensors()[Constants::parambase];
    ro_base->memory_offset()->set_base_offset(DUMP_RO_BASE_ADDRESS);
    ro_base->memory_offset()->relative_offset = 0;
    ro_base->memory_offset()->size = 0;

    auto desc_base = m_graph->global_tensors()[Constants::descriptorbase];
    desc_base->memory_offset()->set_base_offset(DUMP_RO_BASE_ADDRESS);
    desc_base->memory_offset()->relative_offset = 0;
    desc_base->memory_offset()->size = 0;

    std::vector<TensorPtr> not_reuse_tensor;
    if (m_graph->global_tensors()["printf_fifo"]->mem_size() > 0)
    {
        printf_fifo = m_graph->global_tensors()["printf_fifo"];
    }
    if (m_graph->global_tensors()["op_cnt"]->mem_size() > 0)
    {
        op_cnt = m_graph->global_tensors()["op_cnt"];
    }
    if (m_graph->global_tensors()["error_code"]->mem_size() > 0)
    {
        error_code = m_graph->global_tensors()["error_code"];
    }
    // if use soc sram, the output would not be reuse to avoid output use reuse buffer.
    // temp disable output if using gm due to how to use sync out feature is not quite clear
    if (extsram_infos.size() || aipubt::BuilderConfig::get<bool>(Constants::disable_output_buffer_reuse) || alloc_output_on_extsram || aipubt::BuilderConfig::get<bool>(Constants::enable_custom_smmu) || aipubt::BuilderConfig::get<uint32_t>(Constants::multi_batch))
    {
        for (auto t : m_graph->get_output_tensors())
        {
            not_reuse_tensor.push_back(t);
        }
    }
    if (extsram_infos.size() || aipubt::BuilderConfig::get<bool>(Constants::disable_input_buffer_reuse) || aipubt::BuilderConfig::get<bool>(Constants::enable_custom_smmu) || aipubt::BuilderConfig::get<uint32_t>(Constants::multi_batch))
    {
        not_reuse_tensor.insert(not_reuse_tensor.end(), inputs.begin(), inputs.end());
    }
    // zerocpy constant is  not reuse
    std::set<TensorPtr> skip_tensors;
    m_graph->walk_on_node([&skip_tensors](NodePtr n) {
        if (n->type() == OpType_InternalConstant)
        {
            skip_tensors.insert(n->outputs()[0]);
        }
    });

    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        // allocate 8KB for temp_param_base by default
        auto tp_base = m_graph->global_tensors()[Constants::temp_param_base];
        tp_base->shape()[0] = 8192;

        uint32_t total_shape_size = 0;
        for (auto out : m_graph->get_output_tensors())
        {
            total_shape_size += 4 * out->shape().dim();
        }
        m_graph->walk_on_node([&total_shape_size](NodePtr n) {
            if (n->attrs().count(Constants::output_shape_size))
            {
                total_shape_size += n->get_attr<uint32_t>(Constants::output_shape_size);
            }
        });
        ts_base = m_graph->global_tensors()[Constants::tensor_shape_base];
        ts_base->shape()[0] = total_shape_size;
    }

    // set break descriptor chain if necessary
    bool break_descriptor_chain = false;
    m_graph->walk_on_node([&break_descriptor_chain](NodePtr n) {
        if (break_descriptor_chain)
        {
            return;
        }
        if (n->attrs().count(Constants::plugin_attrs_name) &&
            n->attrs().count(Constants::max_batch_size))
        {
            break_descriptor_chain = true;
        }
        if (n->type() == OpType_Loop ||
            n->type() == OpType_If)
        {
            break_descriptor_chain = true;
        }
    });
    bool dump_rt_tensor = BuilderConfig::get<bool>(Constants::dump_rt_output_tensors) || BuilderConfig::get<bool>(Constants::dump_rt_input_tensors) || (BuilderConfig::get<int>(Constants::dump_rt_tensor_id) != -1);
    if (BuilderConfig::get<bool>("DEBUG") ||
        BuilderConfig::get<bool>(Constants::enable_debug_info) ||
        AIPUBuilder::get_profiler()->enable_profile() ||
        BuilderConfig::get<bool>(Constants::enable_ds) ||
        dump_rt_tensor ||
        break_descriptor_chain)
    {
        BuilderConfig::set(Constants::break_descriptor_chain, true);
    }

    set_loop_scan(m_graph);

    // statistic the graph memory requirement
    DDRFootPrintEstimation ddrfp;
    ddrfp(m_graph);

    // print estimated total cycles for cost model evaluation
    if (AIPUBuilder::get_profiler()->enable_profile())
    {
        PerfEstimator perf;
        perf(m_graph);
        INFO() << perf.summary();
    }

    std::vector<NodePtr> level;
    std::vector<std::vector<NodePtr>> sub_levels;
    std::vector<std::vector<std::vector<NodePtr>>> sub_model;
    auto max_wt_size = aipubt::BuilderConfig::get<uint64_t>(Constants::max_wt_size);
    std::vector<TensorPtr> cross_submodel;
    std::vector<TensorPtr> total_submodel_tensors;
    auto max_act_size = aipubt::BuilderConfig::get<uint64_t>(Constants::max_act_size);
    if (max_wt_size || max_act_size)
    {
        auto get_level_constant = [](const std::vector<NodePtr> &level) -> uint64_t {
            uint64_t total_constant_size = 0;
            for (auto &node : level)
            {
                total_constant_size += node->get_constant_size();
            }
            return total_constant_size;
        };
        auto get_level_act = [](const std::vector<NodePtr> &level) -> uint64_t {
            uint64_t total_act_size = 0;
            for (auto &node : level)
            {
                auto sgn = SubGraphNode::from_node(node);
                if (sgn)
                {
                    auto sg_nodes = sgn->subgraph()->nodes();
                    for (auto n : sg_nodes)
                    {
                        // total_act_size += n->get_output_size();
                        if (n->type() == OpType_InternalConcatenated ||
                            n->type() == OpType_InternalCrop ||
                            n->type() == OpType_InternalSlice ||
                            n->type() == OpType_InternalSpliter ||
                            n->type() == OpType_InternalPad ||
                            n->type() == OpType_InternalConstant ||
                            (n->type() == OpType_LayoutConvertor &&
                             !is_a_remapping_convertor(n)))
                        {
                            continue;
                        }
                        else
                        {
                            total_act_size += n->get_output_size();
                        }
                    }
                }
                else
                {
                    ERROR();
                }
            }
            return total_act_size;
        };
        uint64_t max_size = max_wt_size ? max_wt_size : max_act_size;
        uint64_t original_total_size = 0;
        uint64_t total_size = 0;
        for (size_t i = 0; i < m_levels.size(); ++i)
        {
            auto level_size = max_wt_size ? get_level_constant(m_levels[i]) : get_level_act(m_levels[i]);
            if (total_size + level_size > max_size)
            {
                if (level_size > max_size)
                {
                    WARN() << "level size " << level_size << " is larger than max size";
                }
                sub_model.push_back(sub_levels);
                sub_levels.clear();
                total_size = 0;
            }
            sub_levels.push_back(m_levels[i]);
            total_size += level_size;
            original_total_size += level_size;
        }
        if (sub_levels.size())
        {
            sub_model.push_back(sub_levels);
        }

        for (size_t i = 0; i < sub_model.size(); ++i)
        {
            sub_levels = sub_model[i];
            std::vector<TensorPtr> new_submodel_tensors;
            for (auto s : sub_levels)
            {
                for (auto n : s)
                {
                    n->set_attr(Constants::bss_id, i);
                    auto sgn = SubGraphNode::from_node(n);
                    if (sgn)
                    {
                        auto sg = sgn->subgraph();
                        for (auto sg_node : sg->nodes())
                        {
                            sg_node->set_attr(Constants::bss_id, i);
                        }
                    }
                    for (auto t : n->inputs())
                    {
                        if (contains(total_submodel_tensors, t))
                        {
                            cross_submodel.push_back(t);
                        }
                    }
                    for (auto t : n->outputs())
                    {
                        new_submodel_tensors.push_back(t);
                    }
                }
            }
            total_submodel_tensors.insert(total_submodel_tensors.end(), new_submodel_tensors.begin(), new_submodel_tensors.end());
        }

        if (aipubt::BuilderConfig::get<bool>(Constants::disable_cross_buffer_reuse))
        {
            not_reuse_tensor.insert(not_reuse_tensor.end(), cross_submodel.begin(), cross_submodel.end());
        }

        INFO() << "total size is " << utils::get_human_readable_size(original_total_size) << " max size is " << utils::get_human_readable_size(max_size) << " so split into " << sub_model.size() << " parts";
        // for (size_t i = 0; i < sub_model.size(); ++i)
        // {
        //     sub_levels = sub_model[i];
        //     DEBUG() << "part " << i << " level size " << sub_levels.size();
        //     for (size_t j = 0; j < sub_levels.size(); ++j)
        //     {
        //         for (size_t k = 0; k < sub_levels[j].size(); ++k)
        //         {
        //             DEBUG() << sub_levels[j][k];
        //         }
        //     }
        // }
    }
    else
    {
        sub_model.push_back(m_levels);
    }

    std::vector<std::vector<MemoryObjectPtr>> data_slots;
    std::map<NodePtr, TensorPtr> node_tensor_map;

    for (size_t i = 0; i < sub_model.size(); ++i)
    {
        auto sub_levels = sub_model[i];
        Scheduler *scheduler;

        if (get_target()->isainfo().type >= ISA_ZYv5)
        {
            if (aipubt::BuilderConfig::get<bool>(Constants::use_amat_scheduler))
            {
                scheduler = new MCScheduler();
            }
            else
            {
                scheduler = new SOMSScheduler();
            }
            uint32_t gms = BuilderConfig::get<uint32_t>(Constants::gm_size);
            if (BuilderConfig::exist(Constants::preserve_gm_size))
            {
                auto preserve_gm_size = BuilderConfig::get<int>(Constants::preserve_gm_size);
                if (preserve_gm_size > (int)gms)
                {
                    ERROR();
                }
                else
                {
                    gms = gms - preserve_gm_size;
                }
            }
            scheduler->set_gm_size({gms});
        }
        else
        {
            scheduler = new Scheduler();
        }
        scheduler->set_graph(m_graph);
        scheduler->set_levels(sub_levels);
        std::vector<TensorPtr> submodel_tensors;
        std::set<TensorPtr> submodel_not_reuse_tensor;
        for (auto s : sub_levels)
        {
            for (auto n : s)
            {
                for (auto t : n->outputs())
                {
                    if (t->mem_size() > 0xffffffff)
                    {
                        ERROR() << t->name() << " tensor size is larger than 32bit";
                    }
                    else
                    {
                        submodel_tensors.push_back(t);
                    }
                }
            }
        }
        for (auto t : not_reuse_tensor)
        {
            if (contains(submodel_tensors, t))
            {
                submodel_not_reuse_tensor.insert(t);
            }
        }

        std::set<TensorPtr> submodel_inputs;
        for (auto t : inputs)
        {
            if (contains(submodel_tensors, t))
            {
                submodel_inputs.insert(t);
            }
        }

        std::set<TensorPtr> submodel_keeps_tensors;
        for (auto t : keepnodes)
        {
            if (contains(submodel_tensors, t))
            {
                submodel_keeps_tensors.insert(t);
            }
        }

        for (auto t : cross_submodel)
        {
            if (contains(submodel_tensors, t))
            {
                submodel_keeps_tensors.insert(t);
            }
        }

        // allocate global tensors
        if (i == 0)
        {
            // after schedule the profiler can calculate correct alloc buffer size.
            if (enable_profile)
            {
                profile = AIPUBuilder::get_profiler()->interface()->allocate_buffer(m_graph);
                auto ptr = profile->memory_offset()->get_base_object();
                ptr->align = 64;
                m_graph->global_tensors()["_profiler"] = profile;
                submodel_not_reuse_tensor.insert(profile);
            }
            // submodel_not_reuse_tensor.insert(printf_fifo);
            // submodel_not_reuse_tensor.insert(op_cnt);
            // submodel_not_reuse_tensor.insert(error_code);
            for (auto gt : m_graph->global_tensors())
            {
                if (gt.second)
                {
                    if (gt.first == "mmu_reserve" || gt.second->mem_size() <= 0)
                    {
                        continue;
                    }
                    submodel_not_reuse_tensor.insert(gt.second);
                }
            }
        }

        //according to EMU requirements, all input/output must aligned to HW bitwidth
        auto bit_width = get_target()->axi_bandwidth();
        uint32_t emu_align = bit_width / 8;
        for (auto &tensors : {submodel_inputs, submodel_keeps_tensors, submodel_not_reuse_tensor})
        {
            for (auto &t : tensors)
            {
                auto ptr = t->memory_offset()->get_base_object();
                if (emu_align > ptr->align)
                {
                    ptr->align = emu_align;
                }
            }
        }
        auto start = std::chrono::high_resolution_clock::now();
        auto slots = scheduler->mem_allocate(submodel_not_reuse_tensor, submodel_inputs, submodel_keeps_tensors, skip_tensors);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        if (sub_model.size() > 1)
        {
            for (auto t : submodel_tensors)
            {
                skip_tensors.insert(t);
            }
        }
        INFO() << "[EVAL] duration time of mem allocation: " << duration.count() << " ms";

        // put output to the first if enable alloc_output_on_extsram
        if (alloc_output_on_extsram)
        {
            // put output slot to the first, so that UMD can allocate the buffer in priority.
            std::vector<MemoryObjectPtr> output_slots;
            for (auto tensor : m_graph->get_output_tensors())
            {
                for (auto slot : slots)
                {
                    if (slot->is_shared_by(tensor->memory_offset()))
                    {
                        output_slots.push_back(slot);
                        break;
                    }
                }
            }
            std::vector<MemoryObjectPtr> slots_in_order(output_slots.begin(), output_slots.end());
            for (auto slot : slots)
            {
                auto find_it = std::find(slots_in_order.begin(), slots_in_order.end(), slot);
                if (find_it == slots_in_order.end())
                {
                    slots_in_order.push_back(slot);
                }
            }
            slots = std::move(slots_in_order);
        }
        data_slots.push_back(slots);

        if (aipubt::BuilderConfig::get<bool>(Constants::enable_x2_sm_opt))
        {
            for (auto n : m_graph->nodes())
            {
                x2_sm_opt(SubGraphNode::from_node(n));
            }
        }

        if (aipubt::BuilderConfig::get<bool>(Constants::enable_gm_enhance) || aipubt::BuilderConfig::get<bool>(Constants::enable_dynamic_gm))
        {
            scheduler->print_gm_avg_usage();
            scheduler->gm_enhance();
            auto gm = scheduler->get_gm_regions()[0];
            auto sub_node_tensor_map = update_node_output_for_dynamic_gm(m_graph, gm);
            scheduler->print_gm_avg_usage();

            for (const auto &pair : sub_node_tensor_map)
            {
                node_tensor_map.insert(pair);
            }
        }

        if (i == 0)
        {
            m_gm_regions = scheduler->get_gm_regions();
            m_gm_size = scheduler->gm_size();
            m_slots = slots;
        }
        delete scheduler;
    }

    if (BuilderConfig::get<bool>(Constants::dump_tune_strategy))
    {
        MemoryObjectPtr gm;
        size_t gm_size = 0;
        if (m_gm_regions.size() > 0)
        {
            gm = m_gm_regions[0];
            gm_size = BuilderConfig::get<uint32_t>(Constants::gm_size);
        }
        dump_tune_strategy(sub_model[0], gm, gm_size);
    }

    check_symbol_shape(m_graph);

    set_layer_id(m_graph);

    pid_t pid = getpid();
    DEBUG() << "before assembly pid: " << pid;

    // partition_graph();
    assembly();

    pid = getpid();
    DEBUG() << "after assembly pid: " << pid;

    INFO() << "peak mem (assembly) " << getPeakMemory() << " MB";
    if (aipubt::BuilderConfig::get<bool>(Constants::enable_dynamic_gm))
    {
        restore_node_output_for_dynamic_gm(m_graph, node_tensor_map, m_ro);
    }

    // record max readonly param size of plugins
    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        uint32_t max_param_size = 0;
        m_graph->walk_on_node([&max_param_size](NodePtr n) {
            if (!n->attrs().count(Constants::plugin_attrs_name) &&
                n->type() != OpType_Reshape &&
                n->type() != OpType_LayoutConvertor)
            {
                return;
            }
            uint32_t param_size = 0;
            if (n->attrs().count(Constants::ro_size))
            {
                param_size += aligned(n->get_attr<uint32_t>(Constants::ro_size), 256);
            }
            if (n->attrs().count(Constants::desc_size))
            {
                param_size += aligned(n->get_attr<uint32_t>(Constants::desc_size), 256);
            }
            if (n->attrs().count(Constants::descriptor_param))
            {
                param_size += aligned(4 * n->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_param).size(), 256);
            }
            if (n->attrs().count(Constants::descriptor_act))
            {
                param_size += aligned(4 * n->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_act).size(), 256);
            }
            max_param_size = std::max(max_param_size, param_size);
        });
        uint32_t core_num = get_target()->core_num();
        auto tp_base = m_graph->global_tensors()[Constants::temp_param_base];
        tp_base->shape()[0] = core_num * (256 + max_param_size);
    }

    auto get_size_string = [](uint64_t size, uint32_t precision = 2) -> std::string {
        return utils::get_human_readable_size(size, utils::DataUnit::DU_Undefined, precision);
    };
    auto ddr_fp_res = ddrfp.get_result();
    INFO() << "The graph DDR Footprint requirement(estimation) of feature maps:";
    INFO() << "    Read and Write:" << get_size_string(ddr_fp_res[0].read + ddr_fp_res[0].write);

    //push inputs/outputs/dumps and others tensors which may not in ro to ro
    for (auto inp : inputs)
    {
        m_ro.push_back(inp);
    }
    for (auto t : cross_submodel)
    {
        m_ro.push_back(t);
    }
    for (auto t : keepnodes)
    {
        m_ro.push_back(t);
    }
    // push all global tensors in ro
    for (auto kv : m_graph->global_tensors())
    {
        if (kv.second->mem_size() > 0)
        {
            m_ro.push_back(kv.second);
        }
    }
    if (BuilderConfig::get<bool>(Constants::enable_debug_info) && get_target()->isainfo().type >= ISA_ZYv5)
    {
        m_graph->walk_on_node(
            [&](NodePtr n) {
                if (n->attrs().count(Constants::plugin_attrs_name))
                {
                    for (auto t : n->inputs())
                    {
                        m_ro.push_back(t);
                    }
                    for (auto t : n->outputs())
                    {
                        m_ro.push_back(t);
                    }
                }
            });
    }
    // push output tensor shape buffer into ro
    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        PlaceholderPtr pb_base = ts_base;
        uint32_t relative_offset = 0;
        auto out_tensors = m_graph->get_output_tensors();
        for (uint32_t i = 0; i < out_tensors.size(); ++i)
        {
            uint32_t shape_size = sizeof(uint32_t) * out_tensors[i]->shape().dim();
            PlaceholderPtr pb = std::make_shared<Tensor>(TensorShape(shape_size));
            *pb = *pb_base + relative_offset;
            pb->set_attr(Constants::output_tensor_id, i);
            m_ro.push_back(pb);
            relative_offset += shape_size;
        }
    }
    if (int res = m_ro.size() % DEFAULT_RO_NUMBER_PER_LAYER)
    {
        for (int n = DEFAULT_RO_NUMBER_PER_LAYER; n > res; --n)
        {
            m_ro.push_back(0);
        }
    }

    // descriptor is just after ro in mini runtime.
    desc_base->memory_offset()->relative_offset = sizeof(uint32_t) * m_ro.size();

    // update all ro alignment
    for (auto ro_list : {m_ro, m_descriptor})
    {
        for (auto r : ro_list)
        {
            if (r.tensor)
            {
                r.tensor->memory_offset()->get_base_object()->align = r.tensor->memory_offset()->get_max_aligned();
            }
        }
    }

    // evaluate exprs in ro and desc if necessary
    // if (get_target()->isainfo().type >= ISA_ZYv5)
    {
        update_params(m_graph, m_ro);
        update_params(m_graph, m_descriptor);
    }

    // build remap table
    std::map<RemapcSection::SectionType, TensorPtr> base_tensors;
    base_tensors[RemapcSection::RO] = ro_base;
    base_tensors[RemapcSection::DESC] = desc_base;
    size_t remap_size = 0;
    build_remap(m_aipubin->remap, base_tensors, m_ro, m_descriptor, remap_size);

    //memory layout for continue case:
    // data:[stack,margin,constant_data,workspace,feature map]
    // first forzen weights
    // set stacksize
    m_aipubin->bss.stack_size = MAX(BuilderConfig::get<uint32_t>("stack_size"), 8192);
    uint32_t weight_offset = m_aipubin->bss.stack_size;

    std::vector<std::vector<MemoryObjectPtr>> const_slots;
    std::vector<std::vector<TensorPtr>> const_tensors;
    uint32_t fm_offset = 0;
    HashConstant()(m_graph);
    for (size_t i = 0; i < sub_model.size(); ++i)
    {
        auto sub_levels = sub_model[i];
        reduce_constant(sub_levels);
        TensorPtr random_reg;
        if (BuilderConfig::get<bool>("rand-reg") && i == 0)
        {
            auto target = Target::get_the_target();
            random_reg = m_graph->global_tensors()[Constants::random_reg];
            auto tec_num = target->tec_num();
            auto treg_num = 32;
            if (target->name() == "Z1_0701" || target->name() == "Z1_0701_P")
            {
                treg_num = 16;
            }
            const int acc_num = 4;
            const int gpr_num = 28;
            const int preg_num = 8;
            int size_random_data = treg_num * 64 * tec_num + //64bytes for a T reg
                                   acc_num * 64 +
                                   gpr_num * 4 +
                                   preg_num * 4;
            int32_t *random_data = new int32_t[size_random_data / 4 + 3];
            uint8_t *ptr = (uint8_t *)random_data;
            srand(BuilderConfig::get<uint32_t>("rand-seed"));
            for (auto i = 0; i < size_random_data + 3 * 4; ++i)
            {
                ptr[i] = rand() % 256;
            }
            random_data[0] = tec_num;
            random_data[1] = treg_num;
            random_data[2] = size_random_data;
            random_reg->shape() = TensorShape(size_random_data + 3 * 4);
            random_reg->set_data(random_data, size_random_data + 3 * 4);
        }
        else
        {
            random_reg = nullptr;
        }
        // allocate constants data to slots
        std::vector<TensorPtr> all_constants;
        auto slots = allocate_constant_data(weight_offset, sub_levels, random_reg, all_constants, fm_offset, i);
        const_slots.push_back(slots);
        const_tensors.push_back(all_constants);
    }
    // add workspace
    // weight_offset += 4 * 1024 * 1024;
    // set first slot base
    // fix slot size
    uint64_t current_offset = fm_offset;
    if (get_target()->isainfo().type >= ISA_ZYv5)
    {
        // use dump 2
        current_offset = 0;
    }
    // reallocate workspace size

    size_t ws_size = 0;
    for (auto node : m_graph->nodes())
    {
        size_t ws = node->get_attr<int>("workspace_size");
        ws_size = ws_size > ws ? ws_size : ws;
        auto sgn = SubGraphNode::from_node(node);
        if (sgn)
        {
            auto t = sgn->subgraph()->global_tensors()[Constants::workspace];
            t->shape()[0] = ws;
            t->memory_offset()->size = ws;
        }
    }
    ws_size = aligned(ws_size);
    m_graph->global_tensors()["workspace"]->shape()[0] = ws_size;

    std::vector<std::vector<MemoryObjectPtr>> new_data_slots;
    for (auto slots : data_slots)
    {
        std::set<MemoryObjectPtr> global_slots = {};
        std::set<MemoryObjectPtr> input_slots = {};
        std::set<MemoryObjectPtr> output_slots = {};
        for (auto slot : slots)
        {
            for (auto gt : m_graph->global_tensors())
            {
                if (gt.second && slot->is_shared_by(gt.second->memory_offset()))
                {
                    if (gt.first == "workspace")
                    {
                        slot->size = ws_size;
                    }
                    if (gt.first == Constants::temp_param_base)
                    {
                        slot->size = gt.second->mem_size();
                    }
                    global_slots.insert(slot);
                    break;
                }
            }
            for (auto tensor : m_graph->get_input_tensors())
            {
                if (slot->is_shared_by(tensor->memory_offset()))
                {
                    input_slots.insert(slot);
                    break;
                }
            }
            for (auto tensor : m_graph->get_output_tensors())
            {
                if (slot->is_shared_by(tensor->memory_offset()))
                {
                    output_slots.insert(slot);
                    break;
                }
            }
        }
        uint32_t extsram_offset = 0;
        uint32_t extsram_end = 0;
        if (extsram_infos.size() > 0)
        {
            extsram_offset = extsram_infos[0].first;
            extsram_end = extsram_infos[0].first + extsram_infos[0].second;
        }
        std::vector<size_t> used_size = {};
        std::vector<size_t> used_addr = {};
        std::vector<MemoryObjectPtr> unalloc_input_slots;
        std::vector<MemoryObjectPtr> slots_none_soc_sram = {}; //slots that should not be on SoC SRAM
        for (auto slot : slots)
        {
            // slots with global_tensors/input_tensors/output_tensors should NOT be on SoC SRAM
            // If enable alloc_output_on_extsram, output slot would alloc on SoC SRAM.
            if (((extsram_infos.size() > 0) &&
                 (global_slots.count(slot) == 0) &&
                 (input_slots.count(slot) == 0) &&
                 (output_slots.count(slot) == 0) &&
                 (extsram_end - aligned(extsram_offset, slot->align) >= slot->size)) ||
                (alloc_output_on_extsram && output_slots.count(slot)))
            {
                extsram_offset = aligned(extsram_offset, slot->align);
                slot->m_base = extsram_offset;
                extsram_offset += slot->size;
                used_addr.push_back(extsram_offset);
                used_size.push_back(slot->size);
            }
            else
            {
                if (get_target()->isainfo().type < ISA_ZYv5)
                {
                    if (BuilderConfig::get<bool>("individual-inputs") && input_slots.count(slot) > 0 && !contains(unalloc_input_slots, slot))
                    {
                        unalloc_input_slots.push_back(slot);
                    }
                    else
                    {
                        current_offset = aligned(current_offset, slot->align);
                        slot->m_base = current_offset;
                        if (slot->size == 0)
                        {
                            slot->size = 64;
                        }
                        current_offset += slot->size;
                    }
                }
                slots_none_soc_sram.push_back(slot);
            }
            slot->base_offset = nullptr;
            slot->relative_offset = 0;
        }
        new_data_slots.push_back(slots_none_soc_sram);

        // show_buffer_slots(m_graph, slots_none_soc_sram);
        // ensure addr of ddr slots for input is after stack and const, so that we can separate them
        if (BuilderConfig::get<bool>("individual-inputs"))
        {
            for (auto slot : unalloc_input_slots)
            {
                current_offset = aligned(current_offset, slot->align);
                slot->m_base = current_offset;
                current_offset += slot->size;
            }
        }
        if (used_size.size() > 0)
        {
            std::ostringstream oss_addr, oss_size;
            oss_addr << "[";
            oss_size << "[";
            // for (auto size: used_size)
            for (size_t i = 0; i < used_size.size(); i++)
            {
                if (i != 0)
                {
                    oss_addr << ", ";
                    oss_size << ", ";
                }
                oss_addr << "0x" << std::hex << used_addr[i];
                oss_size << "0x" << std::hex << used_size[i];
            }
            oss_addr << "]";
            oss_size << "]";
            INFO("Using external sram to accelerate AIPU. External sram addr: %#x, total size: %#x Bytes",
                 extsram_infos[0].first, extsram_infos[0].second);
            INFO("External sram used by %d buffers. Addr: ", used_size.size()) << oss_addr.str() << ", Size: " << oss_size.str();
        }
    }

    // build buffer map and tensor map
    for (uint32_t idx = 0; idx < m_ro.size(); ++idx)
    {
        auto tensor = m_ro[idx].tensor;
        if (tensor)
        {
            auto boffset = tensor->memory_offset()->get_base_object();
            if (!m_tensor_map.count(tensor))
            {
                m_buffer_map[boffset].push_back(m_ro[idx].tensor);
            }
            m_tensor_map[tensor].push_back(idx);
        }
    }
    for (uint32_t idx = 0; idx < m_descriptor.size(); ++idx)
    {
        auto tensor = m_descriptor[idx].tensor;
        if (tensor)
        {
            auto boffset = tensor->memory_offset()->get_base_object();
            if (!m_tensor_map.count(tensor))
            {
                m_buffer_map[boffset].push_back(m_descriptor[idx].tensor);
            }
            m_tensor_map[tensor].push_back(idx + m_ro.size());
            // INFO() << "desc_idx: "<< idx + m_ro.size()
            //       << ". addr : 0x" << std::hex << tensor->offset();
        }
    }

    //build buffer
    // std::map<BssSubSectionDescriptor::SubSectionType, std::vector<TensorPtr>> special_marked_tensor;
    m_marked_tensors[BssSubSectionDescriptor::input] = inputs;
    m_marked_tensors[BssSubSectionDescriptor::output] = outputs;
    m_marked_tensors[BssSubSectionDescriptor::dump] = std::vector<TensorPtr>(all_dumps.begin(), all_dumps.end());
    m_marked_tensors[BssSubSectionDescriptor::profile_data] = {profile};
    m_marked_tensors[BssSubSectionDescriptor::printf_fifo] = {printf_fifo};
    m_marked_tensors[BssSubSectionDescriptor::op_cnt] = {op_cnt};
    m_marked_tensors[BssSubSectionDescriptor::error_code] = {error_code};
    m_marked_tensors[BssSubSectionDescriptor::cross_submodel] = cross_submodel;
    m_marked_tensors[BssSubSectionDescriptor::dump_tensor_shape] = {ts_base};
    std::vector<BssSection> bss(sub_model.size(), m_aipubin->bss);
    for (size_t i = 0; i < sub_model.size(); ++i)
    {
        auto data_slot = new_data_slots[i];
        auto const_slot = const_slots[i];
        build_bss_section(bss[i], data_slot, const_slot, m_buffer_map, m_tensor_map, m_marked_tensors, i);
        DEBUG() << "submodel " << i;
        uint64_t total_slot_size = 0;
        for (auto slot : data_slot)
        {
            DEBUG() << "data slot" << slot->get_base_object()->buffer_id << " size: " << utils::get_human_readable_size(slot->get_base_object()->size) << " fp: " << utils::get_human_readable_size(slot->foot_print);
            total_slot_size += slot->get_base_object()->size;
        }
        DEBUG() << "total data slot size: " << utils::get_human_readable_size(total_slot_size);
        for (auto slot : m_gm_regions)
        {
            DEBUG() << "gm slot" << slot->get_base_object()->buffer_id << " fp: " << utils::get_human_readable_size(slot->foot_print);
        }
        for (auto slot : const_slot)
        {
            auto tensors = m_buffer_map[slot->get_base_object()];
            for (auto t : tensors)
            {
                DEBUG() << "constant slot" << slot->get_base_object()->buffer_id << " " << t->name() << " size: " << utils::get_human_readable_size(slot->get_base_object()->size);
            }
        }
    }

    if (aipubt::BuilderConfig::get<bool>(Constants::dump_constant_info))
    {
        dump_const_info_csv(m_graph, sub_model, const_slots, m_buffer_map);
    }

    m_bss_vec = bss;
    m_aipubin->bss = bss[0];
    m_aipubin->header.bss_size += remap_size;

    build_ro(m_aipubin->ro, m_ro);
    // build descriptor
    build_ro(m_aipubin->desc, m_descriptor);
    build_text();
    m_const_tensors = const_tensors;
    m_const_offsets.resize(const_tensors.size());
    build_data(m_aipubin->data, const_tensors[0], m_const_offsets[0]);
    for (size_t i = 1; i < const_tensors.size(); ++i)
    {
        BinSection data_bin;
        build_data(data_bin, const_tensors[i], m_const_offsets[i]);
        m_aipubin->add_extra_data(data_bin);
    }

    //// header
    ABHeader &header = m_aipubin->header;
    if (BuilderConfig::get<bool>("enable-asid"))
    {
        //ASID use 1 for default config
        header.flag |= 1;
    }
    // enable remapping
    header.flag |= (1 << 4);
    if (BuilderConfig::get<bool>(Constants::preallocated_soc_sram))
    {
        header.flag |= (1 << 5);
    }
    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        header.flag |= (1 << 6);
    }

    header.arch = get_target()->gen_arch_number();

    // unsigned int bss_size = 0;
    // for (unsigned int i = 0; i < m_aipubin->bss.num_buffer_descriptor; i++)
    // {
    // INFO() << "size " << i << " = 0x" << std::hex << m_aipubin->bss.buffer_descriptor[i].size;
    // bss_size += m_aipubin->bss.buffer_descriptor[i].size;
    // }
    uint32_t file_offset = sizeof(m_aipubin->header);
    m_aipubin->header.text_offset = file_offset;
    m_aipubin->header.text_size = m_aipubin->text.bin_size;
    file_offset += m_aipubin->text.bin_size;
    m_aipubin->header.rodata_offset = file_offset;
    m_aipubin->header.rodata_size = m_aipubin->ro.bin_size;
    file_offset += m_aipubin->ro.bin_size;
    m_aipubin->header.dcr_offset = file_offset;
    m_aipubin->header.dcr_size = m_aipubin->desc.bin_size;
    file_offset += m_aipubin->desc.bin_size;
    m_aipubin->header.data_offset = file_offset;
    m_aipubin->header.data_size = m_aipubin->data.bin_size;
    file_offset += m_aipubin->data.bin_size;
    m_aipubin->header.bss_offset = file_offset;
    m_aipubin->header.bss_size = m_aipubin->bss.size();
    file_offset += m_aipubin->header.bss_size;
    file_offset += m_aipubin->remap.size();
    m_aipubin->header.filesize = file_offset;

    if (BuilderConfig::get<bool>(Constants::enable_tcm))
    {
        uint32_t tcms = BuilderConfig::get<uint32_t>(Constants::tcm_size) / 1024;
        memcpy(m_aipubin->header.extra_data + 4, &tcms, sizeof(uint32_t));
    }

    //clean build time temp data
    for (auto node : *m_graph)
    {
        //free subgraph default text
        if (SubGraphNode::is_subgraph_node(node))
        {
            SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(node);
            if (sgn->attrs().count("default_text"))
            {
                if (sgn->get_attr<bool>("default_text"))
                {
                    BuilderText txt = sgn->get_attr<BuilderText>("text");
                    if (txt.first)
                    {
                        delete[] txt.first;
                    }
                    sgn->set_attr("default_text", false);
                    sgn->erase_attr("text");
                }
            }
        }
        node->erase_attr("workspace");
    }

    INFO() << "peak mem (aipu.bin) " << getPeakMemory() << " MB";

    if (get_target()->isainfo().type < ISA_ZYv5)
    {
        show_memory_info();
    }
    return error_cnt == GCompilerLogger::get_error_count();
}

void GraphBuilder::show_memory_info()
{
    unsigned int total_size = get_text_size() +
                              m_aipubin->ro.bin_size +
                              m_aipubin->desc.bin_size +
                              m_aipubin->data.bin_size +
                              m_aipubin->bss.stack_size +
                              m_aipubin->header.bss_size;
    for (auto extra_data : m_aipubin->extra_data)
    {
        total_size += extra_data.bin_size;
    }
    INFO() << "memory statistics for this graph"
           << " (" << m_graph->name() << ")";
    // keep the same data_unit
    auto data_unit = utils::get_best_data_unit(total_size);
    show_section_size("Total memory     :\t", total_size);
    show_section_size("Text      section:\t", get_text_size(), data_unit);
    show_section_size("RO        section:\t", m_aipubin->ro.bin_size, data_unit);
    show_section_size("Desc      section:\t", m_aipubin->desc.bin_size, data_unit);
    show_section_size("Data      section:\t", m_aipubin->data.bin_size, data_unit);
    for (auto extra_data : m_aipubin->extra_data)
    {
        show_section_size("ExtraData section:\t", extra_data.bin_size, data_unit);
    }
    show_section_size("BSS       section:\t", m_aipubin->header.bss_size, data_unit);
    show_section_size("Stack            :\t", m_aipubin->bss.stack_size, data_unit);
    show_section_size("Workspace(BSS)   :\t", m_graph->global_tensors()["workspace"]->shape()[0], data_unit);
    INFO();
}

static void delete_unused_const(NodePtr node, const BuilderParams &ro, const BuilderParams &desc, size_t ro_idx, size_t ro_end, size_t desc_idx, size_t desc_end)
{
    // if (ro_end == ro_idx && desc_idx == desc_end)
    // {
    //     return;
    // }
    std::set<MemoryObjectPtr> used;
    for (auto i = ro_idx; i < ro_end; ++i)
    {
        if (ro[i].tensor)
        {
            used.insert(ro[i].tensor->memory_offset()->get_base_object());
        }
    }
    for (auto i = desc_idx; i < desc_end; ++i)
    {
        if (desc[i].tensor)
        {
            used.insert(desc[i].tensor->memory_offset()->get_base_object());
        }
    }
    std::set<std::string> to_removed;
    for (auto kv : node->constants())
    {
        if (kv.second)
        {
            if (kv.second->attrs().count(Constants::zerocopy_cst) && kv.second->get_attr(Constants::zerocopy_cst).cast<bool>())
            {
                continue;
            }
            if (!used.count(kv.second->memory_offset()->get_base_object()))
            {
                to_removed.insert(kv.first);
            }
        }
    }
    for (auto k : to_removed)
    {
        // INFO() << "delete const " << k << " of node " << node->name();
        node->constants().erase(k);
    }
}
void GraphBuilder::assembly()
{
    m_codetree = CodeTree();
    m_ro.clear();
    m_codes.clear();
    m_kernels.clear();
    m_text.clear();
    m_descriptor.clear();
    m_ro.reserve(DEFAULT_RO_NUMBER_PER_LAYER * m_graph->nodes().size());
    m_ro.resize(DEFAULT_RO_NUMBER_PER_LAYER, 0);
    PlaceholderPtr ro_base = std::static_pointer_cast<Placeholder>(m_graph->global_tensors()[Constants::parambase]);
    PlaceholderPtr descriptor_base = std::static_pointer_cast<Placeholder>(m_graph->global_tensors()[Constants::descriptorbase]);
    TensorPtr tp_base = m_graph->global_tensors()[Constants::temp_param_base];
    TensorPtr ts_base = m_graph->global_tensors()[Constants::tensor_shape_base];
    std::string terminator = BuilderConfig::get<std::string>("terminator");
    PlaceholderPtr next_descriptor_length = nullptr;
    PlaceholderPtr next_descriptor_offset = nullptr;
    PlaceholderPtr next_descriptor_eof = nullptr;
    bool enable_perf = aipubt::AIPUBuilder::get_profiler()->enable_profile();
    uint32_t operator_id = 1;
    auto target = Target::get_the_target();
    // uint32_t pb_offset = m_ro.size();
    // uint32_t db_offset = m_descriptor.size();
    // uint32_t code_offset = m_text.size();
    for (auto node : *m_graph)
    {
        auto desc_act_base = std::make_shared<Placeholder>();
        auto desc_param_base = std::make_shared<Placeholder>();
        auto desc_dt_base = std::make_shared<Placeholder>();
        auto ro_generator = get_ro_generator(node->type());
        auto code_generator = get_code_generator(node->type());
        auto kernel_generator = get_kernel_generator(node->type());
        auto descriptor_generator = get_descriptor_generator(node->type());
        if (!node->attrs().count(Constants::parambase))
        {
            node->set_attr(Constants::parambase, std::make_shared<Placeholder>());
        }
        if (!node->attrs().count(Constants::descriptorbase))
        {
            node->set_attr(Constants::descriptorbase, std::make_shared<Placeholder>());
        }
        node->set_attr(Constants::descriptor_param_base, desc_param_base);
        node->set_attr(Constants::descriptor_act_base, desc_act_base);
        node->set_attr(Constants::descriptor_dt_base, desc_dt_base);
        node->set_attr(Constants::first_descriptor_length, next_descriptor_length);
        node->set_attr(Constants::first_descriptor_offset, next_descriptor_offset);
        node->set_attr(Constants::previous_descriptor_eof, next_descriptor_eof);
        auto pb = node->get_attr<PlaceholderPtr>(Constants::parambase);
        *pb = *ro_base + sizeof(uint32_t) * m_ro.size();
        auto db = node->get_attr<PlaceholderPtr>(Constants::descriptorbase);
        *db = *descriptor_base + sizeof(uint32_t) * m_descriptor.size();

        node->set_attr(Constants::node_perf, false);
        if (SubGraphNode::is_subgraph_node(node) && !node->attrs().count(Constants::plugin_attrs_name))
        {
            // for virtual subgraph node, setup the opid counter
            node->set_attr(Constants::opid_counter, operator_id);
        }
        node->set_attr(Constants::temp_param_base, tp_base);
        node->set_attr(Constants::tensor_shape_base, ts_base);
        if (!node->attrs().count(Constants::codetree))
        {
            node->set_attr(Constants::codetree, std::make_shared<CodeTree>(node->name(), node));
        }
        auto codetree = node->get_attr<CodeTreePtr>(Constants::codetree);
        // descriptor is optional
        if (ro_generator && code_generator)
        {
            // core order: descriptor, code, ro
            size_t des_start = m_descriptor.size();
            size_t ro_start = m_ro.size();
            bool dump_rt_tensor = BuilderConfig::get<bool>(Constants::dump_rt_output_tensors) || BuilderConfig::get<bool>(Constants::dump_rt_input_tensors) || (BuilderConfig::get<int>(Constants::dump_rt_tensor_id) != -1);
            if (descriptor_generator || dump_rt_tensor)
            {
                node->set_attr(Constants::desc_offset, des_start * 4); // offset is in byte
                descriptor_generator(node, m_descriptor);
                // concat descriptor for x2
                if (node->attrs().count(Constants::descriptor_param))
                {
                    *desc_param_base = *descriptor_base + (m_descriptor.size() * sizeof(uint32_t));
                    auto desc_param = node->get_attr<BuilderParams>(Constants::descriptor_param);
                    if (desc_param.size())
                    {
                        m_descriptor.insert(m_descriptor.end(), desc_param.begin(), desc_param.end());
                    }
                    node->erase_attr(Constants::descriptor_param);
                }
                if (node->attrs().count(Constants::descriptor_act))
                {
                    *desc_act_base = *descriptor_base + (m_descriptor.size() * sizeof(uint32_t));
                    auto desc_act = node->get_attr<BuilderParams>(Constants::descriptor_act);
                    if (desc_act.size())
                    {
                        m_descriptor.insert(m_descriptor.end(), desc_act.begin(), desc_act.end());
                    }
                    node->erase_attr(Constants::descriptor_act);
                }
                if (node->attrs().count(Constants::descriptor_dt))
                {
                    *desc_dt_base = *descriptor_base + (m_descriptor.size() * sizeof(uint32_t));
                    auto desc_dt = node->get_attr<BuilderParams>(Constants::descriptor_dt);
                    if (desc_dt.size())
                    {
                        m_descriptor.insert(m_descriptor.end(), desc_dt.begin(), desc_dt.end());
                    }
                    node->erase_attr(Constants::descriptor_dt);
                }
                //align to 64
                if (int res = m_descriptor.size() % DEFAULT_RO_NUMBER_PER_LAYER)
                {
                    for (int n = 64; n > res; --n)
                    {
                        m_descriptor.push_back(0);
                    }
                }
                node->set_attr(Constants::desc_size, (m_descriptor.size() - des_start) * sizeof(uint32_t)); // offset is in byte
            }
            auto generated_code = code_generator(node);
            if (generated_code.empty())
            {
                if (terminator == node->name())
                {
                    m_codes.push_back({"_exit"});
                }
            }
            else
            {
                auto node_codes = utils::split_string(generated_code, '\n');
                std::vector<uint32_t> operator_ids;
                if (!utils::startswith(generated_code, ".function"))
                {
                    for (auto code : node_codes)
                    {
                        if (code == Constants::profiler_code)
                        {
                            continue;
                        }
                        if (target->isainfo().type < ISA_ZYv5 && code.size() && !utils::startswith(code, ".") && code != "_exit")
                        {
                            auto text = get_text_content_by_filename(code);
                            if (!text.first)
                            {
                                ERROR() << node->name() << " generate a code \""
                                        << code
                                        << "\" which does not exist. Please double check the plugin.";
                            }
                        }
                        operator_ids.push_back(operator_id);
                        operator_id++;
                    }
                    node->set_attr(Constants::operator_id, operator_ids);
                }
                else
                {
                    if (SubGraphNode::is_subgraph_node(node) && !node->attrs().count(Constants::plugin_attrs_name))
                    {
                        // for virtual subgraph node, setup the opid counter
                        operator_id = node->get_attr<uint32_t>(Constants::opid_counter);
                    }
                }

                std::vector<std::string> valid_codes;
                std::copy_if(node_codes.begin(), node_codes.end(), std::back_inserter(valid_codes), is_valid_code);
                m_codes.push_back(valid_codes);
                node->set_attr(Constants::ro_offset, ro_start * 4); // offset is in byte
                ro_generator(node, m_ro);
                if (int res = m_ro.size() % DEFAULT_RO_NUMBER_PER_LAYER)
                {
                    for (int n = 64; n > res; --n)
                    {
                        m_ro.push_back(0);
                    }
                }

                node->set_attr(Constants::ro_size, (m_ro.size() - ro_start) * 4); // offset is in byte

                if (kernel_generator)
                {
                    auto kernels = kernel_generator(node);
                    auto node_kernels = utils::split_string(kernels, '\n');

                    std::vector<std::string> valid_kernels;
                    std::copy_if(node_kernels.begin(), node_kernels.end(), std::back_inserter(valid_kernels), is_valid_kernel);
                    m_kernels.push_back(valid_kernels);

                    if (!codetree->is_leaf()) // skip empty task
                    {
                        m_codetree.codes.push_back(codetree);
                    }
                }

                // remove useless constants
                delete_unused_const(node, m_ro, m_descriptor, ro_start, m_ro.size(), des_start, m_descriptor.size());
            }
        }
        else
        {
            auto bk = BuilderConfig::get<bool>(Constants::show_location);
            BuilderConfig::set(Constants::show_location, true);
            ERROR() << "Unsupport node type: "
                    << optype_to_string(node->type()) << "\tname:"
                    << node->name() << " definition:\n"
                    << node->to_string();
            std::vector<std::string> &logs = node->get_attr(Constants::build_error_log);
            std::vector<std::string> &warns = node->get_attr(Constants::build_warn_log);
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
            BuilderConfig::set(Constants::show_location, bk);
        }

        if (terminator == node->name())
        {
            m_codes.push_back({"_exit"});
        }

        // if debug mode or profiler enable, set next descriptor nullptr to break the chain
        if (BuilderConfig::get<bool>(Constants::enable_gm_enhance) ||
            BuilderConfig::get<bool>(Constants::enable_dynamic_gm) ||
            BuilderConfig::get<bool>(Constants::enable_x2_sm_opt) ||
            BuilderConfig::get<uint64_t>(Constants::max_wt_size) ||
            BuilderConfig::get<uint64_t>(Constants::max_act_size) ||
            BuilderConfig::get<bool>(Constants::break_descriptor_chain))
        {
            next_descriptor_length = nullptr;
            next_descriptor_offset = nullptr;
            next_descriptor_eof = nullptr;
        }
        else
        {
            next_descriptor_length = node->get_attr<PlaceholderPtr>(Constants::next_descriptor_length);
            next_descriptor_offset = node->get_attr<PlaceholderPtr>(Constants::next_descriptor_offset);
            next_descriptor_eof = node->get_attr<PlaceholderPtr>(Constants::next_descriptor_eof);
        }

        if (enable_perf)
        {
            auto func = [](const NodePtr &n) { return n->attrs().count(Constants::node_perf) && n->get_attr<bool>(Constants::node_perf); };
            if ((SubGraphNode::is_subgraph_node(node) && node->attrs().count(Constants::plugin_attrs_name)) || !SubGraphNode::is_subgraph_node(node))
            {
                bool perf = func(node);
                node->set_attr(Constants::node_perf, perf);
            }
            else
            {
                SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
                std::vector<NodePtr> operation_nodes;
                AIPUBuilder::get_operation_nodes(sgnode->subgraph(), operation_nodes);
                auto found = std::find_if(operation_nodes.begin(), operation_nodes.end(), func);
                bool perf = found != operation_nodes.end();
                node->set_attr(Constants::node_perf, perf);
            }
        }
    }

    DEBUG() << "graph codetree:\n"
            << m_codetree;

    // uint32_t pb_size = m_ro.size() - pb_offset;
    // uint32_t db_size = m_descriptor.size() - db_offset;
    // uint32_t code_size = m_text.size() - code_offset;
    // subgraph->set_attr("pb_size", pb_size);
    // subgraph->set_attr("db_size", db_size);
    // subgraph->set_attr("code_size", code_size);
    // subgraph->set_attr("pb_offset", pb_offset);
    // subgraph->set_attr("db_offset", db_offset);
    // subgraph->set_attr("code_offset", code_offset);
    // break the chain for each subgraph
    next_descriptor_length = nullptr;
    next_descriptor_offset = nullptr;
    next_descriptor_eof = nullptr;
    // build global ro
    // 64 global params
    // 0-5 for normal events profiler
    // 6 for breakdump
    // 7 for profiler aiff events
    // 8 for profiler xbu events
    // 9 for random init
    // 10 for printf fifo
    // 12 for error_code
    // 13 for op_cnt (DEBUG mode only)

    // some global info is generated after other nodes ro is generated
    // so put the global init ro at the end
    if (enable_perf)
    {
        auto interface = AIPUBuilder::get_profiler()->interface();
        auto fn = interface->get_function("generate_profiler_ro");
        if (fn)
        {
            fn(&m_ro);
        }
    }
    // dumpspace for 6th ro
    m_ro[6] = 0;
    m_ro[9] = m_graph->global_tensors()[Constants::random_reg];
    // print fifo for 10th ro
    m_ro[10] = m_graph->global_tensors()[Constants::printf_fifo];
    //print fifo's size
    m_ro[11] = aipubt::BuilderConfig::get<uint64_t>("printf_fifo_size");
    m_ro[12] = m_graph->global_tensors()[Constants::error_code];
    if (BuilderConfig::get<bool>("DEBUG"))
    {
        m_ro[13] = m_graph->global_tensors()[Constants::op_cnt];
    }
    if (get_target()->isainfo().type >= ISA_ZYv5) // ro size need at least be 1K
    {
        while (m_ro.size() < 256)
        {
            m_ro.emplace_back(0);
        }
    }
}

const std::pair<char *, size_t> GraphBuilder::get_build_text() const
{
    if (m_aipubin)
        return {m_aipubin->text.bin, m_aipubin->text.bin_size};
    return {nullptr, 0};
}
GraphBuilder::~GraphBuilder()
{
    if (m_aipubin)
    {
        delete m_aipubin;
    }
}
} // namespace aipubt

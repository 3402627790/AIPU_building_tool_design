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
 * @Author: Kyle Huang
 * @Date: 2023-03-20 17:00:00
 * @Last Modified by: kyle.huang
 * @Last Modified time: 2024-08-14 17:24:16
 */
#include "builder/dumper/dumper_x.h"
#include "builder/load_balancer.h"
#include "builder/scheduler.h"
#include "builder/soms_scheduler.h"
#include "config_manager.h"
#include "core/constants.h"
#include "core/proxy_tensor.h"
#include "core/subgraph.h"
#include "dump_manager.h"
#include "tcb_manager.h"
#include <fstream>
#include <utils/utils.h>

#include <iostream>

#define DUMPER_VERSION 1
namespace aipubt
{
#define REMAP_SEGMENT0 0xf0000000

DumperX::DumperX(BuilderConfig *configs, AIPUELFPtr elf, Target *target, GraphPtr graph, const std::vector<ReadOnlyParam> &ro)
{
    if (graph)
    {
        m_mode = Run_Mode::run_by_builder;
    }
    else
    {
        m_mode = Run_Mode::run_by_dumper;
    }
    m_arg_map[Constants::load_addr_cfg] = (m_mode == Run_Mode::run_by_builder) ? "" : configs->get<std::string>(Constants::load_addr_cfg);
    m_arg_map[Constants::dump_addr_cfg] = configs->get<bool>(Constants::dump_addr_cfg);
    m_arg_map[Constants::profile] = configs->get<bool>(Constants::profile);
    m_arg_map[Constants::ddr_data_gather] = configs->get<bool>(Constants::ddr_data_gather);
    m_arg_map[Constants::insert_init_reg] = configs->get<bool>(Constants::insert_init_reg);
    m_arg_map[Constants::set_asid] = configs->get<string>(Constants::set_asid);
    m_arg_map[Constants::dump_all_tensors] = configs->get<bool>(Constants::dump_all_tensors);
    m_arg_map[Constants::disable_gm] = configs->get<bool>(Constants::disable_gm);
    m_arg_map[Constants::enable_gm_enhance] = configs->get<bool>(Constants::enable_gm_enhance);
    m_arg_map[Constants::enable_dynamic_gm] = configs->get<bool>(Constants::enable_dynamic_gm);
    m_arg_map[Constants::enable_x2_sm_opt] = configs->get<bool>(Constants::enable_x2_sm_opt);
    m_arg_map[Constants::enable_weight_align] = configs->get<uint32_t>(Constants::enable_weight_align);
    m_arg_map[Constants::ddr_data_mode] = configs->get<uint32_t>(Constants::ddr_data_mode);
    m_arg_map[Constants::enable_slc_ocm_mode] = configs->get<bool>(Constants::enable_slc_ocm_mode);
    m_arg_map[Constants::enable_custom_smmu] = configs->get<bool>(Constants::enable_custom_smmu);
    m_arg_map[Constants::gm_size] = configs->get<uint32_t>(Constants::gm_size);
    m_arg_map[Constants::ocm_addr] = configs->get<uint64_t>(Constants::ocm_addr);
    m_arg_map[Constants::ocm_size] = configs->get<uint64_t>(Constants::ocm_size);
    m_arg_map["random-init"] = configs->get<bool>("random-init");
    m_arg_map[Constants::multi_batch] = configs->get<uint32_t>(Constants::multi_batch);
    m_arg_map[Constants::put_weight_gm] = configs->get<bool>(Constants::put_weight_gm);
    m_arg_map[Constants::multi_core] = (uint32_t)target->core_num();
    m_arg_map[Constants::target_name] = (string)target->name();
    m_arg_map[Constants::multi_tec] = (uint32_t)target->tec_num();
    m_arg_map[Constants::enable_ds] = configs->get<bool>(Constants::enable_ds);
    m_arg_map[Constants::put_desc_gm] = configs->get<bool>(Constants::put_desc_gm);
    m_arg_map[Constants::put_ws_gm] = configs->get<bool>(Constants::put_ws_gm);
    m_arg_map[Constants::set_l2_flush] = configs->get<bool>(Constants::set_l2_flush);
    m_arg_map[Constants::native_float_simulator] = configs->get<bool>(Constants::native_float_simulator);
    m_arg_map[Constants::ws_mem_opt] = configs->get<bool>(Constants::ws_mem_opt);
    m_arg_map[Constants::reuse_buffer_expand] = configs->get<bool>(Constants::reuse_buffer_expand);
    m_arg_map[Constants::specify_wt_gm] = configs->get<string>(Constants::specify_wt_gm);
    m_arg_map[Constants::set_profile_option] = configs->get<string>(Constants::set_profile_option);
    m_arg_map[Constants::max_wt_size] = configs->get<uint32_t>(Constants::max_wt_size);
    m_arg_map[Constants::max_act_size] = configs->get<uint32_t>(Constants::max_act_size);
    m_arg_map[Constants::split_wt_act] = configs->get<bool>(Constants::split_wt_act);
    m_arg_map["DEBUG"] = configs->get<bool>("DEBUG");
    m_elf = elf;
    m_graph = graph;
    m_ro = ro;
}

inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

struct GmInfo
{
    uint64_t addr;
    uint32_t size;
    uint32_t sync_in_offset;
    uint32_t sync_in_size;
};

uint32_t DumperX::get_ro_offset(uint32_t offset)
{
    return m_ro_offset_map[offset];
};
uint32_t DumperX::get_desc_offset(uint32_t offset)
{
    return m_desc_offset_map[offset];
};
uint32_t DumperX::get_ro_id(uint32_t offset)
{
    return m_ro_id_map[offset];
};
uint32_t DumperX::get_desc_id(uint32_t offset)
{
    return m_desc_id_map[offset];
};
bool DumperX::set_ro_offset(uint32_t offset, uint32_t new_offset, uint32_t bss_id)
{
    m_ro_offset_map[offset] = new_offset;
    m_ro_id_map[offset] = bss_id;

    // INFO() << "set ro offset old: 0x" << std::hex << offset << " new: 0x" << new_offset;

    return true;
};
bool DumperX::set_desc_offset(uint32_t offset, uint32_t new_offset, uint32_t bss_id)
{
    m_desc_offset_map[offset] = new_offset;
    m_desc_id_map[offset] = bss_id;

    // INFO() << "set desc offset old: 0x" << std::hex << offset << " new: 0x" << new_offset;
    return true;
};

static std::vector<MemInfo>
gen_reuse_cfg(const BssSection &bss, uint32_t bss_id)
{
    vector<BssSubSectionDescriptor> sub_sections;
    uint32_t addr = 0;
    std::vector<std::string> type_vec;
    std::string type_str;
    // INFO() << "reuse addr";
    std::vector<MemInfo> cfg_vec;
    for (uint32_t i = 0; i < bss.num_buffer_descriptor; ++i)
    {
        auto address = aligned(addr, bss.buffer_descriptor[i].align_bytes);
        MemInfo mem_info;
        mem_info.address = address;
        mem_info.align = bss.buffer_descriptor[i].align_bytes;
        mem_info.size = bss.buffer_descriptor[i].size;
        mem_info.slot_idx = i;
        mem_info.mem_type = 0;
        mem_info.data_offset = 0;
        mem_info.bss_id = bss_id;
        mem_info.asid = 0;
        sub_sections = bss.buffer_descriptor[i].sub_sections;
        for (uint32_t j = 0; j < sub_sections.size(); ++j)
        {
            auto it = mem_info.type.find(sub_sections[j].type);
            if (it == mem_info.type.end())
            {
                mem_info.type.insert(std::make_pair(sub_sections[j].type, sub_sections[j].id));
            }
        }
        // INFO() << "index: " << i << " addr: 0x" << hex << mem_info.address << " size: 0x" << mem_info.size << " align: 0x" << mem_info.align;
        // for (auto type : mem_info.type)
        //     INFO() << type;

        mem_info.address = aligned(mem_info.address, mem_info.align);
        cfg_vec.push_back(mem_info);
        // mem_info.batch_vec.clear();

        type_vec.clear();
        addr = mem_info.address + mem_info.size;
    }
    return cfg_vec;
}

/*get tensor name from m_ro by ro offset*/
static string get_name_from_ro(uint32_t ro_offset, const std::vector<ReadOnlyParam> &ro)
{
    int idx = ro_offset / 4;
    if (idx > (int)ro.size())
    {
        return "";
    }
    auto tensor = ro[idx].tensor;
    if (tensor)
    {
        string name = tensor->name();
        return utils::replace(name, "/", "_");
    }
    return "";
}
// static int get_slot_from_ro(uint32_t ro_offset, const std::vector<ReadOnlyParam> &ro)
// {
//     int idx = ro_offset / 4;
//     if (idx > (int)ro.size())
//     {
//         return -2;
//     }
//     auto tensor = ro[idx].tensor;
//     if (tensor)
//     {
//         auto root = tensor->memory_offset()->get_base_object();
//         return root->buffer_id;
//     }
//     return -1;
// }
static std::vector<MemInfo> gen_const_cfg(const BssSection &bss, uint32_t bss_id, const std::vector<uint32_t> &gm_wt_vec, bool is_zerocpy, bool is_gm)
{
    std::vector<MemInfo> const_cfg;
    vector<BssSubSectionDescriptor> sub_sections;
    uint32_t addr = 0;
    for (uint32_t i = 0; i < bss.num_data_descriptor; ++i)
    {
        auto type = bss.data_descriptor[i].sub_sections[0].type;
        if (((type == BssSubSectionDescriptor::SubSectionType::zerocpy_constant) && !is_zerocpy) ||
            ((type != BssSubSectionDescriptor::SubSectionType::zerocpy_constant) && is_zerocpy))
        {
            continue;
        }
        if (is_gm && !contains(gm_wt_vec, i))
        {
            continue;
        }
        if (!is_gm && contains(gm_wt_vec, i))
        {
            continue;
        }

        MemInfo mem_info;
        mem_info.address = aligned(addr, bss.data_descriptor[i].align_bytes);
        mem_info.align = bss.data_descriptor[i].align_bytes;
        mem_info.size = bss.data_descriptor[i].size;
        mem_info.slot_idx = i;
        mem_info.mem_type = is_gm ? 2 : 0;
        mem_info.bss_id = bss_id;
        mem_info.asid = is_zerocpy ? 0 : 1;
        mem_info.data_offset = bss.data_descriptor[i].data_offset;
        const_cfg.push_back(mem_info);
        // INFO() << "index: " << i << " addr: 0x" << hex << mem_info.address << " size: 0x" << mem_info.size << " align 0x" << mem_info.align;
        addr = mem_info.address + bss.data_descriptor[i].size;
    }
    return const_cfg;
}

// static void init_regs(ELFIO::section *text_section)
// {
//     auto data = text_section->get_data();
//     // insert init regs bin
//     const unsigned char m_d[] = {0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x20, 0x06, 0xc0, 0x00, 0x80, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x21, 0x06, 0xc0, 0x00, 0x81, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x22, 0x06, 0xc0, 0x00, 0x82, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x23, 0x06, 0xc0, 0x00, 0x83, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x24, 0x06, 0xc0, 0x00, 0x84, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x25, 0x06, 0xc0, 0x00, 0x85, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x26, 0x06, 0xc0, 0x00, 0x86, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x27, 0x06, 0xc0, 0x00, 0x87, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x28, 0x06, 0xc0, 0x00, 0x88, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x29, 0x06, 0xc0, 0x00, 0x89, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2a, 0x06, 0xc0, 0x00, 0x8a, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2b, 0x06, 0xc0, 0x00, 0x8b, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2c, 0x06, 0xc0, 0x00, 0x8c, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2d, 0x06, 0xc0, 0x00, 0x8d, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2e, 0x06, 0xc0, 0x00, 0x8e, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2f, 0x06, 0xc0, 0x00, 0x8f, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x30, 0x06, 0xc0, 0x00, 0x90, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x31, 0x06, 0xc0, 0x00, 0x91, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x32, 0x06, 0xc0, 0x00, 0x92, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x33, 0x06, 0xc0, 0x00, 0x93, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x34, 0x06, 0xc0, 0x00, 0x94, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x35, 0x06, 0xc0, 0x00, 0x95, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x36, 0x06, 0xc0, 0x00, 0x96, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x37, 0x06, 0xc0, 0x00, 0x97, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x38, 0x06, 0xc0, 0x00, 0x98, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x39, 0x06, 0xc0, 0x00, 0x99, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3a, 0x06, 0xc0, 0x00, 0x9a, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3b, 0x06, 0xc0, 0x00, 0x9b, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3c, 0x06, 0xc0, 0x00, 0x9c, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3d, 0x06, 0xc0, 0x00, 0x9d, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3e, 0x06, 0xc0, 0x00, 0x9e, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x9f, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xf8, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xf9, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfa, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfb, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfc, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfd, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfe, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x01, 0xf6, 0xfd, 0x00, 0x80, 0x4f, 0x01, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0xc0};
//     auto tsize = text_section->get_size();
//     char *new_data = new char[tsize + sizeof(m_d)];
//     std::copy(data, data + tsize, new_data);
//     std::copy(m_d, m_d + sizeof(m_d), new_data + tsize);
//     text_section->set_data(new_data, tsize + sizeof(m_d));
//     delete[] new_data;
// }

bool update_by_subsec(char *ro, char *desc, uint32_t ro_size, const std::vector<BssSubSectionDescriptor> &subsec, uint32_t addr)
{
    char *ptr;
    uint32_t new_offset;
    for (auto t : subsec)
    {
        for (auto offset_in_ro : t.offset_in_ro)
        {
            if (offset_in_ro >= ro_size)
            {
                // INFO() << "[DEBUG]desc: ";
                ptr = desc;
                offset_in_ro -= ro_size;
                new_offset = offset_in_ro;
            }
            else
            {
                // INFO() << "[DEBUG]ro: ";
                ptr = ro;
                new_offset = offset_in_ro;
            }

            *(uint32_t *)(ptr + new_offset) = ((*(uint32_t *)(ptr + new_offset)) & (~t.addr_mask)) |
                                              ((addr + t.relative_offset) & (t.addr_mask));
            // INFO() << "[DEBUG]offset:0x" << hex << new_offset << "(0x" << new_offset << ") addr:0x" << addr << " offset:0x" << t.relative_offset;
        }
    }
    return true;
}

uint32_t DumperX::gen_ro_data(char *ro, char *bin, uint32_t size, std::vector<SubGraphDesc> &sgs, uint32_t bss_id)
{
    uint32_t new_offset = 0;
    uint32_t ro_offset, ro_size;
    // INFO() << "gen ro data";
    if (bss_id == 0)
    {
        for (size_t i = 0; i * 4 < 0x100; ++i)
        {
            memcpy(ro + i * 4, bin + i * 4, 4);
            set_ro_offset(i * 4, i * 4, bss_id);
        }
        new_offset = 0x100;
    }

    for (auto &sg : sgs)
    {
        if (sg.fm_desc_offset != bss_id)
        {
            continue;
        }
        ro_offset = sg.rodata_offset;
        ro_size = sg.rodata_size;
        // INFO() << "sg " << sg.id << " ro offset: 0x" << std::hex << sg.rodata_offset << " size: 0x" << sg.rodata_size;
        memcpy(ro + new_offset, bin + ro_offset, ro_size);
        for (size_t i = 0; i * 4 < ro_size; ++i)
        {
            set_ro_offset(ro_offset + i * 4, new_offset + i * 4, bss_id);
        }
        sg.rodata_offset = new_offset;
        new_offset = new_offset + ro_size;
    }

    if (bss_id == 0)
    {
        uint32_t last_base_offset = size - 0x100;
        for (size_t i = 0; i * 4 < 0x100; ++i)
        {
            memcpy(ro + new_offset + i * 4, bin + last_base_offset + i * 4, 4);
            set_ro_offset(last_base_offset + i * 4, new_offset + i * 4, bss_id);
        }
        new_offset = new_offset + 0x100;
    }

    return new_offset;
}

uint32_t DumperX::gen_desc_data(char *desc, char *bin, std::vector<SubGraphDesc> &sgs, uint32_t bss_id)
{
    uint32_t new_offset = 0;
    uint32_t desc_offset, desc_size;
    // INFO() << "gen ro data";
    for (auto &sg : sgs)
    {
        if (sg.fm_desc_offset != bss_id)
        {
            continue;
        }
        desc_offset = sg.descriptor_offset;
        desc_size = sg.descriptor_size;
        // INFO() << "sg " << sg.id << " desc offset: 0x" << std::hex << desc_offset << " size: 0x" << desc_size;
        memcpy(desc + new_offset, bin + desc_offset, desc_size);
        for (size_t i = 0; i * 4 < desc_size; ++i)
        {
            set_desc_offset(desc_offset + i * 4, new_offset + i * 4, bss_id);
        }
        sg.descriptor_offset = new_offset;
        new_offset = new_offset + desc_size;
    }
    return new_offset;
}

static std::vector<BssSubSectionDescriptor> gen_data_subsec(const BssSection &bss, const std::vector<MemInfo> &cfg)
{
    std::vector<BssSubSectionDescriptor> ret;
    for (uint32_t i = 0; i < cfg.size(); ++i)
    {
        auto sub_sections = bss.data_descriptor[cfg[i].slot_idx].sub_sections;
        for (size_t j = 0; j < sub_sections.size(); ++j)
        {
            BssSubSectionDescriptor subsec;
            subsec.offset_in_ro = sub_sections[j].offset_in_ro;
            subsec.relative_offset = cfg[i].address + sub_sections[j].relative_offset;
            subsec.addr_mask = sub_sections[j].addr_mask;
            subsec.size = sub_sections[j].size;
            subsec.type = sub_sections[j].type;
            subsec.id = sub_sections[j].id;
            ret.push_back(subsec);
        }
    }
    return ret;
}

static std::vector<BssSubSectionDescriptor> gen_buffer_subsec(const BssSection &bss, const std::vector<MemInfo> &cfg)
{
    std::vector<BssSubSectionDescriptor> ret;
    for (uint32_t i = 0; i < cfg.size(); ++i)
    {
        auto sub_sections = bss.buffer_descriptor[cfg[i].slot_idx].sub_sections;
        for (size_t j = 0; j < sub_sections.size(); ++j)
        {
            BssSubSectionDescriptor subsec;
            subsec.offset_in_ro = sub_sections[j].offset_in_ro;
            subsec.relative_offset = cfg[i].address + sub_sections[j].relative_offset;
            subsec.addr_mask = sub_sections[j].addr_mask;
            subsec.size = sub_sections[j].size;
            subsec.type = sub_sections[j].type;
            subsec.id = sub_sections[j].id;
            ret.push_back(subsec);
            // INFO() << "set slot: " << cfg[i].slot_idx << " type: " << sub_sections[j].type << " addr: 0x" << std::hex << addr << " offset: 0x" << cfg[i].address;
        }
    }
    return ret;
}

/*update the value in ro/desc/text based on remap table and new address*/
bool DumperX::update_by_remap(const RemapcSection &remap, char *ro_bin, char *desc_bin, char *text_bin, uint32_t ro_addr, uint32_t desc_addr, uint32_t text_addr)
{
    uint32_t loc_offset;
    for (auto entry : remap.table)
    {
        uint32_t dest_addr = entry.dest_type == RemapEntry::RemapType_RO ? ro_addr : entry.dest_type == RemapEntry::RemapType_Descriptor ? desc_addr
                                                                                                                                         : text_addr;

        char *loc_bin = entry.loc_type == RemapEntry::RemapType_RO ? ro_bin : entry.loc_type == RemapEntry::RemapType_Descriptor ? desc_bin
                                                                                                                                 : text_bin;
        if (entry.loc_type == RemapEntry::RemapType_RO)
        {
            loc_bin = ro_bin;
            loc_offset = entry.loc_offset;
        }
        else if (entry.loc_type == RemapEntry::RemapType_Descriptor)
        {
            loc_bin = desc_bin;
            loc_offset = entry.loc_offset;
        }
        else //text
        {
            loc_bin = text_bin;
            loc_offset = entry.loc_offset;
        }
        *(uint32_t *)(loc_bin + loc_offset) = dest_addr + entry.dest_offset;
        // INFO() << "[DEBUG]loc_type:" << entry.loc_type << " dest_type:" << entry.dest_type << " addr[0x" << hex << entry.loc_offset << "] = 0x" << dest_addr + entry.dest_offset;
    }
    return true;
}

static int32_t get_layerid_from_ro(uint32_t ro_offset, const std::vector<ReadOnlyParam> &ro)
{
    int idx = ro_offset / 4;
    auto tensor = ro[idx].tensor;
    if (tensor)
    {

        int32_t id = -1;
        auto node = tensor->node();
        auto sgn = SubGraphNode::from_node(node);
        if (sgn)
        {
            TensorPtr proxied_outp;
            // ProxyTensorPtr outp_p = std::dynamic_pointer_cast<ProxyTensor>(tensor);
            // if (outp_p)
            // {
            //     proxied_outp = outp_p->proxied_tensor();
            // }
            // auto sg = sgn->subgraph();
            for (auto n : sgn->get_real_node())
            {
                for (auto o : n->outputs())
                {
                    ProxyTensorPtr outp_p = std::dynamic_pointer_cast<ProxyTensor>(o);
                    if (outp_p)
                    {
                        proxied_outp = outp_p->proxied_tensor(true);
                    }
                    if (tensor == proxied_outp)
                    {
                        id = n->get_attr("layer_id");
                    }
                }
            }
        }
        else
        {
            id = node->get_attr("layer_id");
        }
        return id;
    }
    return -1;
}

static vector<MemInfo> gen_gm_cfg(vector<MemInfo> &reuse_cfg, const GMConfig &gm_config, GmInfo &gm_info)
{
    vector<MemInfo> cfg;
    uint32_t gm_control = gm_config.GM_control;
    uint32_t gm_en = gm_control & 0xf;
    if (!gm_en)
    {
        return cfg;
    }

    uint gm_idx = gm_config.GM_r0_idx.buf_index;
    for (int32_t i = 0; i < int32_t(reuse_cfg.size()); ++i)
    {
        auto slot = reuse_cfg[i].slot_idx;
        if (slot == gm_idx)
        {
            auto info = reuse_cfg[i];
            auto type = reuse_cfg[i].type;
            reuse_cfg.erase(reuse_cfg.begin() + i);
            i--;
            cfg.push_back(info);
            // INFO() << "move buffer " << info.slot_idx << " to gm";
            break;
        }
    }

    auto sync_control = gm_config.GM_r0_control;
    uint32_t sync_en = sync_control >> 30;
    if (sync_en)
    {
        gm_info.sync_in_offset = sync_control & 0xffff000;
        gm_info.sync_in_size = (sync_control & 0xfff) << 12;
        WARN() << "gm_sync: 0x" << std::hex << gm_info.sync_in_offset << " size: 0x" << gm_info.sync_in_size;
    }

    uint32_t reuse_start_addr = 0;
    for (int32_t i = 0; i < int32_t(reuse_cfg.size()); ++i)
    {
        auto &info = reuse_cfg[i];
        info.address = aligned(reuse_start_addr, info.align);
        reuse_start_addr = info.address + info.size;
    }

    uint32_t gm_start_addr = 0;
    for (int32_t i = 0; i < int32_t(cfg.size()); ++i)
    {
        auto &info = cfg[i];
        info.address = aligned(gm_start_addr, info.align);
        info.mem_type = 1;
        gm_start_addr = info.address + info.size;
    }

    return cfg;
}

static vector<MemInfo> gen_cross_cfg(vector<MemInfo> &reuse_cfg)
{
    vector<MemInfo> mem_cfg;
    bool has_cross_tensor;
    for (int32_t i = 0; i < int32_t(reuse_cfg.size()); ++i)
    {
        auto info = reuse_cfg[i];

        auto type = reuse_cfg[i].type;
        if (type.empty())
        {
            continue;
        }
        has_cross_tensor = false;
        for (const auto &pair : type)
        {
            if (pair.first == 17) //cross_submodel_tensor
            {
                has_cross_tensor = true;
            }
        }

        if (!has_cross_tensor)
        {
            continue;
        }

        reuse_cfg.erase(reuse_cfg.begin() + i);
        i--;
        mem_cfg.push_back(info);
    }

    uint32_t reuse_start_addr = 0;
    for (int32_t i = 0; i < int32_t(reuse_cfg.size()); ++i)
    {
        auto &info = reuse_cfg[i];
        info.address = aligned(reuse_start_addr, info.align);
        reuse_start_addr = info.address + info.size;
    }

    uint32_t cross_start_addr = 0;
    for (int32_t i = 0; i < int32_t(mem_cfg.size()); ++i)
    {
        auto &info = mem_cfg[i];
        info.address = aligned(cross_start_addr, info.align);
        cross_start_addr = info.address + info.size;
    }

    return mem_cfg;
}

static vector<MemInfo> gen_mem_cfg(vector<MemInfo> &cfg, uint32_t gm_size, uint32_t mem_type, bool is_const_data = false)
{
    vector<MemInfo> mem_cfg;
    uint32_t total_addr;
    uint32_t gm_addr = 0;
    bool only_intermediate_data;
    for (int32_t i = 0; i < int32_t(cfg.size()); ++i)
    {
        auto info = cfg[i];
        if (!is_const_data)
        {
            auto type = cfg[i].type;
            if (type.empty())
            {
                continue;
            }
            only_intermediate_data = true;
            for (const auto &pair : type)
            {
                if (pair.first != 3)
                {
                    only_intermediate_data = false;
                }
            }

            if (!only_intermediate_data)
            {
                continue;
            }
        }

        total_addr = aligned(gm_addr, info.align) + info.size;

        if (total_addr <= gm_size)
        {
            gm_addr = total_addr;
            // INFO() << "move buffer " << info.slot_idx << " to mem type: " << mem_type << " total addr: " << total_addr;
        }
        else
        {
            continue;
        }

        cfg.erase(cfg.begin() + i);
        i--;
        info.mem_type = mem_type;
        mem_cfg.push_back(info);
    }

    uint32_t reuse_start_addr = 0;
    for (int32_t i = 0; i < int32_t(cfg.size()); ++i)
    {
        auto &info = cfg[i];
        info.address = aligned(reuse_start_addr, info.align);
        reuse_start_addr = info.address + info.size;
    }

    uint32_t gm_start_addr = 0;
    for (int32_t i = 0; i < int32_t(mem_cfg.size()); ++i)
    {
        auto &info = mem_cfg[i];
        info.address = aligned(gm_start_addr, info.align);
        gm_start_addr = info.address + info.size;
    }

    return mem_cfg;
}

static uint32_t get_ws_offset_opt(const std::vector<SubGraphDesc> &sg_vec, std::map<uint32_t, uint32_t> &ws_offset)
{
    //construct graph
    GraphPtr ws_graph = Graph::create();
    std::vector<std::vector<NodePtr>> ws_levels;
    uint32_t ws_size = 0, ws_align = 0;
    std::map<uint32_t, NodePtr> node_map;
    for (auto sg : sg_vec)
    {
        if (sg.private_buffer_cnt == 0)
        {
            ws_size = 0;
            ws_align = 0;
        }
        else
        {
            ws_size = sg.private_buffer[0].size;
            ws_align = sg.private_buffer[0].align_bytes;
        }

        NodePtr node = Node::create(to_string(sg.id), OpType_NoOp);
        TensorPtr out = std::make_shared<Tensor>(to_string(sg.id) + "out", TensorShape(0), Tensor::Dtype::UINT8);
        TensorPtr ws = std::make_shared<Tensor>(to_string(sg.id) + "ws", TensorShape(ws_size), Tensor::Dtype::UINT8);
        ws->memory_offset()->get_base_object()->align = ws_align;
        node->add_output(out);
        node->add_output(ws);

        for (auto id : sg.precursors)
        {
            auto n = node_map[id];
            node->add_input(n->outputs()[0]);
        }
        ws_graph->add_node(node);
        node_map[sg.id] = node;
    }

    BuilderConfig::set(Constants::load_balance_strategy, 0);
    BuilderConfig::set(Constants::dump_schedule_cfg, false);
    BuilderConfig::set(Constants::load_schedule_cfg, "");
    BuilderConfig::set(Constants::dump_tune_strategy, false);
    BuilderConfig::set(Constants::load_tune_strategy, "");
    LoadBalancer *load_balancer = new LoadBalancer();
    load_balancer->set_graph(ws_graph);
    load_balancer->schedule();
    ws_levels = load_balancer->get_levels();
    delete load_balancer;

    if (ws_levels.size() > 1)
    {
        ERROR();
    }

    //mem allocate
    BuilderConfig::set(Constants::dump_prior_in_gm, false);
    BuilderConfig::set(Constants::load_prior_in_gm, "");
    SOMSScheduler *scheduler = new SOMSScheduler();
    scheduler->set_gm_size({0});
    scheduler->disable_sm();
    scheduler->set_graph(ws_graph);
    scheduler->set_levels(ws_levels);
    auto slots = scheduler->mem_allocate(std::set<TensorPtr>(), std::set<TensorPtr>(), std::set<TensorPtr>(), std::set<TensorPtr>());

    if (slots.size() > 1)
    {
        ERROR();
    }

    //get ws_offset
    for (auto sg : sg_vec)
    {
        if (sg.private_buffer_cnt == 0)
        {
            continue;
        }
        auto n = node_map[sg.id];
        auto ws = n->outputs()[1];
        auto ptr = ws->memory_offset();
        auto offset = ptr->relative_offset;
        ws_offset[sg.id] = offset;
    }

    // for (auto sg : sg_vec)
    // {
    //     if (sg.private_buffer_cnt == 0)
    //     {
    //         ws_size = 0;
    //     }else
    //     {
    //         ws_size = sg.private_buffer[0].size;
    //     }

    //     INFO() << sg.id << " offset: 0x" << std::hex << ws_offset[sg.id] << " size: 0x" << ws_size;
    // }

    uint32_t total_ws_size;
    if (slots.size() == 0)
    {
        total_ws_size = 0;
    }
    else
    {
        total_ws_size = slots[0]->size;
    }

    return total_ws_size;
}

static uint32_t get_ws_offset(const std::vector<SubGraphDesc> &sg_vec, std::map<uint32_t, uint32_t> &ws_offset)
{
    uint32_t offset = 0;
    for (size_t i = 0; i < sg_vec.size(); ++i)
    {
        auto sg = sg_vec[i];
        if (sg.private_buffer_cnt == 0)
        {
            continue;
        }
        auto sg_buf = sg.private_buffer[0]; //for now the only private buffer is workspace
        offset = aligned(offset, sg_buf.align_bytes);
        ws_offset[sg.id] = offset;
        offset += sg_buf.size;
    }
    return offset;
}

static std::vector<uint32_t> get_gm_wt_vec(const string &gm_wt_str)
{
    std::vector<uint32_t> gm_wt_vec;
    if (gm_wt_str != "")
    {
        string line = gm_wt_str;
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            int num = std::stoi(token);
            gm_wt_vec.push_back(num);
        }
    }
    return gm_wt_vec;
}

static std::vector<uint64_t> get_asid(const string &asid_str)
{
    std::vector<uint64_t> asid_vec;

    asid_vec.push_back(0);
    asid_vec.push_back(0);
    asid_vec.push_back(0);
    asid_vec.push_back(0);

    if (asid_str != "")
    {
        string line = asid_str; // + ',';//add , at the end
        std::istringstream iss(line);
        std::string id_str, addr_str;
        int id;
        uint64_t addr;

        while (iss.good())
        {
            std::getline(iss, id_str, '='); // asid
            id_str = utils::trim(id_str);
            auto asid_str = id_str.substr(0, 4);
            if (asid_str != "asid")
            {
                WARN() << "may have typo in option --set_asid for \'" << asid_str << "\'";
            }
            std::getline(iss, addr_str, ','); // addr
            addr = std::stoul(utils::trim(addr_str), 0, 0);
            id = id_str.at(4) - '0';
            if (id >= (int)asid_vec.size())
            {
                WARN() << "doesn't support set asid" << id << " which is larger than 3, this set asid will be ignored";
            }
            else
            {
                asid_vec[id] = addr;
            }
        }
    }

    for (size_t i = 0; i < asid_vec.size(); ++i)
    {
        if ((asid_vec[i] & 0xffffffff00000000) != asid_vec[i])
        {
            WARN() << "asid" << i << " not aligned to 4G";
        }
    }

    return asid_vec;
}

static std::vector<string> get_profile_option(const string &options)
{
    std::vector<string> profile_option;
    if (options != "")
    {
        string line = options;
        std::istringstream ss(line);
        std::string token;
        while (std::getline(ss, token, ','))
        {
            profile_option.push_back(token);
        }
    }
    return profile_option;
}

bool DumperX::set_asid_vec(uint32_t bssid, const std::vector<uint64_t> &asid_vec)
{
    if (m_asid_vec.size() < bssid + 1)
    {
        m_asid_vec.resize(bssid + 1);
    }
    m_asid_vec[bssid] = asid_vec;
    return true;
}

std::vector<uint64_t> DumperX::get_asid_vec(uint32_t bssid, uint32_t mode)
{
    if (mode == 0)
    {
        return m_asid_vec[0];
    }
    return m_asid_vec[bssid];
}

bool DumperX::set_data_info(char *data, uint32_t size, DataType type, int bssid, uint32_t asid, MemType mem = MemType::ddr, std::string name = "")
{
    uint32_t idx = 0;
    if (type_string.find(type) == type_string.end())
    {
        ERROR() << "not found type in type string";
        return false;
    }
    if (name == "")
    {
        if (bssid == -1)
        {
            name = type_string.at(type);
        }
        else
        {
            name = type_string.at(type) + std::to_string(bssid);
        }
        auto it = std::find(m_data_name.begin(), m_data_name.end(), name);
        while (it != m_data_name.end())
        {
            name = type_string.at(type) + std::to_string(bssid) + "_" + std::to_string(idx);
            it = std::find(m_data_name.begin(), m_data_name.end(), name);
            idx++;
        }
    }
    m_data_name.push_back(name);
    m_data_info.push_back(DataInfo{data, size, asid, type, bssid, mem, name, std::vector<MemInfo>()});
    return true;
}

bool DumperX::set_data_info(const std::vector<MemInfo> &cfg, DataType type, int bssid, uint32_t asid, MemType mem = MemType::ddr, std::string name = "")
{
    if (cfg.size() == 0)
    {
        return false;
    }
    if (name == "")
    {
        name = type_string.at(type);
        uint32_t idx = 0;
        if (type == DataType::constant && asid == 0)
        {
            name = "zerocpy_" + name;
        }
        if (mem != MemType::ddr)
        {
            name = mem_string.at(mem) + "_" + name;
        }
        if (asid == m_smmu_asid) //smmu
        {
            name = "smmu_" + name;
        }
        if (bssid != -1)
        {
            name = name + std::to_string(bssid);
        }
        auto it = std::find(m_data_name.begin(), m_data_name.end(), name);
        while (it != m_data_name.end())
        {
            name = type_string.at(type) + std::to_string(bssid) + "_" + std::to_string(idx);
            it = std::find(m_data_name.begin(), m_data_name.end(), name);
            idx++;
        }
    }
    m_data_name.push_back(name);
    uint32_t size = cfg.back().address + cfg.back().size - cfg.front().address;
    m_data_info.push_back(DataInfo{nullptr, size, asid, type, bssid, mem, name, cfg});
    return true;
}

DataInfo DumperX::get_data_info(DataType type, int bssid, MemType mem)
{
    DataInfo data_info;
    for (auto &info : m_data_info)
    {
        if ((info.bssid == -1 || info.bssid == bssid) && info.type == type && info.mem == mem)
        {
            data_info = info;
            // found = true;
            break;
        }
    }
    return data_info;
}

std::vector<DataInfo> DumperX::get_data_info(DataType type, int bssid)
{
    std::vector<DataInfo> data_info_vec;
    for (auto &info : m_data_info)
    {
        if ((info.bssid == -1 || info.bssid == bssid) && info.type == type)
        {
            data_info_vec.push_back(info);
            // found = true;
        }
    }
    return data_info_vec;
}
bool DumperX::update_info_data(std::string name, char *data)
{
    bool found = false;
    for (auto &info : m_data_info)
    {
        if (info.name == name)
        {
            info.data = data;
            found = true;
            break;
        }
    }
    if (!found)
    {
        return false;
    }
    return true;
}
bool DumperX::update_info_size(std::string name, uint32_t size)
{
    bool found = false;
    for (auto &info : m_data_info)
    {
        if (info.name == name)
        {
            info.size = size;
            found = true;
            break;
        }
    }
    if (!found)
    {
        return false;
    }
    return true;
}

bool DumperX::update_info_cfg(std::string name, const std::vector<MemInfo> &cfg)
{
    bool found = false;
    for (auto &info : m_data_info)
    {
        if (info.name == name)
        {
            info.cfg = cfg;
            info.size = cfg.back().address + cfg.back().size;
            found = true;
            break;
        }
    }
    if (!found)
    {
        return false;
    }
    return true;
}

bool DumperX::switch_info_after_gm(DataType type, uint32_t asid)
{
    DataInfo gm_info;
    bool found = false;
    for (size_t i = 0; i < m_data_info.size(); ++i)
    {
        auto info = m_data_info[i];
        if (info.type == type && info.asid == asid && info.mem == MemType::gm)
        {
            gm_info = info;
            m_data_info.erase(m_data_info.begin() + i);
            found = true;
            break;
        }
    }
    if (found)
    {
        m_data_info.push_back(gm_info);
    }
    else
    {
        return false;
    }

    return true;
}

bool DumperX::gen_dumper_files(const std::vector<TensorPtr> &input_tensors, const string &dump_prefix, const string &output_file)
{
    bool ret = true;
    std::string target = m_arg_map[Constants::target_name];

    uint32_t gm_size = cast_any<uint32_t>(m_arg_map[Constants::gm_size]);
    uint32_t ocm_size = cast_any<uint32_t>(m_arg_map[Constants::ocm_size]);
    uint32_t batch_num = cast_any<uint32_t>(m_arg_map[Constants::multi_batch]);
    uint32_t core_num = cast_any<uint32_t>(m_arg_map[Constants::multi_core]);
    uint32_t max_wt_size = cast_any<uint32_t>(m_arg_map[Constants::max_wt_size]);
    uint32_t max_act_size = cast_any<uint32_t>(m_arg_map[Constants::max_act_size]);
    uint32_t smmu_offset = 0xf0000000;
    bool split_wt_act = cast_any<bool>(m_arg_map[Constants::split_wt_act]);

    std::vector<BssSection> bss_vec;
    std::vector<SubGraphDesc> sgs;
    if (batch_num > 1)
    {
        auto bss = m_elf->get_fm_list().fmdesc[0];
        if (m_elf->get_fm_list().fmdesc.size() > 1)
        {
            ERROR() << "cannot support bss size > 1 in multi_batch";
        }
        auto sg = m_elf->get_subgraphs().sgs[0];
        if (m_elf->get_subgraphs().sgs.size() > 1)
        {
            ERROR() << "cannot support sgs size > 1 in multi_batch";
        }
        for (size_t i = 0; i < batch_num; ++i)
        {
            bss_vec.push_back(bss);
            if ((i > 0) && (i % core_num != 0))
            {
                sg.precursor_cnt = 0;
            }
            else
            {
                sg.precursor_cnt = -1;
            }
            sg.fm_desc_offset = i;
            sgs.push_back(sg);
        }
    }
    else
    {
        bss_vec = m_elf->get_fm_list().fmdesc;
        sgs = m_elf->get_subgraphs().sgs;
    }

    if (sgs.size() == 0)
    {
        SubGraphDesc sg;
        sg.id = 0;
        sg.type = 0;
        sg.elf_text_offset = (m_elf)->get_function_offset("_start");
        sg.fm_desc_offset = 0;
        sg.rodata_offset = 0;
        sg.rodata_size = 0;
        sg.descriptor_offset = 0;
        sg.descriptor_size = 0;
        sg.printfifo_size = 0;
        sg.profiler_buf_size = 0;
        sg.private_data_size = 0;
        sg.warmup_len = 0;
        sg.precursor_cnt = 0;
        sg.private_buffer_cnt = 0;
        sgs.push_back(sg);
    }
    //-1: no multi bss; 0: multi_batch; 1: split weight; 2: split featuremap; 3: split weight & featuremap
    int32_t sub_model_mode = bss_vec.size() == 0 ? -1 : batch_num > 1               ? 0
                                                    : max_wt_size && !split_wt_act  ? 1
                                                    : max_act_size && !split_wt_act ? 2
                                                    : split_wt_act                  ? 3
                                                                                    : 4;

    auto text_section = m_elf->get_elf_section(".text");
    uint32_t text_size = text_section->get_size();
    if (sub_model_mode == 2 || sub_model_mode == 3)
    {
        for (size_t i = 0; i < bss_vec.size(); ++i)
        {
            char *text_data_bin = new char[text_size];
            memcpy(text_data_bin, text_section->get_data(), text_size);
            set_data_info(text_data_bin, text_size, DataType::text, i, 0);
            if (cast_any<bool>(m_arg_map[Constants::insert_init_reg]))
            {
                // insert init regs bin
                unsigned char m_d[] = {0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x20, 0x06, 0xc0, 0x00, 0x80, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x21, 0x06, 0xc0, 0x00, 0x81, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x22, 0x06, 0xc0, 0x00, 0x82, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x23, 0x06, 0xc0, 0x00, 0x83, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x24, 0x06, 0xc0, 0x00, 0x84, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x25, 0x06, 0xc0, 0x00, 0x85, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x26, 0x06, 0xc0, 0x00, 0x86, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x27, 0x06, 0xc0, 0x00, 0x87, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x28, 0x06, 0xc0, 0x00, 0x88, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x29, 0x06, 0xc0, 0x00, 0x89, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2a, 0x06, 0xc0, 0x00, 0x8a, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2b, 0x06, 0xc0, 0x00, 0x8b, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2c, 0x06, 0xc0, 0x00, 0x8c, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2d, 0x06, 0xc0, 0x00, 0x8d, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2e, 0x06, 0xc0, 0x00, 0x8e, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2f, 0x06, 0xc0, 0x00, 0x8f, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x30, 0x06, 0xc0, 0x00, 0x90, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x31, 0x06, 0xc0, 0x00, 0x91, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x32, 0x06, 0xc0, 0x00, 0x92, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x33, 0x06, 0xc0, 0x00, 0x93, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x34, 0x06, 0xc0, 0x00, 0x94, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x35, 0x06, 0xc0, 0x00, 0x95, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x36, 0x06, 0xc0, 0x00, 0x96, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x37, 0x06, 0xc0, 0x00, 0x97, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x38, 0x06, 0xc0, 0x00, 0x98, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x39, 0x06, 0xc0, 0x00, 0x99, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3a, 0x06, 0xc0, 0x00, 0x9a, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3b, 0x06, 0xc0, 0x00, 0x9b, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3c, 0x06, 0xc0, 0x00, 0x9c, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3d, 0x06, 0xc0, 0x00, 0x9d, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3e, 0x06, 0xc0, 0x00, 0x9e, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x9f, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xf8, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xf9, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfa, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfb, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfc, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfd, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfe, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x01, 0xf6, 0xfd, 0x00, 0x80, 0x4f, 0x01, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0xc0};
                uint32_t init_text_size = sizeof(m_d);
                char *init_text_data = new char[init_text_size];
                memcpy(init_text_data, m_d, init_text_size);
                set_data_info(init_text_data, init_text_size, DataType::text_init, i, 0);
            }
        }
    }
    else
    {
        char *text_data_bin = new char[text_size];
        memcpy(text_data_bin, text_section->get_data(), text_size);
        set_data_info(text_data_bin, text_size, DataType::text, -1, 0);
        if (cast_any<bool>(m_arg_map[Constants::insert_init_reg]))
        {
            // insert init regs bin
            unsigned char m_d[] = {0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x20, 0x06, 0xc0, 0x00, 0x80, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x21, 0x06, 0xc0, 0x00, 0x81, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x22, 0x06, 0xc0, 0x00, 0x82, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x23, 0x06, 0xc0, 0x00, 0x83, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x24, 0x06, 0xc0, 0x00, 0x84, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x25, 0x06, 0xc0, 0x00, 0x85, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x26, 0x06, 0xc0, 0x00, 0x86, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x27, 0x06, 0xc0, 0x00, 0x87, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x28, 0x06, 0xc0, 0x00, 0x88, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x29, 0x06, 0xc0, 0x00, 0x89, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2a, 0x06, 0xc0, 0x00, 0x8a, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2b, 0x06, 0xc0, 0x00, 0x8b, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2c, 0x06, 0xc0, 0x00, 0x8c, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2d, 0x06, 0xc0, 0x00, 0x8d, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2e, 0x06, 0xc0, 0x00, 0x8e, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x2f, 0x06, 0xc0, 0x00, 0x8f, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x30, 0x06, 0xc0, 0x00, 0x90, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x31, 0x06, 0xc0, 0x00, 0x91, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x32, 0x06, 0xc0, 0x00, 0x92, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x33, 0x06, 0xc0, 0x00, 0x93, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x34, 0x06, 0xc0, 0x00, 0x94, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x35, 0x06, 0xc0, 0x00, 0x95, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x36, 0x06, 0xc0, 0x00, 0x96, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x37, 0x06, 0xc0, 0x00, 0x97, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x38, 0x06, 0xc0, 0x00, 0x98, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x39, 0x06, 0xc0, 0x00, 0x99, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3a, 0x06, 0xc0, 0x00, 0x9a, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3b, 0x06, 0xc0, 0x00, 0x9b, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3c, 0x06, 0xc0, 0x00, 0x9c, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3d, 0x06, 0xc0, 0x00, 0x9d, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x3e, 0x06, 0xc0, 0x00, 0x9e, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0x9f, 0x00, 0xf0, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xf8, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xf9, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfa, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfb, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfc, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfd, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xfe, 0x01, 0xf6, 0xfd, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x01, 0xf6, 0xfd, 0x00, 0x80, 0x4f, 0x01, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0x40, 0xff, 0x7f, 0x0c, 0xc0};
            uint32_t init_text_size = sizeof(m_d);
            char *init_text_data = new char[init_text_size];
            memcpy(init_text_data, m_d, init_text_size);
            set_data_info(init_text_data, init_text_size, DataType::text_init, -1, 0);
        }
    }

    auto rodata = m_elf->get_elf_section(".rodata");
    if (rodata)
    {
        if (sub_model_mode == 2 || sub_model_mode == 3)
        {
            for (size_t i = 0; i < bss_vec.size(); ++i)
            {
                char *rodata_bin = new char[rodata->get_size()];
                memcpy(rodata_bin, rodata->get_data(), rodata->get_size());
                set_data_info(rodata_bin, rodata->get_size(), DataType::rodata, i, 0);
            }
        }
        else
        {
            char *rodata_bin = new char[rodata->get_size()];
            memcpy(rodata_bin, rodata->get_data(), rodata->get_size());
            set_data_info(rodata_bin, rodata->get_size(), DataType::rodata, -1, 0);
        }
    }
    else
    {
        set_data_info(nullptr, 0, DataType::rodata, -1, 0);
    }

    //global_param
    auto global_param = m_elf->get_globalparam();
    std::vector<std::vector<int>> shapes;
    for (auto inp : input_tensors)
    {
        shapes.push_back(inp->shape());
    }
    global_param.update(shapes);
    uint32_t global_param_size = global_param.size();
    char *global_param_data = new char[global_param_size];
    global_param.write(global_param_data);
    uint32_t global_param_asid = sub_model_mode == 2 || sub_model_mode == 3 ? m_smmu_asid : 0;
    set_data_info(global_param_data, global_param_size, DataType::global_param, -1, global_param_asid);

    DEBUG() << "sub_model_mode: " << sub_model_mode;

    auto gm_wt_vec = get_gm_wt_vec(m_arg_map[Constants::specify_wt_gm]);

    GmInfo gm_info{0, 0, 0, 0};
    uint32_t gm_used = 0;
    uint32_t share_reuse_size = 0;
    MemType const_mem = cast_any<bool>(m_arg_map[Constants::put_weight_gm]) ? MemType::gm : MemType::ddr;
    for (size_t i = 0; i < bss_vec.size(); ++i)
    {
        // INFO() << "bss " << i;
        auto bss = bss_vec[i];
        auto reuse_cfg = gen_reuse_cfg(bss, i);
        auto const_cfg = gen_const_cfg(bss, i, gm_wt_vec, false, false);
        auto zerocpy_const_cfg = gen_const_cfg(bss, i, gm_wt_vec, true, false);
        auto gm_const_cfg = gen_const_cfg(bss, i, gm_wt_vec, false, true);
        auto gm_zerocpy_const_cfg = gen_const_cfg(bss, i, gm_wt_vec, true, true);

        std::vector<MemInfo> gm_reuse_cfg;
        if (i == 0)
        {
            auto gm_config = m_elf->get_GMconfig();
            gm_reuse_cfg = gen_gm_cfg(reuse_cfg, gm_config, gm_info);
        }
        else
        {
            if (sub_model_mode == 1)
                gm_reuse_cfg = gen_mem_cfg(reuse_cfg, gm_size, MemType::gm);
        }
        if (gm_reuse_cfg.size())
        {
            // MemType gm_mem = cast_any<bool>(m_arg_map[Constants::disable_gm]) ? MemType::gm : MemType::ddr;
            set_data_info(gm_reuse_cfg, DataType::reuse, i, 0, MemType::gm);
        }

        std::vector<MemInfo> cross_reuse_cfg;
        if (sub_model_mode == 2 || sub_model_mode == 3)
        {
            cross_reuse_cfg = gen_cross_cfg(reuse_cfg);
            if (cross_reuse_cfg.size())
            {
                // MemType gm_mem = cast_any<bool>(m_arg_map[Constants::disable_gm]) ? MemType::gm : MemType::ddr;
                set_data_info(cross_reuse_cfg, DataType::reuse, i, m_smmu_asid, MemType::ddr);
            }
        }

        if (sub_model_mode == 1)
        {
            auto share_reuse_cfg = gen_mem_cfg(reuse_cfg, (uint32_t)3 * 1024 * 1024 * 1024, 0);
            set_data_info(share_reuse_cfg, DataType::reuse, i, 0, MemType::share);
            auto share_reuse_info = get_data_info(DataType::reuse, i, MemType::share);
            if (share_reuse_size < share_reuse_info.size)
            {
                share_reuse_size = share_reuse_info.size;
            }
        }

        if (cast_any<bool>(m_arg_map[Constants::reuse_buffer_expand]))
        {
            for (auto &reuse : reuse_cfg)
            {
                reuse.address = 0;
                auto new_cfg = {reuse};
                std::string name = "buf" + to_string(reuse.slot_idx) + "_" + "reuse" + to_string(i);
                for (auto sub_sec : reuse.type)
                {
                    INFO() << "buf" << reuse.slot_idx << " has type " << sub_sec_type.at(sub_sec.first);
                }
                set_data_info(new_cfg, DataType::reuse, i, 0, MemType::ddr, name);
            }
        }
        else
        {
            set_data_info(reuse_cfg, DataType::reuse, i, 0, MemType::ddr);
        }

        if (sub_model_mode == 0)
        {
            if (i == 0)
            {
                set_data_info(const_cfg, DataType::constant, -1, 1, const_mem);
                set_data_info(zerocpy_const_cfg, DataType::constant, -1, 0);
            }
        }
        else
        {
            set_data_info(const_cfg, DataType::constant, i, 1, const_mem);
            set_data_info(gm_const_cfg, DataType::constant, i, 1, MemType::gm);
            set_data_info(zerocpy_const_cfg, DataType::constant, i, 0);
            set_data_info(gm_zerocpy_const_cfg, DataType::constant, i, 0, MemType::gm);
        }
    }

    auto ro_bin = m_elf->get_rodata();
    auto desc_bin = m_elf->get_descriptor();
    MemType desc_mem = cast_any<bool>(m_arg_map[Constants::put_desc_gm]) ? MemType::gm : MemType::ddr;
    for (size_t i = 0; i < bss_vec.size(); ++i)
    {
        if (sub_model_mode == 0 || sub_model_mode == 2 || sub_model_mode == 3)
        {

            char *ro_data_bin = new char[ro_bin.bin_size];
            memcpy(ro_data_bin, ro_bin.bin, ro_bin.bin_size);
            set_data_info(ro_data_bin, ro_bin.bin_size, DataType::ro, i, 0);

            char *desc_data_bin = new char[desc_bin.bin_size];
            memcpy(desc_data_bin, desc_bin.bin, desc_bin.bin_size);
            set_data_info(desc_data_bin, desc_bin.bin_size, DataType::desc, i, 0, desc_mem);
        }
        else
        {
            if (i == 0)
            {
                char *ro_data_bin = new char[ro_bin.bin_size];
                memcpy(ro_data_bin, ro_bin.bin, ro_bin.bin_size);
                set_data_info(ro_data_bin, ro_bin.bin_size, DataType::ro, -1, 0);

                char *desc_data_bin = new char[desc_bin.bin_size];
                memcpy(desc_data_bin, desc_bin.bin, desc_bin.bin_size);
                set_data_info(desc_data_bin, desc_bin.bin_size, DataType::desc, -1, 0, desc_mem);
            }
        }
    }

    uint32_t ws_size = 0, max_ws_size = 0;
    std::map<uint32_t, uint32_t> ws_offset;
    std::vector<SubGraphDesc> sg_vec;
    for (auto sg : sgs)
    {
        if (sg.precursor_cnt == -1 && sg_vec.size() > 0)
        {
            if (cast_any<bool>(m_arg_map[Constants::ws_mem_opt]))
            {
                ws_size = get_ws_offset_opt(sg_vec, ws_offset);
            }
            else
            {
                ws_size = get_ws_offset(sg_vec, ws_offset);
            }
            max_ws_size = std::max(max_ws_size, ws_size);
            sg_vec.clear();
        }
        sg_vec.push_back(sg);
    }
    if (sg_vec.size() > 0)
    {
        if (cast_any<bool>(m_arg_map[Constants::ws_mem_opt]))
        {
            ws_size = get_ws_offset_opt(sg_vec, ws_offset);
        }
        else
        {
            ws_size = get_ws_offset(sg_vec, ws_offset);
        }
        max_ws_size = std::max(max_ws_size, ws_size);
        sg_vec.clear();
    }

    // INFO() << "max ws size 0x" << std::hex << max_ws_size;

    MemType ws_mem = cast_any<bool>(m_arg_map[Constants::put_ws_gm]) ? MemType::gm : MemType::ddr;
    if (sub_model_mode == 2 || sub_model_mode == 3)
    {
        for (size_t i = 0; i < bss_vec.size(); ++i)
        {
            set_data_info(nullptr, max_ws_size, DataType::workspace, i, 0, ws_mem);
        }
    }
    else
    {
        set_data_info(nullptr, max_ws_size, DataType::workspace, -1, 0, ws_mem);
    }

    if (cast_any<bool>(m_arg_map[Constants::put_desc_gm]) || cast_any<bool>(m_arg_map[Constants::put_weight_gm]) || gm_wt_vec.size())
    {
        switch_info_after_gm(DataType::reuse, 0);
    }

    if (cast_any<bool>(m_arg_map[Constants::put_ws_gm]))
    {
        switch_info_after_gm(DataType::workspace, 0);
    }

    if (sub_model_mode == 2 || sub_model_mode == 3)
    {
        set_data_info(nullptr, 0, DataType::stack, -1, m_smmu_asid);
    }
    else
    {
        set_data_info(nullptr, 0, DataType::stack, -1, 0);
    }

    uint32_t tcb_asid = sub_model_mode == 2 || sub_model_mode == 3 ? m_smmu_asid : 0;
    set_data_info(nullptr, 0, DataType::tcb, -1, tcb_asid);

    MemManager mem_manager;
    if (cast_any<std::string>(m_arg_map[Constants::load_addr_cfg]) != "")
    {
        std::string addr_cfg_file = cast_any<std::string>(m_arg_map[Constants::load_addr_cfg]);
        DumpManager dump_manager(target, DUMPER_VERSION);
        dump_manager.load_addr_cfg(addr_cfg_file);
        for (size_t i = 0; i < bss_vec.size(); ++i)
        {
            set_asid_vec(i, dump_manager.get_asid(i));
        }
        auto mem_vec = dump_manager.get_mem_vec();
        for (auto mem : mem_vec)
        {
            bool found = false;
            DataInfo data_info;
            for (auto &info : m_data_info)
            {
                if (info.name == mem.name)
                {
                    found = true;
                    data_info = info;
                    break;
                }
            }
            if (found)
            {
                uint32_t bss_id = data_info.bssid == -1 ? 0 : data_info.bssid;
                uint64_t asid_addr = get_asid_vec(bss_id, sub_model_mode)[data_info.asid];
                mem_manager.insert_mem(mem.name, mem.addr, mem.size, asid_addr, mem.align, true);
                if (data_info.mem == MemType::gm)
                {
                    gm_info.addr = mem_manager.get_abs_addr(mem.name);
                }
            }
            else if (mem.mem_type == MemType::ocm)
            {
                uint32_t bss_id = data_info.bssid == -1 ? 0 : data_info.bssid;
                uint64_t asid_addr = get_asid_vec(bss_id, sub_model_mode)[data_info.asid];
                mem_manager.insert_mem(mem.name, mem.addr, mem.size, asid_addr, mem.align, true);
                ocm_size = mem.size;
                INFO() << "ocm_size: 0x" << std::hex << ocm_size;
                DataInfo reuse_info = get_data_info(DataType::reuse, bss_id, MemType::ddr);
                auto ocm_reuse_cfg = gen_mem_cfg(reuse_info.cfg, ocm_size, MemType::ocm, 0);
                set_data_info(ocm_reuse_cfg, DataType::reuse, bss_id, 0, MemType::ocm);
                auto info = get_data_info(DataType::reuse, bss_id, MemType::ocm);
                update_info_size(info.name, ocm_size);
                update_info_cfg(reuse_info.name, reuse_info.cfg);
            }
            else
            {
                ERROR() << "cannot find " << mem.name;
            }
        }
    }
    else
    {
        bool gm_assigned = false;
        bool share_assigned = false;
        std::string gm_info_name;
        std::string share_info_name;
        for (size_t i = 0; i < bss_vec.size(); ++i)
        {
            std::vector<uint64_t> asid_vec(4, 0);

            if (bss_vec.size() > 1)
            {
                if (sub_model_mode == 1)
                {
                    asid_vec[1] = (i + 1) * 0x100000000;
                }
                else if (sub_model_mode == 2)
                {
                    asid_vec[0] = i * 0x100000000;
                    asid_vec[1] = bss_vec.size() * 0x100000000;
                    asid_vec[2] = (bss_vec.size() + 1) * 0x100000000;
                }
                else if (sub_model_mode == 3)
                {
                    asid_vec[0] = i * 0x100000000;
                    asid_vec[1] = (bss_vec.size() + i) * 0x100000000;
                    asid_vec[2] = (bss_vec.size() + bss_vec.size()) * 0x100000000;
                }
            }
            else
            {
                std::string asid_str = cast_any<std::string>(m_arg_map[Constants::set_asid]);
                asid_vec = get_asid(asid_str);
            }

            set_asid_vec(i, asid_vec);

            if (cast_any<uint32_t>(m_arg_map[Constants::ocm_addr]) && cast_any<uint32_t>(m_arg_map[Constants::ocm_size]))
            {
                INFO() << "ocm_size: 0x" << std::hex << ocm_size;
                DataInfo reuse_info = get_data_info(DataType::reuse, i, MemType::ddr);
                auto ocm_reuse_cfg = gen_mem_cfg(reuse_info.cfg, ocm_size, MemType::ocm, 0);
                uint32_t ocm_addr = cast_any<uint32_t>(m_arg_map[Constants::ocm_addr]);
                if (set_data_info(ocm_reuse_cfg, DataType::reuse, i, 0, MemType::ocm))
                {
                    auto info = get_data_info(DataType::reuse, i, MemType::ocm);
                    update_info_size(info.name, ocm_size);
                    mem_manager.insert_mem(info.name, ocm_addr, info.size, info.asid, 0x0, true);
                    update_info_cfg(reuse_info.name, reuse_info.cfg);
                }
                else
                {
                    mem_manager.insert_mem("ocm", ocm_addr, ocm_size, asid_vec[0], 0x0, true);
                }
            }
        }

        if (sub_model_mode == 2 || sub_model_mode == 3)
        {
            uint64_t smmu_asid_addr = get_asid_vec(0, sub_model_mode)[m_smmu_asid];
            // uint64_t smmu_asid_addr = 0x200000000;
            mem_manager.set_smmu(smmu_asid_addr, smmu_offset);
        }

        for (auto &info : m_data_info)
        {
            uint32_t bss_id = info.bssid == -1 ? 0 : info.bssid;
            auto asid_addr = get_asid_vec(bss_id, sub_model_mode)[info.asid];
            bool fixed = (info.type == DataType::stack || info.type == DataType::tcb) ? false : true;
            uint32_t align = (info.type == DataType::tcb && info.asid == 3) ? 0x4000 : 0x1000;

            if (info.mem == MemType::gm)
            {
                // INFO() << "assigning gm for " << info.name << " 0x" << std::hex << info.size;
                if (!gm_assigned)
                {
                    mem_manager.assign_mem(info.name, gm_size, asid_addr, align, fixed);
                    gm_info.addr = mem_manager.get_abs_addr(info.name);
                    gm_info.size = gm_size;
                    gm_info_name = info.name;
                    gm_assigned = true;
                    gm_used += info.size;
                }
                else
                {

                    if (info.bssid > 0) //multi bss will not put desc/constant/ws into gm
                    {
                        mem_manager.share_mem(info.name, gm_info_name);

                        if (gm_used < info.size)
                        {
                            gm_used = info.size;
                        }
                    }
                    else
                    {
                        uint32_t info_addr = aligned(gm_info.addr + gm_used, align);
                        if (info_addr + info.size > gm_info.addr + gm_info.size)
                        {
                            ERROR() << "no GM space for " << info.name << ". gm left size: 0x" << std::hex << gm_size - gm_used << " required size: 0x" << info.size;
                        }
                        uint32_t rest_size = gm_info.size - gm_used;
                        mem_manager.insert_mem(info.name, info_addr, rest_size, asid_addr, align, fixed);
                        gm_used += info.size;
                    }
                }

                if (info.type == DataType::desc || info.type == DataType::constant)
                {
                    gm_info.sync_in_size += info.size;
                }
            }
            else if (info.mem == MemType::share)
            {
                if (share_assigned)
                {
                    mem_manager.share_mem(info.name, share_info_name);
                }
                else
                {
                    mem_manager.assign_mem(info.name, share_reuse_size, asid_addr, align, fixed);
                    share_info_name = info.name;
                    share_assigned = true;
                }
            }
            else
            {
                // INFO() << "assigning ddr for " << info.name << " 0x" << std::hex << info.size;
                mem_manager.assign_mem(info.name, info.size, asid_addr, align, fixed);
            }
        }
    }
    DEBUG() << "[GM] actual used size: 0x" << std::hex << gm_used << "(" << utils::get_human_readable_size(gm_used) << ")";
    // for (auto &info : m_data_info)
    // {
    //     if (info.type == DataType::reuse)
    //     {
    //         INFO() << info.name;
    //         for (auto obj : info.cfg)
    //         {
    //             INFO() << "offset: 0x" << std::hex << obj.address << " size: 0x" << obj.size << " slot: " << obj.slot_idx;
    //             for (auto t : obj.type)
    //             {
    //                 INFO() << " type: " << t.first;
    //             }
    //         }
    //     }
    // }
    if (cast_any<bool>(m_arg_map[Constants::disable_gm]))
    {
        gm_info.size = 0;
        gm_info.sync_in_size = 0;
    }

    if (gm_info.size)
    {
        INFO() << "[GM] addr: 0x" << hex << gm_info.addr << " size: 0x" << gm_info.size;
        INFO() << "[GM] sync offset: 0x" << hex << gm_info.sync_in_offset << " size: 0x" << gm_info.sync_in_size;
    }

    if (m_mode == Run_Mode::run_by_builder)
    {
        auto update_addr_vec = [](std::vector<uint32_t> &vec, const std::vector<MemInfo> &cfg, uint32_t addr) -> bool {
            for (size_t i = 0; i < cfg.size(); ++i)
            {
                auto obj = cfg[i];
                if (vec.size() < obj.slot_idx + 1)
                {
                    vec.resize(obj.slot_idx + 1);
                }
                vec[obj.slot_idx] = obj.address + addr;
            }
            return true;
        };

        m_graph->set_attr("gm_addr0", gm_info.addr);
        uint32_t gm_size = gm_info.size;
        if (gm_size)
        {
            m_graph->set_attr("gm_size0", gm_size);
        }

        std::vector<uint64_t> gm_addr_vec(bss_vec.size(), gm_info.addr);
        std::vector<uint32_t> gm_size_vec(bss_vec.size(), gm_size);
        m_graph->set_attr("gm_addr_vec", gm_addr_vec);
        m_graph->set_attr("gm_size_vec", gm_size_vec);

        std::vector<std::vector<uint32_t>> addr_vec_list(bss_vec.size());
        std::vector<std::vector<uint32_t>> w_vec_list(bss_vec.size());

        for (size_t i = 0; i < bss_vec.size(); ++i)
        {
            auto reuse_info_vec = get_data_info(DataType::reuse, i);
            std::vector<uint32_t> addr_vec;
            for (auto info : reuse_info_vec)
            {
                update_addr_vec(addr_vec, info.cfg, mem_manager.get_addr(info.name));
            }
            addr_vec_list[i] = addr_vec;

            std::vector<uint32_t> w_vec;
            auto const_info_vec = get_data_info(DataType::constant, i);
            for (auto info : const_info_vec)
            {
                update_addr_vec(w_vec, info.cfg, mem_manager.get_addr(info.name));
            }
            w_vec_list[i] = w_vec;
        }
        m_graph->set_attr("addr_vec_list", addr_vec_list);
        m_graph->set_attr("w_vec_list", w_vec_list);
    }

    //data update
    auto first_const_bin = m_elf->get_weight();
    auto extra_const_bin = m_elf->get_extra_weight();
    std::vector<BinSection> const_bin_vec;
    const_bin_vec.push_back(first_const_bin);
    const_bin_vec.insert(const_bin_vec.end(), extra_const_bin.begin(), extra_const_bin.end());

    auto remap = m_elf->get_remap_table();
    std::vector<std::pair<BssSubSectionDescriptor, DataInfo>> reuse_subsec_vec;

    for (auto &info : m_data_info)
    {
        if (info.type == DataType::constant)
        {
            char *const_data = new char[info.size];
            memset(const_data, 0x0, info.size);
            uint32_t bss_id = info.bssid == -1 ? 0 : info.bssid;
            auto const_bin = const_bin_vec[bss_id];
            auto const_cfg = info.cfg;
            for (size_t j = 0; j < const_cfg.size(); ++j)
            {
                if (const_cfg[j].data_offset + const_cfg[j].size > const_bin.bin_size)
                {
                    ERROR() << "out of weight buffer bound 0x" << std::hex << const_bin.bin_size;
                }
                memcpy(const_data + const_cfg[j].address - const_cfg[0].address, const_bin.bin + const_cfg[j].data_offset, const_cfg[j].size);
                // INFO() << "copy from 0x" << std::hex << const_cfg[j].data_offset << " to 0x" << const_cfg[j].address - const_cfg[0].address << " size:0x" << const_cfg[j].size;
            }
            update_info_data(info.name, const_data);
        }
    }

    for (size_t i = 0; i < bss_vec.size(); ++i)
    {
        auto ro_info = get_data_info(DataType::ro, i, MemType::ddr);
        auto desc_info = get_data_info(DataType::desc, i, desc_mem);
        auto text_info = get_data_info(DataType::text, i, MemType::ddr);
        for (auto &info : m_data_info)
        {
            if (info.bssid != (int)i && info.bssid != -1 && info.asid != m_smmu_asid)
            {
                continue;
            }
            if (info.type != DataType::constant && info.type != DataType::reuse)
            {
                continue;
            }
            int32_t bss_idx = sub_model_mode == 0 ? i : info.bssid;
            if (bss_idx == -1)
            {
                ERROR();
            }
            std::vector<BssSubSectionDescriptor> subsec;
            auto bss = bss_vec[bss_idx];
            if (info.type == DataType::constant)
            {
                subsec = gen_data_subsec(bss, info.cfg);
            }
            else if (info.type == DataType::reuse)
            {
                subsec = gen_buffer_subsec(bss, info.cfg);
                for (auto &ss : subsec)
                {
                    reuse_subsec_vec.push_back(std::make_pair(ss, info));
                }
            }
            update_by_subsec(ro_info.data, desc_info.data, ro_info.size, subsec, mem_manager.get_addr(info.name));
        }
        if (sub_model_mode == 0 || sub_model_mode == 2 || sub_model_mode == 3)
        {
            update_by_remap(remap, ro_info.data, desc_info.data, text_info.data, mem_manager.get_addr(ro_info.name), mem_manager.get_addr(desc_info.name), mem_manager.get_addr(text_info.name));
        }
        else
        {
            if (i == 0) //all shared
            {
                update_by_remap(remap, ro_info.data, desc_info.data, text_info.data, mem_manager.get_addr(ro_info.name), mem_manager.get_addr(desc_info.name), mem_manager.get_addr(text_info.name));
            }
        }
    }

    //generate TCB

    uint32_t tec_num = cast_any<uint32_t>(m_arg_map[Constants::multi_tec]);

    TcbManager tcb_manager(core_num, tec_num, target);
    tcb_manager.set_text_size(text_section->get_size());

    uint32_t prof_addr = 0;
    uint32_t prof_size = 0;
    for (const auto &pair : reuse_subsec_vec)
    {
        if (pair.first.type == (uint32_t)BssSubSectionDescriptor::SubSectionType::profile_data)
        {
            prof_addr = mem_manager.get_addr(pair.second.name) + pair.first.relative_offset;
            prof_size = sgs.size() == 0 ? 0 : sgs.size() == 1 ? pair.first.size
                                                              : sgs[sgs.size() - 1].profiler_buf_size - sgs[sgs.size() - 2].profiler_buf_size;
        }
    }

    uint32_t bss_id = 0;
    uint32_t smmu_remap_addr = 0; //mem_manager.get_addr("tcb");
    uint32_t smmu_remap_size = sub_model_mode == 2 || sub_model_mode == 3 ? 128 * 1024 * 1024 : 0;
    uint32_t smmu_remap_asid = 2;

    if (smmu_remap_size)
    {
        INFO() << "[SMMU] addr: 0x" << std::hex << smmu_remap_addr + get_asid_vec(0, sub_model_mode)[smmu_remap_asid] << " size: 0x" << smmu_remap_size;
    }

    tcb_manager.set_stack_size(bss_vec[0].stack_size);
    if (!cast_any<bool>(m_arg_map[Constants::set_l2_flush]))
    {
        tcb_manager.set_l2_flush();
    }
    bool multi_grid = false;
    if ((target.find("X2") != std::string::npos) && (batch_num <= 1))
    {
        multi_grid = true;
    }
    if (multi_grid)
    {
        for (size_t i = 0; i < bss_vec.size(); ++i)
        {
            tcb_manager.set_asid(get_asid_vec(i, sub_model_mode));
            tcb_manager.set_smmu(smmu_remap_addr, smmu_remap_size, smmu_remap_asid);
            int group_size = cast_any<bool>(m_arg_map[Constants::insert_init_reg]) ? sgs.size() + 1 : sgs.size();
            tcb_manager.add_grid_init_tcb(gm_info.addr, gm_info.size, gm_info.sync_in_size, group_size);

            auto text_info = get_data_info(DataType::text, i, MemType::ddr);
            auto rodata_info = get_data_info(DataType::rodata, i, MemType::ddr);
            auto globalparam_info = get_data_info(DataType::global_param, i, MemType::ddr);
            auto stack_info = get_data_info(DataType::stack, i, MemType::ddr);
            auto ro_info = get_data_info(DataType::ro, i, MemType::ddr);

            if (i == 0 && cast_any<bool>(m_arg_map[Constants::insert_init_reg]))
            {
                auto text_init_info = get_data_info(DataType::text_init, i, MemType::ddr);
                for (size_t j = 0; j < core_num; ++j)
                {
                    tcb_manager.add_task_tcb(0, true, std::vector<uint32_t>(), mem_manager.get_addr(text_init_info.name),
                                             0, 0, 0,
                                             0, 0);
                }
            }

            for (size_t j = 0; j < sgs.size(); ++j)
            {
                auto sg = sgs[j];
                bss_id = sg.fm_desc_offset;
                if (bss_id != i)
                {
                    continue;
                }

                bool depd_all = (j == 0) ? 0 : sg.precursor_cnt == -1;
                // INFO() << "DEBUG " << i << " depd all " << depd_all;
                auto depd_vec = sg.precursors;
                tcb_manager.set_data_size(sg.private_data_size);
                tcb_manager.set_print_size(sg.printfifo_size);
                tcb_manager.set_asid(get_asid_vec(bss_id, sub_model_mode));
                tcb_manager.add_group_init_tcb(sg.id, depd_all, depd_vec);
                tcb_manager.add_task_tcb(sg.id, depd_all, depd_vec, mem_manager.get_addr(text_info.name) + sg.elf_text_offset,
                                         mem_manager.get_addr(rodata_info.name), mem_manager.get_addr(globalparam_info.name), prof_addr + prof_size * j,
                                         mem_manager.get_addr(ro_info.name) + sg.rodata_offset, sg.warmup_len);
            }
        }
    }
    else
    {
        tcb_manager.set_asid(get_asid_vec(bss_id, sub_model_mode));
        tcb_manager.set_smmu(smmu_remap_addr, smmu_remap_size, smmu_remap_asid);
        int group_size = cast_any<bool>(m_arg_map[Constants::insert_init_reg]) ? sgs.size() + 1 : sgs.size();
        tcb_manager.add_grid_init_tcb(gm_info.addr, gm_info.size, gm_info.sync_in_size, group_size);

        if (cast_any<bool>(m_arg_map[Constants::insert_init_reg]))
        {
            tcb_manager.set_data_size(0);
            tcb_manager.set_print_size(0);
            tcb_manager.set_asid(get_asid_vec(0, 0));
            tcb_manager.add_group_init_tcb(0, true, std::vector<uint32_t>());
            auto text_init_info = get_data_info(DataType::text_init, 0, MemType::ddr);
            for (size_t j = 0; j < core_num; ++j)
            {
                tcb_manager.add_task_tcb(0, true, std::vector<uint32_t>(), mem_manager.get_addr(text_init_info.name),
                                         0, 0, 0,
                                         0, 0);
            }
        }

        for (size_t i = 0; i < sgs.size(); ++i)
        {
            auto sg = sgs[i];
            bss_id = sg.fm_desc_offset;
            auto text_info = get_data_info(DataType::text, i, MemType::ddr);
            auto rodata_info = get_data_info(DataType::rodata, i, MemType::ddr);
            auto globalparam_info = get_data_info(DataType::global_param, i, MemType::ddr);
            auto ro_info = get_data_info(DataType::ro, i, MemType::ddr);

            // INFO() << "sg " << i << " is in bss " << bss_id;
            bool depd_all = (i == 0) ? 0 : sg.precursor_cnt == -1;
            int sg_id = sg.id;
            if (cast_any<bool>(m_arg_map[Constants::insert_init_reg]))
            {
                if (i == 0)
                {
                    depd_all = true;
                }
                sg_id = sg.id + 1;
            }
            // depd_all = true;
            auto depd_vec = sg.precursors;
            // INFO() << "[DEBUG] " << sg.id;
            // if(depd_all)
            // {
            //     INFO() << "[DEBUG] " << "depd all";
            // }else
            // {
            //     for(auto dped : depd_vec)
            //     {
            //         INFO() << "[DEBUG] " << "depends on " << dped;
            //     }
            // }
            tcb_manager.set_data_size(sg.private_data_size);
            tcb_manager.set_print_size(sg.printfifo_size);
            tcb_manager.set_asid(get_asid_vec(bss_id, sub_model_mode));
            tcb_manager.add_group_init_tcb(sg.id, depd_all, depd_vec);
            tcb_manager.add_task_tcb(sg_id, depd_all, depd_vec, mem_manager.get_addr(text_info.name) + sg.elf_text_offset,
                                     mem_manager.get_addr(rodata_info.name), mem_manager.get_addr(globalparam_info.name), prof_addr + prof_size * i,
                                     mem_manager.get_addr(ro_info.name) + sg.rodata_offset, sg.warmup_len);
        }
    }

    auto stack_info = get_data_info(DataType::stack, 0, MemType::ddr);
    uint32_t private_buf_size = tcb_manager.get_private_buf_size();
    mem_manager.fix_mem(stack_info.name, private_buf_size);

    uint32_t tcb_size = tcb_manager.get_tcb_size();
    char *tcb = new char[tcb_size];
    auto tcb_info = get_data_info(DataType::tcb, 0, MemType::ddr);
    mem_manager.fix_mem(tcb_info.name, tcb_size);

    tcb_manager.gen_tcb(tcb, mem_manager.get_addr(tcb_info.name), mem_manager.get_addr(stack_info.name));

    update_info_data(tcb_info.name, tcb);
    update_info_size(tcb_info.name, tcb_size);

    for (auto sg : sgs)
    {
        bss_id = sg.fm_desc_offset;

        if (sg.private_buffer_cnt == 0)
        {
            continue;
        }

        auto sg_buf = sg.private_buffer[0]; //for now the only private buffer is workspace

        // update_ws(sgs, ro_bin, desc_bin, ws_buf_offset, mem_manager.get_addr("workspace"));
        auto sub_sections = sg.private_buffer[0].sub_sections;

        // set_subsection_addr(ws_addr[i] + addr, sub_sections, ro, desc);
        std::vector<BssSubSectionDescriptor> private_subsec;
        for (size_t i = 0; i < sub_sections.size(); ++i)
        {
            BssSubSectionDescriptor sec;
            sec.offset_in_ro = sub_sections[i].offset_in_ro;
            sec.relative_offset = ws_offset[sg.id] + sub_sections[i].relative_offset;
            sec.addr_mask = sub_sections[i].addr_mask;
            private_subsec.push_back(sec);
            // INFO() << "set sub_sec: " << i << " type: " << sub_sections[i].type;
        }
        auto ws_info = get_data_info(DataType::workspace, bss_id, ws_mem);
        auto ro_info = get_data_info(DataType::ro, bss_id, MemType::ddr);
        auto desc_info = get_data_info(DataType::desc, bss_id, desc_mem);
        update_by_subsec(ro_info.data, desc_info.data, ro_info.size, private_subsec, mem_manager.get_addr(ws_info.name));
    }

    // if (need_sync)
    // {
    //     WARN() << "data in GM, need sync out";
    // }

    mem_manager.show();

    //dump_addr_cfg
    // dump_addr_cfg_file(mem_manager);
    if (cast_any<bool>(m_arg_map[Constants::dump_addr_cfg]))
    {
        DumpManager dump_manager(target, DUMPER_VERSION);
        for (size_t i = 0; i < bss_vec.size(); ++i)
        {
            dump_manager.add_asid(get_asid_vec(i, sub_model_mode), i);
        }
        for (auto &info : m_data_info)
        {
            if (info.type == DataType::reuse || info.type == DataType::constant)
            {
                dump_manager.add_cfg(info.name, mem_manager.get_addr(info.name), info.cfg, info.mem);
            }
            else
            {
                dump_manager.add_buf(info.name, mem_manager.get_addr(info.name), mem_manager.get_size(info.name), mem_manager.get_align(info.name));
            }
        }
        dump_manager.dump_addr_cfg("addr_cfg_default.ini");
    }

    //dump temp files
    ConfigManager config_manager(target, core_num);
    gm_size = gm_size / 1024 / 1024;
    config_manager.set_gm_size(gm_size);
    config_manager.set_log_level(0);
    uint64_t max_asid_addr = mem_manager.get_max_addr() / (1024 * 1024 * 1024);
    config_manager.set_ddr_size(max_asid_addr + 4);
    config_manager.set_calloc(!cast_any<bool>(m_arg_map["random-init"]));
    config_manager.set_fp_mode(cast_any<bool>(m_arg_map[Constants::native_float_simulator]));
    config_manager.set_log_verbose(true);
    config_manager.set_profile(prof_size > 0);

    for (auto &info : m_data_info)
    {
        if (info.size > 0 && info.type != DataType::reuse && info.type != DataType::stack && info.type != DataType::workspace)
        {
            config_manager.add_input(dump_prefix + "." + info.name, mem_manager.get_abs_addr(info.name), info.data, info.size);
        }
    }

    config_manager.set_tcbp(mem_manager.get_abs_addr(tcb_info.name));
    uint32_t tcb_num = tcb_manager.get_tcb_num();
    config_manager.set_tcb_num(tcb_num);

    for (const auto &pair : reuse_subsec_vec)
    {
        auto subsec = pair.first;
        auto info = pair.second;
        if (subsec.type == (uint32_t)BssSubSectionDescriptor::SubSectionType::input)
        {
            if (subsec.id >= input_tensors.size())
            {
                ERROR() << "this model requires " << subsec.id + 1 << " input but we only have " << input_tensors.size() << " input";
                continue;
            }
            auto input_t = input_tensors[subsec.id];
            std::string input_name;
            char *input = input_t->data<char>();
            uint32_t input_size = cast_any<bool>(m_arg_map[Constants::enable_ds]) ? input_t->mem_size() : subsec.size;
            if (batch_num > 1)
            {
                input_name = dump_prefix + ".b" + to_string(info.bssid) + ".input" + std::to_string(subsec.id);
                input = (char *)(input_tensors[subsec.id]->data<char>() + info.bssid * subsec.size);
            }
            else
            {
                if ((subsec.size != input_t->mem_size()) && !cast_any<bool>(m_arg_map[Constants::enable_ds]))
                {
                    WARN() << "input size 0x" << std::hex << input_t->mem_size() << " is not equal to model input size 0x" << subsec.size;
                }
                input_name = dump_prefix + ".input" + std::to_string(subsec.id);
            }
            config_manager.add_input(input_name, mem_manager.get_abs_addr(info.name) + subsec.relative_offset, input, input_size);
        }
        else if (subsec.type == (uint32_t)BssSubSectionDescriptor::SubSectionType::output)
        {
            std::string name = (subsec.id == 0) ? output_file : output_file + std::to_string(subsec.id);
            std::string output_name;
            if (batch_num > 1)
            {
                output_name = "b" + to_string(info.bssid) + "." + name;
            }
            else
            {
                output_name = name;
            }
            config_manager.add_output(output_name, mem_manager.get_abs_addr(info.name) + subsec.relative_offset, subsec.size);
        }
        else if (prof_size > 0 && subsec.type == (uint32_t)BssSubSectionDescriptor::SubSectionType::profile_data)
        {
            std::string profile_name;
            if (batch_num > 1)
            {
                profile_name = "b" + to_string(info.bssid) + ".profile_data.bin";
            }
            else
            {
                profile_name = "profile_data.bin";
            }
            config_manager.add_output(profile_name, mem_manager.get_abs_addr(info.name) + subsec.relative_offset, subsec.size);
        }
        else if (cast_any<bool>(m_arg_map[Constants::enable_ds]) && subsec.type == (uint32_t)BssSubSectionDescriptor::SubSectionType::dump_tensor_shape)
        {
            config_manager.add_output("output_shape.bin", mem_manager.get_abs_addr(info.name) + subsec.relative_offset, subsec.size);
        }

        if (cast_any<bool>(m_arg_map[Constants::dump_all_tensors]) && (subsec.type == (uint32_t)BssSubSectionDescriptor::SubSectionType::dump || subsec.type == (uint32_t)BssSubSectionDescriptor::SubSectionType::output))
        {
            std::string name = "";
            uint32_t layerid = 0;
            if (m_ro.size())
            {
                name = get_name_from_ro(subsec.offset_in_ro[0], m_ro);
                layerid = get_layerid_from_ro(subsec.offset_in_ro[0], m_ro);
            }
            else
            {
                WARN() << "cannot dump tensors without ro";
            }
            auto dump_name = "dumptensor_" + to_string(layerid) + "_" + utils::verify_name(name) + ".bin";
            config_manager.add_output(dump_name, mem_manager.get_abs_addr(info.name) + subsec.relative_offset, subsec.size);
        }
    }

    std::string graph_json_name;
    std::vector<string> profile_option;
    if (prof_size > 0)
    {
        graph_json_name = config_manager.dump_graph_json(dump_prefix, m_elf->get_graph_json());
        profile_option = get_profile_option(cast_any<string>(m_arg_map[Constants::set_profile_option]));
    }

    config_manager.gen_config_files(dump_prefix + ".cfg", graph_json_name, profile_option);

    if (sgs.size() == 0)
    {
        config_manager.dump_relative_output();
        WARN() << "no need dumper or simulator";
        return false;
    }

    // for(auto &info : m_data_info)
    // {
    //     //delete
    // }

    return ret;
}
}; // namespace aipubt
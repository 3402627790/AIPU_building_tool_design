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
 * @Date: 2024-01-25 14:26:32
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-01-25 14:26:32
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "utils/utils.h"

namespace aipubt
{

std::string get_mem_info(TensorPtr out, GraphPtr top_g, bool show_full_mem_info)
{
    std::stringstream oss;
    NodePtr n = out->node();

    std::vector<std::vector<uint32_t>> addr_vec_list, w_vec_list;
    if (top_g)
    {
        addr_vec_list = cast_any<std::vector<std::vector<uint32_t>>>(top_g->get_attr("addr_vec_list"));
        w_vec_list = cast_any<std::vector<std::vector<uint32_t>>>(top_g->get_attr("w_vec_list"));
    }
    else
    {
        DEBUG() << "origin graph doesn't exist";
    }

    auto bss_id = n->get_attr<size_t>(Constants::bss_id);
    auto mmu = out->memory_offset()->get_mmu_object();
    std::ostringstream buf_idx;
    if (show_full_mem_info)
    {
        if (out->memory_offset()->get_base_object()->buffer_id >= 0)
        {
            buf_idx << ", buf_idx: " << out->memory_offset()->get_base_object()->buffer_id;
        }
        else if (mmu && mmu->next_block->get_base_object()->buffer_id >= 0)
        {
            buf_idx << ", buf_idx: " << mmu->next_block->get_base_object()->buffer_id;
        }
    }
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
    auto get_sram_info = [addr_to_sram_id, n, bss_id, top_g](uint32_t addr, uint32_t size, bool enable) -> std::string {
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
        if (!enable)
        {
            return "";
        }
        std::string addr_str = "(ddr)";
        auto gsram_id = addr_to_sram_id(addr, gsram_bases, 0xFF800000);
        auto lsram_id = addr_to_sram_id(addr, lsram_bases, 0xFFF80000);
        if (gsram_id >= 0)
        {
            addr_str = ("(gsram_" + std::to_string(gsram_id) + ")");
        }
        else if (lsram_id >= 0)
        {
            addr_str = ("(lsram_" + std::to_string(lsram_id) + ")");
        }
        else if (BuilderConfig::get<bool>(Constants::enable_tcm))
        {
            uint32_t dtcm_base = 0xD0000000;
            uint32_t dtcm_size = BuilderConfig::get<uint32_t>(Constants::tcm_size);
            if (addr >= dtcm_base && addr < (dtcm_base + dtcm_size))
            {
                addr_str = "(dtcm)";
            }
        }
        else if (aipubt::get_target()->isainfo().type >= aipubt::ISA_ZYv5)
        {
            auto upper_graph = n->graph();
            auto sg = std::dynamic_pointer_cast<SubGraph>(upper_graph);
            uint32_t gm_addr0 = 0;
            uint32_t gm_size0 = 0;
            uint32_t ocm_addr = BuilderConfig::get<uint64_t>("ocm_addr");
            uint32_t ocm_size = BuilderConfig::get<uint64_t>("ocm_size");

            std::vector<uint64_t> gm_addr_vec = top_g->get_attr("gm_addr_vec");
            if (bss_id < gm_addr_vec.size() && bss_id >= 0)
            {
                gm_addr0 = gm_addr_vec[bss_id];
            }
            std::vector<uint32_t> gm_size_vec = top_g->get_attr("gm_size_vec");
            if (bss_id < gm_size_vec.size() && bss_id >= 0)
            {
                gm_size0 = gm_size_vec[bss_id];
            }

            if (addr >= ocm_addr && addr < ocm_addr + ocm_size)
            {
                addr_str = "(ocm)";
            }
            else if (!BuilderConfig::get<bool>(Constants::disable_gm))
            {
                if (gm_size0 && (addr >= gm_addr0) && (addr < gm_addr0 + gm_size0))
                {
                    addr_str = "(gm0)";
                    if (addr + size > gm_addr0 + gm_size0)
                    {
                        std::stringstream ss;
                        ss << addr_str << ", 0x" << std::hex << gm_addr0 + gm_size0 << "(ddr)";
                        addr_str = ss.str();
                    }
                }
            }
        }
        if (addr_str == "(ddr)" && n->attrs().count("with_IC_child"))
        {
            addr_str = "(ddr_)";
        }
        return addr_str;
    };

    uint32_t real_addr;
    if (out->memory_offset()->get_base_object()->get_buf_type() == MemoryObject::type::constant)
    {
        if (bss_id < w_vec_list.size() && bss_id >= 0)
        {
            auto w_vec = w_vec_list[bss_id];
            if (out->memory_offset()->get_base_object()->buffer_id < (int)w_vec.size() && out->memory_offset()->get_base_object()->buffer_id >= 0)
            {
                //tensor offset contains stack size, so we cannot use it, we just assume no constant shares the same data
                real_addr = w_vec[out->memory_offset()->get_base_object()->buffer_id];
            }
            else
            {
                DEBUG() << "no such constant tensor: " << n->name() << "!";
                real_addr = out->memory_offset()->offset();
            }
        }
        else
        {
            DEBUG() << "no such constant tensor: " << n->name() << "!";
            real_addr = out->memory_offset()->offset();
        }
    }
    else
    {

        if (bss_id < addr_vec_list.size() && bss_id >= 0)
        {
            auto addr_vec = addr_vec_list[bss_id];
            if (out->memory_offset()->get_base_object()->buffer_id < (int)addr_vec.size() && out->memory_offset()->get_base_object()->buffer_id >= 0)
            {
                real_addr = addr_vec[out->memory_offset()->get_base_object()->buffer_id] + out->memory_offset()->offset();
            }
            else
            {
                real_addr = out->memory_offset()->offset();
            }
        }
        else
        {
            real_addr = out->memory_offset()->offset();
        }
    }
    oss << "address: 0x" << std::hex << std::setw(8) << std::setfill('0') << real_addr
        << get_sram_info(real_addr, out->mem_size(), show_full_mem_info);
    if (mmu)
    {
        oss << ", 0x" << std::hex << std::setw(8) << std::setfill('0') << mmu->next_block->offset()
            << get_sram_info(mmu->next_block->offset(), out->mem_size(), show_full_mem_info);
    }
    oss << " size: 0x" << out->mem_size() << "(" << utils::get_human_readable_size(out->mem_size()) << ")" << buf_idx.str();

    return oss.str();
}

bool _add_mem_info_to_dot(GraphPtr g, GraphPtr top_g)
{
    bool show_full_mem_info = false;
    if (BuilderConfig::exist(Constants::show_info))
    {
        if (BuilderConfig::get<std::set<std::string>>(Constants::show_info).count("mem"))
        {
            show_full_mem_info = true;
        }
    }

    auto m_nodes = g->nodes();
    for (auto n : m_nodes)
    {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn && !sgn->has_plugin())
        {
            _add_mem_info_to_dot(sgn->subgraph(), top_g);
            continue;
        }

        if (n->attrs().count(Constants::dot_address))
        {
            std::ostringstream oss;
            auto bss_id = n->get_attr<size_t>(Constants::bss_id);
            for (auto out : n->outputs())
            {
                oss << get_mem_info(out, top_g, show_full_mem_info) << "\\n";
            }

            uint32_t gm_addr = 0;
            std::vector<uint64_t> gm_addr_vec = top_g->get_attr("gm_addr_vec");
            if (bss_id < gm_addr_vec.size() && bss_id >= 0)
            {
                gm_addr = gm_addr_vec[bss_id];
            }
            if (show_full_mem_info && aipubt::BuilderConfig::get<bool>(Constants::enable_gm_enhance))
            {
                uint32_t w_addr = 0;
                if (n->attrs().count(Constants::mov_w_to_gm))
                {
                    auto start_node = n->get_attr<NodeRef>(Constants::mov_w_to_gm).lock();
                    auto end_node = start_node->get_attr<NodeRef>(Constants::mov_w_to_gm_wait).lock();
                    uint32_t start_node_offset = n->get_attr<uint32_t>(Constants::mov_w_gm_offset);
                    uint32_t start_node_size = n->get_attr<uint32_t>(Constants::mov_w_gm_size);
                    w_addr = gm_addr + start_node_offset;
                    oss << "|";
                    oss << "mov_w_to_gm_front: ";
                    oss << end_node->name() << " addr: 0x" << std::hex << w_addr << " size: 0x" << std::hex << start_node_size;
                }
                if (n->attrs().count(Constants::mov_w_to_gm_wait))
                {
                    auto root_node = n->get_attr<NodeRef>(Constants::mov_w_to_gm_wait).lock();
                    std::stringstream sec_value;
                    oss << "|";
                    bool wait_w_in_front = n->get_attr(Constants::wait_w_in_front);
                    if (wait_w_in_front)
                    {
                        oss << "wait_w_to_gm_front: ";
                    }
                    else
                    {
                        oss << "wait_w_to_gm_back: ";
                    }
                    oss << root_node->name();
                }
                if (n->attrs().count(Constants::same_w_same_level))
                {
                    auto root_node = n->get_attr<NodeRef>(Constants::same_w_same_level).lock();
                    if (root_node && root_node->attrs().count(Constants::ld_w_from_gm))
                    {
                        uint32_t m_node_offset = root_node->get_attr<uint32_t>(Constants::ld_w_gm_offset);
                        w_addr = gm_addr + m_node_offset;
                        oss << "|";
                        oss << "ld_w_from_gm addr: 0x" << std::hex << w_addr;
                    }
                }
                else if (n->attrs().count(Constants::ld_w_from_gm))
                {
                    uint32_t m_node_offset = n->get_attr<uint32_t>(Constants::ld_w_gm_offset);
                    w_addr = gm_addr + m_node_offset;
                    oss << "|";
                    oss << "ld_w_from_gm addr: 0x" << std::hex << w_addr;
                }
            }

            if (show_full_mem_info && aipubt::BuilderConfig::get<bool>(Constants::enable_x2_sm_opt))
            {
                if (n->attrs().count(Constants::movwtosm))
                {
                    auto next_node = n->get_attr<NodeRef>(Constants::movwtosm).lock();
                    uint32_t node_size = n->get_attr<uint32_t>("_preload_sm_size");
                    oss << "|";
                    oss << "mv_w_to_sm: " << next_node->name() << " size: " << std::hex << node_size;
                }
                if (n->attrs().count(Constants::ldwfromsm))
                {
                    int sm = n->get_attr<int>("_use_sm");
                    oss << "|";
                    oss << "ld_w_from_sm " << sm;
                }
            }

            if (show_full_mem_info && aipubt::BuilderConfig::get<bool>(Constants::enable_dynamic_gm))
            {
                if (n->attrs().count(Constants::wr_out_to_gm))
                {
                    auto offset = n->get_attr<uint32_t>(Constants::wr_out_to_gm);
                    oss << "|";
                    oss << "wr_out_to_gm address: 0x" << std::hex << gm_addr + offset;
                }

                if (n->attrs().count(Constants::mov_d_to_ddr))
                {
                    auto end_node = n->get_attr<NodeRef>(Constants::mov_d_to_ddr).lock();
                    auto root_node = end_node->get_attr<NodeRef>(Constants::mov_d_to_ddr_wait).lock();
                    oss << "|";
                    oss << "mov_d_to_ddr: " << root_node->name();
                }

                if (n->attrs().count(Constants::mov_d_to_ddr_wait))
                {
                    auto root_node = n->get_attr<NodeRef>(Constants::mov_d_to_ddr_wait).lock();
                    oss << "|";
                    oss << "mov_d_to_ddr_wait: " << root_node->name();
                }
            }

            n->add_dot_section(Constants::dot_address, oss.str());
        }
    }

    return false;
}

bool add_mem_info_to_dot(GraphPtr g)
{
    return _add_mem_info_to_dot(g, g);
}

} // namespace aipubt

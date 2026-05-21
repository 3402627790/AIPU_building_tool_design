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
 * @Author: Weibin Li
 * @Date: 2026-04-14 09:26:15
 * @Last Modified by:  Weibin Li
 * @Last Modified time: 2026-04-15 10:45:29
 */

#ifndef __CPS_ELF_H__
#define __CPS_ELF_H__

#include <elfio/elfio.hpp>
#include <iostream>
#include <map>
#include <memory>
#include <stdint.h>
#include <string>
#include <vector>

namespace aipubt
{

struct sync_ctrl_t
{
    int set_mask;
    int wait_mask;
};

enum
{
    TASK_TYPE_KERNEL,
    TASK_TYPE_DMA,
    TASK_TYPE_BARRIER_SET,
    TASK_TYPE_BARRIER_WAIT,
    TASK_TYPE_CTRL_FLOW,
    TASK_TYPE_CACHE_MGR
};
struct task_t
{
    int type;
    uint16_t invalid_offset;
    uint16_t id;
    union
    {
        struct
        {
            uint32_t offset;
            uint16_t grid_dim[3];
            uint16_t block_dim[3];
            int surfb_desp_num;
            int gb_param_offset;
            int gb_param_size;
            uint32_t ib_size;
            uint16_t static_sm_bytes;
            uint16_t param_size;
            uint8_t sreg_param_size;
            uint8_t sreg_cnt;
            uint8_t vreg_cnt;
            uint8_t head_wg_vreg_cnt;
            uint8_t isa_set;
            uint8_t subnormal_en;
            uint8_t saturation_en;
            uint8_t padding;
            int policy;
            sync_ctrl_t sync;
            int l1d_op;
            int l1i_prefetch_stride;
            uint8_t pfc_group_en;
            uint64_t pfc_addr;
        } kernel;

        struct
        {
            uint64_t src_addr;
            size_t size;
            int osd_ctrl;
            sync_ctrl_t sync;
        } dma;

        struct
        {
            uint64_t addr;
            uint32_t data;
            uint32_t mask;
            int interrupt;
            int timer_mode;
            bool bar_src;
        } barrier_set;

        struct
        {
            uint64_t addr;
            uint32_t ref;
            uint32_t mask;
            int func;
            int pool_interval;
            bool bar_src;
        } barrier_wait;

        struct
        {
            bool ctrl;
            uint64_t addr;
            int value;
            int offset;
            sync_ctrl_t sync;
        } ctrl_flow;

        struct
        {
            int op;
            sync_ctrl_t sync;
        } cache_mgr;
    };
};

struct stream_t
{
    uint32_t id;
    std::vector<task_t> tasks;
};
struct tensor_info_t
{
    uint32_t id;
    uint8_t dtype;
    std::vector<uint32_t> shape;
};
struct memory_plan_t
{
    std::vector<uint64_t> slots;
    std::map<uint32_t, std::pair<uint32_t, uint64_t>> tso; // tensor to slot and with offset;
};

class CpsElf
{
public:
    CpsElf() { elfio_.create(ELFCLASS64, ELFDATA2LSB); };
    ~CpsElf() = default;

    bool load(const std::string &filename)
    {
        return elfio_.load(filename);
    }
    bool load(std::istream &stream)
    {
        return elfio_.load(stream);
    }
    bool save(const std::string &filename)
    {
        return elfio_.save(filename);
    }
    bool save(std::ostream &stream)
    {
        return elfio_.save(stream);
    }

    std::pair<const void *, size_t> get_text() const
    {
        std::pair<const void *, size_t> t{nullptr, 0};
        auto text = elfio_.sections[".text"];
        if (text != nullptr)
            t = {text->get_data(), text->get_size()};
        return t;
    }
    void set_text(const void *txt, size_t size)
    {
        auto text = elfio_.sections[".text"];
        if (text == nullptr)
        {
            text = elfio_.sections.add(".text");
            text->set_type(SHT_PROGBITS);
            text->set_flags(SHF_ALLOC | SHF_EXECINSTR);
        }
        text->set_data((const char *)txt, size);
    }

    std::vector<stream_t> get_graph() const
    {
        std::vector<stream_t> streams;
        auto graph = get_note_section("graph");
        if (graph.first == nullptr)
            return streams;
        constexpr auto k_task_size = sizeof(task_t);
        const uint8_t *p = (const uint8_t *)graph.first;
        const auto &streams_count = *((const uint32_t *)p);
        p += sizeof(streams_count);
        for (uint32_t i = 0; i < streams_count; ++i)
        {
            stream_t stream;
            stream.id = *((const uint32_t *)p);
            p += sizeof(stream.id);

            auto task_count = *((const uint32_t *)p);
            stream.tasks.resize(task_count);
            p += sizeof(task_count);

            const auto tasks_bytes = task_count * k_task_size;
            memcpy(stream.tasks.data(), p, tasks_bytes);
            p += tasks_bytes;
            streams.emplace_back(std::move(stream));
        }
        return streams;
    }

    void set_graph(const std::vector<stream_t> &streams)
    {
        // streams_size + (streamid + task_size + task) * task_count
        constexpr auto k_task_size = sizeof(task_t);
        uint32_t tasks_count = 0;
        for (const auto &stream : streams)
        {
            tasks_count += stream.tasks.size();
        }
        const auto total_size = sizeof(uint32_t) + (sizeof(uint32_t) + sizeof(uint32_t)) * streams.size() + k_task_size * tasks_count;
        auto buffer = std::make_unique<uint8_t[]>(total_size);
        uint8_t *p = buffer.get();
        auto &streams_count = *((uint32_t *)p);
        streams_count = streams.size();
        p += sizeof(streams_count);
        for (const auto &stream : streams)
        {
            auto &stream_id = *((uint32_t *)p);
            stream_id = stream.id;
            p += sizeof(stream_id);

            auto &task_count = *((uint32_t *)p);
            task_count = stream.tasks.size();
            p += sizeof(task_count);

            auto tasks_bytes = task_count * k_task_size;
            memcpy(p, stream.tasks.data(), tasks_bytes);
            p += tasks_bytes;
        }
        auto note = force_get_note();
        ELFIO::note_section_accessor nsa(elfio_, note);

        nsa.add_note(0, "graph", buffer.get(), total_size);
    }

    std::pair<const void *, size_t> get_data() const
    {
        return get_note_section("data");
    }
    void set_data(const void *data, size_t size)
    {
        auto note = force_get_note();
        ELFIO::note_section_accessor nsa(elfio_, note);
        nsa.add_note(0, "data", data, size);
    }

    std::pair<const void *, size_t> get_param() const
    {
        return get_note_section("param");
    }
    void set_param(const void *param, size_t size)
    {
        auto note = force_get_note();
        ELFIO::note_section_accessor nsa(elfio_, note);
        nsa.add_note(0, "param", param, size);
    }
    std::map<uint32_t, uint32_t> get_param_tab() const
    {
        return get_tab<uint32_t, uint32_t>("param_tab");
    }
    void set_param_tab(const std::map<uint32_t, uint32_t> &pt)
    {
        set_tab("param_tab", pt);
    }
    std::map<uint32_t, uint64_t> get_weight_tab() const
    {
        return get_tab<uint32_t, uint64_t>("weight_tab");
    }
    void set_weight_tab(const std::map<uint32_t, uint64_t> &wt)
    {
        set_tab("weight_tab", wt);
    }
    std::multimap<uint8_t, tensor_info_t> get_tensor_tab() const
    {
        std::multimap<uint8_t, tensor_info_t> tis;
        auto ns = get_note_section("tensor_tab");
        if (ns.first == nullptr)
            return tis;
        const uint8_t *p = (const uint8_t *)ns.first;
        const auto count = *((const int *)p);
        p += sizeof(count);
        for (int i = 0; i < count; ++i)
        {
            const auto key = *p;
            p += sizeof(key);

            const auto id = *((const uint32_t *)p);
            p += sizeof(id);

            const auto dtype = *p;
            p += sizeof(dtype);

            const auto dim = *p;
            p += sizeof(dim);
            std::vector<uint32_t> shape(dim);
            if (dim != 0)
            {
                const auto shape_size = sizeof(uint32_t) * dim;
                memcpy(shape.data(), p, shape_size);
                p += shape_size;
            }
            tis.emplace(key, tensor_info_t{id, dtype, std::move(shape)});
        }
        return tis;
    }

    void set_tensor_tab(const std::multimap<uint8_t, tensor_info_t> &tis)
    {
        // count + (type + id + dtype + dim + shape) * size
        size_t total_size = sizeof(uint32_t);
        for (const auto &t : tis)
        {
            total_size += (sizeof(uint8_t) + sizeof(uint32_t) + sizeof(uint8_t) +
                           sizeof(uint8_t) + t.second.shape.size() * sizeof(uint32_t));
        }

        auto buffer = std::make_unique<uint8_t[]>(total_size);
        uint8_t *p = buffer.get();

        auto &count = *((uint32_t *)p);
        count = (uint32_t)tis.size();
        p += sizeof(count);

        for (const auto &ti : tis)
        {
            *p = ti.first;
            p += 1;

            std::memcpy(p, &ti.second.id, sizeof(uint32_t));
            p += sizeof(uint32_t);

            *p = ti.second.dtype;
            p += 1;

            const uint8_t shape_size = ti.second.shape.size();
            *p = shape_size;
            p += 1;

            if (shape_size > 0)
            {
                const size_t data_size = shape_size * sizeof(uint32_t);
                std::memcpy(p, ti.second.shape.data(), data_size);
                p += data_size;
            }
        }
        auto note = force_get_note();
        ELFIO::note_section_accessor nsa(elfio_, note);

        nsa.add_note(0, "tensor_tab", buffer.get(), total_size);
    }
    memory_plan_t get_memory_plan() const
    {
        memory_plan_t mem_plan;
        auto mp = get_note_section("memory_plan");
        if (mp.first == nullptr)
            return mem_plan;

        const uint8_t *p = (const uint8_t *)mp.first;
        const auto &slots_count = *((const int *)p);
        mem_plan.slots.resize(slots_count);
        p += sizeof(slots_count);
        const auto slot_data_size = sizeof(uint64_t) * slots_count;
        memcpy(mem_plan.slots.data(), p, slot_data_size);
        p += slot_data_size;
        const auto &tso_count = *((const int *)p);
        p += sizeof(tso_count);
        for (int i = 0; i < tso_count; ++i)
        {
            auto &tensor_id = *((uint32_t *)p);
            p += sizeof(tensor_id);

            auto &slot_off = *((std::pair<uint32_t, uint64_t> *)p);

            mem_plan.tso.emplace(tensor_id, slot_off);

            p += sizeof(slot_off);
        }
        return mem_plan;
    }

    void set_memory_plan(const memory_plan_t &mem_plan)
    {
        // memory_plan_t
        //{
        //    slot_size:int
        //    slot_data : uint64_t*
        //    map :uint8_t[]
        // } per stream
        const auto slot_data_size = sizeof(uint64_t) * mem_plan.slots.size();
        const auto total_size = sizeof(int) + slot_data_size + sizeof(int) +
                                (sizeof(uint32_t) + sizeof(std::pair<uint32_t, uint64_t>)) * mem_plan.tso.size();
        auto buffer = std::make_unique<uint8_t[]>(total_size);
        uint8_t *p = buffer.get();
        auto &slots_count = *((int *)p);
        slots_count = mem_plan.slots.size();
        p += sizeof(slots_count);
        memcpy(p, mem_plan.slots.data(), slot_data_size);
        p += slot_data_size;
        auto &tso_count = *((int *)p);
        tso_count = mem_plan.tso.size();
        p += sizeof(tso_count);
        for (const auto &tso : mem_plan.tso)
        {
            auto &tensor_id = *((uint32_t *)p);
            tensor_id = tso.first;
            p += sizeof(tensor_id);

            auto &slot_off = *((std::pair<uint32_t, uint64_t> *)p);
            slot_off = tso.second;
            p += sizeof(slot_off);
        }
        auto note = force_get_note();
        ELFIO::note_section_accessor nsa(elfio_, note);

        nsa.add_note(0, "memory_plan", buffer.get(), total_size);
    }

    ELFIO::section *get_section(const std::string &name)
    {
        return elfio_.sections[name];
    }
    std::pair<const void *, size_t> get_note_section(const ELFIO::section *section,
                                                     const std::string &key_name,
                                                     uint32_t key_type = 0) const
    {
        ELFIO::const_note_section_accessor nsa(elfio_, section);
        auto num = nsa.get_notes_num();
        for (decltype(num) i = 0; i < num; ++i)
        {
            std::string name;
            void *desc = nullptr;
            ELFIO::Elf_Word type = 0;
            ELFIO::Elf_Word descSize = 0;
            nsa.get_note(i, type, name, desc, descSize);

            if (key_name == name && key_type == type)
                return {desc, descSize};
        }
        return {nullptr, 0};
    }
    std::pair<uint64_t, uint64_t> get_symbol(const ELFIO::section *section, const std::string &key_name)
    {
        ELFIO::const_symbol_section_accessor ssa(elfio_, section);
        auto num = ssa.get_symbols_num();
        for (decltype(num) i = 0; i < num; ++i)
        {
            std::string name;
            aipubt::ELFIO::Elf64_Addr value = 0;
            aipubt::ELFIO::Elf_Xword size = 0;
            unsigned char bind = 0, type = 0, other = 0;
            aipubt::ELFIO::Elf_Half section_index = 0;
            ssa.get_symbol(i, name, value, size, bind, type, section_index, other);
            if (key_name == name)
                return {value, size};
        }
        return {0, 0};
    }

private:
    ELFIO::section *force_get_note()
    {
        auto note = elfio_.sections[".note.zhouyi"];
        if (note == nullptr)
        {
            note = elfio_.sections.add(".note.zhouyi");
            note->set_type(SHT_NOTE);
        }
        return note;
    }
    std::pair<const void *, size_t> get_note_section(const std::string &name) const
    {
        return get_note_section(elfio_.sections[".note.zhouyi"], name, 0);
    }
    template <class _Key, class _Val, template <class...> class _Tab = std::map, class... _Args>
    _Tab<_Key, _Val, _Args...> get_tab(const std::string &name) const
    {
        using tab_elem_t = std::pair<_Key, _Val>;
        _Tab<_Key, _Val, _Args...> tab;
        const auto p = get_note_section(name);
        const auto *end = (const tab_elem_t *)(((uint8_t *)p.first) + p.second);
        for (const auto *elem = (const tab_elem_t *)p.first; elem < end; ++elem)
            tab.insert(*elem);
        return tab;
    }
    template <class _Key, class _Val, template <class...> class _Tab = std::map, class... _Args>
    void set_tab(const std::string &name, const _Tab<_Key, _Val, _Args...> &tab)
    {
        using tab_elem_t = std::pair<_Key, _Val>;
        ELFIO::note_section_accessor nsa(elfio_, force_get_note());
        auto buf = std::make_unique<tab_elem_t[]>(tab.size());
        tab_elem_t *p = buf.get();
        for (auto it = tab.begin(); it != tab.end(); ++it)
            *p++ = *it;
        nsa.add_note(0, name, buf.get(), tab.size() * sizeof(tab_elem_t));
    }

    ELFIO::elfio elfio_;
};
using CpsElfPtr = std::shared_ptr<CpsElf>;
} // namespace aipubt

#endif //!__CPS_ELF_H__
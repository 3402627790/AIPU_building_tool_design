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
#include <sstream>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <utility>
#include <vector>

#ifndef ELF_MAJOR_VERSION
#define ELF_MAJOR_VERSION 1
#endif

#ifndef ELF_MIN_VERSION
#define ELF_MIN_VERSION 1
#endif

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
    TASK_TYPE_CACHE_MGR,
    TASK_TYPE_MAX
};

enum
{
    BUF_TYPE_PARAM,
    BUF_TYPE_WEIGHT,
    BUF_TYPE_SLOT,
    BUF_TYPE_STREAM,
};

struct buf_desc_t
{
    uint8_t type;
    uint32_t index;
    size_t size;
    size_t alignment;
    void *data;
};

struct buf_loc_t
{
    uint8_t type;
    uint32_t index;
    size_t offset;
};

struct task_t
{
    int type;
    uint16_t invalid_offset;
    uint16_t id;
    union {
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
            uint8_t mode;
            uint8_t evict_priority;
            int osd_ctrl;
            sync_ctrl_t sync;
        } dma;

        struct
        {
            uint64_t addr;
            uint32_t data;
            int interrupt;
            uint8_t wr_be;
            int timer_mode;
            bool bar_src;
        } barrier_set;

        struct
        {
            uint64_t addr;
            uint32_t ref;
            int func;
            int pool_interval;
            uint8_t rd_be;
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

class CpsElf;
using CpsElfPtr = std::shared_ptr<CpsElf>;
using CpsElfRef = std::weak_ptr<CpsElf>;
class CpsElf
{
public:
    CpsElf() { elfio_.create(ELFCLASS64, ELFDATA2LSB); }
    ~CpsElf() = default;

    bool load(std::istream &stream)
    {
        bool ret = elfio_.load(stream);
        if (ret)
            load_buffer_tab();
        return ret;
    }
    bool load(const std::string &filename)
    {
        std::ifstream in(filename);
        return load(in);
    }

    bool save(std::ostream &stream)
    {
        save_buffer_tab();
        return elfio_.save(stream);
    }
    bool save(const std::string &filename)
    {
        std::ofstream out(filename);
        return save(out);
    }

    const buf_desc_t *get_buffer(uint8_t type, uint32_t idx) const
    {
        uint32_t key = ((uint32_t)type << 24) | (idx & 0xffffff);
        auto it = buffers_.find(key);
        return it != buffers_.end() ? &(it->second) : nullptr;
    }
    void set_buffer(uint8_t type, uint32_t idx, size_t size,
                    size_t alignment = 1, const void *data = nullptr)
    {
        buf_desc_t desc;
        desc.type = type;
        desc.index = idx;
        desc.size = size;
        desc.alignment = alignment;
        desc.data = (void *)data;

        uint32_t key = ((uint32_t)type << 24) | (idx & 0xffffff);
        buffers_[key] = desc;

        buffer_total_size_ += (data ? size : 0);
    }

    std::map<uint32_t, buf_loc_t> get_tensor_offsets() const
    {
        return get_tab<uint32_t, buf_loc_t>("tensor_offset_tab");
    }
    void set_tensor_offsets(const std::map<uint32_t, buf_loc_t> &offsets)
    {
        set_tab("tensor_offset_tab", offsets);
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

    std::map<uint32_t, uint32_t> get_param_tab() const
    {
        return get_tab<uint32_t, uint32_t>("param_tab");
    }
    void set_param_tab(const std::map<uint32_t, uint32_t> &pt)
    {
        set_tab("param_tab", pt);
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
        static_assert(std::is_trivial_v<_Key> && std::is_standard_layout_v<_Key>,
                      "'key' must be a plain data type.");
        static_assert(std::is_trivial_v<_Val> && std::is_standard_layout_v<_Val>,
                      "'value' must be a plain data type.");
        using tab_elem_t = typename _Tab<_Key, _Val, _Args...>::value_type;
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
        static_assert(std::is_trivial_v<_Key> && std::is_standard_layout_v<_Key>,
                      "'key' must be a plain data type.");
        static_assert(std::is_trivial_v<_Val> && std::is_standard_layout_v<_Val>,
                      "'value' must be a plain data type.");
        using tab_elem_t = typename _Tab<_Key, _Val, _Args...>::value_type;
        ELFIO::note_section_accessor nsa(elfio_, force_get_note());
        auto buf = std::make_unique<tab_elem_t[]>(tab.size());
        tab_elem_t *p = buf.get();
        for (auto it = tab.begin(); it != tab.end(); ++it, ++p)
            memcpy((void *)p, &(*it), sizeof(tab_elem_t));
        nsa.add_note(0, name, buf.get(), tab.size() * sizeof(tab_elem_t));
    }

    void save_buffer_tab()
    {
        auto data = std::make_unique<uint8_t[]>(buffer_total_size_);
        uint64_t offset = 0;
        for (auto it = buffers_.begin(); it != buffers_.end(); ++it)
        {
            auto &desc = it->second;
            if (desc.data == nullptr)
                continue;

            memcpy(data.get() + offset, desc.data, desc.size);
            desc.data = (void *)((offset << 1) | 1);
            offset += desc.size;
        }

        ELFIO::note_section_accessor nsa(elfio_, force_get_note());
        nsa.add_note(0, "buffer_data", data.get(), buffer_total_size_);

        set_tab("buffer_tab", buffers_);

        buffers_.clear();
        buffer_total_size_ = 0;
    }
    void load_buffer_tab()
    {
        auto duo = get_note_section("buffer_data");
        uint8_t *data = (uint8_t *)duo.first;
        buffer_total_size_ = duo.second;
        auto tab = get_tab<uint32_t, buf_desc_t>("buffer_tab");
        for (auto it = tab.begin(); it != tab.end(); ++it)
        {
            auto &desc = it->second;
            uint64_t offset = (uint64_t)desc.data;
            if (!(offset & 1))
                continue;
            offset >>= 1;
            desc.data = data + offset;
        }
        buffers_ = std::move(tab);
    }

    ELFIO::elfio elfio_;
    std::map<uint32_t, buf_desc_t> buffers_;
    uint64_t buffer_total_size_ = 0;
};

class HybridELF;
using HybridELFPtr = std::shared_ptr<HybridELF>;
using HybridELFRef = std::weak_ptr<HybridELF>;
class HybridELF
{
public:
    enum CompilerMsgFlag
    {
        HAS_HOST_CODE = 0x1 << 0,
        ENABLE_DS = 0x1 << 1,
    };

    struct CompilerMsg
    {
        uint32_t file_version = ((ELF_MAJOR_VERSION & 0xff) << 24) | ((ELF_MIN_VERSION & 0xff) << 16);
        uint32_t build_version = 0;
        uint32_t arch = 0;
        uint32_t flag = 0;

        bool has_host_code() const
        {
            return flag & HAS_HOST_CODE;
        }
        bool enable_ds() const
        {
            return flag & ENABLE_DS;
        }

        void load(std::istream &stream)
        {
            stream.read((char *)&file_version, sizeof(uint32_t));
            stream.read((char *)&build_version, sizeof(uint32_t));
            stream.read((char *)&arch, sizeof(uint32_t));
            stream.read((char *)&flag, sizeof(uint32_t));
        }
        void write(std::ostream &stream)
        {
            stream.write((char *)&file_version, sizeof(uint32_t));
            stream.write((char *)&build_version, sizeof(uint32_t));
            stream.write((char *)&arch, sizeof(uint32_t));
            stream.write((char *)&flag, sizeof(uint32_t));
        }
    };

    struct Section
    {
        operator bool() const
        {
            return data && size;
        }

        void *data = nullptr;
        size_t size = 0;
    };

    struct SectionBuf : public std::streambuf
    {
        SectionBuf(const void *data = nullptr, size_t size = 0)
        {
            char *p = (char *)data;
            setg(p, p, p + size);
        }
        SectionBuf(const Section &sec) : SectionBuf(sec.data, sec.size) {}

        std::streambuf::pos_type seekoff(std::streambuf::off_type off,
                                         std::ios_base::seekdir way,
                                         std::ios_base::openmode which)
        {
            if (!(which & std::ios::in) || gptr() == nullptr)
                return std::streambuf::pos_type(std::streambuf::off_type(-1));

            std::streambuf::off_type newoff = 0;
            char *seekhigh = egptr();
            switch (way)
            {
            case std::ios::beg:
                newoff = 0;
                break;
            case std::ios::cur:
                newoff = gptr() - eback();
                break;
            case std::ios::end:
                newoff = seekhigh - eback();
                break;
            default:
                break;
            }
            newoff += off;
            if (newoff < 0 || newoff > seekhigh - eback())
                return std::streambuf::pos_type(std::streambuf::off_type(-1));

            char *newpos = eback() + newoff;
            setg(eback(), newpos, std::max(newpos, egptr()));
            return std::streambuf::pos_type(newoff);
        }

        std::streambuf::pos_type seekpos(std::streambuf::pos_type sp,
                                         std::ios_base::openmode which)
        {
            if (!(which & std::ios::in) || gptr() == nullptr)
                return std::streambuf::pos_type(std::streambuf::off_type(-1));

            if (sp < 0 || sp > egptr() - eback())
                return std::streambuf::pos_type(off_type(-1));

            char *newpos = eback() + sp;

            setg(eback(), newpos, std::max(newpos, egptr()));

            return sp;
        }
    };

    HybridELF() { m_elfio.create(ELFCLASS64, ELFDATA2LSB); }
    ~HybridELF() = default;

    bool load(std::istream &stream)
    {
        bool ret = m_elfio.load(stream);
        load_msg();
        return ret;
    }
    bool load(const std::string &filename)
    {
        std::ifstream ifs(filename);
        return load(ifs);
    }

    bool save(std::ostream &stream)
    {
        save_msg();
        save_cpself();
        return m_elfio.save(stream);
    }
    bool save(const std::string &filename)
    {
        std::ofstream ofs(filename);
        return save(ofs);
    }

    void set_cpself(CpsElfPtr elf)
    {
        m_cpself = elf;
    }
    CpsElfPtr get_cpself()
    {
        if (!m_cpself)
        {
            load_cpself();
        }
        return m_cpself;
    }
    Section get_cpself_note()
    {
        return get_note("cpself");
    }
    void remove_cpself_note()
    {
        remove_note("cpself");
    }

    void set_note(const std::string &note,
                  const std::string &note_name)
    {
        set_note(note, note_name, 0);
    }
    Section get_note(const std::string &note_name)
    {
        ELFIO::Elf_Word idx;
        return get_note(note_name, 0, idx);
    }
    void remove_note(const std::string &note_name)
    {
        remove_note(note_name, 0);
    }

    CompilerMsg &get_compiler_msg()
    {
        return m_msg;
    }
    const CompilerMsg &get_compiler_msg() const
    {
        return m_msg;
    }

private:
    void load_msg()
    {
        auto note = get_note("compilermsg");
        if (note)
        {
            SectionBuf buf(note);
            std::istream stream(&buf);
            m_msg.load(stream);
        }
    }
    void save_msg()
    {
        remove_note("compilermsg");
        std::stringstream oss;
        m_msg.write(oss);
        set_note(oss.str(), "compilermsg");
    }

    void load_cpself()
    {
        auto note = get_note("cpself");
        if (note)
        {
            SectionBuf buf(note);
            std::istream stream(&buf);
            m_cpself = std::make_shared<CpsElf>();
            m_cpself->load(stream);
        }
    }
    void save_cpself()
    {
        remove_note("cpself");
        if (m_cpself)
        {
            std::stringstream oss;
            m_cpself->save(oss);
            set_note(oss.str(), "cpself");
        }
    }

    void set_note_section()
    {
        if (!m_note)
        {
            m_note = m_elfio.sections[".note.zhouyi"];
            if (!m_note)
            {
                m_note = m_elfio.sections.add(".note.zhouyi");
                m_note->set_type(SHT_NOTE);
            }
        }
    }
    void set_note(const std::string &note,
                  const std::string &note_name,
                  size_t note_type)
    {
        set_note_section();
        ELFIO::note_section_accessor nsa(m_elfio, m_note);
        nsa.add_note(note_type, note_name, note.c_str(), note.size());
    }
    Section get_note(const std::string &note_name,
                     size_t note_type,
                     ELFIO::Elf_Word &idx)
    {
        set_note_section();
        ELFIO::const_note_section_accessor nsa(m_elfio, m_note);
        auto num = nsa.get_notes_num();
        for (ELFIO::Elf_Word i = 0; i < num; ++i)
        {
            std::string name;
            void *desc = nullptr;
            ELFIO::Elf_Word type = 0;
            ELFIO::Elf_Word size = 0;
            nsa.get_note(i, type, name, desc, size);
            if (note_name == name && note_type == type)
            {
                idx = i;
                return {desc, size};
            }
        }
        return Section();
    }
    void remove_note(const std::string &note_name,
                     size_t note_type)
    {
        ELFIO::Elf_Word idx;
        if (get_note(note_name, note_type, idx))
        {
            ELFIO::note_section_accessor nsa(m_elfio, m_note);
            nsa.remove_note(idx);
        }
    }

    ELFIO::elfio m_elfio;
    ELFIO::section *m_note = nullptr;
    CpsElfPtr m_cpself;
    CompilerMsg m_msg;
};

} // namespace aipubt

#endif //!__CPS_ELF_H__

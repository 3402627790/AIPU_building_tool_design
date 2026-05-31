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
 * @Date: 2021-07-20 10:27:47
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-27 11:30:22
 */
#include "builder/aipu_elf.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/logger.h"
#include "utils/utils.h"
#include <fstream>
#include <sstream>

namespace aipubt
{
using namespace ELFIO;

struct MemBuf : std::streambuf
{
    MemBuf(char *base, std::ptrdiff_t n)
    {
        this->setg(base, base, base + n);
    }
};

static bool dump_buf_to_bin(char *buf, uint32_t size, std::string name)
{
    std::ofstream file;
    file.open(name, std::ios::out | std::ios::binary);
    file.write(buf, size);
    bool ret = !file.fail();
    if (!ret)
    {
        ERROR() << "file open for " << name << " failed! ";
    }
    file.close();
    return ret;
}

static bool load_bin_from_buf(char *&buf, uint32_t &size, std::string name)
{
    std::ifstream file;
    file.open(name, std::ios::binary);
    if (!file)
    {
        ERROR() << "file open for " << name << " failed! ";
        return false;
    }
    file.seekg(0, std::ios::end);
    std::streamsize buf_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size = buf_size;
    buf = new char[size];
    file.read(buf, size);
    file.close();
    return true;
}

static std::vector<char> serialize_str_vec_to_mem(const std::vector<std::string> &vec)
{
    std::ostringstream oss;
    size_t vecSize = vec.size();
    oss.write(reinterpret_cast<const char *>(&vecSize), sizeof(vecSize));

    for (const auto &str : vec)
    {
        size_t str_length = str.length();
        oss.write(reinterpret_cast<const char *>(&str_length), sizeof(str_length));
        oss.write(str.c_str(), str_length);
    }

    std::string str = oss.str();

    std::vector<char> ret(str.begin(), str.end());

    return ret;
}

static std::vector<std::string> deserialize_str_vec_from_mem(const std::vector<char> &data)
{
    std::vector<std::string> vec;
    std::istringstream iss(std::string(data.begin(), data.end()));

    size_t size;
    iss.read(reinterpret_cast<char *>(&size), sizeof(size));
    vec.resize(size);

    for (size_t i = 0; i < size; ++i)
    {
        size_t str_length;
        iss.read(reinterpret_cast<char *>(&str_length), sizeof(str_length));

        vec[i].resize(str_length);
        iss.read(&vec[i][0], str_length);
    }

    return vec;
}

void AIPUELF::load(const char *base, uint32_t n)
{
    std::stringstream ss;
    ss.write(base, n);
    load(ss);
}

void AIPUELF::load(const std::string &filename)
{
    std::ifstream stream(filename);

    std::string path = "";
    size_t found = filename.find_last_of("/");
    if (found != std::string::npos)
    {
        path = filename.substr(0, found + 1);
    }

    m_elf_folder = path;

    load(stream);
}

void AIPUELF::load(std::istream &stream)
{
    m_elf.load(stream);
    m_symtab.clear();

    auto bin = get_bin_note("extra_weight_name");
    if (bin.bin_size)
    {
        std::vector<char> str_mem(bin.bin, bin.bin + bin.bin_size);
        std::vector<std::string> str_vec = deserialize_str_vec_from_mem(str_mem);

        for (size_t i = 0; i < str_vec.size(); ++i)
        {
            size_t pos = str_vec[i].find('+');
            if (pos == std::string::npos)
            {
                ERROR() << "no sha32";
            }
            std::string str = str_vec[i].substr(0, pos);
            std::string hash = str_vec[i].substr(pos + 1);
            char *ptr = nullptr;
            uint32_t size = 0;
            auto bin_path = m_elf_folder + str;
            if (utils::file_exists(bin_path))
            {
                load_bin_from_buf(ptr, size, bin_path);
            }
            else
            {
                ERROR() << bin_path << " doesn't exist";
            }
            auto file_hash = utils::sha256_hex(ptr, size);
            if (file_hash != hash)
            {
                ERROR() << "integrity check failed";
            }
            // INFO() << "get extra weight: " << str_vec[i] << " size: " << size;
            BinSection wt;
            wt.bin_size = size;
            wt.bin = ptr;
            m_extra_weight_vec.push_back(wt);
        }
    }

    auto raw_bin = get_bin_note("compilermsg");
    if (raw_bin.bin_size)
    {
        MemBuf buf(raw_bin.bin, raw_bin.bin_size);
        std::istream stream(&buf);
        m_comment.load(stream);
    }

    raw_bin = get_bin_note("gmconfig");
    if (raw_bin.bin_size)
    {
        MemBuf buf(raw_bin.bin, raw_bin.bin_size);
        std::istream stream(&buf);
        m_gm.load(stream);
    }

    raw_bin = get_bin_note("ir");
    if (raw_bin.bin_size)
    {
        m_ir = std::string(raw_bin.bin, raw_bin.bin_size);
    }
    raw_bin = get_bin_note("dumpir");
    if (raw_bin.bin_size)
    {
        m_dump_ir = std::string(raw_bin.bin, raw_bin.bin_size);
    }
    raw_bin = get_bin_note("graphjson");
    if (raw_bin.bin_size)
    {
        m_graph_json = std::string(raw_bin.bin, raw_bin.bin_size);
    }
}

void AIPUELF::write(const std::string &filename)
{
    std::ofstream stream(filename);

    if (m_extra_weight_vec.size())
    {
        std::string path = "";
        std::string new_name = "";
        size_t found = filename.find_last_of("/");
        if (found != std::string::npos)
        {
            path = filename.substr(0, found + 1);
            new_name = filename.substr(found + 1);
        }
        else
        {
            new_name = filename;
        }

        BinSection extra_wt_name;
        std::vector<std::string> str_vec;
        for (size_t i = 0; i < m_extra_weight_vec.size(); ++i)
        {
            std::string str = new_name + ".extraweight" + std::to_string(i);
            dump_buf_to_bin(m_extra_weight_vec[i].bin, m_extra_weight_vec[i].bin_size, path + str);
            auto hash = utils::sha256_hex(m_extra_weight_vec[i].bin, m_extra_weight_vec[i].bin_size);
            str_vec.push_back(str + "+" + hash);
        }
        auto str_mem = serialize_str_vec_to_mem(str_vec);

        extra_wt_name.bin_size = str_mem.size();
        extra_wt_name.bin = new char[str_mem.size()];
        memcpy(extra_wt_name.bin, str_mem.data(), sizeof(char) * str_mem.size());
        set_bin_note(extra_wt_name, "extra_weight_name");
    }

    write(stream);
}

void AIPUELF::write(std::ostream &stream)
{
    //before write, should update compiler msg and other section
    std::ostringstream ss;
    m_comment.write(ss);
    set_note(ss.str(), "compilermsg");

    ss.str("");
    m_gm.write(ss);
    set_note(ss.str(), "gmconfig");

    if (m_ir.size())
    {
        set_note(m_ir, "ir");
    }

    if (m_dump_ir.size())
    {
        set_note(m_dump_ir, "dumpir");
    }

    if (m_graph_json.size())
    {
        set_note(m_graph_json, "graphjson");
    }
    std::stringstream oss;
    m_elf.save(oss);
    auto tmp = oss.str();
    stream.write(tmp.c_str(), tmp.size());
}

std::string AIPUELF::get_text() const
{
    section *text = get_elf_section(".text");
    if (!text)
    {
        return "";
    }
    return std::string(text->get_data(), text->get_size());
}

BinSection AIPUELF::get_bin_note(const std::string &note_name) const
{
    BinSection ro = {0, 0};
    if (!m_note)
    {
        m_note = get_elf_section(".note.aipu");
    }
    if (m_note)
    {
        uint32_t idx = 0;
        get_note(note_name, idx, true, ro);
    }
    return ro;
}

void AIPUELF::get_cps_section_note(const aipubt::ELFIO::section *s, std::string kernel_name, uint32_t info_type, char *&buf, uint32_t &size)
{
    aipubt::ELFIO::const_note_section_accessor nsa(m_elf, s);

    std::string name;
    void *desc = nullptr;
    aipubt::ELFIO::Elf_Word type = 0;
    aipubt::ELFIO::Elf_Word descSize = 0;
    auto num = nsa.get_notes_num();
    for (int i = 0; i < (int)num; ++i)
    {
        nsa.get_note(i, type, name, desc, descSize);

        if (name == kernel_name && type == info_type)
        {
            size = descSize;
            if (descSize > 0)
            {
                buf = new char[descSize]; // 分配新内存
                std::memcpy(buf, desc, descSize);
            }
            else
            {
                buf = nullptr;
            }
            INFO() << "get note name: " << name;
            return;
        }
    }
    ERROR() << "cannot find " << kernel_name;
}

void AIPUELF::set_comment(const std::string &comment)
{
    section *cmt_sec = get_elf_section(".comment");
    if (!cmt_sec)
    {
        return;
    }
    cmt_sec->append_data(comment);
    return;
}

void AIPUELF::set_cps_bin_note(BinSection &ro, const std::string &note_name)
{
    add_aipu_note();
    remove_note(note_name);
    note_section_accessor note_writer(m_elf, m_note);
    note_writer.add_note(0x01, note_name, ro.bin, ro.bin_size);
}

void AIPUELF::set_bin_note(BinSection &ro, const std::string &note_name)
{
    add_aipu_note();
    remove_note(note_name);
    note_section_accessor note_writer(m_elf, m_note);
    note_writer.add_note(0x01, note_name, ro.bin, ro.bin_size);
}

void AIPUELF::set_note(const std::string &note, const std::string &note_name)
{
    add_aipu_note();
    remove_note(note_name);
    note_section_accessor note_writer(m_elf, m_note);
    note_writer.add_note(0x01, note_name, (char *)note.c_str(), note.size());
}

bool AIPUELF::get_note(const std::string &note_name,
                       uint32_t &idx,
                       bool copy_desc,
                       BinSection &sec) const
{
    if (m_note)
    {
        note_section_accessor notes(m_elf, m_note);
        Elf_Word no_notes = notes.get_notes_num();
        for (Elf_Word i = 0; i < no_notes; ++i)
        {
            Elf_Word type;
            std::string name;
            void *desc;
            Elf_Word descsz;

            if (notes.get_note(i, type, name, desc, descsz))
            {
                if (name == note_name)
                {
                    if (copy_desc)
                    {
                        sec.bin = new char[descsz];
                        memcpy(sec.bin, (char *)desc, descsz);
                        sec.bin_size = descsz;
                    }
                    idx = i;
                    return true;
                }
            }
        }
    }
    return false;
}

void AIPUELF::remove_note(const std::string &note_name)
{
    uint32_t idx = 0;
    BinSection sec = {0, 0};
    if (get_note(note_name, idx, false, sec))
    {
        note_section_accessor notes(m_elf, m_note);
        notes.remove_note(idx);
    }
}

BinSection &AIPUELF::get_rodata() const
{
    if (!m_ro.bin_size || !m_ro.bin)
    {
        m_ro = get_bin_note("rodata");
    }
    return m_ro;
}

void AIPUELF::rewrite_ro(uint32_t offset, uint32_t value, bool write_to_file)
{
    get_rodata();
    if ((offset)*4 >= m_ro.bin_size)
    {
        return;
    }
    memcpy((uint8_t *)m_ro.bin + offset * 4, &value, sizeof(value));
    if (write_to_file)
    {
        set_rodata(m_ro);
    }
}
void AIPUELF::rewrite_ro(uint32_t offset, std::vector<uint32_t> values, bool write_to_file)
{
    get_rodata();
    if (offset * 4 + values.size() * 4 > m_ro.bin_size)
    {
        return;
    }
    memcpy((uint8_t *)m_ro.bin + (offset)*4, &values[0], sizeof(uint32_t) * values.size());
    if (write_to_file)
    {
        set_rodata(m_ro);
    }
}

void AIPUELF::set_rodata(BinSection &ro)
{
    m_ro = ro;
    set_bin_note(m_ro, "rodata");
}

BinSection AIPUELF::get_descriptor() const
{
    return get_bin_note("desc");
}

void AIPUELF::set_descriptor(BinSection &ro)
{
    set_bin_note(ro, "desc");
}

BinSection AIPUELF::get_weight() const
{
    return get_bin_note("weight");
}

void AIPUELF::set_weight(BinSection &ro)
{
    set_bin_note(ro, "weight");
}

void AIPUELF::set_extra_weight(const std::vector<BinSection> extra_data_vec)
{
    m_extra_weight_vec = extra_data_vec;
}

std::vector<BinSection> AIPUELF::get_extra_weight() const
{
    return m_extra_weight_vec;
}

void AIPUELF::set_llminfo(const LlmInfo &info)
{
    auto bin = get_bin_note("llm_info");
    if (bin.bin && bin.bin_size > 0)
    {
        ERROR() << "llm_info already exists so cannot add new llm_info";
        return;
    }
    bin = info.ser_to_bin();
    set_bin_note(bin, "llm_info");
}

LlmInfo AIPUELF::get_llminfo() const
{
    auto bin = get_bin_note("llm_info");
    LlmInfo info;
    if (!bin.bin || bin.bin_size == 0)
    {
        ERROR() << "llm_info not exists";
        return info;
    }
    info.deser_from_bin(bin, m_elf_folder);
    return info;
}

void AIPUELF::set_aipu_note(BinSection &bin, const std::string &name)
{
    set_bin_note(bin, name);
}

BinSection AIPUELF::get_aipu_note(const std::string &name) const
{
    return get_bin_note(name);
}

bool AIPUELF::has_aipu_note(const std::string &name) const
{
    uint32_t idx = 0;
    BinSection ro = {0, 0};
    return get_note(name, idx, false, ro);
}

void AIPUELF::remove_aipu_note(const std::string &name)
{
    remove_note(name);
}

BinSection AIPUELF::get_bin_section(const std::string &name)
{
    return get_bin_note(name);
}

FeatureMapList AIPUELF::get_fm_list() const
{
    auto raw_bin = get_bin_note("fmlist");
    FeatureMapList fm = {0};
    MemBuf buf(raw_bin.bin, raw_bin.bin_size);
    std::istream stream(&buf);
    fm.load(stream);
    return fm;
}

void AIPUELF::set_fm_list(FeatureMapList &ro)
{
    std::ostringstream ss;
    ro.write(ss);
    set_note(ss.str(), "fmlist");
}

RemapcSection AIPUELF::get_remap_table() const
{
    auto raw_bin = get_bin_note("remap");
    MemBuf buf(raw_bin.bin, raw_bin.bin_size);
    std::istream stream(&buf);
    RemapcSection remap;
    remap.load(stream);
    return remap;
}

void AIPUELF::set_remap_table(RemapcSection &ro)
{
    std::ostringstream ss;
    ro.write(ss);
    set_note(ss.str(), "remap");
}

SGList AIPUELF::get_subgraphs() const
{
    auto raw_bin = get_bin_note("subgraphs");
    MemBuf buf(raw_bin.bin, raw_bin.bin_size);
    std::istream stream(&buf);
    SGList sglist;
    sglist.load(stream);
    return sglist;
}

void AIPUELF::set_subgraphs(SGList &ro)
{
    std::ostringstream ss;
    ro.write(ss);
    set_note(ss.str(), "subgraphs");
}

GMConfig AIPUELF::get_GMconfig() const
{
    return m_gm;
}
void AIPUELF::set_GMconfig(const GMConfig &ro)
{
    m_gm = ro;
}

ModelGlobalParam AIPUELF::get_globalparam() const
{
    auto raw_bin = get_bin_note("globalparam");
    MemBuf buf(raw_bin.bin, raw_bin.bin_size);
    std::istream stream(&buf);
    ModelGlobalParam param;
    param.load(stream);
    return param;
}
void AIPUELF::set_globalparam(ModelGlobalParam &ro)
{
    std::ostringstream ss;
    ro.write(ss);
    set_note(ss.str(), "globalparam");
}

std::vector<std::vector<uint32_t>> AIPUELF::get_input_shapes() const
{
    auto param = get_input_shape_constraint();
    std::vector<std::vector<uint32_t>> input_shapes;
    for (auto &c : param.constrains)
    {
        input_shapes.push_back(c.shapes);
    }
    return input_shapes;
}
InputShapeConstraint AIPUELF::get_input_shape_constraint() const
{
    auto raw_bin = get_bin_note("inputshapeconstraint");
    MemBuf buf(raw_bin.bin, raw_bin.bin_size);
    std::istream stream(&buf);
    InputShapeConstraint param = {0};
    param.load(stream);
    return param;
}
void AIPUELF::set_input_shape_constraint(InputShapeConstraint &ro)
{
    std::ostringstream ss;
    ro.write(ss);
    set_note(ss.str(), "inputshapeconstraint");
}

std::vector<std::vector<uint32_t>> AIPUELF::get_output_shapes() const
{
    auto param = get_output_tensor_shape();
    std::vector<std::vector<uint32_t>> output_shapes;
    for (auto &s : param.shapes)
    {
        output_shapes.push_back(s.shapes);
    }
    return output_shapes;
}
OutputTensorShape AIPUELF::get_output_tensor_shape() const
{
    auto raw_bin = get_bin_note("outputtensorshape");
    MemBuf buf(raw_bin.bin, raw_bin.bin_size);
    std::istream stream(&buf);
    OutputTensorShape param = {0};
    param.load(stream);
    return param;
}
void AIPUELF::set_output_tensor_shape(OutputTensorShape &ro)
{
    std::ostringstream ss;
    ro.write(ss);
    set_note(ss.str(), "outputtensorshape");
}

ConstantHashTable AIPUELF::get_constant_hash_table() const
{
    auto raw_bin = get_bin_note("constanthashtable");
    MemBuf buf(raw_bin.bin, raw_bin.bin_size);
    std::istream stream(&buf);
    ConstantHashTable table;
    table.load(stream);
    return table;
}
void AIPUELF::set_constant_hash_table(ConstantHashTable &table)
{
    std::ostringstream ss;
    table.write(ss);
    set_note(ss.str(), "constanthashtable");
}

std::string AIPUELF::get_lora_info() const
{
    auto raw_bin = get_bin_note("lorainfo");
    MemBuf buf(raw_bin.bin, raw_bin.bin_size);
    std::istream stream(&buf);
    return std::string(std::istreambuf_iterator<char>(stream), {});
}
void AIPUELF::set_lora_info(const std::string &info)
{
    set_note(info, "lorainfo");
}

void AIPUELF::add_aipu_note()
{
    if (!m_note)
    {
        m_note = m_elf.sections.add(".note.aipu");
        m_note->set_type(SHT_NOTE);
    }
}

void AIPUELF::set_elf_file(std::istream &stream)
{
    m_elf.load(stream);
}

void AIPUELF::set_elf_file(const std::string &filename)
{
    m_elf.load(filename);
}

const std::string &AIPUELF::get_ir() const
{
    return m_ir;
}
void AIPUELF::set_ir(const std::string &ir)
{
    m_ir = ir;
}

const std::string &AIPUELF::get_graph_json() const
{
    return m_graph_json;
}
void AIPUELF::set_graph_json(const std::string &perf_meta)
{
    m_graph_json = perf_meta;
}
const std::string &AIPUELF::get_dump_ir() const
{
    return m_dump_ir;
}
void AIPUELF::set_dump_ir(const std::string &ir)
{
    m_dump_ir = ir;
}

LayerDebugInfos AIPUELF::get_layer_debug_info() const
{
    auto raw_bin = get_bin_note("debuginfo");
    LayerDebugInfos info;
    if (raw_bin.bin_size)
    {
        MemBuf buf(raw_bin.bin, raw_bin.bin_size);
        std::istream stream(&buf);
        info.load(stream);
    }
    return info;
}
void AIPUELF::set_layer_debug_info(const LayerDebugInfos &info)
{
    std::ostringstream ss;
    info.write(ss);
    set_note(ss.str(), "debuginfo");
}

ELFIO::section *AIPUELF::get_elf_section(const std::string &section_name) const
{
    Elf_Half no = m_elf.sections.size();
    for (Elf_Half i = 0; i < no; ++i)
    { // For all sections
        section *sec = m_elf.sections[i];
        if (section_name == sec->get_name())
        {
            return sec;
        }
    }
    return nullptr;
}

uint32_t AIPUELF::get_text_size() const
{
    section *text = get_elf_section(".text");
    if (!text)
    {
        return 0;
    }
    return text->get_size();
}

uint32_t AIPUELF::get_func_size(const std::string &func_name) const
{
    get_symtab();
    if (m_sym_size.count(func_name))
    {
        return m_sym_size.at(func_name);
    }
    section *psec = get_elf_section(".symtab");
    if (!psec)
    {
        return 0;
    }
    // Check section type
    if (psec->get_type() == SHT_SYMTAB)
    {
        const ELFIO::symbol_section_accessor symbols(m_elf, psec);
        for (unsigned int j = 0; j < symbols.get_symbols_num(); ++j)
        {
            std::string name;
            Elf64_Addr value;
            Elf_Xword size = 0;
            unsigned char bind = 0;
            unsigned char type = 0;
            Elf_Half section_index;
            unsigned char other;

            // Read symbol properties
            symbols.get_symbol(j, name, value, size, bind, type,
                               section_index, other);
            if (type == STT_FUNC && bind == STB_GLOBAL)
            {
                if (name == func_name)
                {
                    return size;
                }
            }
        }
    }
    return 0;
}

const std::map<std::string, uint64_t> &AIPUELF::get_symtab() const
{
    if (m_symtab.size())
    {
        return m_symtab;
    }
    section *text = get_elf_section(".text");
    if (!text)
    {
        return m_symtab;
    }
    uint32_t text_offset = text->get_address(); // - text->get_offset();
    section *psec = get_elf_section(".symtab");
    if (!psec)
    {
        return m_symtab;
    }
    // Check section type
    if (psec->get_type() == SHT_SYMTAB)
    {
        const ELFIO::symbol_section_accessor symbols(m_elf, psec);
        for (unsigned int j = 0; j < symbols.get_symbols_num(); ++j)
        {
            std::string name;
            Elf64_Addr value;
            Elf_Xword size;
            unsigned char bind = 0;
            unsigned char type = 0;
            Elf_Half section_index;
            unsigned char other;

            // Read symbol properties
            symbols.get_symbol(j, name, value, size, bind, type,
                               section_index, other);
            if ((type == STT_FUNC && bind == STB_GLOBAL) || name == "_start")
            {
                m_symtab[name] = value - text_offset;
                m_sym_size[name] = size;
            }
        }
    }
    return m_symtab;
}

uint32_t AIPUELF::get_function_offset(const std::string &func_name) const
{
    get_symtab();
    if (m_symtab.count(func_name))
    {
        return m_symtab.at(func_name);
    }

    return 0;
}

} // namespace aipubt
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
 * @Date: 2020-01-29 14:26:09 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-13 17:12:57
 */
#include "builder/aipu_binfile.h"
#include "builder/linker/linker_v1.h"
#include "core/logger.h"
#include "core/tensor.h"
#include <cstddef>
#include <map>
#include <string.h>
#include <string>
#define FORMAT_ADDR(value) "0x" << std::setw(8) << std::setfill('0') << std::hex << value << std::setfill(' ') << std::dec

namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}
static void set_ro_desc_text_offset(AIPUBin &bin, uint32_t offset, uint32_t type)
{
    // change remaping
    for (auto entry : bin.remap.table)
    {
        if (entry.dest_type == type)
        {
            switch (entry.loc_type)
            {
            case RemapEntry::RemapType_RO:
                *(uint32_t *)(bin.ro.bin + entry.loc_offset) = offset + entry.dest_offset;
                break;
            case RemapEntry::RemapType_Descriptor:
                *(uint32_t *)(bin.desc.bin + entry.loc_offset) = offset + entry.dest_offset;
                break;
            case RemapEntry::RemapType_Text:
                *(uint32_t *)(bin.text.bin + entry.loc_offset) = offset + entry.dest_offset;
                break;
            default:
                break;
            }
        }
    }
}

BinSection &AIPUBin::get_text()
{
    return text;
}
BinSection &AIPUBin::get_ro()
{
    return ro;
}
BinSection &AIPUBin::get_desc()
{
    return desc;
}
BinSection &AIPUBin::get_const()
{
    return data;
}

AddressInfo AIPUBin::get_text_info()
{
    return {32, text.bin_size, 0};
}
AddressInfo AIPUBin::get_ro_info()
{
    return {4, ro.bin_size, 0};
}
AddressInfo AIPUBin::get_desc_info()
{
    return {4, desc.bin_size, 0};
}

AddressInfo AIPUBin::get_const_info()
{
    AddressInfo ret;
    ret.align = 32;
    ret.address = 0xffffffff;
    ret.size = data.bin_size;
    for (uint32_t i = 0; i < bss.num_data_descriptor; ++i)
    {
        if (ret.align < bss.data_descriptor[i].align_bytes)
        {
            ret.align = bss.data_descriptor[i].align_bytes;
        }
        auto address = get_const_address(i);
        if (ret.address > address)
        {
            ret.address = address;
        }
    }
    return ret;
}

AddressInfo AIPUBin::get_reuse_info()
{
    AddressInfo ret;
    ret.align = 32;
    ret.size = 0;
    ret.address = 0xffffffff;
    for (uint32_t i = 0; i < bss.num_buffer_descriptor; ++i)
    {
        auto &buf = bss.buffer_descriptor[i];
        ret.align = buf.align_bytes > ret.align ? buf.align_bytes : ret.align;
        ret.size = aligned(ret.size, buf.align_bytes) + buf.size;
        auto address = get_reuse_address(i);
        if (ret.address > address)
        {
            ret.address = address;
        }
    }
    return ret;
}
AddressInfo AIPUBin::get_stack_info()
{
    return {bss.stack_align_bytes, bss.stack_size, 0};
}

uint32_t AIPUBin::get_num_const()
{
    return bss.num_data_descriptor;
}
uint32_t AIPUBin::get_num_reuse()
{
    return bss.num_buffer_descriptor;
}
uint32_t AIPUBin::get_num_input()
{
    return get_num_typed_buffer(0);
}
uint32_t AIPUBin::get_num_output()
{
    return get_num_typed_buffer(1);
}

AddressInfo AIPUBin::get_input_info(uint32_t index)
{
    AddressInfo ret;
    for (size_t i = 0; i < get_num_reuse(); ++i)
    {
        auto &desc = bss.buffer_descriptor[i];
        for (size_t j = 0; j < desc.sub_section_cnt; ++j)
        {
            auto &subdesc = desc.sub_sections[j];
            if (subdesc.type == 0 && subdesc.id == index && subdesc.offset_in_ro_cnt != 0)
            {
                ret.align = desc.align_bytes;
                ret.size = subdesc.size;
                ret.address = ((uint32_t *)(ro.bin))[subdesc.offset_in_ro[0]];
                break;
            }
        }
    }
    return ret;
}

std::vector<AddressInfo> AIPUBin::get_inputs_info()
{
    std::vector<AddressInfo> ret(1);
    for (size_t i = 0; i < get_num_reuse(); ++i)
    {
        auto &desc = bss.buffer_descriptor[i];
        for (size_t j = 0; j < desc.sub_section_cnt; ++j)
        {
            auto &subdesc = desc.sub_sections[j];
            if (subdesc.type == 0 && subdesc.offset_in_ro_cnt != 0)
            {
                if (ret.size() <= subdesc.id)
                {
                    ret.resize(subdesc.id + 1);
                }
                AddressInfo &info = ret[subdesc.id];
                info.align = desc.align_bytes;
                info.size = subdesc.size;
                info.address = *(uint32_t *)(ro.bin + subdesc.offset_in_ro[0]);
            }
        }
    }
    return ret;
}

AddressInfo AIPUBin::get_output_info(uint32_t index)
{
    AddressInfo ret;
    for (size_t i = 0; i < get_num_reuse(); ++i)
    {
        auto &desc = bss.buffer_descriptor[i];
        for (size_t j = 0; j < desc.sub_section_cnt; ++j)
        {
            auto &subdesc = desc.sub_sections[j];
            if (subdesc.type == 1 && subdesc.id == index && subdesc.offset_in_ro_cnt != 0)
            {
                ret.align = desc.align_bytes;
                ret.size = subdesc.size;
                ret.address = *(uint32_t *)(ro.bin + subdesc.offset_in_ro[0]);
                break;
            }
        }
    }
    return ret;
}

std::vector<AddressInfo> AIPUBin::get_outputs_info()
{
    std::vector<AddressInfo> ret(1);
    for (size_t i = 0; i < get_num_reuse(); ++i)
    {
        auto &desc = bss.buffer_descriptor[i];
        for (size_t j = 0; j < desc.sub_section_cnt; ++j)
        {
            auto &subdesc = desc.sub_sections[j];
            if (subdesc.type == 1 && subdesc.offset_in_ro_cnt != 0)
            {
                if (ret.size() <= subdesc.id)
                {
                    ret.resize(subdesc.id + 1);
                }
                AddressInfo &info = ret[subdesc.id];
                info.align = desc.align_bytes;
                info.size = subdesc.size;
                info.address = *(uint32_t *)(ro.bin + subdesc.offset_in_ro[0]);
            }
        }
    }
    return ret;
}

bool AIPUBin::set_reuse_address(uint32_t addr)
{
    if (addr & (get_reuse_info().align - 1))
    {
        return false;
    }
    bool ret = true;
    for (uint32_t i = 0; i < bss.num_buffer_descriptor; ++i)
    {
        addr = aligned(addr, bss.buffer_descriptor[i].align_bytes);
        ret = ret && set_reuse_address(i, addr);
        addr += bss.buffer_descriptor[i].size;
    }
    return ret;
}

bool AIPUBin::set_const_address(uint32_t addr)
{
    if (addr & (get_const_info().align - 1))
    {
        return false;
    }
    bool ret = true;
    for (uint32_t i = 0; i < bss.num_data_descriptor; ++i)
    {
        ret = ret && set_const_address(i, addr + bss.data_descriptor[i].data_offset);
    }
    return ret;
}

bool AIPUBin::set_ro_address(uint32_t addr)
{
    if (addr & (get_ro_info().align - 1))
    {
        return false;
    }
    set_ro_desc_text_offset(*this, addr, RemapEntry::RemapType_RO);
    return true;
}

bool AIPUBin::set_desc_address(uint32_t addr)
{
    if (addr & (get_desc_info().align - 1))
    {
        return false;
    }
    set_ro_desc_text_offset(*this, addr, RemapEntry::RemapType_Descriptor);
    return true;
}

bool AIPUBin::set_text_address(uint32_t addr)
{
    if (addr & (get_text_info().align - 1))
    {
        return false;
    }
    set_ro_desc_text_offset(*this, addr, RemapEntry::RemapType_Text);
    return true;
}

AddressInfo AIPUBin::get_reuse_info(uint32_t index)
{
    if (index >= bss.num_buffer_descriptor)
    {
        return {0, 0, 0};
    }
    auto &buf = bss.buffer_descriptor[index];
    AddressInfo info;
    info.size = buf.size;
    info.align = buf.align_bytes;
    info.address = get_reuse_address(index);
    return info;
}

AddressInfo AIPUBin::get_const_info(uint32_t index)
{
    if (index >= bss.num_data_descriptor)
    {
        return {0, 0, 0};
    }
    auto &buf = bss.data_descriptor[index];
    AddressInfo info;
    info.size = buf.size;
    info.align = buf.align_bytes;
    info.address = get_const_address(index);
    return info;
}

bool AIPUBin::set_reuse_address(uint32_t index, uint32_t address)
{
    // offset not align
    if (index >= bss.num_buffer_descriptor)
    {
        return false;
    }
    else if ((address & (bss.buffer_descriptor[index].align_bytes - 1)))
    {
        return false;
    }
    auto &buffer = bss.buffer_descriptor[index];
    for (size_t i = 0; i < buffer.sub_sections.size(); ++i)
    {
        for (size_t j = 0; j < buffer.sub_sections[i].offset_in_ro.size(); ++j)
        {
            auto ro_offset = buffer.sub_sections[i].offset_in_ro[j];
            auto ptr = ro.bin;
            if (ro_offset >= ro.bin_size)
            {
                ptr = desc.bin;
                ro_offset -= ro.bin_size;
            }
            *(uint32_t *)(ptr + ro_offset) = ((*(uint32_t *)(ptr + ro_offset)) & ~buffer.sub_sections[i].addr_mask) | ((address + buffer.sub_sections[i].relative_offset) & (buffer.sub_sections[i].addr_mask));
        }
    }
    return true;
}

bool AIPUBin::set_const_address(uint32_t index, uint32_t address)
{
    // offset not align
    if (index >= bss.num_data_descriptor)
    {
        return false;
    }
    else if ((address & (bss.data_descriptor[index].align_bytes - 1)))
    {
        return false;
    }
    auto &buffer = bss.data_descriptor[index];
    for (size_t i = 0; i < buffer.sub_sections.size(); ++i)
    {
        for (size_t j = 0; j < buffer.sub_sections[i].offset_in_ro.size(); ++j)
        {
            auto ro_offset = buffer.sub_sections[i].offset_in_ro[j];
            auto ptr = ro.bin;
            if (ro_offset >= ro.bin_size)
            {
                ptr = desc.bin;
                ro_offset -= ro.bin_size;
            }
            *(uint32_t *)(ptr + ro_offset) = ((*(uint32_t *)(ptr + ro_offset)) & (~buffer.sub_sections[i].addr_mask)) |
                                             ((address + buffer.sub_sections[i].relative_offset) & (buffer.sub_sections[i].addr_mask));
        }
    }
    return true;
}

uint32_t AIPUBin::get_num_typed_buffer(uint32_t type)
{
    uint32_t num_inp = 0;
    for (size_t i = 0; i < get_num_reuse(); ++i)
    {
        auto &desc = bss.buffer_descriptor[i];
        for (size_t j = 0; j < desc.sub_section_cnt; ++j)
        {
            auto &subdesc = desc.sub_sections[j];
            if (subdesc.type == type)
            {
                num_inp++;
            }
        }
    }
    return num_inp;
}

std::vector<uint32_t> AIPUBin::get_typed_buffer(uint32_t type)
{
    std::vector<uint32_t> ret;
    for (size_t i = 0; i < get_num_reuse(); ++i)
    {
        auto &desc = bss.buffer_descriptor[i];
        for (size_t j = 0; j < desc.sub_section_cnt; ++j)
        {
            auto &subdesc = desc.sub_sections[j];
            if (subdesc.type == type)
            {
                ret.push_back(i);
                break;
            }
        }
    }
    return ret;
}

const BssBufferDescriptor &AIPUBin::get_reuse_buffer(uint32_t index)
{
    if (index >= bss.num_buffer_descriptor)
    {
        static BssBufferDescriptor zero;
        return zero;
    }
    return bss.buffer_descriptor[index];
}

const BssDataDescriptor &AIPUBin::get_const_buffer(uint32_t index)
{
    if (index >= bss.num_data_descriptor)
    {
        static BssDataDescriptor zero;
        return zero;
    }
    return bss.data_descriptor[index];
}

uint32_t AIPUBin::get_reuse_address(uint32_t index)
{
    if (index >= bss.num_buffer_descriptor)
    {
        return 0;
    }
    auto &buf = bss.buffer_descriptor[index];
    if (buf.sub_section_cnt > 0)
    {
        for (uint32_t i = 0; i < buf.sub_section_cnt; i++)
        {
            auto &sub = buf.sub_sections[i];
            if (sub.offset_in_ro_cnt > 0)
            {
                auto ro_offset = sub.offset_in_ro[0];
                auto ptr = ro.bin + ro_offset;
                if (ro_offset >= ro.bin_size)
                {
                    ptr = desc.bin + ro_offset - ro.bin_size;
                }
                return (*(uint32_t *)ptr) - sub.relative_offset;
            }
        }
    }
    return 0;
}

uint32_t AIPUBin::get_const_address(uint32_t index)
{
    if (index >= bss.num_data_descriptor)
    {
        return 0;
    }
    auto &buf = bss.data_descriptor[index];
    if (buf.sub_section_cnt > 0)
    {
        for (uint32_t i = 0; i < buf.sub_section_cnt; i++)
        {
            auto &sub = buf.sub_sections[i];
            if (sub.offset_in_ro_cnt > 0)
            {
                auto ro_offset = sub.offset_in_ro[0];
                auto ptr = ro.bin + ro_offset;
                if (ro_offset >= ro.bin_size)
                {
                    ptr = desc.bin + ro_offset - ro.bin_size;
                }
                return (*(uint32_t *)ptr) - sub.relative_offset;
            }
        }
    }
    return 0;
}

char *AIPUBin::get_const_data(uint32_t index)
{
    if (index >= bss.num_data_descriptor)
    {
        return nullptr;
    }
    return data.bin + bss.data_descriptor[index].data_offset;
}

bool AIPUBin::add_extra_data(BinSection data)
{
    extra_data.push_back(data);
    return true;
}

std::vector<BinSection> AIPUBin::get_extra_data()
{
    return extra_data;
}

std::string bss_type_str(uint32_t type)
{
    static std::map<uint32_t, std::string> strmap = {
        {0, "input"},
        {1, "output"},
        {2, "dump"},
        {3, "inter"},
        {4, "weight"},
        {5, "bias"},
        {6, "scale"},
        {7, "shift"},
        {8, "LUT"},
        {9, "negative"},
        {10, "profiler"},
        {12, "profiler_fifo"},
        {13, "op_cnt"},
        {14, "error_code"},
        {15, "zcpy_const"},
        {16, "out_shape"},
        {17, "submodel"},
        {18, "dump_shape"},
        {20, "breakdump"},
    };
    if (strmap.count(type))
        return strmap[type];
    return "unkown:" + std::to_string(type);
}

void AIPUBin::write(std::ostream &ofs)
{
    ofs.seekp(sizeof(header), std::ios::beg);
    header.text_offset = ofs.tellp();
    header.text_size = text.bin_size;
    ofs.write(text.bin, text.bin_size);

    header.rodata_offset = ofs.tellp();
    header.rodata_size = ro.bin_size;
    ofs.write(ro.bin, ro.bin_size);

    header.dcr_offset = ofs.tellp();
    header.dcr_size = desc.bin_size;
    ofs.write(desc.bin, desc.bin_size);

    header.data_offset = ofs.tellp();
    header.data_size = data.bin_size;
    ofs.write(data.bin, data.bin_size);

    header.bss_offset = ofs.tellp();
    bss.write(ofs, (header.version >> 24));
    header.bss_size = static_cast<int>(ofs.tellp()) - header.bss_offset;
    ofs.seekp(0, std::ios::end);
    // write remapping section
    remap.write(ofs);
    ofs.seekp(0, std::ios::end);
    header.filesize = ofs.tellp();
    ofs.seekp(0, std::ios::beg);
    header.write(ofs);
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))

static inline void update_offset_in_ro(AIPUBin &out, size_t ro_size1, size_t ro_size2, int &inp_id, int &out_id)
{
    for (auto &&sec : out.bss.data_descriptor)
    {
        for (auto &&sub : sec.sub_sections)
        {
            for (size_t i = 0; i < sub.offset_in_ro.size(); ++i)
            {
                if (sub.offset_in_ro[i] > ro_size1)
                {
                    sub.offset_in_ro[i] += ro_size2;
                }
            }
        }
    }
    for (auto &&sec : out.bss.buffer_descriptor)
    {
        for (auto &&sub : sec.sub_sections)
        {
            if (sub.type == 0)
            {
                inp_id++;
            }
            else if (sub.type == 1)
            {
                out_id++;
            }
            for (size_t i = 0; i < sub.offset_in_ro.size(); ++i)
            {
                if (sub.offset_in_ro[i] > ro_size1)
                {
                    sub.offset_in_ro[i] += ro_size2;
                }
            }
        }
    }
}

static BinSection merge_bin_sec(const BinSection &bin1, const BinSection &bin2)
{
    BinSection out;
    out.bin_size = bin1.bin_size + bin2.bin_size;
    out.bin = new char[out.bin_size];
    memcpy(out.bin, bin1.bin, bin1.bin_size);
    memcpy(out.bin + bin1.bin_size, bin2.bin, bin2.bin_size);
    return out;
}
bool AIPUBin::merge(AIPUBin &out, const AIPUBin &bin1, const AIPUBin &bin2)
{
    //the arch and graph version must be same
    if (bin1.header.arch != bin2.header.arch || bin1.header.version != bin2.header.version)
    {
        return false;
    }
    uint32_t entry1 = bin1.header.type, entry2 = bin2.header.type, new_entry = 0;
    if (entry1 == 0 || entry2 == 0)
    {
        DEBUG() << "Cannot merge aipu.bin due to not find entry point.";
        return false;
    }
    // merge header
    out = bin1;
    out.text = LinkerV1::merge_text(bin1.text, bin2.text, entry1, entry2, new_entry);
    out.header.type = new_entry;
    out.ro = merge_bin_sec(bin1.ro, bin2.ro);
    out.desc = merge_bin_sec(bin1.desc, bin2.desc);
    out.data = merge_bin_sec(bin1.data, bin2.data);
    // update bss
    out.bss.stack_size = MAX(bin1.bss.stack_size, bin2.bss.stack_size);
    out.bss.stack_align_bytes = MAX(bin1.bss.stack_align_bytes, bin2.bss.stack_align_bytes);
    const auto data_offset = bin1.data.bin_size;
    const auto ro_size2 = bin2.ro.bin_size;
    const auto ro_size1 = bin1.ro.bin_size;
    const auto desc1 = bin1.desc.bin_size;
    int inp_id = 0;
    int out_id = 0;
    update_offset_in_ro(out, ro_size1, ro_size2, inp_id, out_id);
    for (auto sec : bin2.bss.data_descriptor)
    {
        sec.data_offset += data_offset;
        for (auto &&sub : sec.sub_sections)
        {
            for (size_t i = 0; i < sub.offset_in_ro.size(); ++i)
            {
                if (sub.offset_in_ro[i] > ro_size2)
                {
                    sub.offset_in_ro[i] += desc1;
                }
                sub.offset_in_ro[i] += ro_size1;
            }
        }
        out.bss.add_data_desc(sec);
    }
    for (auto sec : bin2.bss.buffer_descriptor)
    {
        for (auto &&sub : sec.sub_sections)
        {
            if (sub.type == 0)
            {
                sub.id += inp_id;
            }
            else if (sub.type == 1)
            {
                sub.id += out_id;
            }
            for (size_t i = 0; i < sub.offset_in_ro.size(); ++i)
            {
                if (sub.offset_in_ro[i] > ro_size2)
                {
                    sub.offset_in_ro[i] += desc1;
                }
                sub.offset_in_ro[i] += ro_size1;
            }
        }
        out.bss.add_buff_desc(sec);
    }

    //update remap
    for (auto entry : bin2.remap.table)
    {
        // runtime will do: base[loc_type]+loc_offset <= base[dest_type]+dest_offset
        if (entry.loc_type == RemapcSection::TEXT)
        {
            entry.loc_offset += bin1.text.bin_size;
        }
        else if (entry.loc_type == RemapcSection::RO)
        {
            entry.loc_offset += ro_size1;
        }
        else if (entry.loc_type == RemapcSection::DESC)
        {
            entry.loc_offset += desc1;
        }
        if (entry.dest_type == RemapcSection::TEXT)
        {
            entry.dest_offset += bin1.text.bin_size;
        }
        else if (entry.dest_type == RemapcSection::RO)
        {
            entry.dest_offset += ro_size1;
        }
        else if (entry.dest_type == RemapcSection::DESC)
        {
            entry.dest_offset += desc1;
        }
        out.remap.table.push_back(entry);
        out.remap.entry_num = out.remap.table.size();
    }

    return true;
}

std::ostream &operator<<(std::ostream &os, const BssSection &bss)
{
    os << "Stack size: 0x" << std::hex << bss.stack_size
       << " aligned in 0x" << bss.stack_align_bytes
       << " bytes.\n";
    os << "Static buffer number: " << std::dec << bss.num_data_descriptor << "\n";
    os << "Static buffer remapping:\n";
    os << "=========================================================================\n";

    os << std::setw(6) << "idx"
       << "\t"
       << std::setw(10) << "size"
       << "\t"
       << std::setw(10) << "offset"
       << "\t"
       << std::setw(10) << "aligned"
       << "\t"
       << std::setw(6) << "tensor num\n";
    os << "\t" << std::setw(6) << "idx"
       << "\t"
       << std::setw(10) << "type"
       << "\t"
       << std::setw(10) << "dtype"
       << "\t"
       << std::setw(10) << "size"
       << "\t"
       << std::setw(10) << "rela_off"
       << "\t"
       << std::setw(6) << "id"
       << "\t"
       << std::setw(10) << "mask"
       << "\t"
       << std::setw(8) << "remapping locs"
       << "\n";
    for (uint32_t i = 0; i < bss.num_data_descriptor; ++i)
    {
        auto &desc = bss.data_descriptor[i];
        os << std::setw(6) << i << "\t"
           << FORMAT_ADDR(desc.size) << "\t"
           << FORMAT_ADDR(desc.data_offset) << "\t"
           << FORMAT_ADDR(desc.align_bytes) << "\t" << std::dec
           << std::setw(6) << desc.sub_section_cnt << "\n";
        os << "\t-------------------------------------------------------------------------\n";

        for (uint32_t j = 0; j < desc.sub_section_cnt; ++j)
        {
            auto &sub = desc.sub_sections[j];
            os << "\t" << std::setw(6) << j
               << "\t" << std::setw(10) << bss_type_str(sub.type)
               << "\t" << std::setw(10) << (Tensor::Dtype)(sub.dtype)
               << "\t" << FORMAT_ADDR(sub.size)
               << "\t" << FORMAT_ADDR(sub.relative_offset)
               << "\t" << std::setw(6) << sub.id
               << "\t" << FORMAT_ADDR(sub.addr_mask) << "\t";
            for (uint32_t k = 0; k < sub.offset_in_ro_cnt; ++k)
            {
                if (k != 0)
                {
                    os << ",";
                }
                os << std::hex << "0x" << sub.offset_in_ro[k];
            }
            os << std::dec << "\n";
        }
        os << "\t-------------------------------------------------------------------------\n";
    }
    os << "\n\n";

    os << "Reuse buffer number: " << std::dec << bss.num_buffer_descriptor << "\n";
    os << "Reuse buffer remapping:\n";
    os << "=========================================================================\n";

    os << std::setw(6) << "idx"
       << "\t"
       << std::setw(10) << "size"
       << "\t"
       << std::setw(10) << "aligned"
       << "\t"
       << std::setw(6) << "tensor num\n";
    os << "\t" << std::setw(6) << "idx"
       << "\t"
       << std::setw(10) << "type"
       << "\t"
       << std::setw(10) << "dtype"
       << "\t"
       << std::setw(10) << "size"
       << "\t"
       << std::setw(10) << "rela_off"
       << "\t"
       << std::setw(6) << "id"
       << "\t"
       << std::setw(10) << "mask"
       << "\t"
       << std::setw(8) << "remapping locs"
       << "\n";
    for (uint32_t i = 0; i < bss.num_buffer_descriptor; ++i)
    {
        auto &desc = bss.buffer_descriptor[i];
        os << std::setw(6) << i << "\t"
           << FORMAT_ADDR(desc.size) << "\t"
           << FORMAT_ADDR(desc.align_bytes) << "\t" << std::dec
           << std::setw(6) << desc.sub_section_cnt << "\n";
        os << "\t-------------------------------------------------------------------------\n";

        for (uint32_t j = 0; j < desc.sub_section_cnt; ++j)
        {
            auto &sub = desc.sub_sections[j];
            os << "\t" << std::setw(6) << j
               << "\t" << std::setw(10) << bss_type_str(sub.type)
               << "\t" << std::setw(10) << (Tensor::Dtype)(sub.dtype)
               << "\t" << FORMAT_ADDR(sub.size)
               << "\t" << FORMAT_ADDR(sub.relative_offset)
               << "\t" << std::setw(6) << sub.id
               << "\t" << FORMAT_ADDR(sub.addr_mask) << "\t";
            for (uint32_t k = 0; k < sub.offset_in_ro_cnt; ++k)
            {
                if (k != 0)
                {
                    os << ",";
                }
                os << std::hex << "0x" << sub.offset_in_ro[k];
            }
            os << std::dec << "\n";
        }
        os << "\t-------------------------------------------------------------------------\n";
    }
    os << "\n\n";
    return os;
}

}; //namespace aipubt
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
 * @Date: 2020-01-29 14:26:13
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-13 16:36:46
 */
#ifndef BUILDER_AIPU_BINFILE_H
#define BUILDER_AIPU_BINFILE_H

#include <core/buildtool_version.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <vector>

#define AIPUBIN_FILE_MAGIC_NUM "AIPU BIN"
// BUILD_VERSION IS DIFFERENT FROM BUILDTOOL_VERSION
// #define MAJOR_VERSION 2
// #define MINOR_VERSION 0
// #define BUILD_NUMBER 3000
// #define BUILD_VERSION ((MAJOR_VERSION & 0xFF) | (MINOR_VERSION & 0xFF) << 8 | (BUILD_NUMBER & 0xFFFF) << 16)

#ifndef GRAPH_VERSION
#define GRAPH_VERSION 5
#endif

#define STACK_SIZE (256 << 10)
#define STACK_MARGIN (1 << 10)

#define TOTAL_STACK_SIZE (STACK_SIZE + STACK_MARGIN)
namespace aipubt
{
struct ABHeader
{
    char ident[16] = AIPUBIN_FILE_MAGIC_NUM;
    uint32_t arch = 0;
    uint32_t version = (GRAPH_VERSION << 24);
    uint32_t build_version = AIPUBIN_BT_VERSION_NUMBER;
    uint32_t hdrsize = sizeof(ABHeader);
    uint32_t filesize;
    uint32_t type = 0; // currently use this field to store the main function offset
    /*
    The Flag
    [6] dynamic shape
        0: disable dynamic shape
        1: enable dynamic shape
    [5] preallocate soc SRAM
        0: disable preallocated soc SRAM
        1: enable preallocated soc SRAM
    [4] linked list description flag
    •	0: no such description in BSS;
    •	1: has such description in BSS;
    [3:0] ASID flag
    •	0: ASID disabled;
    •	>=1: ASID feature (TBD)
    */
    uint32_t flag = 0;
    uint32_t entry = 0;
    uint32_t text_offset;
    uint32_t text_size;
    uint32_t rodata_offset;
    uint32_t rodata_size;
    uint32_t dcr_offset = 0;
    uint32_t dcr_size = 0;
    uint32_t data_offset;
    uint32_t data_size;
    uint32_t bss_offset;
    uint32_t bss_size;
    uint32_t misc_offset = 0;
    uint32_t misc_size = 0;
    int8_t extra_data[8] = {0};
    void load(std::istream &stream)
    {
        stream.read((char *)this, sizeof(ABHeader));
    }
    void write(std::ostream &stream)
    {
        stream.write((char *)this, sizeof(ABHeader));
    }
};

struct BssSubSectionDescriptor
{
    enum ASID
    {
        ASID_0 = 0,
        ASID_1 = 1,
        ASID_2 = 2,
        ASID_3 = 3,
    };
    enum SubSectionType
    {
        input = 0,
        output = 1,
        dump = 2,
        intermediate = 3,
        constant = 4,
        profile_data = 10,
        dump_instruction = 11,
        printf_fifo = 12,
        op_cnt = 13,
        error_code = 14,

        zerocpy_constant = 15,
        /**
         * @brief  the buffer is store the real output tensor shape for dynamic shape
         * data stored in uint32_t, and we can get the dimension by dim = size/4.
         * for example, a buffer size=16, then it's 4 dim tensor,
         * and the value=[0x0000_0001,0x0000_0002,0x0000_0003, 0x0000_0004]  indicates that the shape is  [1,2,3,4]
         */
        output_tensor_shape = 16,

        cross_submodel = 17,

        dump_tensor_shape = 18,

        segmmu_buffer = 255, // for X2 or higher segmmu
        /*
        for segmmu type, the id has following definition:
        ID field	meaning
        16:31	Identify  core0-core15 use this buffer, for example 0b11 means, core0 and core1 share this buffer for segmmu.
        8:15	Identify which segment [0-3] are available, [4-255] are reserved.
        0:7	Identify which control [0-1] are available,  value [2,255] are reserved.

        and addr_mask is also used as mask for rewrite segmmu controls.
        */
    };
    uint32_t relative_offset = 0; //sub-section offset in the section containing it
    uint32_t size = 1;
    uint32_t type = 3 | ASID_0 << 24; //0: input; 1: output; 2: saved/dump; 3: intermediate; 4: weight; 5: bias; 6: scales; 7: shifts; 8: LUT; 9: neg_slope; 10: profile_data; 11: dump_instruction
    uint32_t id = 0;                  //used for identifying different inputs/outputs
    uint32_t dtype = 0;               //0: None; 1: bool; 2: uint8; 3: int8; 4: uint16; 5: int16; 6: uint32; 7: int32; 8: uint64; 9: int64; 10: float16; 11: float32; 12: float64;
    float scale = 1.;                 //0: None; 1: NHWC; 2: NCHW
    int32_t zp = 0.;                  //0: None; 1: NHWC; 2: NCHW
    uint32_t reserve[3] = {0};
    uint32_t addr_mask = 0xFFFFFFFF;
    uint32_t offset_in_ro_cnt = 0;
    std::vector<uint32_t> offset_in_ro; // offset where stores absolute addr of this sub-section, base is data addr register 0

    uint32_t file_size()
    {
        uint32_t s = sizeof(uint32_t) * 12 + sizeof(uint32_t) * offset_in_ro.size();
        return s;
    }
    void load(std::istream &stream)
    {
        stream.read((char *)&relative_offset, sizeof(uint32_t));
        stream.read((char *)&size, sizeof(uint32_t));
        stream.read((char *)&type, sizeof(uint32_t));
        stream.read((char *)&id, sizeof(uint32_t));
        stream.read((char *)&dtype, sizeof(uint32_t));
        stream.read((char *)&scale, sizeof(uint32_t));
        stream.read((char *)&zp, sizeof(uint32_t));
        stream.read((char *)&reserve, sizeof(uint32_t) * 3);
        stream.read((char *)&addr_mask, sizeof(uint32_t));
        stream.read((char *)&offset_in_ro_cnt, sizeof(uint32_t));
        offset_in_ro.resize(offset_in_ro_cnt);
        for (uint32_t i = 0; i < offset_in_ro_cnt; ++i)
        {
            stream.read((char *)&(offset_in_ro[i]), sizeof(uint32_t));
        }
    }
    void write(std::ostream &stream)
    {
        stream.write((char *)&relative_offset, sizeof(uint32_t));
        stream.write((char *)&size, sizeof(uint32_t));
        stream.write((char *)&type, sizeof(uint32_t));
        stream.write((char *)&id, sizeof(uint32_t));
        stream.write((char *)&dtype, sizeof(uint32_t));
        stream.write((char *)&scale, sizeof(uint32_t));
        stream.write((char *)&zp, sizeof(uint32_t));
        stream.write((char *)&reserve, sizeof(uint32_t) * 3);
        stream.write((char *)&addr_mask, sizeof(uint32_t));
        stream.write((char *)&offset_in_ro_cnt, sizeof(uint32_t));
        for (uint32_t i = 0; i < offset_in_ro_cnt; ++i)
        {
            stream.write((char *)&(offset_in_ro[i]), sizeof(uint32_t));
        }
    }
    void add_ro_offset(uint32_t off)
    {
        offset_in_ro.push_back(off);
        offset_in_ro_cnt = offset_in_ro.size();
    }
};

struct BssDataDescriptor
{                             //for static data: weight/bias
    uint32_t data_offset = 0; //static data offset in data section, base is the data section in this segment
    uint32_t size = 0;
    uint32_t align_bytes = 16;
    uint32_t sub_section_cnt = 0;
    std::vector<BssSubSectionDescriptor> sub_sections;
    uint32_t file_size()
    {
        uint32_t s = sizeof(uint32_t) * 4;
        for (auto &&d : sub_sections)
        {
            s += d.file_size();
        }
        return s;
    }
    void load(std::istream &stream)
    {
        stream.read((char *)&data_offset, sizeof(uint32_t));
        stream.read((char *)&size, sizeof(uint32_t));
        stream.read((char *)&align_bytes, sizeof(uint32_t));
        stream.read((char *)&sub_section_cnt, sizeof(uint32_t));
        sub_sections.resize(sub_section_cnt);
        for (uint32_t i = 0; i < sub_section_cnt; ++i)
        {
            sub_sections[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        stream.write((char *)&data_offset, sizeof(uint32_t));
        stream.write((char *)&size, sizeof(uint32_t));
        stream.write((char *)&align_bytes, sizeof(uint32_t));
        stream.write((char *)&sub_section_cnt, sizeof(uint32_t));
        for (uint32_t i = 0; i < sub_section_cnt; ++i)
        {
            sub_sections[i].write(stream);
        }
    }
    void add_subsection(const BssSubSectionDescriptor &sub)
    {
        sub_sections.push_back(sub);
        sub_section_cnt = sub_sections.size();
    }
};

struct BssBufferDescriptor
{
    uint32_t size = 0;
    uint32_t align_bytes = 16;
    int32_t buf_type = -1; // for input/output monopolize. Valid for version >= 5
    uint32_t sub_section_cnt = 0;
    std::vector<BssSubSectionDescriptor> sub_sections;
    uint32_t file_size()
    {
        uint32_t s = sizeof(uint32_t) * 4;
        for (auto &&d : sub_sections)
        {
            s += d.file_size();
        }
        return s;
    }
    bool has_typed_tensor(uint32_t t) const
    {
        for (auto sub : sub_sections)
        {
            if (sub.type == t)
            {
                return true;
            }
        }
        return false;
    }

    void load(std::istream &stream, uint32_t version = GRAPH_VERSION)
    {
        stream.read((char *)&size, sizeof(uint32_t));
        stream.read((char *)&align_bytes, sizeof(uint32_t));
        if (version >= 5)
        {
            stream.read((char *)&buf_type, sizeof(uint32_t));
        }
        stream.read((char *)&sub_section_cnt, sizeof(uint32_t));
        sub_sections.resize(sub_section_cnt);
        for (uint32_t i = 0; i < sub_section_cnt; ++i)
        {
            sub_sections[i].load(stream);
        }
    }
    void write(std::ostream &stream, uint32_t version = GRAPH_VERSION)
    {
        stream.write((char *)&size, sizeof(uint32_t));
        stream.write((char *)&align_bytes, sizeof(uint32_t));
        if (version >= 5)
        {
            stream.write((char *)&buf_type, sizeof(uint32_t));
            ;
        }
        stream.write((char *)&sub_section_cnt, sizeof(uint32_t));
        for (uint32_t i = 0; i < sub_section_cnt; ++i)
        {
            sub_sections[i].write(stream);
        }
    }
    void add_subsection(const BssSubSectionDescriptor &sub)
    {
        if (sub_sections.size() == 0)
        {
            buf_type = sub.type;
        }
        else if (buf_type != (int32_t)sub.type)
        {
            buf_type = -1;
        }
        sub_sections.push_back(sub);
        sub_section_cnt = sub_sections.size();
    }
};

struct BssSection
{
    uint32_t stack_size = 0;
    uint32_t stack_align_bytes = 16;
    uint32_t num_data_descriptor = 0;
    uint32_t num_buffer_descriptor = 0;
    std::vector<BssDataDescriptor> data_descriptor;
    std::vector<BssBufferDescriptor> buffer_descriptor;
    uint32_t size()
    {
        uint32_t s = sizeof(uint32_t) * 4;
        for (auto &&d : data_descriptor)
        {
            s += d.file_size();
        }
        for (auto &&d : buffer_descriptor)
        {
            s += d.file_size();
        }
        return s;
    }
    void load(std::istream &stream, uint32_t version = GRAPH_VERSION)
    {
        stream.read((char *)&stack_size, sizeof(uint32_t));
        stream.read((char *)&stack_align_bytes, sizeof(uint32_t));
        stream.read((char *)&num_data_descriptor, sizeof(uint32_t));
        stream.read((char *)&num_buffer_descriptor, sizeof(uint32_t));
        data_descriptor.resize(num_data_descriptor);
        buffer_descriptor.resize(num_buffer_descriptor);
        for (uint32_t i = 0; i < num_data_descriptor; ++i)
        {
            data_descriptor[i].load(stream);
        }
        for (uint32_t i = 0; i < num_buffer_descriptor; ++i)
        {
            buffer_descriptor[i].load(stream, version);
        }
    }
    void write(std::ostream &stream, uint32_t version = GRAPH_VERSION)
    {
        stream.write((char *)&stack_size, sizeof(uint32_t));
        stream.write((char *)&stack_align_bytes, sizeof(uint32_t));
        stream.write((char *)&num_data_descriptor, sizeof(uint32_t));
        stream.write((char *)&num_buffer_descriptor, sizeof(uint32_t));
        for (uint32_t i = 0; i < num_data_descriptor; ++i)
        {
            data_descriptor[i].write(stream);
        }
        for (uint32_t i = 0; i < num_buffer_descriptor; ++i)
        {
            buffer_descriptor[i].write(stream, version);
        }
    }
    void add_data_desc(const BssDataDescriptor &desc)
    {
        data_descriptor.push_back(desc);
        num_data_descriptor = data_descriptor.size();
    }
    void add_buff_desc(const BssBufferDescriptor &desc)
    {
        buffer_descriptor.push_back(desc);
        num_buffer_descriptor = buffer_descriptor.size();
    }
};

std::ostream &operator<<(std::ostream &os, const BssSection &bss);
struct BinSection
{
    char *bin = 0;
    uint32_t bin_size = 0;
    std::shared_ptr<bool> _ref_flag = std::make_shared<bool>();
    ~BinSection()
    {
        if (_ref_flag.use_count() <= 1 && bin)
        {
            delete[] bin;
            bin = nullptr;
        }
    }
    void load(std::istream &stream)
    {
        if (_ref_flag.use_count() <= 1 && bin)
        {
            delete[] bin;
            bin = nullptr;
        }
        bin = new char[bin_size];
        _ref_flag = std::make_shared<bool>();
        stream.read(bin, bin_size);
    }
    void write(std::ostream &stream)
    {
        if (bin)
        {
            stream.write(bin, bin_size);
        }
    }
};

struct RemapEntry
{
    /*
    member	type	size	description
type	uInt32	4	Type of a section where this node stores in
•	0: RO data section
•	1: Descriptor section
•	2: Text section
next_addr_entry_offset	uInt32	4	This field stores an offset (after base address of a section specified by type) where absolute address of next node should be filled into.
next_type	uInt32	4	Type of a section where next node stores in (supported values are the same as type).
next_offset	uInt32	4	This field stores an offset (after base address of a section specified by next_type) pointed to an address where stores the next node.

    */
    // runtime will do: base[loc_type]+loc_offset <= base[dest_type]+dest_offset
    enum RemapSecType
    {
        RemapType_RO = 0,
        RemapType_Descriptor = 1,
        RemapType_Text = 2,
    };
    uint32_t loc_type;
    uint32_t loc_offset;
    uint32_t dest_type;
    uint32_t dest_offset;
    void load(std::istream &stream)
    {
        stream.read((char *)&loc_type, sizeof(uint32_t));
        stream.read((char *)&loc_offset, sizeof(uint32_t));
        stream.read((char *)&dest_type, sizeof(uint32_t));
        stream.read((char *)&dest_offset, sizeof(uint32_t));
    }
    void write(std::ostream &stream)
    {
        stream.write((char *)&loc_type, sizeof(uint32_t));
        stream.write((char *)&loc_offset, sizeof(uint32_t));
        stream.write((char *)&dest_type, sizeof(uint32_t));
        stream.write((char *)&dest_offset, sizeof(uint32_t));
    }
};

struct RemapcSection
{
    enum SectionType
    {
        RO = 0,
        DESC = 1,
        TEXT = 2,
    };
    uint32_t entry_num = 0;
    std::vector<RemapEntry> table;
    uint32_t size()
    {
        return sizeof(uint32_t) + entry_num * 4 * sizeof(uint32_t);
    }
    void load(std::istream &stream)
    {
        stream.read((char *)&entry_num, sizeof(uint32_t));
        table.resize(entry_num);
        for (uint32_t i = 0; i < entry_num; ++i)
        {
            table[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        entry_num = table.size();
        stream.write((char *)&entry_num, sizeof(uint32_t));
        for (uint32_t i = 0; i < entry_num; ++i)
        {
            table[i].write(stream);
        }
    }
};

struct AddressInfo
{
    uint32_t align;
    uint32_t size;
    uint32_t address;
};

struct GraphBin
{
    virtual ~GraphBin() = default;
    virtual BinSection &get_text() = 0;
    virtual BinSection &get_ro() = 0;
    virtual BinSection &get_desc() = 0;
    virtual BinSection &get_const() = 0;

    virtual AddressInfo get_text_info() = 0;
    virtual AddressInfo get_ro_info() = 0;
    virtual AddressInfo get_desc_info() = 0;
    virtual AddressInfo get_const_info() = 0;
    virtual AddressInfo get_reuse_info() = 0; // after alignment
    virtual AddressInfo get_stack_info() = 0;

    virtual uint32_t get_num_const() = 0;
    virtual uint32_t get_num_reuse() = 0;
    virtual uint32_t get_num_input() = 0;
    virtual uint32_t get_num_output() = 0;
    virtual AddressInfo get_input_info(uint32_t index = 0) = 0;
    virtual std::vector<AddressInfo> get_inputs_info() = 0;
    virtual AddressInfo get_output_info(uint32_t index = 0) = 0;
    virtual std::vector<AddressInfo> get_outputs_info() = 0;

    virtual bool set_reuse_address(uint32_t addr) = 0;
    virtual bool set_const_address(uint32_t addr) = 0;
    virtual bool set_ro_address(uint32_t addr) = 0;
    virtual bool set_desc_address(uint32_t addr) = 0;
    virtual bool set_text_address(uint32_t addr) = 0;

    virtual AddressInfo get_reuse_info(uint32_t index) = 0;
    virtual AddressInfo get_const_info(uint32_t index) = 0;
    virtual char *get_const_data(uint32_t index) = 0;
    virtual bool set_reuse_address(uint32_t index, uint32_t address) = 0;
    virtual bool set_const_address(uint32_t index, uint32_t address) = 0;
};

struct AIPUBin : GraphBin
{
    ABHeader header;
    BinSection text;
    BinSection ro;
    BinSection desc;
    BinSection data;
    BssSection bss;
    RemapcSection remap;
    std::vector<BinSection> extra_data;

    void write(std::ostream &ofs);
    void write(const std::string &filename)
    {
        std::ofstream ofs;
        ofs.open(filename, std::ios::out | std::ios::binary);
        write(ofs);
        ofs.close();
    }

    void load(std::istream &is)
    {
        header.load(is);
        text.bin_size = header.text_size;
        ro.bin_size = header.rodata_size;
        desc.bin_size = header.dcr_size;
        data.bin_size = header.data_size;
        text.load(is);
        ro.load(is);
        desc.load(is);
        data.load(is);
        bss.load(is, header.version >> 24);
        remap.load(is);
    }

    void load(const std::string &filename)
    {
        std::ifstream ifs;
        ifs.open(filename, std::ios::in | std::ios::binary);
        load(ifs);
        ifs.close();
    }

    /**
     * @brief merge two aipu.bin to single one, the bin1 and bin2 will be merged into out
     *  
     * @param out  the output aipu.bin 
     * @param bin1  the aipu.bin to be merged
     * @param bin2  the aipu.bin to be merged
     * @return true if successful
     * @return false if failed
     */
    static bool merge(AIPUBin &out, const AIPUBin &bin1, const AIPUBin &bin2);

    BinSection &get_text();
    BinSection &get_ro();
    BinSection &get_desc();
    BinSection &get_const();

    AddressInfo get_text_info();
    AddressInfo get_ro_info();
    AddressInfo get_desc_info();
    AddressInfo get_const_info();
    AddressInfo get_reuse_info(); // after alignment
    AddressInfo get_stack_info();

    uint32_t get_num_const();
    uint32_t get_num_reuse();
    uint32_t get_num_input();
    uint32_t get_num_output();
    template <uint32_t reusetype>
    uint32_t get_num_typed_reuse()
    {
        return get_num_typed_buffer(reusetype);
    }

    template <uint32_t reusetype>
    std::vector<AddressInfo> get_typed_reuse_info()
    {
        std::vector<AddressInfo> ret;
        for (size_t i = 0; i < get_num_reuse(); ++i)
        {
            auto &desc = bss.buffer_descriptor[i];
            for (size_t j = 0; j < desc.sub_section_cnt; ++j)
            {
                auto &subdesc = desc.sub_sections[j];
                if (subdesc.type == reusetype && subdesc.offset_in_ro_cnt != 0)
                {
                    AddressInfo info;
                    info.align = desc.align_bytes;
                    info.size = subdesc.size;
                    info.address = *(uint32_t *)(ro.bin + subdesc.offset_in_ro[0]);
                    ret.push_back(info);
                }
            }
        }
        return ret;
    }
    AddressInfo get_input_info(uint32_t index = 0);
    std::vector<AddressInfo> get_inputs_info();
    AddressInfo get_output_info(uint32_t index = 0);
    std::vector<AddressInfo> get_outputs_info();

    bool set_reuse_address(uint32_t addr);
    bool set_const_address(uint32_t addr);
    bool set_ro_address(uint32_t addr);
    bool set_desc_address(uint32_t addr);
    bool set_text_address(uint32_t addr);

    AddressInfo get_reuse_info(uint32_t index);
    AddressInfo get_const_info(uint32_t index);
    char *get_const_data(uint32_t index);
    uint32_t get_reuse_address(uint32_t index);
    uint32_t get_const_address(uint32_t index);
    bool set_reuse_address(uint32_t index, uint32_t address);
    bool set_const_address(uint32_t index, uint32_t address);

    bool add_extra_data(BinSection data);
    std::vector<BinSection> get_extra_data();

    uint32_t get_num_typed_buffer(uint32_t type);          //private
    std::vector<uint32_t> get_typed_buffer(uint32_t type); //private
    std::vector<uint32_t> get_inputs();
    std::vector<uint32_t> get_outputs();
    const BssBufferDescriptor &get_reuse_buffer(uint32_t index);
    const BssDataDescriptor &get_const_buffer(uint32_t index);
};

std::string bss_type_str(uint32_t type);
}; // namespace aipubt

#endif //!BUILDER_AIPU_BINFILE_H
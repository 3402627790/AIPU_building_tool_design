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
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-09-03 23:55:25
 */
#ifndef BUILDER_AIPU_ELF_H
#define BUILDER_AIPU_ELF_H

#include "aipu_binfile.h"
#include "core/logger.h"
#include "utils/utils.h"
#include <core/buildtool_version.h>
#include <cstdint>
#include <elfio/elfio.hpp>
#include <vector>
#define AIPUBIN_FILE_MAGIC_NUM "AIPU BIN"

#ifndef ELF_GRAPH_MIN_VERSION
#define ELF_GRAPH_MIN_VERSION 2
#endif

#ifndef ELF_GRAPH_MAJOR_VERSION
#define ELF_GRAPH_MAJOR_VERSION 1
#endif

#define STACK_SIZE (256 << 10)
#define STACK_MARGIN (1 << 10)

#define TOTAL_STACK_SIZE (STACK_SIZE + STACK_MARGIN)
namespace aipubt
{

struct AIPUCompilerMsg
{
    uint32_t file_version = (ELF_GRAPH_MIN_VERSION << 24) | ((ELF_GRAPH_MAJOR_VERSION & 0xff) << 16);
    uint32_t build_version = AIPUBIN_BT_VERSION_NUMBER;
    uint32_t arch = 0;
    uint32_t flag = 0;
    int32_t reserve0[8] = {0};
    void load(std::istream &stream)
    {
        stream.read((char *)&file_version, sizeof(uint32_t));
        stream.read((char *)&build_version, sizeof(uint32_t));
        stream.read((char *)&arch, sizeof(uint32_t));
        stream.read((char *)&flag, sizeof(uint32_t));
        for (uint32_t i = 0; i < 8; ++i)
        {
            stream.read((char *)&(reserve0[i]), sizeof(uint32_t));
        }
    }
    void write(std::ostream &stream)
    {
        stream.write((char *)&file_version, sizeof(uint32_t));
        stream.write((char *)&build_version, sizeof(uint32_t));
        stream.write((char *)&arch, sizeof(uint32_t));
        stream.write((char *)&flag, sizeof(uint32_t));
        for (uint32_t i = 0; i < 8; ++i)
        {
            stream.write((char *)&(reserve0[i]), sizeof(uint32_t));
        }
    }
};

// struct global_param
// {
//     std::vector<uint32_t> reserve0;
//     std::vector<uint32_t> input_shape;
//     std::vector<uint32_t> reserve1;
// };
#define GP_RESERVE0_OFFSET (2 * 4)
#define GP_RESERVE0_COUNT 128
#define GP_INPUT_SHAPE_OFFSET (GP_RESERVE0_OFFSET + GP_RESERVE0_COUNT * 4)
#define GP_INPUT_SHAPE_COUNT 896
#define GP_RESERVE1_OFFSET (GP_INPUT_SHAPE_OFFSET + GP_INPUT_SHAPE_COUNT * 4)
#define GP_RESERVE1_COUNT 1024
#define GLOBAL_PARAM_COUNT (GP_RESERVE0_COUNT + GP_INPUT_SHAPE_COUNT + GP_RESERVE1_COUNT)

struct ModelGlobalParam
{
    uint32_t input_shape_offset;
    uint32_t num_params;
    std::vector<uint32_t> params;
    uint32_t size() const
    {
        return sizeof(uint32_t) * (2 + params.size());
    }
    void load(std::istream &stream)
    {
        stream.read((char *)&input_shape_offset, sizeof(uint32_t));
        stream.read((char *)&num_params, sizeof(uint32_t));
        params.resize(num_params);
        for (uint32_t i = 0; i < num_params; ++i)
        {
            stream.read((char *)&(params[i]), sizeof(uint32_t));
        }
    }
    void write(std::ostream &stream)
    {
        num_params = params.size();
        stream.write((char *)&input_shape_offset, sizeof(uint32_t));
        stream.write((char *)&num_params, sizeof(uint32_t));
        for (uint32_t i = 0; i < num_params; ++i)
        {
            stream.write((char *)&params[i], sizeof(uint32_t));
        }
    }
    void write(void *dst)
    {
        if (!dst)
        {
            return;
        }
        uint32_t *ptr = reinterpret_cast<uint32_t *>(dst);
        num_params = params.size();
        ptr[0] = input_shape_offset;
        ptr[1] = num_params;
        for (uint32_t i = 0; i < num_params; ++i)
        {
            ptr[2 + i] = params[i];
        }
    }
    void update(const std::vector<std::vector<int>> &shapes)
    {
        uint32_t start = input_shape_offset / sizeof(uint32_t) - 2;
        uint32_t index = 0;
        for (auto &s : shapes)
        {
            for (auto d : s)
            {
                params[start + index++] = d;
            }
        }
    }
};

struct AIPUTensorShape
{
    AIPUTensorShape() = default;
    AIPUTensorShape(const std::vector<uint32_t> &s)
    {
        dim = s.size();
        shapes = s;
    }
    uint32_t dim = 0;
    std::vector<uint32_t> shapes = {};
    void load(std::istream &stream)
    {
        stream.read((char *)&dim, sizeof(uint32_t));
        shapes.resize(dim);
        for (uint32_t i = 0; i < dim; ++i)
        {
            stream.read((char *)&(shapes[i]), sizeof(uint32_t));
        }
    }
    void write(std::ostream &stream)
    {
        dim = shapes.size();
        stream.write((char *)&dim, sizeof(uint32_t));
        for (uint32_t i = 0; i < dim; ++i)
        {
            stream.write((char *)&shapes[i], sizeof(uint32_t));
        }
    }
    std::string to_string() const
    {
        std::ostringstream oss;
        oss << "[";
        for (uint32_t i = 0; i < shapes.size(); ++i)
        {
            oss << shapes[i];
            if (i != shapes.size() - 1)
            {
                oss << ",";
            }
        }
        oss << "]";
        return oss.str();
    }
};

struct InputShapeConstraint
{
    uint32_t num_inputs;
    // store in order as : [inp0_min_shape,inp0_max_shape,inp1_min_shape,inp1_max_shape,...]
    std::vector<AIPUTensorShape> constrains; // the constrains contains num_input*2 shape, the first is min shape, the second is max shape.
    void load(std::istream &stream)
    {
        stream.read((char *)&num_inputs, sizeof(uint32_t));
        constrains.resize(num_inputs);
        for (uint32_t i = 0; i < num_inputs; ++i)
        {
            constrains[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        num_inputs = constrains.size();
        stream.write((char *)&num_inputs, sizeof(uint32_t));
        for (uint32_t i = 0; i < num_inputs; ++i)
        {
            constrains[i].write(stream);
        }
    }
};

struct OutputTensorShape
{
    uint32_t num_outputs;
    std::vector<AIPUTensorShape> shapes;
    void load(std::istream &stream)
    {
        stream.read((char *)&num_outputs, sizeof(uint32_t));
        shapes.resize(num_outputs);
        for (uint32_t i = 0; i < num_outputs; ++i)
        {
            shapes[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        num_outputs = shapes.size();
        stream.write((char *)&num_outputs, sizeof(uint32_t));
        for (uint32_t i = 0; i < num_outputs; ++i)
        {
            shapes[i].write(stream);
        }
    }
};

struct ConstantHashItem
{
    uint32_t bin_offset = 0;
    uint32_t bin_size = 0;
    std::vector<uint8_t> hash_code;

    void load(std::istream &stream)
    {
        stream.read((char *)&bin_offset, sizeof(uint32_t));
        stream.read((char *)&bin_size, sizeof(uint32_t));
        hash_code.resize(32);
        for (uint32_t i = 0; i < hash_code.size(); ++i)
        {
            stream.read((char *)&(hash_code[i]), sizeof(uint8_t));
        }
    }
    void write(std::ostream &stream)
    {
        stream.write((char *)&bin_offset, sizeof(uint32_t));
        stream.write((char *)&bin_size, sizeof(uint32_t));
        hash_code.resize(32);
        for (uint32_t i = 0; i < hash_code.size(); ++i)
        {
            stream.write((char *)&(hash_code[i]), sizeof(uint8_t));
        }
    }
};

struct ConstantHashItems
{
    uint32_t num_items = 0;
    std::vector<ConstantHashItem> items;

    void load(std::istream &stream)
    {
        stream.read((char *)&num_items, sizeof(uint32_t));
        items.resize(num_items);
        for (uint32_t i = 0; i < num_items; ++i)
        {
            items[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        num_items = items.size();
        stream.write((char *)&num_items, sizeof(uint32_t));
        for (uint32_t i = 0; i < num_items; ++i)
        {
            items[i].write(stream);
        }
    }
};

struct ConstantHashTable
{
    uint32_t num_items = 0;
    std::vector<ConstantHashItems> items;

    void load(std::istream &stream)
    {
        stream.read((char *)&num_items, sizeof(uint32_t));
        items.resize(num_items);
        for (uint32_t i = 0; i < num_items; ++i)
        {
            items[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        num_items = items.size();
        stream.write((char *)&num_items, sizeof(uint32_t));
        for (uint32_t i = 0; i < num_items; ++i)
        {
            items[i].write(stream);
        }
    }
};

struct FeatureMapList
{
    uint32_t reserve0 = 0;
    uint32_t reserve1 = 0;
    uint32_t num_fm_descriptor = 0;
    std::vector<BssSection> fmdesc;
    void load(std::istream &stream)
    {
        stream.read((char *)&reserve0, sizeof(uint32_t));
        stream.read((char *)&reserve1, sizeof(uint32_t));
        stream.read((char *)&num_fm_descriptor, sizeof(uint32_t));
        fmdesc.resize(num_fm_descriptor);
        for (uint32_t i = 0; i < num_fm_descriptor; ++i)
        {
            fmdesc[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        num_fm_descriptor = fmdesc.size();
        stream.write((char *)&reserve0, sizeof(uint32_t));
        stream.write((char *)&reserve1, sizeof(uint32_t));
        stream.write((char *)&num_fm_descriptor, sizeof(uint32_t));
        for (uint32_t i = 0; i < num_fm_descriptor; ++i)
        {
            fmdesc[i].write(stream);
        }
    }
    void add_bss(BssSection bss)
    {
        fmdesc.push_back(bss);
        num_fm_descriptor = fmdesc.size();
    }
};

struct LlmInfo
{
    char *vocab;
    uint32_t vocab_size;

    uint32_t hidden_size;
    uint32_t cache_head;
    uint32_t cache_dim;
    uint32_t logits_dim;
    uint32_t hidden_layers;
    bool use_cache;
    std::string model_type;

    char *embd_wt = 0;
    uint32_t embd_wt_size = 0;
    std::string embd_dtype = "float16";
    void set_vocab(const char *data, size_t size)
    {
        vocab = new char[size];
        vocab_size = size;
        memcpy(vocab, data, size);
    }
    void set_embd_wt(const char *data, size_t size)
    {
        embd_wt = new char[size];
        embd_wt_size = size;
        memcpy(embd_wt, data, size);
    }
    BinSection ser_to_bin() const
    {
        BinSection bin;
        uint32_t offset = 0;
        std::string str_mem;

        std::string name = "embd_wt.bin";
        std::ofstream file;
        file.open(name, std::ios::out | std::ios::binary);
        file.write(embd_wt, embd_wt_size);
        if (file.fail())
        {
            ERROR() << "file open for " << name << " failed! ";
            return bin;
        }
        file.close();
        auto hash = utils::sha256_hex(embd_wt, embd_wt_size);
        str_mem = name + "+" + hash;
        uint32_t str_size = str_mem.size();
        bin.bin_size = vocab_size + 8 * sizeof(uint32_t) + str_size + sizeof(uint8_t) + model_type.size();
        bin.bin = new char[bin.bin_size];
        memcpy(bin.bin + offset, &vocab_size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(bin.bin + offset, vocab, vocab_size);
        offset += vocab_size;
        memcpy(bin.bin + offset, &hidden_size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(bin.bin + offset, &cache_head, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(bin.bin + offset, &cache_dim, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(bin.bin + offset, &logits_dim, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(bin.bin + offset, &hidden_layers, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        uint8_t cache = use_cache;
        memcpy(bin.bin + offset, &cache, sizeof(uint8_t));
        offset += sizeof(uint8_t);
        uint32_t model_type_size = model_type.size();
        memcpy(bin.bin + offset, &model_type_size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(bin.bin + offset, model_type.data(), model_type_size);
        offset += model_type_size;
        memcpy(bin.bin + offset, &str_size, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(bin.bin + offset, str_mem.data(), str_size);
        offset += str_size;
        if (offset != bin.bin_size)
        {
            ERROR() << "bin size not correct";
        }
        return bin;
    }
    void deser_from_bin(const BinSection &bin, const std::string path)
    {
        uint32_t offset = 0;
        uint32_t str_size = 0;
        uint32_t model_type_size = 0;
        std::string str_mem;
        memcpy(&vocab_size, bin.bin + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        vocab = new char[vocab_size];
        if (offset + vocab_size > bin.bin_size)
        {
            ERROR() << "llm info de-serial error";
        }
        memcpy(vocab, bin.bin + offset, vocab_size);
        offset += vocab_size;
        memcpy(&hidden_size, bin.bin + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(&cache_head, bin.bin + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(&cache_dim, bin.bin + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(&logits_dim, bin.bin + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(&hidden_layers, bin.bin + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        memcpy(&use_cache, bin.bin + offset, sizeof(uint8_t));
        offset += sizeof(uint8_t);
        memcpy(&model_type_size, bin.bin + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        model_type.assign(bin.bin + offset, model_type_size);
        offset += model_type.size();
        memcpy(&str_size, bin.bin + offset, sizeof(uint32_t));
        offset += sizeof(uint32_t);
        if (offset + str_size > bin.bin_size)
        {
            ERROR() << "llm info de-serial error";
            return;
        }
        str_mem.assign(bin.bin + offset, str_size);
        // INFO() << "str_mem " << str_mem << " size: " << str_size;
        size_t pos = str_mem.find('+');
        if (pos == std::string::npos)
        {
            ERROR() << "no sha32";
        }
        std::string name = str_mem.substr(0, pos);
        std::string hash = str_mem.substr(pos + 1);

        std::string full_name = path + "/" + name;

        if (utils::file_exists(full_name))
        {
            std::ifstream file;
            file.open(full_name, std::ios::binary);
            if (!file)
            {
                ERROR() << "file open for " << full_name << " failed! ";
                return;
            }
            file.seekg(0, std::ios::end);
            std::streamsize buf_size = file.tellg();
            file.seekg(0, std::ios::beg);

            embd_wt_size = buf_size;
            embd_wt = new char[buf_size];
            file.read(embd_wt, embd_wt_size);
            file.close();

            auto file_hash = utils::sha256_hex(embd_wt, embd_wt_size);
            if (file_hash != hash)
            {
                ERROR() << "integrity check failed";
            }
        }
        else
        {
            ERROR() << "file not exist " << full_name;
        }
    }
};

struct GMConfig
{
    enum GMSyncDir
    {
        NoSync = 0,
        DDR2GM = 1,
        GM2DDR = 2,
    };
    struct BssBufferIndex
    {
        uint32_t fm_index;  //the idx of feature map list
        uint32_t buf_type;  //0:reuse buffer, 1: static(constant) buffer
        uint32_t buf_index; //the index of buffer
        uint32_t resver0;
    };
    uint32_t GM_control = 0;
    uint32_t GM_r0_control = 0;
    uint32_t GM_r1_control = 0;
    uint32_t reserve0 = 0;
    uint32_t reserve1 = 0;
    uint32_t reserve2 = 0;
    BssBufferIndex GM_r0_idx = {0};
    BssBufferIndex GM_r1_idx = {0};
    void load(std::istream &stream)
    {
        stream.read((char *)this, sizeof(GMConfig));
    }
    void write(std::ostream &stream)
    {
        stream.write((char *)this, sizeof(GMConfig));
    }
    void set_region_sync(uint32_t rid, uint32_t offset, uint32_t size, GMSyncDir dir)
    {
        uint32_t control = 0;
        if (dir != GMSyncDir::NoSync)
        {
            control = control | ((dir & 0b11) << 30); //set sync
            control = control | (((offset >> 12) & 0xffff) << 12);
            control = control | (((size + 4095) >> 12) & 0xfff);
        }
        if (rid == 0)
        {
            GM_r0_control = control;
        }
        else if (rid == 1)
        {
            GM_r1_control = control;
        }
    }
    void set_region_buffer_idx(uint32_t rid, uint32_t idx, uint32_t buf_type = 0, uint32_t fm_idx = 0)
    {
        BssBufferIndex *region;
        region = (rid == 0) ? &GM_r0_idx : &GM_r1_idx;
        if (rid > 1)
            return;
        region->buf_index = idx;
        region->buf_type = buf_type;
        region->fm_index = fm_idx;
    }
    void set_gm_size(uint32_t in_bytes = 4 * 1024 * 1024)
    {
        // only accept: 512K-64MB, we will set to the closest one
        uint32_t gm_size = 0;
        uint32_t mask = 0xf0000;
        if (in_bytes < 512 * 1024 * 1.5)
        {
            gm_size = 0;
        }
        else if (in_bytes < 1.5 * 1024 * 1024)
        {
            gm_size = 1;
        }
        else if (in_bytes < 3 * 1024 * 1024)
        {
            gm_size = 2;
        }
        else if (in_bytes < 6 * 1024 * 1024)
        {
            gm_size = 3;
        }
        else if (in_bytes < 12 * 1024 * 1024)
        {
            gm_size = 4;
        }
        else if (in_bytes < 24 * 1024 * 1024)
        {
            gm_size = 5;
        }
        else if (in_bytes < 48 * 1024 * 1024)
        {
            gm_size = 6;
        }
        else
        {
            gm_size = 7;
        }
        GM_control = (GM_control & ~mask) | ((gm_size << 16) & mask);
    }
    void set_remap_enable(uint32_t en = 2)
    {
        /**
         * Global memory remapping enable bit
         *  0 	Global Memory Remapping is disabled.
         *  1 	Global Memory Remapping is enabled. Only Region 0 is enabled, and all memory buffer size is allocated to Region 0.
         *  2 	Global Memory Remap is enabled. Both Region 0 and Region 1 are enabled, and memory buffer size is equally allocated to Region 0 and Region 1.
         *  Other      Reserved.
         */
        GM_control = (GM_control & ~0xF) | (en & 0xF);
    }
};

struct SegMMUConfig
{
    struct MMUAddr
    {
        uint32_t control0;
        uint32_t control1;
    };

    MMUAddr seg0 = {0};
    MMUAddr seg1 = {0};
    MMUAddr seg2 = {0};
    MMUAddr seg3 = {0};
    uint32_t SegMMU_ctl = 0;
    uint32_t SegMMU_remap = 0;
    uint32_t reserve0 = 0;
    uint32_t reserve1 = 0;
    uint32_t reserve2 = 0;
    uint32_t reserve3 = 0;
};
struct SegMMUList
{
    uint32_t num_mmu;
    SegMMUConfig *smmu;
};

struct SubGraphDesc
{
    uint32_t id;              // the id for subgraph
    uint32_t type;            // type, default is 0;
    uint32_t elf_text_offset; // start pc offset/ entry point of this subgraph
    uint32_t fm_desc_offset;  // the fm list
    uint32_t rodata_offset;   // the begin ro
    uint32_t rodata_size;     // and ro size
    uint32_t descriptor_offset;
    uint32_t descriptor_size;
    uint32_t printfifo_size;    // print fifo for the subgraph
    uint32_t profiler_buf_size; //buffer size
    uint32_t private_data_size; // private data per tec
    uint32_t warmup_len;        // the warmup len for subgraph
    uint32_t reserve2;
    uint32_t reserve3;
    int32_t precursor_cnt;
    std::vector<uint32_t> precursors;
    uint32_t private_buffer_cnt;
    std::vector<BssBufferDescriptor> private_buffer; // the offset_in_ro in private buffer is same as fmlist's convention

    void add_precursor(uint32_t p)
    {
        precursors.push_back(p);
    }
    void load(std::istream &stream)
    {
        stream.read((char *)&id, sizeof(uint32_t));
        stream.read((char *)&type, sizeof(uint32_t));
        stream.read((char *)&elf_text_offset, sizeof(uint32_t));
        stream.read((char *)&fm_desc_offset, sizeof(uint32_t));
        stream.read((char *)&rodata_offset, sizeof(uint32_t));
        stream.read((char *)&rodata_size, sizeof(uint32_t));
        stream.read((char *)&descriptor_offset, sizeof(uint32_t));
        stream.read((char *)&descriptor_size, sizeof(uint32_t));
        stream.read((char *)&printfifo_size, sizeof(uint32_t));
        stream.read((char *)&profiler_buf_size, sizeof(uint32_t));
        stream.read((char *)&private_data_size, sizeof(uint32_t));
        stream.read((char *)&warmup_len, sizeof(uint32_t));
        stream.read((char *)&reserve2, sizeof(uint32_t));
        stream.read((char *)&reserve3, sizeof(uint32_t));
        stream.read((char *)&precursor_cnt, sizeof(uint32_t));
        if (precursor_cnt > 0)
        {
            precursors.resize(precursor_cnt);
        }
        for (int32_t i = 0; i < precursor_cnt; ++i)
        {
            stream.read((char *)&(precursors[i]), sizeof(uint32_t));
        }
        stream.read((char *)&private_buffer_cnt, sizeof(uint32_t));
        private_buffer.resize(private_buffer_cnt);
        for (uint32_t i = 0; i < private_buffer_cnt; ++i)
        {
            private_buffer[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        if (precursor_cnt >= 0)
        {
            precursor_cnt = precursors.size();
        }
        stream.write((char *)&id, sizeof(uint32_t));
        stream.write((char *)&type, sizeof(uint32_t));
        stream.write((char *)&elf_text_offset, sizeof(uint32_t));
        stream.write((char *)&fm_desc_offset, sizeof(uint32_t));
        stream.write((char *)&rodata_offset, sizeof(uint32_t));
        stream.write((char *)&rodata_size, sizeof(uint32_t));
        stream.write((char *)&descriptor_offset, sizeof(uint32_t));
        stream.write((char *)&descriptor_size, sizeof(uint32_t));
        stream.write((char *)&printfifo_size, sizeof(uint32_t));
        stream.write((char *)&profiler_buf_size, sizeof(uint32_t));
        stream.write((char *)&private_data_size, sizeof(uint32_t));
        stream.write((char *)&warmup_len, sizeof(uint32_t));
        stream.write((char *)&reserve2, sizeof(uint32_t));
        stream.write((char *)&reserve3, sizeof(uint32_t));
        stream.write((char *)&precursor_cnt, sizeof(uint32_t));
        for (int32_t i = 0; i < precursor_cnt; ++i)
        {
            stream.write((char *)&(precursors[i]), sizeof(uint32_t));
        }
        stream.write((char *)&private_buffer_cnt, sizeof(uint32_t));
        for (uint32_t i = 0; i < private_buffer_cnt; ++i)
        {
            private_buffer[i].write(stream);
        }
    }
};

struct SGList
{
    uint32_t num_subgraph;
    std::vector<SubGraphDesc> sgs;
    void load(std::istream &stream)
    {
        stream.read((char *)&num_subgraph, sizeof(uint32_t));
        sgs.resize(num_subgraph);
        for (uint32_t i = 0; i < num_subgraph; ++i)
        {
            sgs[i].load(stream);
        }
    }
    void write(std::ostream &stream)
    {
        num_subgraph = sgs.size();
        stream.write((char *)&num_subgraph, sizeof(uint32_t));
        for (uint32_t i = 0; i < num_subgraph; ++i)
        {
            sgs[i].write(stream);
        }
    }
    void add_sg(const SubGraphDesc &sg)
    {
        sgs.push_back(sg);
    }
};

struct LayerDebugInfo
{
    uint32_t address;     // the entry pc address of this layer
    uint32_t layer_id;    // the layer id of dump IR
    uint32_t subgraph_id; // the layer id of dump IR
    std::vector<int32_t> ids;
    std::vector<int32_t> inputs;
    std::vector<int32_t> outputs;
    void load(std::istream &stream)
    {
        stream.read((char *)&address, sizeof(uint32_t));
        stream.read((char *)&layer_id, sizeof(uint32_t));
        stream.read((char *)&subgraph_id, sizeof(uint32_t));
        uint32_t n_ids, n_inputs, n_outputs;
        stream.read((char *)&n_ids, sizeof(uint32_t));
        ids.resize(n_ids);
        for (uint32_t i = 0; i < n_ids; ++i)
        {
            stream.read((char *)&ids[i], sizeof(int32_t));
        }
        stream.read((char *)&n_inputs, sizeof(uint32_t));
        inputs.resize(n_inputs);
        for (uint32_t i = 0; i < n_inputs; ++i)
        {
            stream.read((char *)&inputs[i], sizeof(int32_t));
        }
        stream.read((char *)&n_outputs, sizeof(uint32_t));
        outputs.resize(n_outputs);
        for (uint32_t i = 0; i < n_outputs; ++i)
        {
            stream.read((char *)&outputs[i], sizeof(int32_t));
        }
    }
    void write(std::ostream &stream) const
    {
        stream.write((char *)&address, sizeof(uint32_t));
        stream.write((char *)&layer_id, sizeof(uint32_t));
        stream.write((char *)&subgraph_id, sizeof(uint32_t));
        uint32_t n_ids, n_inputs, n_outputs;
        n_ids = ids.size();
        stream.write((char *)&n_ids, sizeof(uint32_t));
        for (uint32_t i = 0; i < n_ids; ++i)
        {
            stream.write((char *)&ids[i], sizeof(int32_t));
        }
        n_inputs = inputs.size();
        stream.write((char *)&n_inputs, sizeof(uint32_t));
        for (uint32_t i = 0; i < n_inputs; ++i)
        {
            stream.write((char *)&inputs[i], sizeof(int32_t));
        }
        n_outputs = outputs.size();
        stream.write((char *)&n_outputs, sizeof(uint32_t));
        for (uint32_t i = 0; i < n_outputs; ++i)
        {
            stream.write((char *)&outputs[i], sizeof(int32_t));
        }
    }
};
struct LayerDebugInfos
{
    std::vector<LayerDebugInfo> info;
    void load(std::istream &stream)
    {
        info.clear();
        uint32_t n_layers;
        stream.read((char *)&n_layers, sizeof(uint32_t));
        info.resize(n_layers);
        for (uint32_t i = 0; i < n_layers; ++i)
        {
            info[i].load(stream);
        }
    }
    void write(std::ostream &stream) const
    {
        uint32_t n_layers = info.size();
        stream.write((char *)&n_layers, sizeof(uint32_t));
        for (uint32_t i = 0; i < n_layers; ++i)
        {
            info[i].write(stream);
        }
    }
};
class AIPUELF
{
public:
    AIPUELF()
    {
    }
    AIPUCompilerMsg &compiler_msg()
    {
        return m_comment;
    }
    const AIPUCompilerMsg &compiler_msg() const
    {
        return m_comment;
    }
    GMConfig get_GMconfig() const;
    void set_GMconfig(const GMConfig &ro);

    const std::string &get_ir() const;
    void set_ir(const std::string &ir);

    //profiler meta info
    const std::string &get_graph_json() const;
    void set_graph_json(const std::string &perf_meta);

    const std::string &get_dump_ir() const;
    void set_dump_ir(const std::string &ir);

    LayerDebugInfos get_layer_debug_info() const;
    void set_layer_debug_info(const LayerDebugInfos &info);

    void load(const char *base, uint32_t n);

    void load(const std::string &filename);
    void load(std::istream &stream);
    void write(const std::string &filename);
    void write(std::ostream &stream);

    BinSection &get_rodata() const;
    void set_rodata(BinSection &ro);
    void rewrite_ro(uint32_t offset, uint32_t value, bool write_to_file = false);
    void rewrite_ro(uint32_t offset, std::vector<uint32_t> values, bool write_to_file = false);
    BinSection get_descriptor() const;
    void set_descriptor(BinSection &ro);
    BinSection get_weight() const;
    std::vector<BinSection> get_extra_weight() const;
    void set_weight(BinSection &ro);
    void set_extra_weight(const std::vector<BinSection> extra_data_vec);
    void set_aipu_note(BinSection &bin, const std::string &name);
    BinSection get_aipu_note(const std::string &name) const;
    bool has_aipu_note(const std::string &name) const;
    void remove_aipu_note(const std::string &name);
    BinSection get_bin_section(const std::string &name);
    FeatureMapList get_fm_list() const;
    void set_fm_list(FeatureMapList &ro);
    RemapcSection get_remap_table() const;
    void set_remap_table(RemapcSection &ro);
    SGList get_subgraphs() const;
    void set_subgraphs(SGList &ro);
    void set_llminfo(const LlmInfo &info);
    LlmInfo get_llminfo() const;

    ModelGlobalParam get_globalparam() const;
    void set_globalparam(ModelGlobalParam &ro);

    inline bool is_dynamic_shape() const
    {
        return m_comment.flag >> 6 & 0x1;
    }

    std::vector<std::vector<uint32_t>> get_input_shapes() const;
    InputShapeConstraint get_input_shape_constraint() const;
    void set_input_shape_constraint(InputShapeConstraint &ro);

    std::vector<std::vector<uint32_t>> get_output_shapes() const; // static output shapes
    OutputTensorShape get_output_tensor_shape() const;
    void set_output_tensor_shape(OutputTensorShape &ro);

    ConstantHashTable get_constant_hash_table() const;
    void set_constant_hash_table(ConstantHashTable &table);

    ELFIO::section *get_elf_section(const std::string &section_name) const;
    std::string get_text() const;
    uint32_t get_text_size() const;
    uint32_t get_func_size(const std::string &func_name) const;
    void set_elf_file(std::istream &stream);
    void set_elf_file(const std::string &filename);

    uint32_t get_function_offset(const std::string &func_name) const;
    const std::map<std::string, uint64_t> &get_symtab() const;

    void set_comment(const std::string &comment);

protected:
    BinSection get_bin_note(const std::string &note_name) const;
    void set_bin_note(BinSection &ro, const std::string &note_name);
    void set_note(const std::string &note, const std::string &note_name);
    bool get_note(const std::string &note_name,
                  uint32_t &idx,
                  bool copy_desc,
                  BinSection &sec) const;
    void remove_note(const std::string &note_name);
    void add_aipu_note();
    mutable ELFIO::section *m_note = nullptr;
    GMConfig m_gm;
    AIPUCompilerMsg m_comment;
    std::string m_ir;
    std::string m_dump_ir;
    std::string m_graph_json;
    ELFIO::elfio m_elf;
    mutable std::map<std::string, uint64_t> m_symtab;
    mutable std::map<std::string, uint32_t> m_sym_size;
    mutable BinSection m_ro = {0};
    std::vector<BinSection> m_extra_weight_vec;
    std::string m_elf_folder = "./";
};
using AIPUELFPtr = std::shared_ptr<AIPUELF>;
} // namespace aipubt

#endif //!BUILDER_AIPU_ELF_H
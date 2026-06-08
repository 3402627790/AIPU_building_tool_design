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
 * @Date: 2020-01-29 14:23:41
 * @Last Modified by: Tom Gao
 * @Last Modified time: 2022-06-06 19:10:39
 */
#ifndef BUILDER_BUILDER_H
#define BUILDER_BUILDER_H
#include "builder/aipu_binfile.h"
#include "core/memory_object.h"
#include "core/node.h"
#include "utils/string_utils.h"
#include "utils/utils.h"
#include <builder/bldtypes.h>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace aipubt
{
struct AIPUBin;
class Scheduler;
class Tensor;
class Graph;
class Node;
struct ReadOnlyParam;
struct ABHeader;
struct RemapcSection;
using GraphPtr = std::shared_ptr<Graph>;
using NodePtr = std::shared_ptr<Node>;
using NodeRef = std::weak_ptr<Node>;

struct CodeTree;
using CodeTreePtr = std::shared_ptr<CodeTree>;
using BssSubSecType = BssSubSectionDescriptor::SubSectionType;
/**
 * @brief  here, we introduce a code tree to descript a code structure, top level is a list of entry points
 *
 */
struct CodeTree
{
    CodeTree() = default;
    CodeTree(const std::string &v, NodePtr n) : value(v), node(n) {}
    CodeTree(const std::string &v) : CodeTree(v, nullptr) {}
    CodeTree(const char *v) : CodeTree(v, nullptr) {}

    std::string value;
    NodeRef node;
    std::vector<CodeTreePtr> codes;

    bool is_leaf() const
    {
        return codes.size() == 0;
    }
    void legalize_func_name()
    {
        value = utils::verify_name(value);
        for (auto c : codes)
        {
            c->legalize_func_name();
        }
    }
    // std::vector<CodeTreePtr> leaves()
    // {
    //     std::vector<CodeTreePtr> ret;
    //     return ret;
    // }
    std::string to_string() const
    {
        std::ostringstream oss;
        format_string(oss);
        return oss.str();
    }
    void push_back(const CodeTree &ct)
    {
        auto ctp = std::make_shared<CodeTree>();
        ctp->value = ct.value;
        ctp->node = ct.node;
        codes.push_back(ctp);
    }

private:
    void format_string(std::ostream &os, int depth = 0) const
    {
        std::string prefix(2 * depth, ' ');
        os << prefix << "|- ";
        if (value.size())
        {
            if (node.lock())
            {
                os << value << " " << node.lock()->name();
            }
            else
            {
                os << value;
            }
        }
        else
        {
            os << "(null)";
        }
        if (is_leaf())
        {
            os << "*";
        }
        os << "\n";
        for (auto c : codes)
        {
            c->format_string(os, depth + 1);
        }
    }
};

inline std::ostream &operator<<(std::ostream &os, const CodeTree &ct)
{
    return os << ct.to_string();
}

inline std::ostream &operator<<(std::ostream &os, const CodeTreePtr &ctp)
{
    return os << *ctp;
}

class GraphBuilder
{
public:
    GraphBuilder(GraphPtr graph);
    GraphBuilder(std::string file);
    virtual ~GraphBuilder();
    virtual void set_build_args(const std::string key);
    virtual bool build(const std::string &filename = "aipu.bin",
                       const TensorList &inputs = TensorList(),
                       const TensorList &outputs = TensorList(),
                       const std::vector<int32_t> &dump_ids = std::vector<int32_t>());
    virtual GraphPtr build_graph(const TensorList &inputs = TensorList(),
                                 const TensorList &outputs = TensorList());

    virtual bool build_aipu(const TensorList &inputs = TensorList(),
                            const TensorList &outputs = TensorList(),
                            const std::vector<int32_t> &dump_ids = std::vector<int32_t>());

    virtual void write(std::ostream &ofs);
    virtual void write(const std::string &filename);
    static AIPUBin load(std::istream &is);
    static AIPUBin load(const std::string &filename = "aipu.bin");
    std::vector<TensorPtr> dump(const std::string &filename_prefix = "temp", const TensorList &inputs = TensorList());
    std::vector<TensorPtr> dump(TensorList &dumps, const std::string &filename_prefix = "temp", const TensorList &inputs = TensorList());

    const std::vector<ReadOnlyParam> &get_ro() const
    {
        return m_ro;
    }
    const std::vector<ReadOnlyParam> &get_descriptor() const
    {
        return m_descriptor;
    }
    const std::pair<char *, size_t> get_build_text() const;

    const BuilderTextList &get_text() const
    {
        return m_text;
    }
    const AIPUBin *get_aipubin()
    {
        return m_aipubin;
    }
    std::vector<uint32_t> get_inp_base_list()
    {
        return m_inp_base_list;
    }
    std::map<BssSubSecType, std::vector<TensorPtr>> get_marked_tensors()
    {
        return m_marked_tensors;
    }

    GraphPtr graph()
    {
        return m_graph;
    }
    const std::vector<MemoryObjectPtr> &get_slots() const
    {
        return m_slots;
    }

    size_t getPeakMemory()
    {
        std::ifstream file("/proc/self/status");
        std::string line;
        while (std::getline(file, line))
        {
            if (line.substr(0, 6) == "VmHWM:")
            {
                std::string value_str = line.substr(7);
                size_t kb = std::stoull(value_str);
                return static_cast<double>(kb) / 1024.0;
            }
        }
        return 0;
    }

protected:
    void assembly();
    void show_section_size(std::string str, uint64_t size, utils::DataUnit unit = utils::DU_MB);
    virtual void show_memory_info();
    virtual uint32_t get_text_size();
    virtual void partition_graph();
    virtual void build_data(BinSection &data_bin, std::vector<TensorPtr> &data, std::vector<uint64_t> &offsets);
    virtual void build_text();
    virtual void build_remap(RemapcSection &remap, std::map<RemapcSection::SectionType, TensorPtr> base_tensors,
                             std::vector<ReadOnlyParam> &ro, std::vector<ReadOnlyParam> &decriptor, size_t &remap_size);
    virtual void build_ro(BinSection &ro_bin, std::vector<ReadOnlyParam> &ro);
    virtual void build_bss_section(BssSection &bss, std::vector<MemoryObjectPtr> &slots,
                                   std::vector<MemoryObjectPtr> &constant_slots,
                                   std::map<MemoryObjectPtr, std::vector<TensorPtr>> &buffer_map,
                                   std::map<TensorPtr, std::vector<int>> &tensor_map,
                                   std::map<BssSubSectionDescriptor::SubSectionType, std::vector<TensorPtr>> &special_marked_tensor,
                                   uint32_t bss_id);
    void update_subsec(BssSubSectionDescriptor &subsec, TensorPtr tensor);
    bool desc_add_subsec(const std::vector<TensorPtr> &vec, uint32_t type, BssSubSectionDescriptor &subsec, TensorPtr tensor, BssBufferDescriptor &desc, bool id_is_zero = 0);
    std::vector<ReadOnlyParam> m_ro;
    std::vector<ReadOnlyParam> m_descriptor;
    BuilderTextList m_text;
    std::vector<std::vector<std::string>> m_codes;
    std::vector<std::vector<std::string>> m_kernels;
    std::map<BssSubSecType, std::vector<TensorPtr>> m_marked_tensors;
    std::vector<uint32_t> m_inp_base_list;
    std::map<MemoryObjectPtr, std::vector<TensorPtr>> m_buffer_map;
    std::map<TensorPtr, std::vector<int>> m_tensor_map;
    CodeTree m_codetree;
    AIPUBin *m_aipubin;
    GraphPtr m_origin_graph;
    GraphPtr m_graph;
    std::vector<MemoryObjectPtr> m_slots;
    std::vector<std::vector<NodePtr>> m_levels;
    std::vector<MemoryObjectPtr> m_gm_regions;
    std::vector<uint32_t> m_gm_size;
    std::vector<BssSection> m_bss_vec;
    std::vector<std::vector<TensorPtr>> m_const_tensors;
    std::vector<std::vector<uint64_t>> m_const_offsets;
};

std::vector<std::vector<int>> parse_input_shape(GraphPtr g, const std::string &opt);
void config_dynamic_shape(GraphPtr graph);
void eval_expr_params(GraphPtr graph, BuilderParams &params);
} // namespace aipubt
#endif //!BUILDER_BUILDER_H
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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-04-29 16:28:29
 */
#ifndef BUILDER_MCBUILDER_H
#define BUILDER_MCBUILDER_H
#include "builder.h"
#include <builder/bldtypes.h>
#include <builder/linker/linker_v2.h>
#include <set>
#include <string>
#include <vector>

namespace aipubt
{
class AIPUELF;
using AIPUELFPtr = std::shared_ptr<AIPUELF>;
struct GMConfig;
struct DSCodePkg;
class GraphELFBuilder : public GraphBuilder
{
public:
    GraphELFBuilder(GraphPtr graph);
    GraphELFBuilder(std::string file);
    GraphELFBuilder(AIPUELFPtr elf);
    virtual ~GraphELFBuilder();
    const AIPUELFPtr get_aipuelf()
    {
        return m_elf;
    }
    virtual bool build_aipu(const TensorList &inputs = TensorList(),
                            const TensorList &outputs = TensorList(),
                            const std::vector<int32_t> &dump_ids = std::vector<int32_t>()) override;
    virtual bool build(const std::string &filename,
                       const std::vector<TensorPtr> &inputs_,
                       const std::vector<TensorPtr> &outputs,
                       const std::vector<int32_t> &dump_ids = std::vector<int32_t>()) override;
    virtual void write(std::ostream &ofs) override;
    virtual void write(const std::string &filename) override;
    virtual uint32_t get_text_size() override;
    // dump files for simlator X2
    // bool dump_sim_x2(const std::vector<TensorPtr> &input_tensors,
    //                  const std::string &dump_prefix,
    //                  const std::string &output_file,
    //                  const uint32_t &batch_num = 1,
    //                  const int32_t &addr_cfg_mode = -1,
    //                  const std::string &addr_cfg_file = "addr_cfg_default.ini",
    //                  const std::string &dumper_str = "");

protected:
    void show_memory_info() override;
    void partition_graph() override;
    virtual void build_text() override;
    void codes_to_kernel_object();
    void aipubin_to_elf();
    void gen_init_ctx(std::stringstream &sss);
    // void generate_epilogue(std::stringstream &m_stream);
    void generate_asm(std::stringstream &m_stream);
    uint32_t get_func_size(const std::string &func_name) const;
    void gen_loop_kernel_asm(std::stringstream &m_stream,
                             const std::string &kernel,
                             const DSCodePkg &codes,
                             bool prefetch_ro = true);
    void gen_func_asm(std::stringstream &m_stream,
                      CodeTreePtr code,
                      std::map<int, DSCodePkg> &code_map,
                      int ro_index);
    // prefetch next layer's code and/or descriptor
    void prefetch_cache(std::stringstream &m_stream,
                        const std::string &symbol,
                        const std::vector<uint32_t> prefetch_desc_len = std::vector<uint32_t>());
    GMConfig build_gmconfig();
    BinSection build_segmmu();
    void build_debug_info();
    LinkerV2 m_linker;
    AIPUELFPtr m_elf;
    std::vector<std::string> m_objects;
    std::map<NodePtr, std::string> m_task_map;
    mutable std::map<std::string, uint32_t> m_func_size;
    std::string m_asm;
    std::string m_mainobj;
};
} // namespace aipubt
#endif //!BUILDER_MCBUILDER_H
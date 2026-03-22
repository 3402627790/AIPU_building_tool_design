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
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2023-03-20 17:00:00
 */
#ifndef BUILDER_DUMPER_X_H
#define BUILDER_DUMPER_X_H

#include "builder/aipu_elf.h"
#include "builder/bldtypes.h"
#include "builder/builder2.h"
#include "builder/dumper/dumper.h"
#include "builder/readonly_param.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/graph.h"
#include "core/node.h"
#include "core/tensor.h"
#include "mem_manager.h"
#include "tools/argparser.h"
#include <map>
#include <vector>

namespace aipubt
{
/* describe memory*/
struct MemInfo
{
    uint32_t ro_offset = 0;
    uint32_t size = 0;
    uint32_t address = 0;
    uint32_t align = 0;
    uint32_t slot_idx = 0;
    uint32_t mem_type = 0; //0: ddr 1: ocm 2: gm
    uint64_t data_offset = 0;
    uint32_t bss_id = 0;
    uint32_t asid = 0;
    std::map<uint32_t, uint32_t> type; //BssType <-> SubsecID
};

enum DataType
{
    text,
    text_init,
    rodata,
    bss,
    ro,
    desc,
    reuse,
    constant,
    global_param,
    tcb,
    stack,
    workspace,
};

const std::map<DataType, std::string> type_string = {
    {DataType::text, "text"},
    {DataType::text_init, "text_init"},
    {DataType::rodata, "rodata"},
    {DataType::bss, "bss"},
    {DataType::ro, "ro"},
    {DataType::desc, "desc"},
    {DataType::reuse, "reuse"},
    {DataType::constant, "const"},
    {DataType::global_param, "global_param"},
    {DataType::tcb, "tcb"},
    {DataType::stack, "stack"},
    {DataType::workspace, "workspace"},
};

enum MemType
{
    ddr,
    gm,
    ocm,
    share
};

const std::map<MemType, std::string> mem_string = {
    {MemType::ddr, "ddr"},
    {MemType::gm, "gm"},
    {MemType::ocm, "ocm"},
    {MemType::share, "share"},
};

const std::map<int, std::string> sub_sec_type = {
    {0, "input"},
    {1, "output"},
    {2, "dump"},
    {3, "intermediate"},
    {4, "constant"},
    {10, "profile_data"},
    {11, "dump_instruction"},
    {12, "printf_fifo"},
    {13, "op_cnt"},
    {14, "error_code"},
    {15, "zerocpy_constant"},
    {16, "output_tensor_shape"},
    {17, "cross_submodel"},
    {18, "dump_tensor_shape"},
    {255, "segmmu_buffer"},
};

struct DataInfo
{
    char *data;
    uint32_t size;
    uint32_t asid;
    DataType type;
    int bssid;
    MemType mem;
    std::string name;
    std::vector<MemInfo> cfg;
};

class DumperX : public Dumper
{
    enum Run_Mode
    {
        run_by_builder = 0,
        run_by_dumper,
    };

public:
    DumperX(BuilderConfig *configs, AIPUELFPtr elf, Target *target, GraphPtr graph = nullptr, const std::vector<ReadOnlyParam> &ro = std::vector<ReadOnlyParam>());
    virtual bool gen_dumper_files(const std::vector<TensorPtr> &input_tensors, const string &dump_prefix, const string &output_file);
    virtual ~DumperX()
    {
        for (auto &info : m_data_info)
        {
            if (info.data && info.size)
            {
                delete[] info.data;
            }
        }
    };

private:
    const uint32_t m_smmu_asid = 2;
    std::map<std::string, aipubt::Any> m_arg_map;
    std::vector<DataInfo> m_data_info;
    std::vector<std::string> m_data_name;
    AIPUELFPtr m_elf;
    std::vector<MemoryObjectPtr> m_slots;
    Run_Mode m_mode;
    GraphPtr m_graph;
    std::vector<ReadOnlyParam> m_ro;
    std::vector<ReadOnlyParam> m_desc;
    std::map<uint32_t, uint32_t> m_ro_id_map;
    std::map<uint32_t, uint32_t> m_ro_offset_map;
    std::map<uint32_t, uint32_t> m_desc_id_map;
    std::map<uint32_t, uint32_t> m_desc_offset_map;
    std::vector<std::vector<uint64_t>> m_asid_vec;
    uint32_t get_ro_offset(uint32_t offset);
    uint32_t get_desc_offset(uint32_t offset);
    uint32_t get_ro_id(uint32_t offset);
    uint32_t get_desc_id(uint32_t offset);
    bool set_ro_offset(uint32_t offset, uint32_t new_offset, uint32_t bss_id);
    bool set_desc_offset(uint32_t offset, uint32_t new_offset, uint32_t bss_id);
    bool update_by_subsections(char *ro, char *desc, uint32_t ro_size, const std::vector<BssSubSectionDescriptor> &sec_vec);
    bool update_by_remap(const RemapcSection &remap, char *ro_bin, char *desc_bin, char *text_bin, uint32_t ro_addr, uint32_t desc_addr, uint32_t text_addr);
    uint32_t gen_ro_data(char *ro, char *bin, uint32_t size, std::vector<SubGraphDesc> &sgs, uint32_t bss_id);
    uint32_t gen_desc_data(char *ro, char *bin, std::vector<SubGraphDesc> &sgs, uint32_t bss_id);
    std::vector<uint64_t> get_asid_vec(uint32_t sm_id, uint32_t mode);
    bool set_asid_vec(uint32_t sm_id, const std::vector<uint64_t> &asid_vec);
    bool set_data_info(char *data, uint32_t size, DataType type, int bssid, uint32_t asid, MemType mem, std::string name);
    bool set_data_info(const std::vector<MemInfo> &cfg, DataType type, int bssid, uint32_t asid, MemType mem, std::string name);
    std::vector<DataInfo> get_data_info(DataType type, int bssid);
    DataInfo get_data_info(DataType type, int bssid, MemType mem);
    bool update_info_data(std::string name, char *data);
    bool update_info_size(std::string name, uint32_t size);
    bool update_info_cfg(std::string name, const std::vector<MemInfo> &cfg);
    bool switch_info_after_gm(DataType type, uint32_t asid);
};
};     // namespace aipubt
#endif //!BUILDER_DUMPER_X_H
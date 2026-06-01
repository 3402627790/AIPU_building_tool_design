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
 * @Date: 2024-01-19 10:40:34
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-01-22 14:38:30
 */
#pragma once
#include "builder/aipu_elf.h"
#include "builder/readonly_param.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"

// global param usage
#define GP_DEFAULT_SHAPE_FLAG (GP_RESERVE0_OFFSET + 0)
#define GP_OUTPUT_SHAPE_BASE (GP_RESERVE0_OFFSET + 4)
#define GP_TEMP_PARAM_BASE (GP_RESERVE0_OFFSET + 8)

namespace aipubt
{
struct DSFuncItem
{
    bool operator==(const DSFuncItem &o)
    {
        if (args.size() != o.args.size())
        {
            return false;
        }
        if (arg_types.size() != o.arg_types.size())
        {
            return false;
        }
        if (arg_dims.size() != o.arg_dims.size())
        {
            return false;
        }
        if (name != o.name)
        {
            return false;
        }
        if (out_base != o.out_base)
        {
            return false;
        }
        if (out_offset != o.out_offset)
        {
            return false;
        }
        if (mode != o.mode)
        {
            return false;
        }
        if (status != o.status)
        {
            return false;
        }
        return true;
    }

    bool operator!=(const DSFuncItem &o)
    {
        return *this == o ? false : true;
    }

    Expr expr;
    std::vector<Symbol> args;
    std::vector<uint32_t> arg_types;
    std::vector<uint32_t> arg_dims;
    std::string func;
    std::string name;
    std::string out_base;
    uint32_t out_offset = 0;
    bool mode = false;
    bool status = false;
};

struct DSFuncDesc
{
    bool operator==(const DSFuncDesc &o)
    {
        if (items.size() != o.items.size())
        {
            return false;
        }
        for (uint32_t i = 0; i < items.size(); ++i)
        {
            if (items[i] != o.items[i])
            {
                return false;
            }
        }
        return true;
    }

    bool operator!=(const DSFuncDesc &d)
    {
        return *this == d ? false : true;
    }

    void reset()
    {
        func_name.clear();
        items.clear();
    }

    std::string func_name;
    std::vector<DSFuncItem> items;
};

struct DSCodePkg
{
    DSCodePkg()
    {
        reset();
    }

    void reset()
    {
        data_aware = false;
        ds = std::vector<std::string>(4);
        fixed_loop_count = false;
        mb = std::vector<std::string>(3);
    }

    bool empty() const
    {
        for (auto &c : ds)
        {
            if (c.size())
            {
                return false;
            }
        }
        for (auto &c : mb)
        {
            if (c.size())
            {
                return false;
            }
        }
        return true;
    }

    // dynamic shape
    bool data_aware = false;
    std::vector<std::string> ds;
    // max-batch
    bool fixed_loop_count = false;
    std::vector<std::string> mb;
};

class DSCodeGenerator
{
public:
    static constexpr const char *set_global_param_fn = "_set_global_param";
    static constexpr const char *record_output_shape_fn = "_record_output_shape";

    DSCodeGenerator(GraphPtr graph);

    void generate(std::stringstream &m_stream,
                  const std::vector<ReadOnlyParam> &ro,
                  const std::vector<ReadOnlyParam> &desc,
                  std::map<int, DSCodePkg> &code_map);

    void generate_global_functions(std::stringstream &m_stream);

private:
    std::string gen_backup_plugin_param_asm(std::stringstream &m_stream,
                                            NodePtr node,
                                            const std::vector<ReadOnlyParam> &ro,
                                            const std::vector<ReadOnlyParam> &desc,
                                            const std::vector<ReadOnlyParam> &desc_param,
                                            const std::vector<ReadOnlyParam> &desc_act,
                                            uint32_t real_ro_size,
                                            uint32_t rel_ro_offset,
                                            bool data_aware);

    void gen_expr_func_asm(std::stringstream &m_stream,
                           Expr expr,
                           DSFuncDesc &fd,
                           const std::string &base,
                           uint32_t offset,
                           bool mode);

    void gen_expr_func_asm(std::stringstream &m_stream,
                           const std::vector<Expr> &exprs,
                           DSFuncDesc &fd,
                           const std::string &base);

    void gen_expr_func_asm(std::stringstream &m_stream,
                           const std::vector<ReadOnlyParam> &params,
                           bool desc_base,
                           uint32_t desc_param_index,
                           uint32_t desc_act_index,
                           uint32_t batch_offset_index,
                           DSFuncDesc &fd);

    std::string gen_modify_plugin_param_asm(std::stringstream &m_stream,
                                            const std::vector<ReadOnlyParam> &ro,
                                            const std::vector<ReadOnlyParam> &desc,
                                            uint32_t desc_param_index,
                                            uint32_t desc_act_index,
                                            bool &data_aware);

    std::string gen_backup_output_shape_asm(NodePtr node,
                                            const std::vector<ReadOnlyParam> &ro);

    std::string gen_loop_count_asm(std::stringstream &m_stream,
                                   NodePtr node);

    std::string gen_modify_addr_asm(std::stringstream &m_stream,
                                    NodePtr node,
                                    const std::vector<ReadOnlyParam> &ro,
                                    const std::vector<ReadOnlyParam> &desc,
                                    uint32_t desc_param_index,
                                    uint32_t desc_act_index,
                                    uint32_t batch_offset_index);

    std::string gen_restore_pp_asm(bool data_aware);
    void gen_set_global_param_asm(std::stringstream &m_stream);
    void gen_record_output_shape_asm(std::stringstream &m_stream);

    GraphPtr m_graph;
    TensorPtr m_tp_base;
    TensorPtr m_ro_base;
    TensorPtr m_desc_base;
    std::vector<Expr> m_total_exprs;
    std::vector<std::string> m_total_funcs;
    std::vector<std::vector<Symbol>> m_total_args;
    std::vector<bool> m_total_status;
    std::map<std::string, std::string> m_func_map;
    std::vector<DSFuncDesc> m_func_descs;
};

int get_data_aware_param_index(NodePtr node,
                               const std::vector<ReadOnlyParam> &ro);
std::string gen_copy_output_shape_code(NodePtr node, int param_index);
std::string gen_restore_pp_code(bool data_aware);
std::string gen_movi_code(uint32_t imm, const std::string &reg);
} // namespace aipubt

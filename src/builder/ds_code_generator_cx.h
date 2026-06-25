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
 * @Date: 2026-05-07 10:00:06
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-05-07 10:00:06
 */
#pragma once
#include "builder/plugin/ds_code_wrapper.h"
#include "ds_code_generator.h"

namespace aipubt
{
struct DSParamDesc
{
    enum SymbolType : int
    {
        GLOBAL = 0,
        DATA_AWARE = 1,
    };

    enum ParamType : int
    {
        PARAM = 0,
        DESC = 1,
        DESC_PARAM = 2,
        DESC_ACT = 3,
        STREAM = 4,
        PARAM_LIST = 5,
    };

    enum UpdateMode : int
    {
        FILL = 0,
        ACC = 1,
    };

    struct ArgDesc
    {
        bool operator==(const ArgDesc &o)
        {
            if (type != o.type)
            {
                return false;
            }
            if (id != o.id)
            {
                return false;
            }
            if (offset != o.offset)
            {
                return false;
            }
            return true;
        }
        bool operator!=(const ArgDesc &o)
        {
            return *this == o ? false : true;
        }

        SymbolType type = GLOBAL;
        size_t id = 0;
        size_t offset = 0;
    };

    struct DstDesc
    {
        bool operator==(const DstDesc &o)
        {
            if (type != o.type)
            {
                return false;
            }
            if (offset != o.offset)
            {
                return false;
            }
            if (index != o.index)
            {
                return false;
            }
            if (mode != o.mode)
            {
                return false;
            }
            return true;
        }
        bool operator!=(const DstDesc &o)
        {
            return *this == o ? false : true;
        }

        ParamType type = PARAM;
        size_t offset = 0;
        int index = -1; // param list
        UpdateMode mode = FILL;
    };

    bool operator==(const DSParamDesc &o)
    {
        if (!expr->equal(o.expr))
        {
            return false;
        }
        if (args.size() != o.args.size())
        {
            return false;
        }
        for (size_t i = 0; i < args.size(); ++i)
        {
            if (args[i] != o.args[i])
            {
                return false;
            }
        }
        if (dsts.size() != o.dsts.size())
        {
            return false;
        }
        for (auto &d : o.dsts)
        {
            bool found = false;
            for (auto &it : dsts)
            {
                if (it == d)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const DSParamDesc &o)
    {
        return *this == o ? false : true;
    }

    bool data_aware() const
    {
        for (auto &arg : args)
        {
            if (arg.type == DATA_AWARE)
            {
                return true;
            }
        }
        return false;
    }

    Expr expr;
    std::vector<ArgDesc> args;
    std::vector<DstDesc> dsts;
};

struct DSKernelDesc
{
    enum KernelType : int
    {
        KERNEL = 0,
        EVAL = 1,
        MAX_BATCH = 2,
        MAX_BATCH_TAIL = 3,
    };

    struct DSRemapTable
    {
        bool operator==(const DSRemapTable &o)
        {
            if (type != o.type)
            {
                return false;
            }
            if (size != o.size)
            {
                return false;
            }
            if (table.size() != o.table.size())
            {
                return false;
            }
            for (auto &kv : o.table)
            {
                if (!table.count(kv.first))
                {
                    return false;
                }
                auto &val = table.at(kv.first);
                if (val.first != kv.second.first)
                {
                    return false;
                }
                if (val.second != kv.second.second)
                {
                    return false;
                }
            }
            return true;
        }
        bool operator!=(const DSRemapTable &o)
        {
            return *this == o ? false : true;
        }

        DSParamDesc::ParamType type = DSParamDesc::PARAM;
        size_t size = 0;
        std::map<size_t, std::pair<DSParamDesc::ParamType, size_t>> table;
    };

    bool operator==(const DSKernelDesc &o)
    {
        if (params.size() != o.params.size())
        {
            return false;
        }
        for (auto &p : o.params)
        {
            bool found = false;
            for (auto &it : params)
            {
                if (it == p)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                return false;
            }
        }
        if (tables.size() != o.tables.size())
        {
            return false;
        }
        for (auto &t : o.tables)
        {
            bool found = false;
            for (auto &it : tables)
            {
                if (it == t)
                {
                    found = true;
                    break;
                }
            }
            if (!found)
            {
                return false;
            }
        }
        return true;
    }
    bool operator!=(const DSKernelDesc &o)
    {
        return *this == o ? false : true;
    }

    bool data_aware() const
    {
        for (auto &p : params)
        {
            if (p.data_aware())
            {
                return true;
            }
        }
        return false;
    }

    std::vector<DSParamDesc> global_params() const
    {
        std::vector<DSParamDesc> ret;
        for (auto &p : params)
        {
            if (!p.data_aware())
            {
                ret.push_back(p);
            }
        }
        return ret;
    }
    std::vector<DSParamDesc> data_aware_params() const
    {
        std::vector<DSParamDesc> ret;
        for (auto &p : params)
        {
            if (p.data_aware())
            {
                ret.push_back(p);
            }
        }
        return ret;
    }

    bool check_param_type(const std::set<DSParamDesc::ParamType> &types) const
    {
        for (auto &p : params)
        {
            for (auto &d : p.dsts)
            {
                if (types.count(d.type))
                {
                    return true;
                }
            }
        }
        return false;
    }
    bool check_remap_type(DSParamDesc::ParamType src, DSParamDesc::ParamType dst) const
    {
        for (auto &t : tables)
        {
            if (t.type == src)
            {
                for (auto &kv : t.table)
                {
                    if (kv.second.first == dst)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    const DSRemapTable &get_remap_table(DSParamDesc::ParamType type) const
    {
        static DSRemapTable unknown;
        for (auto &t : tables)
        {
            if (t.type == type)
            {
                return t;
            }
        }
        return unknown;
    }

    NodePtr node;
    std::shared_ptr<GopDSCodeImpl> impl;
    KernelType type = KERNEL;
    size_t offset = 0;
    int index = -1; // data aware
    std::string name;
    PlaceholderPtr param_base;
    PlaceholderPtr stream_base;
    std::vector<DSParamDesc> params;
    std::vector<DSRemapTable> tables;
};

class DSCodeOptimizer
{
public:
    DSCodeOptimizer() {}
    virtual ~DSCodeOptimizer() = default;

    virtual void optimize();

protected:
    std::vector<DSKernelDesc> m_kernels;
};

class DSCodeGeneratorCL : public DSCodeOptimizer
{
public:
    DSCodeGeneratorCL() {}

    void parse(GraphPtr graph,
               const std::vector<ReadOnlyParam> &total_ro,
               const std::vector<ReadOnlyParam> &total_desc);

    void generate(std::map<int, DSCodePkg> &code_map);

protected:
    void build_remap_tables(GraphPtr graph,
                            NodePtr node,
                            size_t index,
                            const std::vector<ReadOnlyParam> &ro,
                            const std::vector<ReadOnlyParam> &desc,
                            const std::vector<ReadOnlyParam> &desc_param,
                            const std::vector<ReadOnlyParam> &desc_act,
                            DSKernelDesc &res);

    void max_batch_preprocess(NodePtr node,
                              std::vector<ReadOnlyParam> &ro,
                              std::vector<ReadOnlyParam> &desc,
                              std::vector<ReadOnlyParam> &desc_param,
                              std::vector<ReadOnlyParam> &desc_act);

    void generate_kernel_code(std::stringstream &oss,
                              const DSKernelDesc &kernel);

    void generate_eval_code(std::stringstream &oss,
                            const DSKernelDesc &kernel);

    size_t m_core_per_cluster = 0;
    size_t m_max_param_size = 0;
};

class DSCodeGeneratorCPS : public DSCodeOptimizer
{
public:
    DSCodeGeneratorCPS() {}

    void generate(GraphPtr graph);

protected:
    void parse(GraphPtr graph);

    void generate_kernel_code(std::stringstream &oss,
                              const DSKernelDesc &kernel);

    PlaceholderPtr m_param_base;
};

const std::string get_ds_code();
} // namespace aipubt

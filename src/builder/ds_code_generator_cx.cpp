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
#include "ds_code_generator_cx.h"
#include "builder/ro_generator.h"
#include "tongue/symbolic.h"

inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

namespace aipubt
{
static std::stringstream ds_code;
const std::string get_ds_code()
{
    auto code = ds_code.str();
    auto arch = get_target()->isainfo().type;
    if (code.size() && (arch == ISA_ZYv5 || arch == ISA_ZYv6))
    {
        std::stringstream oss;
        oss << "typedef unsigned int uint;\n"
            << "void _aipubt_copy_data(uint, uint, uint);\n"
            << "__attribute__((overloadable)) float ceil(float);\n"
            << "__attribute__((overloadable)) float round(float);\n"
            << "__attribute__((overloadable)) float rint(float);\n"
            << "__attribute__((overloadable)) float log2(float);\n"
            << "__attribute__((overloadable)) float sqrt(float);\n"
            << "__attribute__((overloadable)) float rsqrt(float);\n"
            << "__attribute__((overloadable)) float floor(float);\n"
            << "__attribute__((overloadable)) uint max(uint, uint);\n"
            << "__attribute__((overloadable)) uint min(uint, uint);\n"
            << "__attribute__((overloadable)) float pow(float, float);\n"
            << "__attribute__((overloadable)) uint mul_hi(uint, uint);\n";
        return oss.str() + "\n" + code;
    }
    return code;
}

void DSCodeOptimizer::optimize()
{
    for (auto &kernel : m_kernels)
    {
        if (kernel.params.size() < 2)
        {
            continue;
        }
        std::set<size_t> ran_idxs;
        std::vector<DSParamDesc> new_params;
        for (size_t i = 0; i < kernel.params.size(); ++i)
        {
            if (ran_idxs.count(i))
            {
                continue;
            }
            auto p0 = kernel.params[i];
            for (size_t j = i + 1; j < kernel.params.size(); ++j)
            {
                if (ran_idxs.count(j))
                {
                    continue;
                }
                auto &p1 = kernel.params[j];
                if (p0.expr->equal(p1.expr))
                {
                    p0.dsts.insert(p0.dsts.end(), p1.dsts.begin(), p1.dsts.end());
                    ran_idxs.insert(j);
                }
            }
            new_params.push_back(p0);
            ran_idxs.insert(i);
        }
        if (new_params.size() < kernel.params.size())
        {
            kernel.params = std::move(new_params);
        }
    }

    if (m_kernels.size() < 2)
    {
        return;
    }
    std::set<size_t> ran_idxs;
    for (size_t i = 0; i < m_kernels.size(); ++i)
    {
        if (ran_idxs.count(i))
        {
            continue;
        }
        auto &kernel0 = m_kernels[i];
        for (size_t j = i + 1; j < m_kernels.size(); ++j)
        {
            if (ran_idxs.count(j))
            {
                continue;
            }
            auto &kernel1 = m_kernels[j];
            if (kernel0 == kernel1)
            {
                kernel1.name = kernel0.name;
                DEBUG() << "reuse code for node: " << kernel1.node->name();
                ran_idxs.insert(j);
            }
        }
        ran_idxs.insert(i);
    }
}

void DSCodeGeneratorCL::parse(GraphPtr graph,
                              const std::vector<ReadOnlyParam> &total_ro,
                              const std::vector<ReadOnlyParam> &total_desc)
{
    size_t cluster_num = 1; // TODO
    size_t core_num = get_target()->core_num();
    auto tp_base = graph->global_tensors()[Constants::temp_param_base];
    m_core_per_cluster = core_num / cluster_num;
    m_max_param_size = tp_base->mem_size() / core_num;
    m_kernels.clear();

    std::vector<NodePtr> nodes;
    graph->walk_on_node([&nodes](NodePtr n) {
        if ((n->attrs().count(Constants::plugin_attrs_name) ||
             n->type() == OpType_LayoutConvertor ||
             n->type() == OpType_ForRangeEnder ||
             n->type() == OpType_IfEnder) &&
            n->attrs().count(Constants::ro_offset) &&
            n->attrs().count(Constants::ro_size))
        {
            nodes.push_back(n);
        }
    });

    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    for (auto node : nodes)
    {
        auto ro_offset = node->get_attr<size_t>(Constants::ro_offset) / 4;
        auto ro_size = node->get_attr<size_t>(Constants::ro_size) / 4;

        std::vector<ReadOnlyParam> desc, desc_param, desc_act;
        if (node->attrs().count(Constants::desc_offset) &&
            node->attrs().count(Constants::desc_size))
        {
            auto desc_offset = node->get_attr<size_t>(Constants::desc_offset) / 4;
            auto desc_size = node->get_attr<size_t>(Constants::desc_size) / 4;

            for (size_t i = desc_offset; i < desc_offset + desc_size; ++i)
            {
                desc.push_back(total_desc[i]);
            }
        }
        if (node->attrs().count(Constants::descriptor_param))
        {
            desc_param = node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_param);
        }
        if (node->attrs().count(Constants::descriptor_act))
        {
            desc_act = node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_act);
        }

        for (size_t i = 0; i < ro_size / DEFAULT_RO_NUMBER_PER_LAYER; ++i)
        {
            std::vector<ReadOnlyParam> ro;
            size_t offset = ro_offset + i * DEFAULT_RO_NUMBER_PER_LAYER;
            for (size_t j = offset; j < offset + DEFAULT_RO_NUMBER_PER_LAYER; ++j)
            {
                ro.push_back(total_ro[j]);
            }

            int kernel_idx = -1;
            if (enable_ds)
            {
                auto kernel_id = std::to_string(utils::get_increase_id64());
                m_kernels.push_back(DSKernelDesc());
                auto &kernel = m_kernels.back();
                kernel.node = node;
                kernel.type = DSKernelDesc::KERNEL;
                kernel.offset = offset;
                kernel.param_index = get_data_aware_param_index(node, ro);
                kernel.name = "_modify_param" + kernel_id;

                parse_params(graph, ro, DSParamDesc::PARAM, kernel);
                parse_params(graph, desc, DSParamDesc::DESC, kernel);
                parse_params(graph, desc_param, DSParamDesc::DESC_PARAM, kernel);
                parse_params(graph, desc_act, DSParamDesc::DESC_ACT, kernel);

                if (kernel.params.empty() && kernel.param_index == -1)
                {
                    m_kernels.pop_back();
                }
                else
                {
                    build_remap_tables(graph, node, i, ro, desc, desc_param, desc_act, kernel);
                    kernel_idx = m_kernels.size() - 1;
                }
            }
            if (node->attrs().count(Constants::max_batch_size) &&
                node->attrs().count(Constants::kernel_loop_count))
            {
                auto kernel_id = std::to_string(utils::get_increase_id64());
                m_kernels.push_back(DSKernelDesc());
                auto &kernel = m_kernels.back();
                kernel.node = node;
                kernel.type = DSKernelDesc::EVAL;
                kernel.offset = offset;
                kernel.name = "_eval_expr" + kernel_id;

                ReadOnlyParam loop_cnt = node->get_attr<Expr>(Constants::kernel_loop_count);
                parse_params(graph, {loop_cnt}, DSParamDesc::PARAM, kernel);

                auto new_ro = ro;
                auto new_desc = desc;
                auto new_desc_param = desc_param;
                auto new_desc_act = desc_act;
                max_batch_preprocess(node, new_ro, new_desc, new_desc_param, new_desc_act);

                for (size_t j = 0; j < 2; ++j)
                {
                    m_kernels.push_back(DSKernelDesc());
                    auto &kernel = m_kernels.back();
                    kernel.node = node;
                    kernel.type = j == 0 ? DSKernelDesc::MAX_BATCH : DSKernelDesc::MAX_BATCH_TAIL;
                    kernel.offset = offset;
                    kernel.name = "_modify_addr" + kernel_id + "_" + std::to_string(j);

                    parse_params(graph, new_ro, DSParamDesc::PARAM, kernel);
                    parse_params(graph, new_desc, DSParamDesc::DESC, kernel);
                    parse_params(graph, new_desc_param, DSParamDesc::DESC_PARAM, kernel);
                    parse_params(graph, new_desc_act, DSParamDesc::DESC_ACT, kernel);

                    if (kernel_idx != -1)
                    {
                        kernel.tables = m_kernels[kernel_idx].tables;
                    }
                    else
                    {
                        build_remap_tables(graph, node, i, ro, desc, desc_param, desc_act, kernel);
                    }
                }
            }
        }
    }

    optimize();
}

void DSCodeGeneratorCL::parse_params(GraphPtr graph,
                                     const std::vector<ReadOnlyParam> &params,
                                     DSParamDesc::ParamType type,
                                     DSKernelDesc &res)
{
    auto &symbols = graph->symbols();
    for (size_t i = 0; i < params.size(); ++i)
    {
        Expr expr;
        if (params[i].tensor)
        {
            if ((res.type == DSKernelDesc::MAX_BATCH || res.type == DSKernelDesc::MAX_BATCH_TAIL) &&
                params[i].tensor->attrs().count(Constants::max_batch_offset))
            {
                std::vector<Expr> offsets = params[i].tensor->attrs()[Constants::max_batch_offset];
                expr = res.type == DSKernelDesc::MAX_BATCH ? offsets[0] : offsets[1];
            }
            else if (params[i].tensor->attrs().count(Constants::expr_offset))
            {
                expr = params[i].tensor->get_attr(Constants::expr_offset);
            }
        }
        else if (params[i].expr)
        {
            expr = params[i].expr;
        }
        if (expr)
        {
            res.params.push_back(DSParamDesc());
            auto &param = res.params.back();
            param.expr = expr;

            auto vars = expr->variables();
            for (auto &var : vars)
            {
                if (!symbols.count(var.name()))
                {
                    FATAL() << "unknown symbol: " << var.name();
                }
                auto &sym = symbols.at(var.name());
                param.args.push_back(DSParamDesc::ArgDesc());
                auto &arg = param.args.back();
                arg.type = sym.type == 0 ? DSParamDesc::GLOBAL : DSParamDesc::DATA_AWARE;
                arg.id = var.id();
                arg.offset = 4 * sym.dims[0];
            }

            param.dsts.push_back(DSParamDesc::DstDesc());
            auto &dst = param.dsts.back();
            dst.type = type;
            dst.offset = 4 * i;
            dst.mode = params[i].expr ? DSParamDesc::FILL : DSParamDesc::ACC;
        }
    }
}

void DSCodeGeneratorCL::build_remap_tables(GraphPtr graph,
                                           NodePtr node,
                                           size_t index,
                                           const std::vector<ReadOnlyParam> &ro,
                                           const std::vector<ReadOnlyParam> &desc,
                                           const std::vector<ReadOnlyParam> &desc_param,
                                           const std::vector<ReadOnlyParam> &desc_act,
                                           DSKernelDesc &res)
{
    auto ro_root = graph->global_tensors()[Constants::parambase];
    auto ro_root_object = ro_root->memory_offset()->get_base_object();
    size_t ro_base_start = 0, ro_base_end = 0;
    if (node->attrs().count(Constants::parambase))
    {
        auto ro_base = node->get_attr<TensorPtr>(Constants::parambase);
        ro_base_start = ro_base->memory_offset()->offset() - ro_root_object->offset();
        ro_base_start += index * 4 * ro.size();
        ro_base_end = ro_base_start + 4 * ro.size();
    }

    auto desc_root = graph->global_tensors()[Constants::descriptorbase];
    auto desc_root_object = desc_root->memory_offset()->get_base_object();
    size_t desc_base_start = 0, desc_base_end = 0;
    if (node->attrs().count(Constants::descriptorbase))
    {
        auto desc_base = node->get_attr<TensorPtr>(Constants::descriptorbase);
        desc_base_start = desc_base->memory_offset()->offset() - desc_root_object->offset();
        desc_base_end = desc_base_start + 4 * desc.size();
    }

    size_t desc_param_base_start = 0, desc_param_base_end = 0;
    if (node->attrs().count(Constants::descriptor_param_base))
    {
        auto desc_param_base = node->get_attr<TensorPtr>(Constants::descriptor_param_base);
        desc_param_base_start = desc_param_base->memory_offset()->offset() - desc_root_object->offset();
        desc_param_base_end = desc_param_base_start + 4 * desc_param.size();
    }

    size_t desc_act_base_start = 0, desc_act_base_end = 0;
    if (node->attrs().count(Constants::descriptor_act_base))
    {
        auto desc_act_base = node->get_attr<TensorPtr>(Constants::descriptor_act_base);
        desc_act_base_start = desc_act_base->memory_offset()->offset() - desc_root_object->offset();
        desc_act_base_end = desc_act_base_start + 4 * desc_act.size();
    }

    auto build_remap_table = [&](const std::vector<ReadOnlyParam> &params,
                                 DSParamDesc::ParamType type) {
        res.tables.push_back(DSKernelDesc::DSRemapTable());
        auto &table = res.tables.back();
        table.type = type;
        table.size = 4 * params.size();
        for (size_t i = 0; i < params.size(); ++i)
        {
            if (!params[i].tensor)
            {
                continue;
            }

            auto t = params[i].tensor;
            auto root_object = t->memory_offset()->get_base_object();
            if (root_object == ro_root_object)
            {
                auto base_offset = t->memory_offset()->offset() - root_object->offset();
                if (base_offset >= ro_base_start && base_offset < ro_base_end)
                {
                    size_t offset = base_offset - ro_base_start;
                    table.table[i * 4] = {DSParamDesc::PARAM, offset};
                }
            }
            else if (root_object == desc_root_object)
            {
                auto base_offset = t->memory_offset()->offset() - root_object->offset();
                if (base_offset >= desc_base_start && base_offset < desc_base_end)
                {
                    size_t offset = base_offset - desc_base_start;
                    table.table[i * 4] = {DSParamDesc::DESC, offset};
                }
                else if (base_offset >= desc_param_base_start && base_offset < desc_param_base_end)
                {
                    size_t offset = base_offset - desc_param_base_start;
                    table.table[i * 4] = {DSParamDesc::DESC_PARAM, offset};
                }
                else if (base_offset >= desc_act_base_start && base_offset < desc_act_base_end)
                {
                    size_t offset = base_offset - desc_act_base_start;
                    table.table[i * 4] = {DSParamDesc::DESC_ACT, offset};
                }
            }
        }
    };

    build_remap_table(ro, DSParamDesc::PARAM);
    build_remap_table(desc, DSParamDesc::DESC);
    build_remap_table(desc_param, DSParamDesc::DESC_PARAM);
    build_remap_table(desc_act, DSParamDesc::DESC_ACT);
}

void DSCodeGeneratorCL::max_batch_preprocess(NodePtr node,
                                             std::vector<ReadOnlyParam> &ro,
                                             std::vector<ReadOnlyParam> &desc,
                                             std::vector<ReadOnlyParam> &desc_param,
                                             std::vector<ReadOnlyParam> &desc_act)
{
    std::set<TensorPtr> tensors;
    for (auto inp : node->inputs())
    {
        if (inp->attrs().count(Constants::max_batch_offset))
        {
            tensors.insert(inp);
        }
    }
    for (auto out : node->outputs())
    {
        if (out->attrs().count(Constants::max_batch_offset))
        {
            tensors.insert(out);
        }
    }

    auto process_max_batch_params = [&tensors](std::vector<ReadOnlyParam> &params) {
        std::vector<ReadOnlyParam> new_params;
        for (size_t i = 0; i < params.size(); ++i)
        {
            ReadOnlyParam p(0);
            if (params[i].tensor)
            {
                auto t = params[i].tensor;
                if (tensors.count(t))
                {
                    continue;
                }
                for (auto it : tensors)
                {
                    if (it->memory_offset()->is_shared_by(t->memory_offset()))
                    {
                        std::vector<Expr> offsets = it->attrs()[Constants::max_batch_offset];
                        t->set_attr(Constants::max_batch_offset, offsets);
                        break;
                    }
                }
                if (t->attrs().count(Constants::max_batch_offset))
                {
                    p = t;
                }
            }
            new_params.push_back(p);
        }
        params = std::move(new_params);
    };

    process_max_batch_params(ro);
    process_max_batch_params(desc);
    process_max_batch_params(desc_param);
    process_max_batch_params(desc_act);
}

void DSCodeGeneratorCL::generate(std::map<int, DSCodePkg> &code_map)
{
    auto &oss = ds_code;
    oss.clear();

    std::set<std::string> kernel_names;
    for (auto &kernel : m_kernels)
    {
        auto &codes = code_map[kernel.offset];
        if (kernel.type == DSKernelDesc::KERNEL)
        {
            codes.data_aware = kernel.data_aware();
            if (kernel.params.size())
            {
                codes.ds[0] = kernel.name;
                codes.ds[3] = gen_restore_pp_code(codes.data_aware);
                if (!kernel_names.count(kernel.name))
                {
                    generate_kernel_code(oss, kernel);
                    kernel_names.insert(kernel.name);
                }
            }
            codes.ds[2] = gen_copy_output_shape_code(kernel.node, kernel.param_index);
        }
        else if (kernel.type == DSKernelDesc::EVAL)
        {
            auto expr = kernel.params[0].expr;
            codes.fixed_loop_count = expr->is_constant();
            if (codes.fixed_loop_count)
            {
                if ((int64_t)expr->value() > 0)
                {
                    codes.mb[0] = gen_movi_code(expr->value(), "r0");
                }
            }
            else
            {
                codes.mb[0] = kernel.name;
                if (!kernel_names.count(kernel.name))
                {
                    generate_eval_code(oss, kernel);
                    kernel_names.insert(kernel.name);
                }
            }
        }
        else
        {
            int idx = kernel.type == DSKernelDesc::MAX_BATCH ? 1 : 2;
            codes.mb[idx] = kernel.name;
            if (!kernel_names.count(kernel.name))
            {
                generate_kernel_code(oss, kernel);
                kernel_names.insert(kernel.name);
            }
        }
    }
}

void DSCodeGeneratorCL::generate_kernel_code(std::stringstream &oss,
                                             const DSKernelDesc &kernel)
{
    static const std::set<DSParamDesc::ParamType> desc_types = {
        DSParamDesc::DESC,
        DSParamDesc::DESC_PARAM,
        DSParamDesc::DESC_ACT,
    };

    size_t var_id = 0;
    auto tab = std::string(4, ' ');
    auto tab2 = std::string(8, ' ');

    auto tcbp = "tcbp";
    auto pp = "pp";
    auto tid = "tid";
    auto gp = "gp";
    oss << "void " << kernel.name << "()\n{\n"
        << tab << "uint *" << tcbp << " = (uint *)__builtin_aipu_mfctrl0(0x14);\n"
        << tab << "uint *" << pp << " = (uint *)" << tcbp << "[14];\n"
        << tab << "uint " << tid << " = " << tcbp << "[20] >> 16;\n"
        << tab << "uint *" << gp << " = (uint *)" << tcbp << "[28];\n";

    auto data_aware = kernel.data_aware();
    if (kernel.type == DSKernelDesc::KERNEL &&
        !data_aware && BuilderConfig::exist(Constants::default_input_shape))
    {
        oss << tab << "if (" << gp << "[" << GP_DEFAULT_SHAPE_FLAG / 4 << "] != 0) {\n"
            << tab2 << "return;\n"
            << tab << "}\n";
    }

    auto desc = "desc";
    auto desc_param = "desc_param";
    auto desc_act = "desc_act";
    auto desc_size = kernel.get_remap_table(DSParamDesc::DESC).size;
    auto desc_param_size = kernel.get_remap_table(DSParamDesc::DESC_PARAM).size;
    auto desc_act_size = kernel.get_remap_table(DSParamDesc::DESC_ACT).size;
    bool has_desc_param = kernel.check_param_type(desc_types);
    bool copy_param = kernel.type == DSKernelDesc::KERNEL ? true : false;
    if (copy_param)
    {
        auto val = "val" + std::to_string(var_id++);
        auto cluster_id = "cluster_id";
        auto core_id = "core_id";
        auto tp = "tp";
        auto tpp = "tpp";
        oss << tab << "uint " << val << " = " << tcbp << "[19];\n"
            << tab << "uint " << cluster_id << " = " << val << " >> 16;\n"
            << tab << "uint " << core_id << " = " << val << " & 0xf;\n"
            << tab << "uint " << tp << " = " << pp << "[54];\n"
            << tab << "uint *" << tpp << " = (uint *)((" << cluster_id << " * " << m_core_per_cluster
            << " + " << core_id << ") * " << m_max_param_size << " + 256 + " << tp << ");\n";

        auto copy_desc = has_desc_param;
        if (kernel.check_remap_type(DSParamDesc::DESC, DSParamDesc::PARAM) ||
            kernel.check_remap_type(DSParamDesc::DESC_PARAM, DSParamDesc::PARAM) ||
            kernel.check_remap_type(DSParamDesc::DESC_ACT, DSParamDesc::PARAM))
        {
            copy_desc = true;
        }
        size_t ro_size = 4 * DEFAULT_RO_NUMBER_PER_LAYER;
        size_t total_desc_size = 0;
        if (copy_desc)
        {
            auto offset = ro_size;
            if (desc_size)
            {
                oss << tab << "uint *" << desc << " = " << tpp << " + " << offset / 4 << ";\n";
            }
            offset += aligned(desc_size, 256);
            if (desc_param_size)
            {
                oss << tab << "uint *" << desc_param << " = " << tpp << " + " << offset / 4 << ";\n";
            }
            offset += aligned(desc_param_size, 256);
            if (desc_act_size)
            {
                oss << tab << "uint *" << desc_act << " = " << tpp << " + " << offset / 4 << ";\n";
            }
            offset += desc_act_size;
            total_desc_size = offset - ro_size;
        }

        oss << tab << "if (" << tid << " == 0) {\n"
            << tab2 << "*(" << tpp << " - 1) = (uint)" << pp << ";\n"
            << tab2 << "_aipubt_copy_data((uint)" << pp << ", (uint)"
            << tpp << ", " << ro_size << ");\n";

        if (copy_desc)
        {
            // write-back and invalidate data cache
            int cache_len = std::ceil(std::log2(std::max(total_desc_size / 1024.0, 1.0))) + 1;
            oss << tab2 << "__builtin_aipu_cacheop((uint)" << desc << ", 24, " << cache_len << ");\n";

            if (desc_size)
            {
                oss << tab2 << "_aipubt_copy_data(" << pp << "[55], (uint)"
                    << desc << ", " << desc_size << ");\n";
            }
            if (desc_param_size)
            {
                oss << tab2 << "_aipubt_copy_data(" << pp << "[56], (uint)"
                    << desc_param << ", " << desc_param_size << ");\n";
            }
            if (desc_act_size)
            {
                oss << tab2 << "_aipubt_copy_data(" << pp << "[57], (uint)"
                    << desc_act << ", " << desc_act_size << ");\n";
            }
        }

        auto generate_remap_code = [&](const DSKernelDesc::DSRemapTable &table) {
            auto get_param_base = [&](DSParamDesc::ParamType type) {
                if (type == DSParamDesc::DESC)
                {
                    return desc;
                }
                else if (type == DSParamDesc::DESC_PARAM)
                {
                    return desc_param;
                }
                else if (type == DSParamDesc::DESC_ACT)
                {
                    return desc_act;
                }
                return tpp;
            };

            auto src_base = get_param_base(table.type);
            for (auto &kv : table.table)
            {
                if (!copy_desc && desc_types.count(kv.second.first))
                {
                    continue;
                }
                auto dst_base = get_param_base(kv.second.first);
                oss << tab2 << src_base << "[" << kv.first / 4 << "] = (uint)("
                    << dst_base << " + " << kv.second.second / 4 << ");\n";
            }
        };

        generate_remap_code(kernel.get_remap_table(DSParamDesc::PARAM));
        if (copy_desc)
        {
            generate_remap_code(kernel.get_remap_table(DSParamDesc::DESC));
            generate_remap_code(kernel.get_remap_table(DSParamDesc::DESC_PARAM));
            generate_remap_code(kernel.get_remap_table(DSParamDesc::DESC_ACT));
        }
        pp = tpp;
        oss << tab << "}\n";
    }
    else if (has_desc_param)
    {
        if (desc_size)
        {
            oss << tab << "uint *" << desc << " = (uint *)" << pp << "[55];\n";
        }
        if (desc_param_size)
        {
            oss << tab << "uint *" << desc_param << " = (uint *)" << pp << "[56];\n";
        }
        if (desc_act_size)
        {
            oss << tab << "uint *" << desc_act << " = (uint *)" << pp << "[57];\n";
        }
    }


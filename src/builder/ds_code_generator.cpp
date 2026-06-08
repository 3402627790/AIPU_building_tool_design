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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-21 16:16:53
 */
#include "ds_code_generator.h"
#include "builder/builder.h"
#include "builder/plugin.h"
#include "builder/ro_generator.h"
#include "core/plugin_loader.h"
#include "tongue/sbe/ssa.h"
#include "tongue/symbolic.h"

inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

namespace aipubt
{
static void split_imm(uint32_t imm, int16_t &low, int16_t &high)
{
    uint16_t half = imm & 0xffff;
    low = *(int16_t *)(&half);
    half = (imm >> 16) & 0xffff;
    high = *(int16_t *)(&half);
}

static void gen_movi_asm(std::stringstream &m_stream, uint32_t imm, const std::string &reg)
{
    int16_t low, high;
    split_imm(imm, low, high);

    m_stream << "{mov " << reg << "," << low << ";}\n";
    if (low < 0 || high != 0)
    {
        m_stream << "{movh " << reg << "," << high << ";}\n";
    }
}

std::string gen_movi_code(uint32_t imm, const std::string &reg)
{
    std::stringstream oss;
    gen_movi_asm(oss, imm, reg);
    return oss.str();
}

static bool has_desc_exprs(const std::vector<ReadOnlyParam> &desc,
                           const std::vector<ReadOnlyParam> &desc_param,
                           const std::vector<ReadOnlyParam> &desc_act)
{
    auto check_expr = [](const std::vector<ReadOnlyParam> &param) {
        for (uint32_t i = 0; i < param.size(); ++i)
        {
            if (param[i].tensor)
            {
                if (param[i].tensor->attrs().count(Constants::expr_offset))
                {
                    return true;
                }
                if (param[i].tensor->attrs().count(Constants::max_batch_offset))
                {
                    return true;
                }
            }
            else if (param[i].expr)
            {
                return true;
            }
        }
        return false;
    };

    return check_expr(desc) || check_expr(desc_param) || check_expr(desc_act);
}

static bool is_desc_mapped_to_ro(NodePtr node,
                                 TensorPtr ro_root,
                                 uint32_t rel_ro_offset,
                                 const std::vector<ReadOnlyParam> &desc,
                                 const std::vector<ReadOnlyParam> &desc_param,
                                 const std::vector<ReadOnlyParam> &desc_act)
{
    auto ro_base = node->get_attr<TensorPtr>(Constants::parambase);
    uint32_t ro_size = DEFAULT_RO_NUMBER_PER_LAYER * sizeof(uint32_t);

    // compute ro address section
    auto ro_root_object = ro_root->memory_offset()->get_base_object();
    uint32_t ro_base_offset = ro_base->memory_offset()->offset() - ro_root_object->offset() + rel_ro_offset;
    uint32_t ro_base_end = ro_base_offset + ro_size;

    auto check_mapping = [&](const std::vector<ReadOnlyParam> &param) {
        for (uint32_t i = 0; i < param.size(); ++i)
        {
            if (param[i].tensor)
            {
                auto tensor = param[i].tensor;
                auto root_object = tensor->memory_offset()->get_base_object();

                if (root_object == ro_root_object)
                {
                    auto base_offset = tensor->memory_offset()->offset() - root_object->offset();
                    if (base_offset >= ro_base_offset && base_offset < ro_base_end)
                    {
                        return true;
                    }
                }
            }
        }
        return false;
    };

    return check_mapping(desc) || check_mapping(desc_param) || check_mapping(desc_act);
}

static void gen_remap_param_asm(std::stringstream &m_stream,
                                NodePtr node,
                                TensorPtr ro_root,
                                TensorPtr desc_root,
                                uint32_t rel_ro_offset,
                                uint32_t desc_size,
                                uint32_t desc_param_size,
                                uint32_t desc_act_size,
                                const std::vector<ReadOnlyParam> &param,
                                const std::string &dst_base,
                                bool remap_to_desc = true)
{
    auto ro_base = node->get_attr<TensorPtr>(Constants::parambase);
    uint32_t ro_size = DEFAULT_RO_NUMBER_PER_LAYER * sizeof(uint32_t);

    TensorPtr desc_base = nullptr;
    if (node->attrs().count(Constants::descriptorbase))
    {
        desc_base = node->get_attr<TensorPtr>(Constants::descriptorbase);
    }
    TensorPtr desc_param_base = nullptr;
    if (node->attrs().count(Constants::descriptor_param_base))
    {
        desc_param_base = node->get_attr<TensorPtr>(Constants::descriptor_param_base);
    }
    TensorPtr desc_act_base = nullptr;
    if (node->attrs().count(Constants::descriptor_act_base))
    {
        desc_act_base = node->get_attr<TensorPtr>(Constants::descriptor_act_base);
    }

    // compute ro and desc address section
    auto ro_root_object = ro_root->memory_offset()->get_base_object();
    uint32_t ro_base_offset = ro_base->memory_offset()->offset() - ro_root_object->offset() + rel_ro_offset;
    uint32_t ro_base_end = ro_base_offset + ro_size;

    auto desc_root_object = desc_root->memory_offset()->get_base_object();
    uint32_t desc_base_offset = 0;
    if (desc_base)
    {
        desc_base_offset = desc_base->memory_offset()->offset() - desc_root_object->offset();
    }
    uint32_t desc_base_end = desc_base_offset + desc_size;

    uint32_t desc_param_base_offset = 0;
    if (desc_param_base)
    {
        desc_param_base_offset = desc_param_base->memory_offset()->offset() - desc_root_object->offset();
    }
    uint32_t desc_param_base_end = desc_param_base_offset + desc_param_size;

    uint32_t desc_act_base_offset = 0;
    if (desc_act_base)
    {
        desc_act_base_offset = desc_act_base->memory_offset()->offset() - desc_root_object->offset();
    }
    uint32_t desc_act_base_end = desc_act_base_offset + desc_act_size;

    for (uint32_t i = 0; i < param.size(); ++i)
    {
        uint32_t offset = 0, dst_offset = 4 * i;
        std::string base;

        if (param[i].tensor)
        {
            auto tensor = param[i].tensor;
            auto root_object = tensor->memory_offset()->get_base_object();

            if (root_object == ro_root_object)
            {
                auto base_offset = tensor->memory_offset()->offset() - root_object->offset();
                if (base_offset >= ro_base_offset && base_offset < ro_base_end)
                {
                    offset = base_offset - ro_base_offset;
                    base = "r2";
                }
            }
            else if (remap_to_desc && root_object == desc_root_object)
            {
                auto base_offset = tensor->memory_offset()->offset() - root_object->offset();
                if (base_offset >= desc_base_offset && base_offset < desc_base_end)
                {
                    offset = base_offset - desc_base_offset;
                    base = "r3";
                }
                else if (base_offset >= desc_param_base_offset && base_offset < desc_param_base_end)
                {
                    offset = base_offset - desc_param_base_offset;
                    base = "r4";
                }
                else if (base_offset >= desc_act_base_offset && base_offset < desc_act_base_end)
                {
                    offset = base_offset - desc_act_base_offset;
                    base = "r5";
                }
            }
        }

        if (base.size())
        {
            if (offset > 1023)
            {
                gen_movi_asm(m_stream, offset, "r8");
                m_stream << "{add r8,r8," << base << ";}\n";
            }
            else if (offset > 0)
            {
                m_stream << "{add r8," << base << "," << offset << ";}\n";
            }

            if (dst_offset > 1023)
            {
                gen_movi_asm(m_stream, dst_offset, "r9");
                m_stream << "{add r9,r9," << dst_base << ";}\n";
                if (offset > 0)
                {
                    m_stream << "{st r8,r9,0;}\n";
                }
                else
                {
                    m_stream << "{st " << base << ",r9,0;}\n";
                }
            }
            else if (dst_offset > 255)
            {
                m_stream << "{add r9," << dst_base << "," << dst_offset << ";}\n";
                if (offset > 0)
                {
                    m_stream << "{st r8,r9,0;}\n";
                }
                else
                {
                    m_stream << "{st " << base << ",r9,0;}\n";
                }
            }
            else
            {
                if (offset > 0)
                {
                    m_stream << "{st r8," << dst_base << "," << dst_offset << ";}\n";
                }
                else
                {
                    m_stream << "{st " << base << "," << dst_base << "," << dst_offset << ";}\n";
                }
            }
        }
    }
}

// dma trans between ddr and lsram
static void gen_dma_copy_asm(std::stringstream &m_stream,
                             const std::string &descp,
                             const std::string &srcp,
                             const std::string &dstp,
                             uint32_t copy_size,
                             bool init_dma_desc,
                             bool ddr_to_lsram = false,
                             bool lsram_to_ddr = false)
{
    if (copy_size > 0xffffff)
    {
        WARN() << "DMA transfer size should be less than " << 0xffffff << ", but got " << copy_size << "!";
    }

    // set dma descriptor
    auto low = static_cast<int16_t>(copy_size & 0xffff);
    auto high = static_cast<int16_t>((copy_size >> 16) & 0xff);
    m_stream << "{mov r8," << low << ";}\n"
             << "{sth r8," << descp << ",4;}\n"
             << "{sth r8," << descp << ",20; add r8,r31," << high << ";}\n";
    if (init_dma_desc)
    {
        m_stream << "{stb r8," << descp << ",6;}\n"
                 << "{stb r8," << descp << ",22; add r8,r31,32;}\n"
                 << "{sth r8," << descp << ",0;}\n"
                 << "{sth r8," << descp << ",2;}\n"
                 << "{sth r8," << descp << ",16;}\n"
                 << "{sth r8," << descp << ",18; add r8,r31,1;}\n"
                 << "{stb r8," << descp << ",7;}\n"
                 << "{stb r8," << descp << ",23;}\n"
                 << "{st r31," << descp << ",8;}\n"
                 << "{st r31," << descp << ",12;}\n";
    }
    else
    {
        m_stream << "{stb r8," << descp << ",6;}\n"
                 << "{stb r8," << descp << ",22; add r8,r31,1;}\n";
    }

    // start dma trans
    m_stream << "{lsl r9,r8,8;}\n"
             << "{or r9,r9,r8; add r8,r31,2;}\n";
    if (ddr_to_lsram)
    {
        m_stream << "{dma 0,r9,0,4,0,1,0," << descp << "," << dstp << "," << srcp << ";}\n";
    }
    else if (lsram_to_ddr)
    {
        m_stream << "{dma 0,r9,0,4,0,0,0," << descp << "," << srcp << "," << dstp << ";}\n";
    }
    else // ddr to ddr by default
    {
        m_stream << "{dma 0,r9,0,0,0,1,0," << descp << "," << dstp << "," << srcp << ";}\n";
    }

    // wait for dma done
    m_stream << "{xorh r8,r8,0;}\n"
             << "{wfe r8,1;}\n";
}

// reorganize order of expr funcs for result reuse
static void reorganize_items(DSFuncDesc &fd)
{
    if (fd.items.empty())
    {
        return;
    }

    std::map<std::string, DSFuncItem> item_map;
    std::map<std::string, std::vector<std::string>> out_base_map;
    std::map<std::string, std::vector<int>> out_offset_map;
    std::map<std::string, std::vector<bool>> mode_map;
    std::map<std::string, std::vector<bool>> spill_map;

    for (uint32_t i = 0; i < fd.items.size(); ++i)
    {
        auto item = fd.items[i];
        std::string func_id;
        if (item.expr->op() == ExprImpl::Op::Val)
        {
            uint32_t val = item.expr->value();
            func_id = std::to_string(val);
        }
        else
        {
            std::ostringstream oss;
            oss << item.func;
            for (auto &arg : item.args)
            {
                oss << arg.name();
            }
            func_id = oss.str();
        }
        item_map[func_id] = item;
        out_base_map[func_id].push_back(item.out_base);
        out_offset_map[func_id].push_back(item.out_offset);
        mode_map[func_id].push_back(item.mode);
        spill_map[func_id].push_back(item.status);
    }

    fd.reset();
    for (auto &kv : item_map)
    {
        auto &base_list = out_base_map[kv.first];
        auto &offset_list = out_offset_map[kv.first];
        auto &mode_list = mode_map[kv.first];
        auto &spill_list = spill_map[kv.first];
        for (uint32_t i = 0; i < base_list.size(); ++i)
        {
            auto item = kv.second;
            item.name = kv.first;
            item.out_base = base_list[i];
            item.out_offset = offset_list[i];
            item.mode = mode_list[i];
            item.status = spill_list[i];
            fd.items.push_back(item);
        }
    }
}

static void split_items(const DSFuncDesc &fd, DSFuncDesc &fd0, DSFuncDesc &fd1)
{
    fd0.func_name = fd.func_name;
    fd1.func_name = fd.func_name;
    for (uint32_t i = 0; i < fd.items.size(); ++i)
    {
        bool data_aware = false;
        for (auto type : fd.items[i].arg_types)
        {
            if (type == 1)
            {
                data_aware = true;
                break;
            }
        }
        if (data_aware)
        {
            fd1.items.push_back(fd.items[i]);
        }
        else
        {
            fd0.items.push_back(fd.items[i]);
        }
    }
}

static void gen_eval_expr_asm(std::stringstream &m_stream,
                              const DSFuncDesc &fd,
                              uint32_t idx,
                              const std::string &tr)
{
    auto item = fd.items[idx];
    std::vector<std::string> bases;
    std::vector<uint32_t> offsets;
    bool quick_call = !item.status;
    for (size_t i = 0; i < item.args.size(); ++i)
    {
        if (item.arg_types[i] == 1)
        {
            bases.push_back("r10");
        }
        else
        {
            bases.push_back("r9");
        }
        uint32_t offset = 4 * item.arg_dims[i];
        offsets.push_back(offset);
        if (offset > 255)
        {
            quick_call = false;
        }
    }
    if (quick_call)
    {
        if (item.expr->op() == ExprImpl::Op::Val)
        {
            uint32_t val = item.expr->value();
            gen_movi_asm(m_stream, val, "r0");
        }
        else if (item.expr->op() == ExprImpl::Op::Constant)
        {
            m_stream << "{ld r0," << bases[0] << "," << offsets[0] << ";}\n";
        }
        else
        {
            uint32_t size = item.args.size();
            for (uint32_t i = 0; i < size / 2 + size % 2 - 1; ++i)
            {
                m_stream << "{ld r" << 2 * i << "," << bases[2 * i] << "," << offsets[2 * i] << ";"
                         << " ld r" << 2 * i + 1 << "," << bases[2 * i + 1] << "," << offsets[2 * i + 1] << ";}\n";
            }
            if (size % 2 == 0)
            {
                m_stream << "{ld r" << size - 2 << "," << bases[size - 2] << "," << offsets[size - 2] << ";"
                         << " ld r" << size - 1 << "," << bases[size - 1] << "," << offsets[size - 1] << ";"
                         << " bl " << item.func << ";}\n";
            }
            else
            {
                m_stream << "{ld r" << size - 1 << "," << bases[size - 1] << "," << offsets[size - 1] << ";"
                         << " bl " << item.func << ";}\n";
            }
        }
    }
    else
    {
        for (uint32_t i = 0; i < item.args.size(); ++i)
        {
            if (i >= 8)
            {
                break;
            }
            std::string base = bases[i];
            uint32_t offset = offsets[i];
            if (offset > 1023)
            {
                gen_movi_asm(m_stream, offset, tr);
                m_stream << "{add " << tr << "," << tr << "," << base << ";}\n"
                         << "{ld r" << i << "," << tr << ",0;}\n";
            }
            else if (offset > 255)
            {
                m_stream << "{add " << tr << "," << base << "," << offset << ";}\n"
                         << "{ld r" << i << "," << tr << ",0;}\n";
            }
            else
            {
                m_stream << "{ld r" << i << "," << base << "," << offset << ";}\n";
            }
        }

        if (item.args.size() > 8)
        {
            for (int i = item.args.size() - 1; i >= 8; --i)
            {
                std::string base = bases[i];
                uint32_t offset = offsets[i];
                if (offset > 1023)
                {
                    gen_movi_asm(m_stream, offset, tr);
                    m_stream << "{add " << tr << "," << tr << "," << base << ";}\n"
                             << "{ld " << tr << "," << tr << ",0;}\n";
                }
                else if (offset > 255)
                {
                    m_stream << "{add " << tr << "," << base << "," << offset << ";}\n"
                             << "{ld " << tr << "," << tr << ",0;}\n";
                }
                else
                {
                    m_stream << "{ld " << tr << "," << base << "," << offset << ";}\n";
                }
                m_stream << "{sub sp,sp,4;}\n"
                         << "{st " << tr << ",sp,0;}\n";
            }
        }

        if (item.expr->op() == ExprImpl::Op::Val)
        {
            uint32_t val = item.expr->value();
            gen_movi_asm(m_stream, val, "r0");
        }
        else if (item.expr->op() != ExprImpl::Op::Constant)
        {
            if (item.status)
            {
                m_stream << "{sub sp,sp,8;}\n"
                         << "{st fp,sp,0; add fp,sp,0;}\n"
                         << "{bl " << item.func << ";}\n"
                         << "{ld fp,sp,0; add sp,sp,8;}\n";
            }
            else
            {
                m_stream << "{bl " << item.func << ";}\n";
            }
        }

        if (item.args.size() > 8)
        {
            uint32_t offset = 4 * (item.args.size() - 8);
            if (offset > 1023)
            {
                gen_movi_asm(m_stream, offset, tr);
                m_stream << "{add sp,sp," << tr << ";}\n";
            }
            else
            {
                m_stream << "{add sp,sp," << offset << ";}\n";
            }
        }
    }
}

static void gen_modify_param_asm(std::stringstream &m_stream,
                                 const DSFuncDesc &fd,
                                 const std::string &tr0 = "r16",
                                 const std::string &tr1 = "r17")
{
    for (uint32_t i = 0; i < fd.items.size(); ++i)
    {
        gen_eval_expr_asm(m_stream, fd, i, tr0);

        std::vector<std::string> bases;
        std::vector<uint32_t> offsets;
        std::vector<bool> modes;
        for (uint32_t j = i; j < fd.items.size(); ++j)
        {
            if (fd.items[j].name != fd.items[i].name)
            {
                break;
            }
            offsets.push_back(fd.items[j].out_offset);
            bases.push_back(fd.items[j].out_base);
            modes.push_back(fd.items[j].mode);
        }

        for (uint32_t j = 0; j < offsets.size(); ++j)
        {
            if (j + 2 < offsets.size())
            {
                if (modes[j] && modes[j + 1] && modes[j + 2])
                {
                    if (offsets[j] <= 255 && offsets[j + 1] <= 255 && offsets[j + 2] <= 255)
                    {
                        m_stream << "{ld " << tr0 << "," << bases[j] << "," << offsets[j] << ";"
                                 << " ld " << tr1 << "," << bases[j + 1] << "," << offsets[j + 1] << ";}\n"
                                 << "{add " << tr0 << "," << tr0 << ",r0; add " << tr1 << "," << tr1 << ",r0;}\n"
                                 << "{st " << tr0 << "," << bases[j] << "," << offsets[j] << ";"
                                 << " ld " << tr0 << "," << bases[j + 2] << "," << offsets[j + 2] << ";}\n"
                                 << "{st " << tr1 << "," << bases[j + 1] << "," << offsets[j + 1] << ";"
                                 << " add " << tr0 << "," << tr0 << ",r0;}\n"
                                 << "{st " << tr0 << "," << bases[j + 2] << "," << offsets[j + 2] << ";}\n";
                        j += 2;
                        continue;
                    }
                }
            }
            if (j + 1 < offsets.size())
            {
                if (modes[j] && modes[j + 1])
                {
                    if (offsets[j] <= 255 && offsets[j + 1] <= 255)
                    {
                        m_stream << "{ld " << tr0 << "," << bases[j] << "," << offsets[j] << ";"
                                 << " ld " << tr1 << "," << bases[j + 1] << "," << offsets[j + 1] << ";}\n"
                                 << "{add " << tr0 << "," << tr0 << ",r0; add " << tr1 << "," << tr1 << ",r0;}\n"
                                 << "{st " << tr0 << "," << bases[j] << "," << offsets[j] << ";}\n"
                                 << "{st " << tr1 << "," << bases[j + 1] << "," << offsets[j + 1] << ";}\n";
                        j += 1;
                        continue;
                    }
                }
                if (!modes[j] && !modes[j + 1])
                {
                    if (offsets[j] > 1023 && offsets[j] <= 32767 &&
                        offsets[j + 1] > 1023 && offsets[j + 1] <= 32767)
                    {
                        m_stream << "{mov " << tr0 << "," << offsets[j] << ";}\n"
                                 << "{add " << tr0 << "," << tr0 << "," << bases[j] << ";"
                                 << " mov " << tr1 << "," << offsets[j + 1] << ";}\n"
                                 << "{st r0," << tr0 << ",0;"
                                 << " add " << tr1 << "," << tr1 << "," << bases[j + 1] << ";}\n"
                                 << "{st r0," << tr1 << ",0;}\n";
                        j += 1;
                        continue;
                    }
                    if (offsets[j] <= 1023 && offsets[j + 1] <= 1023)
                    {
                        if (offsets[j] > 255 && offsets[j + 1] > 255)
                        {
                            m_stream << "{add " << tr0 << "," << bases[j] << "," << offsets[j] << ";"
                                     << " add " << tr1 << "," << bases[j + 1] << "," << offsets[j + 1] << ";}\n"
                                     << "{st r0," << tr0 << ",0;}\n"
                                     << "{st r0," << tr1 << ",0;}\n";
                            j += 1;
                            continue;
                        }
                        else if (offsets[j] <= 255 && offsets[j + 1] > 255)
                        {
                            m_stream << "{st r0," << bases[j] << "," << offsets[j] << ";"
                                     << " add " << tr0 << "," << bases[j + 1] << "," << offsets[j + 1] << ";}\n"
                                     << "{st r0," << tr0 << ",0;}\n";
                            j += 1;
                            continue;
                        }
                        else if (offsets[j] > 255 && offsets[j + 1] <= 255)
                        {
                            m_stream << "{st r0," << bases[j + 1] << "," << offsets[j + 1] << ";"
                                     << " add " << tr0 << "," << bases[j] << "," << offsets[j] << ";}\n"
                                     << "{st r0," << tr0 << ",0;}\n";
                            j += 1;
                            continue;
                        }
                    }
                }
            }
            if (offsets[j] > 1023)
            {
                gen_movi_asm(m_stream, offsets[j], tr0);
                m_stream << "{add " << tr0 << "," << tr0 << "," << bases[j] << ";}\n";
                if (modes[j])
                {
                    m_stream << "{ld " << tr1 << "," << tr0 << ",0;}\n"
                             << "{add " << tr1 << "," << tr1 << ",r0;}\n"
                             << "{st " << tr1 << "," << tr0 << ",0;}\n";
                }
                else
                {
                    m_stream << "{st r0," << tr0 << ",0;}\n";
                }
            }
            else if (offsets[j] > 255)
            {
                m_stream << "{add " << tr0 << "," << bases[j] << "," << offsets[j] << ";}\n";
                if (modes[j])
                {
                    m_stream << "{ld " << tr1 << "," << tr0 << ",0;}\n"
                             << "{add " << tr1 << "," << tr1 << ",r0;}\n"
                             << "{st " << tr1 << "," << tr0 << ",0;}\n";
                }
                else
                {
                    m_stream << "{st r0," << tr0 << ",0;}\n";
                }
            }
            else
            {
                if (modes[j])
                {
                    m_stream << "{ld " << tr0 << "," << bases[j] << "," << offsets[j] << ";}\n"
                             << "{add " << tr0 << "," << tr0 << ",r0;}\n"
                             << "{st " << tr0 << "," << bases[j] << "," << offsets[j] << ";}\n";
                }
                else
                {
                    m_stream << "{st r0," << bases[j] << "," << offsets[j] << ";}\n";
                }
            }
        }
        i += offsets.size() - 1;
    }
}

static void patch_max_batch_offset(const std::set<TensorPtr> &tensors,
                                   const std::vector<ReadOnlyParam> &params)
{
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        if (params[i].tensor)
        {
            auto tensor = params[i].tensor;
            if (tensors.count(tensor))
            {
                continue;
            }
            for (auto t : tensors)
            {
                if (t->memory_offset()->is_shared_by(tensor->memory_offset()))
                {
                    // DEBUG() << "mapped tensor: " << tensor->name();
                    std::vector<Expr> offsets = t->get_attr(Constants::max_batch_offset);
                    tensor->set_attr(Constants::max_batch_offset, offsets);
                    break;
                }
            }
        }
    }
}

// add max-batch offset if shared memory
static void patch_max_batch_offset(NodePtr node,
                                   const std::vector<ReadOnlyParam> &ro,
                                   const std::vector<ReadOnlyParam> &desc)
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
    patch_max_batch_offset(tensors, ro);
    patch_max_batch_offset(tensors, desc);
}

// mask all exprs except max-batch offsets
static std::vector<ReadOnlyParam> mask_non_max_batch_param(const std::vector<ReadOnlyParam> &params)
{
    std::vector<ReadOnlyParam> masked_params;
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        ReadOnlyParam p(0);
        if (params[i].tensor)
        {
            auto tensor = params[i].tensor;
            if (tensor->attrs().count(Constants::max_batch_offset))
            {
                if (tensor->attrs().count(Constants::expr_offset))
                {
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        if (params[i].tensor)
        {
            auto tensor = params[i].tensor;
            if (tensors.count(tensor))
            {
                continue;
            }
            for (auto t : tensors)
            {
                if (t->memory_offset()->is_shared_by(tensor->memory_offset()))
                {
                    // DEBUG() << "mapped tensor: " << tensor->name();
                    std::vector<Expr> offsets = t->get_attr(Constants::max_batch_offset);
                    tensor->set_attr(Constants::max_batch_offset, offsets);
                    break;
                }
            }
        }
    }
}

// add max-batch offset if shared memory
static void patch_max_batch_offset(NodePtr node,
                                   const std::vector<ReadOnlyParam> &ro,
                                   const std::vector<ReadOnlyParam> &desc)
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
    patch_max_batch_offset(tensors, ro);
    patch_max_batch_offset(tensors, desc);
}

// mask all exprs except max-batch offsets
static std::vector<ReadOnlyParam> mask_non_max_batch_param(const std::vector<ReadOnlyParam> &params)
{
    std::vector<ReadOnlyParam> masked_params;
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        ReadOnlyParam p(0);
        if (params[i].tensor)
        {
            auto tensor = params[i].tensor;
            if (tensor->attrs().count(Constants::max_batch_offset))
            {
                if (tensor->attrs().count(Constants::expr_offset))
                {
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        if (params[i].tensor)
        {
            auto tensor = params[i].tensor;
            if (tensors.count(tensor))
            {
                continue;
            }
            for (auto t : tensors)
            {
                if (t->memory_offset()->is_shared_by(tensor->memory_offset()))
                {
                    // DEBUG() << "mapped tensor: " << tensor->name();
                    std::vector<Expr> offsets = t->get_attr(Constants::max_batch_offset);
                    tensor->set_attr(Constants::max_batch_offset, offsets);
                    break;
                }
            }
        }
    }
}

// add max-batch offset if shared memory
static void patch_max_batch_offset(NodePtr node,
                                   const std::vector<ReadOnlyParam> &ro,
                                   const std::vector<ReadOnlyParam> &desc)
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
    patch_max_batch_offset(tensors, ro);
    patch_max_batch_offset(tensors, desc);
}

// mask all exprs except max-batch offsets
static std::vector<ReadOnlyParam> mask_non_max_batch_param(const std::vector<ReadOnlyParam> &params)
{
    std::vector<ReadOnlyParam> masked_params;
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        ReadOnlyParam p(0);
        if (params[i].tensor)
        {
            auto tensor = params[i].tensor;
            if (tensor->attrs().count(Constants::max_batch_offset))
            {
                if (tensor->attrs().count(Constants::expr_offset))
                {
            m_stream << "{ld " << t0 << ",r1," << 4 * i << ";}\n";
        }
        else
        {
            m_stream << "{st " << t0 << ",r2," << 4 * i0 << "; ld " << t1 << ",r1," << 4 * i << ";}\n";
            if (i == ro.size() - 8)
            {
                m_stream << "{st " << t1 << ",r2," << 4 * i << ";}\n";
            }
            t0.swap(t1);
        }
        i0 = i;
    }

    if (copy_desc)
    {
        uint32_t offset = aligned(ro_size, 256);
        if (desc.size() && desc_param.size() && desc_act.size() &&
            offset + aligned(desc_size, 256) + aligned(desc_param_size, 256) <= 32767)
        {
            uint32_t desc_param_offset = offset + aligned(desc_size, 256);
            uint32_t desc_act_offset = desc_param_offset + aligned(desc_param_size, 256);
            if (desc_param_offset > 1023 && desc_act_offset > 1023)
            {
                m_stream << "{add r3,r2," << offset << "; mov r4," << desc_param_offset << ";}\n"
                         << "{add r4,r4,r2; mov r5," << desc_act_offset << ";}\n"
                         << "{add r5,r5,r2;}\n";
            }
            else if (desc_act_offset > 1023)
            {
                m_stream << "{add r3,r2," << offset << "; add r4,r2," << desc_param_offset << ";"
                         << " mov r5," << desc_act_offset << ";}\n"
                         << "{add r5,r5,r2;}\n";
            }
            else
            {
                m_stream << "{add r3,r2," << offset << "; add r4,r2," << desc_param_offset << ";}\n"
                         << "{add r5,r2," << desc_act_offset << ";}\n";
            }
            offset = desc_act_offset;
        }
        else
        {
            if (desc.size()) // get new desc_base
            {
                if (offset > 1023)
                {
                    gen_movi_asm(m_stream, offset, "r3");
                    m_stream << "{add r3,r3,r2;}\n";
                }
                else
                {
                    m_stream << "{add r3,r2," << offset << ";}\n";
                }
            }

            offset += aligned(desc_size, 256);
            if (desc_param.size()) // get new desc_param_base
            {
                if (offset > 1023)
                {
                    gen_movi_asm(m_stream, offset, "r4");
                    m_stream << "{add r4,r4,r2;}\n";
                }
                else
                {
                    m_stream << "{add r4,r2," << offset << ";}\n";
                }
            }

            offset += aligned(desc_param_size, 256);
            if (desc_act.size()) // get new desc_act_base
            {
                if (offset > 1023)
                {
                    gen_movi_asm(m_stream, offset, "r5");
                    m_stream << "{add r5,r5,r2;}\n";
                }
                else
                {
                    m_stream << "{add r5,r2," << offset << ";}\n";
                }
            }
        }
        offset += desc_act_size;

        // copy desc to temp_param_base
        offset -= aligned(ro_size, 256);
        uint32_t cache_size = std::ceil(std::log2(std::max(offset / 1024.0, 1.0))) + 1;
        m_stream << "{cacheop r3,24," << cache_size << ";}\n"; // write-back and invalidate data cache

        if (desc.size() && desc_param.size() && desc_act.size() &&
            desc_size <= 32767 && desc_param_size <= 32767 && desc_act_size <= 32767)
        {
            // copy desc
            m_stream << "{sub r6,sp,24; mov r8," << desc_size << ";}\n"
                     << "{st r8,r6,4; ld r7,r1,220;}\n"
                     << "{st r8,r6,20; add r8,r31,32;}\n"
                     << "{movh r8,32;}\n"
                     << "{st r8,r6,0;}\n"
                     << "{st r8,r6,16; add r8,r31,1;}\n"
                     << "{stb r8,r6,7;}\n"
                     << "{stb r8,r6,23;}\n"
                     << "{st r31,r6,8;}\n"
                     << "{st r31,r6,12;}\n"
                     << "{lsl r9,r8,8; mov r10," << desc_param_size << ";}\n"
                     << "{or r9,r9,r8; add r8,r31,2;}\n"
                     << "{dma 0,r9,0,0,0,1,0,r6,r3,r7;}\n"
                     << "{xorh r7,r8,0; mov r8," << desc_act_size << ";}\n"
                     << "{wfe r7,1;}\n";
            // copy desc_param
            m_stream << "{sth r10,r6,4;}\n"
                     << "{sth r10,r6,20; ld r10,r1,224;}\n"
                     << "{dma 0,r9,0,0,0,1,0,r6,r4,r10;}\n"
                     << "{wfe r7,1;}\n";
            // copy desc_act
            m_stream << "{sth r8,r6,4; ld r10,r1,228;}\n"
                     << "{sth r8,r6,20;}\n"
                     << "{dma 0,r9,0,0,0,1,0,r6,r5,r10;}\n"
                     << "{wfe r7,1;}\n";
        }
        else
        {
            m_stream << "{sub r6,sp,24;}\n"; // get dma descriptor

            if (desc.size()) // copy desc
            {
                m_stream << "{ld r7,r1,220;}\n"; // get desc_base
                gen_dma_copy_asm(m_stream,
                                 "r6",
                                 "r7",
                                 "r3",
                                 desc_size,
                                 true);
            }

            if (desc_param.size()) // copy desc_param
            {
                m_stream << "{ld r7,r1,224;}\n"; // get desc_param_base
                gen_dma_copy_asm(m_stream,
                                 "r6",
                                 "r7",
                                 "r4",
                                 desc_param_size,
                                 false);
            }

            if (desc_act.size()) // copy desc_act
            {
                m_stream << "{ld r7,r1,228;}\n"; // get desc_act_base
                gen_dma_copy_asm(m_stream,
                                 "r6",
                                 "r7",
                                 "r5",
                                 desc_act_size,
                                 false);
            }
        }

        cache_size = std::min(cache_size, 4u);
        m_stream << "{cacheop r3,8," << cache_size << ";}\n"; // prefetch up to 8KB

        if (desc.size()) // remap desc
        {
            gen_remap_param_asm(m_stream,
                                node,
                                m_ro_base,
                                m_desc_base,
                                rel_ro_offset,
                                desc_size,
                                desc_param_size,
                                desc_act_size,
                                desc,
                                "r3");
        }

        if (desc_param.size()) // remap desc_param
        {
            gen_remap_param_asm(m_stream,
                                node,
                                m_ro_base,
                                m_desc_base,
                                rel_ro_offset,
                                desc_size,
                                desc_param_size,
                                desc_act_size,
                                desc_param,
                                "r4");
        }

        if (desc_act.size()) // remap desc_act
        {
            gen_remap_param_asm(m_stream,
                                node,
                                m_ro_base,
                                m_desc_base,
                                rel_ro_offset,
                                desc_size,
                                desc_param_size,
                                desc_act_size,
                                desc_act,
                                "r5");
        }
    }

    gen_remap_param_asm(m_stream, // remap ro
                        node,
                        m_ro_base,
                        m_desc_base,
                        rel_ro_offset,
                        desc_size,
                        desc_param_size,
                        desc_act_size,
                        ro,
                        "r2",
                        copy_desc);

    m_stream << skip_label << ":\n"
             << "{br ra;}\n";
    return call_name;
}

void DSCodeGenerator::gen_expr_func_asm(std::stringstream &m_stream,
                                        Expr expr,
                                        DSFuncDesc &fd,
                                        const std::string &base,
                                        uint32_t offset,
                                        bool mode)
{
    auto &symbols = m_graph->symbols();
    DSFuncItem item;
    expr = expr->simplify();
    item.expr = expr;
    auto args = expr->variables();
    item.args = args;
    for (auto &arg : args)
    {
        if (!symbols.count(arg.name()))
        {
            FATAL() << "unknown symbol: " << arg.name();
        }
        auto &sym = symbols.at(arg.name());
        item.arg_types.push_back(sym.type);
        item.arg_dims.push_back(sym.dims[0]);
    }
    int idx = -1;
    for (uint32_t j = 0; j < m_total_exprs.size(); ++j)
    {
        if (compare(expr, args, m_total_exprs[j], m_total_args[j], true))
        {
            idx = j;
            break;
        }
    }
    if (idx != -1)
    {
        item.func = m_total_funcs[idx];
        item.status = m_total_status[idx];
        // DEBUG() << "expr func: " << m_total_funcs[idx];
    }
    else
    {
        std::string func_name;
        bool spilled = false;
        if (expr->op() != ExprImpl::Op::Val &&
            expr->op() != ExprImpl::Op::Constant)
        {
            auto func_id = std::to_string(utils::get_increase_id64());
            func_name = "_expr_func" + func_id;
            auto func = expr_to_func(expr, func_name);
            auto code = sbe::emit(func, spilled, true);
            // m_stream << "# " << expr->to_string() << "\n";
            m_stream << ".type " << func_name << ", function\n"
                     << code;
        }
        m_total_exprs.push_back(expr);
        m_total_args.push_back(args);
        m_total_funcs.push_back(func_name);
        m_total_status.push_back(spilled);
        item.func = func_name;
        item.status = spilled;
    }
    item.out_base = base;
    item.out_offset = offset;
    item.mode = mode;
    fd.items.push_back(item);
}

void DSCodeGenerator::gen_expr_func_asm(std::stringstream &m_stream,
                                        const std::vector<Expr> &exprs,
                                        DSFuncDesc &fd,
                                        const std::string &base)
{
    for (uint32_t i = 0; i < exprs.size(); ++i)
    {
        gen_expr_func_asm(m_stream, exprs[i], fd, base, 4 * i, false);
    }
    reorganize_items(fd);
}

void DSCodeGenerator::gen_expr_func_asm(std::stringstream &m_stream,
                                        const std::vector<ReadOnlyParam> &params,
                                        bool desc_base,
                                        uint32_t desc_param_index,
                                        uint32_t desc_act_index,
                                        uint32_t batch_offset_index,
                                        DSFuncDesc &fd)
{
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        Expr expr;
        if (params[i].tensor)
        {
            if (params[i].tensor->attrs().count(Constants::expr_offset))
            {
                expr = params[i].tensor->get_attr(Constants::expr_offset);
            }
            if (batch_offset_index != UINT32_MAX &&
                params[i].tensor->attrs().count(Constants::max_batch_offset))
            {
                std::vector<Expr> offsets = params[i].tensor->get_attr(Constants::max_batch_offset);
                expr = offsets[batch_offset_index];
            }
        }
        else if (params[i].expr)
        {
            expr = params[i].expr;
        }
        if (expr)
        {
            std::string base;
            uint32_t offset = 0;
            if (desc_base)
            {
                if (i >= desc_act_index)
                {
                    base = "r14"; // desc_act_base
                    offset = 4 * (i - desc_act_index);
                }
                else if (i >= desc_param_index)
                {
                    base = "r13"; // desc_param_base
                    offset = 4 * (i - desc_param_index);
                }
                else
                {
                    base = "r12"; // desc_base
                    offset = 4 * i;
                }
            }
            else
            {
                base = "r11"; // pp_base
                offset = 4 * i;
            }
            gen_expr_func_asm(m_stream,
                              expr,
                              fd,
                              base,
                              offset,
                              params[i].expr ? false : true);
        }
    }
    reorganize_items(fd);
}

std::string DSCodeGenerator::gen_modify_plugin_param_asm(std::stringstream &m_stream,
                                                         const std::vector<ReadOnlyParam> &ro,
                                                         const std::vector<ReadOnlyParam> &desc,
                                                         uint32_t desc_param_index,
                                                         uint32_t desc_act_index,
                                                         bool &data_aware)
{
    DSFuncDesc ro_fd, desc_fd;

    gen_expr_func_asm(m_stream,
                      ro,
                      false,
                      desc_param_index,
                      desc_act_index,
                      UINT32_MAX,
                      ro_fd);

    gen_expr_func_asm(m_stream,
                      desc,
                      true,
                      desc_param_index,
                      desc_act_index,
                      UINT32_MAX,
                      desc_fd);

    DSFuncDesc ro_fd0, desc_fd0; // normal exprs
    DSFuncDesc ro_fd1, desc_fd1; // data-aware exprs

    split_items(ro_fd, ro_fd0, ro_fd1);
    split_items(desc_fd, desc_fd0, desc_fd1);
    data_aware = ro_fd1.items.size() || desc_fd1.items.size();

    if (ro_fd.items.empty() && desc_fd.items.empty())
    {
        return std::string();
    }

    std::string call_name, func_id;
    for (uint32_t i = 0; i < m_func_descs.size() / 2; ++i)
    {
        if (m_func_descs[i * 2] == ro_fd && m_func_descs[i * 2 + 1] == desc_fd)
        {
            call_name = m_func_descs[i * 2].func_name;
            break;
        }
    }
    if (call_name.size())
    {
        return call_name;
    }
    else
    {
        func_id = std::to_string(utils::get_increase_id64());
        call_name = "_modify_param" + func_id;
        ro_fd.func_name = call_name;
        desc_fd.func_name = call_name;
        m_func_descs.push_back(ro_fd);
        m_func_descs.push_back(desc_fd);
    }

    auto skip_label = "._skip_func_lbl_" + func_id;
    auto exit_label = "._exit_func_lbl_" + func_id;
    m_stream << ".type " << call_name << ", function\n"
             << call_name << ":\n"
             << "{mfctrl0 r8,0x14;}\n"; // get TCB pointer

    if (!data_aware && BuilderConfig::exist(Constants::default_input_shape))
    {
        m_stream << "{ld r0,r8,80; ld r9,r8,112;}\n"                                            // r9: gp_base
                 << "{lsr r0,r0,16; ld r1,r9," << GP_DEFAULT_SHAPE_FLAG << "; ld r11,r8,56;}\n" // r0: tec id, r11: pp_base
                 << "{cbnz r1," << skip_label << ";}\n"                                         // only not default shape
                 << "{cbnz r0," << skip_label << ";}\n";                                        // only tec0
    }
    else
    {
        m_stream << "{ld r0,r8,80; ld r9,r8,112;}\n"
                 << "{lsr r0,r0,16; ld r11,r8,56;}\n"
                 << "{cbnz r0," << skip_label << ";}\n"; // only tec0
    }

    m_stream << "{add r9,r9," << GP_INPUT_SHAPE_OFFSET << ";"
             << " ld r10,r9," << GP_OUTPUT_SHAPE_BASE << "; ld r12,r11,220;}\n" // r10: os_base, r12: desc_base
             << "{ld r13,r11,224; ld r14,r11,228; add r15,ra,0;}\n";

    if (data_aware) // always modify expr with data-aware symbol
    {
        if (ro_fd1.items.size()) // modify ro
        {
            gen_modify_param_asm(m_stream, ro_fd1);
        }

        if (desc_fd1.items.size()) // modify desc
        {
            gen_modify_param_asm(m_stream, desc_fd1);
        }

        if (ro_fd0.items.size() || desc_fd0.items.size())
        {
            m_stream << "{sub r0,r9," << GP_INPUT_SHAPE_OFFSET << ";}\n"
                     << "{ld r0,r0," << GP_DEFAULT_SHAPE_FLAG << ";}\n"
                     << "{cbnz r0," << exit_label << ";}\n"; // only not default shape
        }
    }

    if (ro_fd0.items.size()) // modify ro
    {
        gen_modify_param_asm(m_stream, ro_fd0);
    }

    if (desc_fd0.items.size()) // modify desc
    {
        gen_modify_param_asm(m_stream, desc_fd0);
    }

    m_stream << exit_label << ":\n"
             << "{add ra,r15,0;}\n"
             << skip_label << ":\n"
             << "{br ra;}\n";
    return call_name;
}

int get_data_aware_param_index(NodePtr node,
                               const std::vector<ReadOnlyParam> &ro)
{
    auto op_type = node->type();
    if (node->attrs().count(Constants::plugin_attrs_name))
    {
        op_type = SubGraphNode::from_node(node)->get_real_node()[0]->type();
    }
    TensorPtr src_base = nullptr;
    if (op_type == OpType_NonZero)
    {
        src_base = node->outputs()[1];
    }
    else if (op_type == OpType_CacheUpdate)
    {
        src_base = node->inputs()[1];
    }
    else if (op_type == OpType_ConstantOfShape)
    {
        src_base = node->inputs()[0];
    }
    else if (op_type == OpType_Slice && node->inputs().size() == 5)
    {
        src_base = node->outputs()[1];
    }
    else if (op_type == OpType_ForRangeEnder)
    {
        src_base = node->inputs()[2];
    }
    else if (op_type == OpType_TopK && node->inputs().size() == 2)
    {
        src_base = node->inputs()[1];
    }
    if (src_base)
    {
        for (size_t i = 0; i < ro.size(); ++i)
        {
            if (ro[i].tensor == src_base)
            {
                return i;
            }
        }
    }
    return -1;
}

std::string gen_copy_output_shape_code(NodePtr node, int param_index)
{
    if (param_index == -1)
    {
        return std::string();
    }

    uint32_t src_offset = param_index * sizeof(uint32_t);
    auto dst_offset = node->get_attr_deep<uint32_t>(Constants::output_shape_offset);
    auto shape_size = node->get_attr_deep<uint32_t>(Constants::output_shape_size) / 4;

    if (shape_size == 0)
    {
        return std::string();
    }

    std::stringstream oss;
    if (src_offset <= 1023 && dst_offset <= 1023 && shape_size <= 1023)
    {
        oss << "{add r0,r31," << src_offset << "; add r1,r31," << dst_offset << ";}\n"
            << "{add r2,r31," << shape_size << ";}\n"
            << "{mov r21, %pc_rel_lo(_aipubt_copy_output_shape);}\n"
            << "{movh r21, %pc_rel_hi(_aipubt_copy_output_shape);}\n"
            << "{brrl r21;}\n";
    }
    else
    {
        gen_movi_asm(oss, src_offset, "r0");
        gen_movi_asm(oss, dst_offset, "r1");
        gen_movi_asm(oss, shape_size, "r2");
        oss << "{mov r21, %pc_rel_lo(_aipubt_copy_output_shape);}\n"
            << "{movh r21, %pc_rel_hi(_aipubt_copy_output_shape);}\n"
            << "{brrl r21;}\n";
    }
    return oss.str();
}

std::string DSCodeGenerator::gen_backup_output_shape_asm(NodePtr node,
                                                         const std::vector<ReadOnlyParam> &ro)
{
    auto param_index = get_data_aware_param_index(node, ro);
    return gen_copy_output_shape_code(node, param_index);
}

// compute max-batch loop count
std::string DSCodeGenerator::gen_loop_count_asm(std::stringstream &m_stream,
                                                NodePtr node)
{
    auto loop_cnt = node->get_attr<Expr>(Constants::kernel_loop_count);

    std::stringstream oss;
    if (loop_cnt->is_constant())
    {
        uint32_t val = loop_cnt->value();
        gen_movi_asm(oss, val, "r0");
    }
    else
    {
        DSFuncDesc fd;
        gen_expr_func_asm(m_stream, {loop_cnt}, fd, "");

        oss << "{mfctrl0 r8,0x14;}\n" // get TCB pointer
            << "{ld r9,r8,112;}\n"    // get gp_base
            << "{add r9,r9," << GP_INPUT_SHAPE_OFFSET << ";"
            << " ld r10,r9," << GP_OUTPUT_SHAPE_BASE << "; add r11,ra,0;}\n"; // r10: os_base

        gen_eval_expr_asm(oss, fd, 0, "r12");
    }
    return oss.str();
}

std::string DSCodeGenerator::gen_modify_addr_asm(std::stringstream &m_stream,
                                                 NodePtr node,
                                                 const std::vector<ReadOnlyParam> &ro,
                                                 const std::vector<ReadOnlyParam> &desc,
                                                 uint32_t desc_param_index,
                                                 uint32_t desc_act_index,
                                                 uint32_t batch_offset_index)
{
    auto masked_ro = mask_non_max_batch_param(ro);
    auto masked_desc = mask_non_max_batch_param(desc);

    DSFuncDesc ro_fd, desc_fd;

    gen_expr_func_asm(m_stream,
                      masked_ro,
                      false,
                      desc_param_index,
                      desc_act_index,
                      batch_offset_index,
                      ro_fd);

    gen_expr_func_asm(m_stream,
                      masked_desc,
                      true,
                      desc_param_index,
                      desc_act_index,
                      batch_offset_index,
                      desc_fd);

    if (ro_fd.items.empty() && desc_fd.items.empty())
    {
        return std::string();
    }

    std::string call_name, func_id;
    for (uint32_t i = 0; i < m_func_descs.size() / 2; ++i)
    {
        if (m_func_descs[i * 2] == ro_fd && m_func_descs[i * 2 + 1] == desc_fd)
        {
            call_name = m_func_descs[i * 2].func_name;
            break;
        }
    }
    if (call_name.size())
    {
        return call_name;
    }
    else
    {
        func_id = std::to_string(utils::get_increase_id64());
        call_name = "_modify_addr" + func_id;
        ro_fd.func_name = call_name;
        desc_fd.func_name = call_name;
        m_func_descs.push_back(ro_fd);
        m_func_descs.push_back(desc_fd);
    }

    auto skip_label = "._skip_func_lbl_" + func_id;
    m_stream << ".type " << call_name << ", function\n"
             << call_name << ":\n"
             << "{mfctrl0 r8,0x14;}\n"              // get TCB pointer
             << "{ld r0,r8,80; ld r9,r8,112;}\n"    // r9: gp_base
             << "{lsr r0,r0,16; ld r11,r8,56;}\n"   // r0: tec id, r11: pp_base
             << "{cbnz r0," << skip_label << ";}\n" // only tec0
             << "{add r9,r9," << GP_INPUT_SHAPE_OFFSET << ";"
             << " ld r10,r9," << GP_OUTPUT_SHAPE_BASE << "; ld r12,r11,220;}\n" // r10: os_base
             << "{ld r13,r11,224; ld r14,r11,228; add r15,ra,0;}\n";

    if (ro_fd.items.size()) // modify ro
    {
        gen_modify_param_asm(m_stream, ro_fd);
    }

    if (desc_fd.items.size()) // modify desc
    {
        gen_modify_param_asm(m_stream, desc_fd);
    }

    m_stream << "{add ra,r15,0;}\n"
             << skip_label << ":\n"
             << "{br ra;}\n";
    return call_name;
}

std::string gen_restore_pp_code(bool data_aware)
{
    auto label_id = std::to_string(utils::get_increase_id64());
    auto skip_label = "._skip_lbl_" + label_id;
    std::ostringstream oss;
    oss << "{mfctrl0 r0,0x14;}\n"; // get TCB pointer
    if (!data_aware && BuilderConfig::exist(Constants::default_input_shape))
    {
        oss << "{ld r2,r0,56; ld r1,r0,112;}\n" // r2: pp_base
            << "{ld r1,r1," << GP_DEFAULT_SHAPE_FLAG << "; }\n"
            << "{cbnz r1," << skip_label << ";}\n"; // only not default shape
    }
    else
    {
        oss << "{ld r2,r0,56;}\n";
    }
    oss << "{ld r2,r2,-4;}\n"
        << "{st r2,r0,56;}\n" // update TCB PP
        << skip_label << ":\n";
    return oss.str();
}

std::string DSCodeGenerator::gen_restore_pp_asm(bool data_aware)
{
    return gen_restore_pp_code(data_aware);
}

void DSCodeGenerator::gen_set_global_param_asm(std::stringstream &m_stream)
{
    auto label_id = std::to_string(utils::get_increase_id64());
    auto skip_label = "._skip_lbl_" + label_id;
    m_stream << ".type " << set_global_param_fn << ", function\n"
             << set_global_param_fn << ":\n"
             << "{mfctrl0 r0,0x14;}\n" // get TCB pointer
             << "{ld r1,r0,80;}\n"
             << "{lsr r1,r1,16;}\n"                 // get tec id
             << "{cbnz r1," << skip_label << ";}\n" // only tec0
             << "{ld r2,r0,56; ld r1,r0,112;}\n"    // r1: gp_base
             << "{ld r2,r2,212; ld r3,r2,216;}\n"   // r2: os_base, r3: tp_base
             << "{st r2,r1," << GP_OUTPUT_SHAPE_BASE << ";}\n"
             << "{st r3,r1," << GP_TEMP_PARAM_BASE << ";}\n";

    // check input shape
    if (BuilderConfig::exist(Constants::default_input_shape))
    {
        auto default_shape = parse_input_shape(m_graph, Constants::default_input_shape);
        uint32_t shape_size = 0;
        for (auto &s : default_shape)
        {
            shape_size += s.size();
        }
        auto loop_label = "._loop_lbl_" + label_id;
        auto loop_break = "._loop_break_lbl_" + label_id;
        auto loop_exit = "._loop_exit_lbl_" + label_id;

        m_stream << "{mov r3," << GP_RESERVE1_OFFSET << ";}\n" // r1: gp_base
                 << "{add r2,r1," << GP_INPUT_SHAPE_OFFSET << "; add r3,r3,r1;}\n"
                 << "{mov r4," << shape_size << ";}\n"
                 << "{mov r5,1;}\n"
                 << loop_label << ":\n"
                 << "{ld r6,r2,0; ld r7,r3,0;}\n"
                 << "{cmp.eq r6,r6,r7;}\n"
                 << "{cbz r6," << loop_break << ";}\n"
                 << "{sub r4,r4,1; add r2,r2,4;}\n"
                 << "{cmp.eq r6,r4,0; add r3,r3,4;}\n"
                 << "{cbz r6," << loop_label << ";}\n"
                 << "{b " << loop_exit << ";}\n"
                 << loop_break << ":\n"
                 << "{mov r5,0;}\n"
                 << loop_exit << ":\n"
                 << "{st r5,r1," << GP_DEFAULT_SHAPE_FLAG << ";}\n";
    }
    else
    {
        m_stream << "{st r31,r1," << GP_DEFAULT_SHAPE_FLAG << ";}\n";
    }

    m_stream << skip_label << ":\n"
             << "{mov r21,-33;}\n"
             << "{wfe r21,0;}\n" // wait lsu idle
             << "{wfe r21,2;}\n"
             << "{br ra;}\n";
}

void DSCodeGenerator::gen_record_output_shape_asm(std::stringstream &m_stream)
{
    std::vector<Expr> exprs;
    DSFuncDesc fd;

    for (auto t : m_graph->get_output_tensors())
    {
        for (int i = 0; i < t->shape().dim(); ++i)
        {
            exprs.push_back(t->shape().symbol()[i]);
        }
    }

    gen_expr_func_asm(m_stream, exprs, fd, "r10");

    auto label_id = std::to_string(utils::get_increase_id64());
    auto skip_label = "._skip_func_lbl_" + label_id;
    m_stream << ".type " << record_output_shape_fn << ", function\n"
             << record_output_shape_fn << ":\n"
             << "{mfctrl0 r8,0x14;}\n"              // get TCB pointer
             << "{ld r0,r8,80; ld r9,r8,112;}\n"    // r9: gp_base
             << "{lsr r0,r0,16;}\n"                 // get tec id
             << "{cbnz r0," << skip_label << ";}\n" // only tec0
             << "{add r9,r9," << GP_INPUT_SHAPE_OFFSET << ";"
             << " ld r10,r9," << GP_OUTPUT_SHAPE_BASE << "; add r11,ra,0;}\n"; // r10: os_base

    gen_modify_param_asm(m_stream, fd, "r12", "r13");

    m_stream << "{add ra,r11,0;}\n"
             << skip_label << ":\n"
             << "{mov r21,-33;}\n"
             << "{wfe r21,0;}\n" // wait lsu idle
             << "{wfe r21,2;}\n"
             << "{br ra;}\n";
}

void DSCodeGenerator::generate(std::stringstream &m_stream,
                               const std::vector<ReadOnlyParam> &ro,
                               const std::vector<ReadOnlyParam> &desc,
                               std::map<int, DSCodePkg> &code_map)
{
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    std::vector<NodePtr> plugin_nodes;
    m_graph->walk_on_node([&plugin_nodes](NodePtr n) {
        if ((n->attrs().count(Constants::plugin_attrs_name) ||
             n->type() == OpType_LayoutConvertor ||
             n->type() == OpType_ForRangeEnder ||
             n->type() == OpType_IfEnder) &&
            n->attrs().count(Constants::ro_offset) &&
            n->attrs().count(Constants::ro_size))
        {
            plugin_nodes.push_back(n);
        }
    });

    for (auto node : plugin_nodes)
    {
        // DEBUG() << "generating: " << node->name() << " (" << node->type() << ")";
        uint32_t ro_index = node->get_attr<size_t>(Constants::ro_offset) / 4;
        uint32_t ro_size = node->get_attr<size_t>(Constants::ro_size) / 4;
        // DEBUG() << "ro_index=" << ro_index << " ro_size=" << ro_size;

        uint32_t desc_index = UINT32_MAX;
        if (node->attrs().count(Constants::desc_offset))
        {
            desc_index = node->get_attr<size_t>(Constants::desc_offset) / 4;
        }
        uint32_t desc_size = 0;
        if (node->attrs().count(Constants::desc_size))
        {
            desc_size = node->get_attr<size_t>(Constants::desc_size) / 4;
        }

        std::vector<ReadOnlyParam> node_desc, desc_sec, desc_param_sec, desc_act_sec;
        if (desc_size && desc_index != UINT32_MAX)
        {
            for (uint32_t i = desc_index; i < desc_index + desc_size; ++i)
            {
                node_desc.push_back(desc[i]);
                desc_sec.push_back(desc[i]);
            }
        }

        uint32_t desc_param_index = node_desc.size();
        if (node->attrs().count(Constants::descriptor_param))
        {
            desc_param_sec = node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_param);
            for (auto &d : desc_param_sec)
            {
                node_desc.push_back(d);
            }
        }

        uint32_t desc_act_index = node_desc.size();
        if (node->attrs().count(Constants::descriptor_act))
        {
            desc_act_sec = node->get_attr<std::vector<ReadOnlyParam>>(Constants::descriptor_act);
            for (auto &d : desc_act_sec)
            {
                node_desc.push_back(d);
            }
        }

        // generate ds code for kernels of a plugin node
        for (uint32_t i = 0; i < ro_size / DEFAULT_RO_NUMBER_PER_LAYER; ++i)
        {
            std::vector<ReadOnlyParam> node_ro;
            uint32_t index = ro_index + i * DEFAULT_RO_NUMBER_PER_LAYER;
            for (uint32_t j = index; j < index + DEFAULT_RO_NUMBER_PER_LAYER; ++j)
            {
                node_ro.push_back(ro[j]);
            }

            DSCodePkg codes;
            if (enable_ds)
            {
                bool data_aware = false;
                auto modify_param_code = gen_modify_plugin_param_asm(m_stream,
                                                                     node_ro,
                                                                     node_desc,
                                                                     desc_param_index,
                                                                     desc_act_index,
                                                                     data_aware);
                std::string backup_param_code;
                if (modify_param_code.size())
                {
                    std::stringstream oss;
                    if (node->attrs().count(Constants::plugin_attrs_name))
                    {
                        BuilderOpPluginInfo *plugin = node->get_attr(Constants::plugin_attrs_name);
                        oss << plugin->name;
                        auto sgnode = SubGraphNode::from_node(node);
                        for (auto m_node : sgnode->get_real_node())
                        {
                            oss << m_node->type();
                        }
                    }
                    else
                    {
                        oss << node->type();
                    }
                    uint32_t real_ro_size = DEFAULT_RO_NUMBER_PER_LAYER;
                    if (node_ro.size() == 1)
                    {
                        real_ro_size = node->get_attr<size_t>(Constants::real_ro_size) / 4;
                    }
                    uint32_t real_desc_size = 0;
                    if (node->attrs().count(Constants::real_desc_size))
                    {
                        real_desc_size = node->get_attr<size_t>(Constants::real_desc_size) / 4;
                    }
                    oss << "_ro_" << real_ro_size << "_desc_" << real_desc_size
                        << "_" << desc_param_sec.size() << "_" << desc_act_sec.size()
                        << "_" << data_aware;
                    auto func_id = oss.str();

                    if (m_func_map.count(func_id))
                    {
                        backup_param_code = m_func_map[func_id];
                    }
                    else
                    {
                        // compute relative ro offset for ro remap
                        uint32_t rel_ro_offset = 4 * i * DEFAULT_RO_NUMBER_PER_LAYER;
                        backup_param_code = gen_backup_plugin_param_asm(m_stream,
                                                                        node,
                                                                        node_ro,
                                                                        desc_sec,
                                                                        desc_param_sec,
                                                                        desc_act_sec,
                                                                        real_ro_size,
                                                                        rel_ro_offset,
                                                                        data_aware);
                        m_func_map[func_id] = backup_param_code;
                    }
                }
                codes.data_aware = data_aware;
                codes.ds[0] = backup_param_code;
                codes.ds[1] = modify_param_code;
                codes.ds[2] = gen_backup_output_shape_asm(node, node_ro);
                if (backup_param_code.size())
                {
                    codes.ds[3] = gen_restore_pp_asm(data_aware);
                }
            }
            if (node->attrs().count(Constants::max_batch_size) &&
                node->attrs().count(Constants::kernel_loop_count))
            {
                codes.mb[0] = gen_loop_count_asm(m_stream, node);
                patch_max_batch_offset(node, node_ro, node_desc);
                for (int i = 0; i < 2; ++i)
                {
                    codes.mb[i + 1] = gen_modify_addr_asm(m_stream,
                                                          node,
                                                          node_ro,
                                                          node_desc,
                                                          desc_param_index,
                                                          desc_act_index,
                                                          i);
                }
            }
            if (!codes.empty())
            {
                code_map[index] = codes;
            }
        }
    }

    if (enable_ds)
    {
        generate_global_functions(m_stream);
    }
}

void DSCodeGenerator::generate_global_functions(std::stringstream &m_stream)
{
    gen_set_global_param_asm(m_stream);
    gen_record_output_shape_asm(m_stream);
}
} // namespace aipubt

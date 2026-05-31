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
 * @Date: 2024-11-20 10:19:05
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-18 14:56:49
 */
#include "pipeline.h"
#include "core/logger.h"
#include <iostream>

namespace ffc
{
using namespace HAL;
namespace HAL
{
    AIFFRegs gen_x3p_regs();
    AIFFRegs gen_x3s_regs();
} // namespace HAL

static bool split_h(const TensorDesc &td, TensorDesc &res0, TensorDesc &res1)
{
    if (td.layout == Layout::NHWC ||
        td.layout == Layout::NCHWC16 || td.layout == Layout::NCHWC32)
    {
        res0 = td;
        res1 = td;
        int idx = td.layout == Layout::NHWC ? 1 : 2;
        if (td.shape[idx] >= 2)
        {
            res0.shape[idx] /= 2;
            res1.shape[idx] -= res0.shape[idx];
        }
        return true;
    }
    return false;
}

static bool split_oc(const TensorDesc &td, TensorDesc &res0, TensorDesc &res1)
{
    if (td.layout == Layout::Flat ||
        td.layout == Layout::NHWC ||
        td.layout == Layout::NCHWC16 || td.layout == Layout::NCHWC32)
    {
        res0 = td;
        res1 = td;
        int oc = td.shape[td.shape.size() - 1];
        if (td.layout == Layout::NCHWC16 || td.layout == Layout::NCHWC32)
        {
            oc = td.shape[1] * td.shape[4];
        }
        int align = 32 / td.element_size();
        if (oc >= align * 2)
        {
            if (td.layout == Layout::NCHWC16 || td.layout == Layout::NCHWC32)
            {
                res0.shape[1] /= 2;
                res1.shape[1] -= res0.shape[1];
            }
            else
            {
                int idx = td.shape.size() - 1;
                res0.shape[idx] = res0.shape[idx] / align / 2 * align;
                res1.shape[idx] -= res0.shape[idx];
            }
        }
        return true;
    }
    return false;
}

Pipeline::Pipeline(const std::string &arch)
{
    resource = AIFFResource::create(arch);
    if (arch == "X3P")
    {
        regs = gen_x3p_regs();
    }
    else if (arch == "X3S")
    {
        regs = gen_x3s_regs();
    }
    else
    {
        ERROR() << "unsupported arch: " << arch;
    }
    m_arch = arch;
}

bool Pipeline::feed(Tensor t)
{
    // on feeding mode,
    // we will try to feed as more node as we can
    // so most case, we will only use mtp0 or mtp1 for mtp calculation
    // on build mode,
    // we will try to maximum the usage of AIFF hardware,
    // so if we find mtp1 is free, the mtp0 task may try to occupy mtp1
    auto mtp0 = resource->get_resource("MTP0");
    if (!mtp0->feed_pipe(t))
    {
        ERROR() << "unsupported tensor:\n"
                << t->to_string();
        return false;
    }
    if (!contains(m_tensors, t))
    {
        m_tensors.push_back(t);
    }
    return true;
}

void Pipeline::build()
{
    // walk through all resources
    // build in order of pipe line
    reallocate();
    create_symbols();
    for (auto name : {"MTP0", "MTP1", "PTP"})
    {
        auto r = resource->get_resource(name);
        r->build_pipe(regs);
    }
}

bool Pipeline::reallocate()
{
    static const std::set<uOp> tile_h_ops = {
        uOp::Add,
        uOp::Sub,
        uOp::Mul,
        uOp::Lut,
        uOp::ReduceMax,
        uOp::ReduceSum,
    };
    static const std::set<uOp> tile_oc_ops = {
        uOp::Conv,
        uOp::Add,
        uOp::Sub,
        uOp::Mul,
        uOp::Cast,
        uOp::LayoutCvt,
    };

    auto mtp0 = resource->get_resource("MTP0");
    auto mtp1 = resource->get_resource("MTP1");

    if (!mtp0->is_pipe_free() && mtp1->is_pipe_free())
    {
        std::array<bool, 2> status = {true, true};
        for (auto t : m_tensors)
        {
            if (!tile_h_ops.count(t->type))
            {
                status[0] = false;
            }
            if (!tile_oc_ops.count(t->type))
            {
                status[1] = false;
            }
        }

        int tile_type = 0; // none
        if (status[0])
        {
            tile_type = 2; // split h
        }
        else if (status[1])
        {
            tile_type = 1; // split oc
        }
        for (auto t : m_tensors)
        {
            t->params["tile_type"] = tile_type;
        }

        if (tile_type == 2) // split h
        {
            TensorDesc input0, input1;
            split_h(m_tensors[0]->inputs[0]->output, input0, input1);
            std::map<int, TensorDesc> other_input0, other_input1;
            for (auto t : m_tensors)
            {
                for (auto inp : t->inputs)
                {
                    if (inp != m_tensors[0]->inputs[0] && !contains(m_tensors, inp))
                    {
                        split_h(inp->output, other_input0[inp->id], other_input1[inp->id]);
                    }
                }
                mtp1->feed_pipe(t);
            }
            mtp0->infer_pipe(input0, other_input0);
            mtp1->infer_pipe(input1, other_input1);
        }
        else if (tile_type == 1) // split oc
        {
            TensorDesc input = m_tensors[0]->inputs[0]->output;
            std::map<int, TensorDesc> other_input0, other_input1;
            for (auto t : m_tensors)
            {
                for (auto inp : t->inputs)
                {
                    if (inp != m_tensors[0]->inputs[0] && !contains(m_tensors, inp))
                    {
                        split_oc(inp->output, other_input0[inp->id], other_input1[inp->id]);
                    }
                }
                mtp1->feed_pipe(t);
            }
            mtp0->infer_pipe(input, other_input0);
            mtp1->infer_pipe(input, other_input1);
        }
        else // none
        {
            auto input0 = m_tensors[0]->inputs[0]->output;
            std::map<int, TensorDesc> other_input0;
            for (auto t : m_tensors)
            {
                for (auto inp : t->inputs)
                {
                    if (inp != m_tensors[0]->inputs[0] && !contains(m_tensors, inp))
                    {
                        other_input0[inp->id] = inp->output;
                    }
                }
            }
            mtp0->infer_pipe(input0, other_input0);
        }
    }

    return true;
}

Symbol Pipeline::set_symbol(const std::string &name, Symbol::Value value)
{
    if (m_symbols.count(name))
    {
        return m_symbols.at(name).first;
    }
    auto sym = Symbol(name);
    m_symbols[name] = {sym, value};
    return sym;
}

Symbol Pipeline::get_symbol(const std::string &name) const
{
    if (!m_symbols.count(name))
    {
        std::cerr << "missing symbol: " << name << "\n";
        return Symbol();
    }
    return m_symbols.at(name).first;
}

bool Pipeline::create_symbols()
{
    int tile_type = m_tensors[0]->params["tile_type"];
    m_symbols.clear();

    int in_cnt = 0;
    for (auto t : m_tensors)
    {
        for (size_t i = 0; i < t->inputs.size(); ++i)
        {
            auto inp = t->inputs[i];
            if (!contains(m_tensors, inp))
            {
                auto name = "in" + std::to_string(in_cnt++) + "_addr";
                if (inp->name.size())
                {
                    name = inp->name;
                }
                auto in_addr = set_symbol(name, 0);
                t->exprs["in" + std::to_string(i) + "_addr"] = in_addr;
            }
        }
    }

    int out_cnt = 0;
    for (size_t i = 0; i < m_tensors.size(); ++i)
    {
        auto t = m_tensors[i];
        for (size_t j = 0; j < t->inputs.size(); ++j)
        {
            auto inp = t->inputs[j];
            if (contains(m_tensors, inp))
            {
                t->exprs["in" + std::to_string(j) + "_addr"] = inp->exprs.at("out_addr");
            }
        }
        if (i == m_tensors.size() - 1 ||
            (i + 1 == m_tensors.size() - 1 &&
             (m_tensors[i + 1]->type == uOp::ReduceMax || m_tensors[i + 1]->type == uOp::ReduceSum)))
        {
            auto name = "out" + std::to_string(out_cnt++) + "_addr";
            if (t->name.size())
            {
                name = t->name;
            }
            auto out_addr = set_symbol(name, 0);
            t->exprs["out_addr"] = out_addr;
        }
        else
        {
            t->exprs["out_addr"] = t->exprs.at("in0_addr");
        }
    }

    if (tile_type == 2) // split h
    {
        for (auto t : m_tensors)
        {
            if (t->type == uOp::Add || t->type == uOp::Sub || t->type == uOp::Mul)
            {
                if (t->inputs.size() == 2)
                {
                    auto out = t->inputs[1]->output;
                    int in1_ic = out.shape[3];
                    if (out.layout == Layout::NCHWC16)
                    {
                        in1_ic = out.shape[1] * out.shape[4];
                    }
                    int in1_es = out.element_size();
                    t->exprs["in1_ic"] = set_symbol("in1_ic", in1_ic);
                    t->exprs["in1_es"] = set_symbol("in1_es", in1_es);
                }
            }
            if (t->type == uOp::Lut)
            {
                t->exprs["lut_addr"] = set_symbol("lut_addr", 0);
            }
        }

        for (auto t : m_tensors)
        {
            auto inp = t->inputs[0];
            int real_iw = t->params["real_iw"];
            int real_ihw = t->params["real_ihw"];
            int ihw_offset = t->params["ihw_offset"];

            int ih = inp->output.shape[2];
            int iw = inp->output.shape[3];
            int ic = inp->output.shape[1] * inp->output.shape[4];
            int step_w = iw;
            int step_h0 = ih >= 2 ? ih / 2 : ih;
            int step_h1 = ih >= 2 ? ih - ih / 2 : ih;

            DEBUG() << t->to_string();
            DEBUG() << "ih: " << ih << ", iw: " << iw << ", ic: " << ic;

            t->exprs["real_iw"] = set_symbol("real_iw", real_iw);
            t->exprs["real_ihw"] = set_symbol("real_ihw", real_ihw);
            t->exprs["ihw_offset"] = set_symbol("ihw_offset", ihw_offset);
            // t->exprs["nhwc_out"] = set_symbol("nhwc_out", 0);
            t->exprs["ih"] = set_symbol("ih", ih);
            t->exprs["iw"] = set_symbol("iw", iw);
            t->exprs["ic"] = set_symbol("ic", ic);
            t->exprs["step_w"] = set_symbol("step_w", step_w);
            t->exprs["step_h0"] = set_symbol("step_h0", step_h0);
            t->exprs["step_h1"] = set_symbol("step_h1", step_h1);
        }

        for (auto t : m_tensors)
        {
            Expr real_iw = t->exprs.at("real_iw");
            Expr real_ihw = t->exprs.at("real_ihw");
            Expr ihw_offset = t->exprs.at("ihw_offset");

            t->exprs["real_ow"] = real_iw;
            t->exprs["real_ohw"] = real_ihw;
            t->exprs["ohw_offset"] = ihw_offset;

            Expr ih = t->exprs.at("ih");
            Expr iw = t->exprs.at("iw");
            Expr step_w = t->exprs.at("step_w");
            Expr step_h0 = t->exprs.at("step_h0");
            Expr step_h1 = t->exprs.at("step_h1");

            Expr ih0 = Expr::select(ih >= 2, ih.floordiv(2), ih);
            Expr ih1 = Expr::select(ih >= 2, ih - ih.floordiv(2), ih);
            t->exprs["ih0"] = ih0;
            t->exprs["ih1"] = ih1;
            t->exprs["step_w"] = Expr::select(step_w == 0, iw, step_w);
            t->exprs["step_h0"] = Expr::select(step_h0 == 0, ih0, step_h0);
            t->exprs["step_h1"] = Expr::select(step_h1 == 0, ih1, step_h1);
        }
    }
    else if (tile_type == 1) // split oc
    {
        auto out = m_tensors[m_tensors.size() - 1];
        int out_es = out->output.element_size();
        int nhwc_out = out->output.layout == Layout::NHWC ? 1 : 0;

        for (auto t : m_tensors)
        {
            auto inp = t->inputs[0];
            int ih = inp->output.shape[2];
            int iw = inp->output.shape[3];
            int ic = inp->output.shape[1] * inp->output.shape[4];
            int in_es = inp->output.element_size();
            int oh = ih, ow = iw;
            int oc = t->output.shape[t->output.shape.size() - 1];
            if (t->output.layout == Layout::NCHWC16 || t->output.layout == Layout::NCHWC32)
            {
                oc = t->output.shape[1] * t->output.shape[4];
            }
            int real_iw = iw;
            if (t->params.count("real_iw"))
            {
                real_iw = t->params["real_iw"];
            }
            int real_ihw = ih * iw;
            if (t->params.count("real_ihw"))
            {
                real_ihw = t->params["real_ihw"];
            }
            int ihw_offset = 0;
            if (t->params.count("ihw_offset"))
            {
                ihw_offset = t->params["ihw_offset"];
            }
            int real_oc = oc;
            if (t->params.count("real_oc"))
            {
                real_oc = t->params["real_oc"];
            }
            int oc_offset = 0;
            if (t->params.count("oc_offset"))
            {
                oc_offset = t->params["oc_offset"];
            }

            DEBUG() << t->to_string();
            DEBUG() << "ih: " << ih << ", iw: " << iw << ", ic: " << ic
                    << ", oh: " << oh << ", ow: " << ow << ", oc: " << oc
                    << ", in_es: " << in_es << ", out_es: " << out_es;

            t->exprs["real_iw"] = set_symbol("real_iw", real_iw);
            t->exprs["real_ihw"] = set_symbol("real_ihw", real_ihw);
            t->exprs["ihw_offset"] = set_symbol("ihw_offset", ihw_offset);
            t->exprs["real_oc"] = set_symbol("real_oc", real_oc);
            t->exprs["oc_offset"] = set_symbol("oc_offset", oc_offset);
            t->exprs["nhwc_out"] = set_symbol("nhwc_out", nhwc_out);
            t->exprs["ih"] = set_symbol("ih", ih);
            t->exprs["iw"] = set_symbol("iw", iw);
            t->exprs["ic"] = set_symbol("ic", ic);
            t->exprs["oc"] = set_symbol("oc", oc);
            t->exprs["in_es"] = set_symbol("in_es", in_es);
            t->exprs["out_es"] = set_symbol("out_es", out_es);
        }

        for (auto t : m_tensors)
        {
            Expr real_iw = t->exprs.at("real_iw");
            Expr real_ihw = t->exprs.at("real_ihw");
            Expr ihw_offset = t->exprs.at("ihw_offset");

            t->exprs["real_ow"] = real_iw;
            t->exprs["real_ohw"] = real_ihw;
            t->exprs["ohw_offset"] = ihw_offset;

            Expr oc = t->exprs.at("oc");
            Expr out_es = t->exprs.at("out_es");

            Expr align = Expr(32).floordiv(out_es);
            Expr oc0 = Expr::select(oc >= align * 2, oc.floordiv(align).floordiv(2) * align, oc);
            Expr oc1 = Expr::select(oc >= align * 2, oc - oc.floordiv(align).floordiv(2) * align, oc);
            t->exprs["oc0"] = oc0;
            t->exprs["oc1"] = oc1;
        }
    }
    return true;
}

std::string Pipeline::desc_func(const std::string &func_name,
                                const std::set<std::string> &ignored_symbols)
{
    auto desc = regs.get_ctrl_regs();
    auto desc_param = regs.get_param_regs();
    auto desc_act = regs.get_act_regs();
    desc.insert(desc.end(), desc_param.begin(), desc_param.end());
    desc.insert(desc.end(), desc_act.begin(), desc_act.end());
    for (size_t i = 0; i < desc.size(); ++i)
    {
        if (desc[i].expr)
        {
            auto vars = desc[i].expr->variables();
            bool ignored = false;
            for (auto &&var : vars)
            {
                if (ignored_symbols.count(var.name()))
                {
                    ignored = true;
                    break;
                }
            }
            if (ignored)
            {
                DEBUG() << "ignored index=" << i << ", expr=" << desc[i].expr->to_string();
                desc[i] = RegisterValue(desc[i].value);
            }
        }
    }
    return regs.get_modify_desc_func(desc, func_name);
}

Descriptor Pipeline::ctrl_desc()
{
    Descriptor ret;
    auto desc = regs.get_ctrl_regs();
    for (auto &&r : desc)
    {
        auto expr = r.expr;
        if (expr)
        {
            std::map<Symbol, Symbol::Value> var_map;
            auto vars = expr->variables();
            for (auto &&var : vars)
            {
                var_map[var] = m_symbols.at(var.name()).second;
            }
            if (!expr->eval(var_map))
            {
                ERROR() << "failed to eval expr: " << expr->to_string();
            }
            ret.push_back(expr->value());
            continue;
        }
        ret.push_back(r);
    }
    return ret;
}

Descriptor Pipeline::act_desc()
{
    Descriptor ret;
    auto desc = regs.get_act_regs();
    for (size_t i = 0; i < desc.size(); ++i)
    {
        if (desc[i].expr)
        {
            DEBUG() << "act index=" << i << ", expr=" << desc[i].expr->to_string();
        }
        ret.push_back(desc[i]);
    }
    return ret;
}

Descriptor Pipeline::param_desc()
{
    Descriptor ret;
    auto desc = regs.get_param_regs();
    for (size_t i = 0; i < desc.size(); ++i)
    {
        if (desc[i].expr)
        {
            DEBUG() << "param index=" << i << ", expr=" << desc[i].expr->to_string();
        }
        ret.push_back(desc[i]);
    }
    return ret;
}

int Pipeline::get_mtp_max_fm_size(int oc, int ic_gsize)
{
    constexpr int max_mtp_iact_buf_size = 64 * 1024;
    constexpr int max_mtp_unb_buf_size = 96 * 1024;

    int iact_gsize = 32 * std::pow(2, ic_gsize);
    int max_iact_fm_size = max_mtp_iact_buf_size / iact_gsize / 16 * 16;
    int max_unb_fm_size = max_mtp_unb_buf_size / (oc * 4) / 16 * 16;

    return std::min(max_iact_fm_size, max_unb_fm_size);
}

bool Pipeline::set_mtp_wdc(int i, const std::array<WDCInfo, 2> &wdc_info)
{
    if (i < 0 || i > 1)
    {
        return false;
    }
    auto mtp = "MTP" + std::to_string(i);
    for (int j = 0; j < 2; ++j)
    {
        uint32_t wt_fmt = wdc_info[j].wt_fmt;
        uint32_t wt_size = wdc_info[j].wt_size;
        uint32_t lut0 = wdc_info[j].lut0;
        uint32_t lut1 = wdc_info[j].lut1;
        if (wt_fmt == 0)
        {
            wt_size = 0;
            lut0 = 0;
            lut1 = 0;
        }
        regs[mtp]["MTP_WEIGHT_SPARSE"]["WT_SPARSE_FMT_" + std::to_string(j)] = wt_fmt;
        regs[mtp]["MTP_WT_COMPRESS" + std::to_string(j)]["COMPRESS_SIZE_" + std::to_string(j)] = wt_size / 32;
        regs[mtp]["MTP_WDC" + std::to_string(j) + "_LUT0"] = lut0;
        regs[mtp]["MTP_WDC" + std::to_string(j) + "_LUT1"] = lut1;
    }
    return true;
}

bool Pipeline::set_mtp_gptq_fixedzp(int i, int fixed_zp)
{
    if (i < 0 || i > 1)
    {
        return false;
    }
    auto mtp = "MTP" + std::to_string(i);
    regs[mtp]["MTP_CTRL"]["ZP_FIXED_EN"] = fixed_zp;
    return true;
}
} // namespace ffc
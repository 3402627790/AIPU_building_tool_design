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
 * @Date: 2024-11-20 10:18:36
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-12-18 14:12:39
 */
#pragma once
#include "hal/base_reg.h"
#include "hal/resource.h"
#include "tensor.h"

namespace ffc
{
using namespace HAL;

struct DescriptorPkg
{
    Descriptor ctrl_desc;
    Descriptor act_desc;
    Descriptor param_desc;
    std::string name;
    std::string modify_func_impl;
};

class Pipeline
{
public:
    Pipeline(const std::string &arch = "X3P");
    bool feed(Tensor t);
    void build();

    AIFFRegs &get_regs()
    {
        return regs;
    }

    DescriptorPkg desc_pkg(const std::string &func_name)
    {
        DescriptorPkg pkg;
        pkg.ctrl_desc = ctrl_desc();
        pkg.act_desc = act_desc();
        pkg.param_desc = param_desc();
        pkg.name = func_name;
        pkg.modify_func_impl = desc_func(func_name);
        return pkg;
    }

    std::string desc_func(const std::string &func_name,
                          const std::set<std::string> &ignored_symbols = {});

    Descriptor ctrl_desc();
    Descriptor act_desc();
    Descriptor param_desc();

    static int get_mtp_max_fm_size(int oc, int ic_gsize);

    bool set_mtp_wdc(int i, const std::array<WDCInfo, 2> &wdc_info);
    bool set_mtp_gptq_fixedzp(int i, int fixed_zp);

protected:
    Symbol set_symbol(const std::string &name, Symbol::Value value);
    Symbol get_symbol(const std::string &name) const;
    bool create_symbols();
    bool reallocate();

    ResourcePtr resource;
    AIFFRegs regs;
    std::vector<Tensor> m_tensors;
    std::string m_arch;
    std::map<std::string, std::pair<Symbol, Symbol::Value>> m_symbols;
};
} //namespace ffc
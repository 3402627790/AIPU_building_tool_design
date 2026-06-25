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
 * @Date: 2026-05-21 16:51:20
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-05-21 16:51:20
 */
#include "ds_code_wrapper.h"

namespace aipubt
{
KernelInfos GopDSCodeImpl::kernel_infos()
{
    Dim3 grid = Expr(1);
    Dim3 block = Expr(32);
    size_t sm_size = 0;
    KernelInfos infos;
    for (size_t i = 0; i < m_kernel_names.size(); ++i)
    {
        infos.push_back({
            m_kernel_names[i],
            grid,
            block,
            sm_size,
            {m_param_root, m_param_bases[i], m_stream_bases[i]},
        });
    }
    return infos;
}

// optional: use for turning performance
void GopDSCodeImpl::update_prefer_cfg(const std::string &key, const Any &value)
{
}

// optional: implement score
uint32_t GopDSCodeImpl::score() const
{
    return 0;
}

// optional: kernel cps bin file name
const std::string GopDSCodeImpl::kernel_cpsbin()
{
    return "";
}

// optional: implement flops
uint64_t GopDSCodeImpl::flops() const
{
    return 0;
}

const std::vector<GPattern> GopDSCodePlugin::patterns()
{
    return {GPattern(OpType_NoOp)};
}

const std::vector<std::vector<Op_Dtype>> GopDSCodePlugin::dtype_cap()
{
    return {{Op_Dtype{{Tensor::FP16}, {Tensor::FP16}}}};
}

bool GopDSCodePlugin::set_target(const std::string &target)
{
    if (target == "X3U")
    {
        m_target = target;
        return true;
    }
    return false;
}

GopImplPtr GopDSCodePlugin::get_impl(const NodePtrList &nodes, const GPattern &pattern)
{
    return nullptr;
}

// REGISTER_PLUGIN(GopDSCodePlugin, 0x01010000, GOPPLUGIN_TYPE);
// REGISTER_PLUGIN(GopDSCodeDSPlugin, 0x01010000, GOPPLUGIN_TYPE | GBPlugin_DS);
} // namespace aipubt

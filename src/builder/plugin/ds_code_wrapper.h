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
#pragma once
#include "gcompiler_api.h"

namespace aipubt
{
#define SETTER(n, t, v) \
    void set_##n(const t &o) { v = o; }
#define GETTER(n, t, v) \
    t &n() { return v; }
#define ADD_MEMBER(n, t, v) \
    SETTER(n, t, v)         \
    GETTER(n, t, v)

class GopDSCodeImpl : public GopImpl
{
public:
    virtual KernelInfos kernel_infos() override;

    // optional: use for turning performance
    virtual void update_prefer_cfg(const std::string &key, const Any &value) override;

    // optional: implement score
    virtual uint32_t score() const override;

    // optional: kernel cps bin file name
    virtual const std::string kernel_cpsbin() override;

    // optional: implement flops
    virtual uint64_t flops() const override;

    ADD_MEMBER(param_root, PlaceholderPtr, m_param_root);
    ADD_MEMBER(kernel_names, std::vector<std::string>, m_kernel_names);
    ADD_MEMBER(param_bases, std::vector<PlaceholderPtr>, m_param_bases);
    ADD_MEMBER(stream_bases, std::vector<PlaceholderPtr>, m_stream_bases);

protected:
    PlaceholderPtr m_param_root;
    std::vector<std::string> m_kernel_names;
    std::vector<PlaceholderPtr> m_param_bases;
    std::vector<PlaceholderPtr> m_stream_bases;
};

class GopDSCodePlugin : public GopPlugin
{
public:
    virtual const std::vector<GPattern> patterns() override;

    virtual const std::vector<std::vector<Op_Dtype>> dtype_cap() override;

    virtual bool set_target(const std::string &target) override;

    virtual GopImplPtr get_impl(const NodePtrList &nodes, const GPattern &pattern = {}) override;

protected:
    std::string m_target;
};

class GopDSCodeDSPlugin : public GopDSCodePlugin
{
};
} // namespace aipubt

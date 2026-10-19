//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Lang Hu
 * @Date: 2026-07-22 09:59:26
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-07-22 09:59:26
 */
#include "cps_host_eltwise.h"

namespace aipubt
{
KernelInfos BTHostEltwiseImpl::kernel_infos()
{
    auto node = m_nodes[0];
    auto input0 = node->inputs()[0];
    auto input1 = node->inputs()[1];
    auto output = node->outputs()[0];
    auto num = output->shape().symbol().size();
    std::string method = node->params()["method"];
    int type = method == "ADD" ? 0 : 1;

    KernelInfo info;
    info.kernel_name = "aipubt_cpshost_eltwise";
    info.args.push_back(input0);
    info.args.push_back(input1);
    info.args.push_back(output);
    info.args.push_back(num);
    info.args.push_back(type);

    return {info};
}

// optional: use for turning performance
void BTHostEltwiseImpl::update_prefer_cfg(const std::string &key, const Any &value)
{
}

// optional: implement score
uint32_t BTHostEltwiseImpl::score() const
{
    return 0;
}

// optional: kernel cps bin file name
const std::string BTHostEltwiseImpl::kernel_cpsbin()
{
    return "libcpshost.so";
}

// optional: implement flops
uint64_t BTHostEltwiseImpl::flops() const
{
    return 0;
}

const std::vector<GPattern> BTHostEltwisePlugin::patterns()
{
    return {GPattern(OpType_Eltwise)};
}

const std::vector<std::vector<Op_Dtype>> BTHostEltwisePlugin::dtype_cap()
{
    return {{Op_Dtype{{Tensor::FP16, Tensor::FP16}, {Tensor::FP16}}}};
}

bool BTHostEltwisePlugin::set_target(const std::string &target)
{
    if (target == "X3U")
    {
        m_target = target;
        return true;
    }
    return false;
}

GopImplPtr BTHostEltwisePlugin::get_impl(const NodePtrList &nodes, const GPattern &pattern)
{
    if (!check_params(nodes, pattern))
    {
        return nullptr;
    }
    auto ret = std::make_shared<BTHostEltwiseImpl>();
    ret->m_target = m_target;
    ret->m_nodes = nodes;
    return ret;
}

bool BTHostEltwisePlugin::check_params(const NodePtrList &nodes, const GPattern &pattern) const
{
    auto node = nodes[0];
    if (node->inputs()[0]->type() != Tensor::FP16 ||
        node->inputs()[1]->type() != Tensor::FP16 ||
        node->outputs()[0]->type() != Tensor::FP16)
    {
        return false;
    }
    std::string method = node->params()["method"];
    if (method != "ADD" && method != "MUL")
    {
        return false;
    }
    if (node->params()["with_activation"] != "NONE")
    {
        return false;
    }
    return true;
}

REGISTER_PLUGIN(BTHostEltwisePlugin, 0x01010000, GOPPLUGIN_TYPE | GBPlugin_Host);
REGISTER_PLUGIN(BTHostDSEltwisePlugin, 0x01010000, GOPPLUGIN_TYPE | GBPlugin_Host | GBPlugin_DS);
} // namespace aipubt

//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------

#include "mlir_add_kernel.h"

#include <cstdlib>

namespace aipubt
{

KernelInfos MlirAddKernelImpl::kernel_infos()
{
    auto node = m_nodes[0];

    KernelInfo info;
    // Entry function emitted by aipugc: `void @model(ptr<1>, ptr<1>, ptr<1>)`.
    info.kernel_name = "model";
    // The generated kernel still contains the whole element loop, so every
    // work item computes the same result. A single work item is enough.
    info.grid = tongue::Expr(1);
    info.block = tongue::Expr(1);
    // Argument order must match the generated signature: in0, in1, out.
    info.args.push_back(node->inputs()[0]);
    info.args.push_back(node->inputs()[1]);
    info.args.push_back(node->outputs()[0]);

    return {info};
}

uint32_t MlirAddKernelImpl::score() const
{
    return 1;   // lower the score to avoid choosing.
}

const std::string MlirAddKernelImpl::kernel_cpsbin()
{
    const char *env = std::getenv("AIPU_MLIR_ADD_OBJ");
    if (env != nullptr && env[0] != '\0')
    {
        return env;
    }

    return "add.o";
}

const std::vector<GPattern> MlirAddKernelPlugin::patterns()
{
    return {GPattern(OpType_Add)};
}

const std::vector<std::vector<Op_Dtype>> MlirAddKernelPlugin::dtype_cap()
{
    return {{Op_Dtype{{Tensor::FP32, Tensor::FP32}, {Tensor::FP32}}}};
}

bool MlirAddKernelPlugin::set_target(const std::string &target)
{
    if (target == "X3U")
    {
        m_target = target;
        return true;
    }
    return false;
}

GopImplPtr MlirAddKernelPlugin::get_impl(const NodePtrList &nodes, const GPattern &pattern)
{
    if (!check_params(nodes, pattern))
    {
        return nullptr;
    }
    auto ret = std::make_shared<MlirAddKernelImpl>();
    ret->m_target = m_target;
    ret->m_nodes = nodes;
    return ret;
}

bool MlirAddKernelPlugin::check_params(const NodePtrList &nodes, const GPattern &pattern) const
{
    if (nodes.size() != 1)
    {
        return false;
    }
    auto node = nodes[0];
    if (node->type() != OpType_Add)
    {
        return false;
    }
    if (node->inputs().size() != 2 || node->outputs().size() != 1)
    {
        return false;
    }
    if (node->inputs()[0]->type() != Tensor::FP32 ||
        node->inputs()[1]->type() != Tensor::FP32 ||
        node->outputs()[0]->type() != Tensor::FP32)
    {
        return false;
    }
    return true;
}

REGISTER_PLUGIN(MlirAddKernelPlugin, 0x01010000, GOPPLUGIN_TYPE);

} // namespace aipubt

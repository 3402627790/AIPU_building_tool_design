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
 * @Author: Samule Han
 * @Date: 2020-03-13 17:15:02
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-11-15 16:38:56
 */
#include "builder/builder.h"
#include "builder/lowering/lowering.h"
#include "builder/mem_remapping.h"
#include "builder/optimizer.h"
#include "builder/param_check.h"
#include "builder/plugin.h"
#include "builder/profiler.h"
#include "builder/ro_generator.h"
#include "builder/target.h"
#include "builder/text_generator.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/logger.h"
#include "utils/utils.h"

namespace aipubt
{

// If the total channel num is not larger than 32 (8bit) or 16 (16bit),
// we don't need LayoutConvertor between NHWC and NCHWC32(T8) (8bit) or NCHWC16(T8) (16bit)
// Refer to: https://jira.armchina.com/browse/CLT-171
static bool are_same_layouts(NodePtr node, std::set<DataLayout> &&same_layout_set)
{
    auto in_layout = node->inputs()[0]->shape().layout();
    if (same_layout_set.find(in_layout) == same_layout_set.end())
    {
        return false;
    }
    auto out_layout = node->outputs()[0]->shape().layout();
    if (same_layout_set.find(out_layout) == same_layout_set.end())
    {
        return false;
    }
    auto in_shape = node->inputs()[0]->shape();
    auto c_base = get_channel_base(node->inputs());
    if (in_shape.c_total() <= c_base)
    {
        return true;
    }
    else if (in_shape.h() == 1 && in_shape.w() == 1)
    {
        return true;
    }
    else if (in_shape.c_total() % c_base == 0)
    {
        int nhwc_count = 0;
        for (auto layout : {in_layout, out_layout})
        {
            if ((layout == DataLayout_NHWC) || (layout == DataLayout_NHWCSparse) || (layout == DataLayout_NDHWC))
            {
                nhwc_count++;
            }
        }
        return (nhwc_count != 1);
    }
    else
    {
        return false;
    }
};

static auto get_z2_same_layout_set(NodePtr node)
{
    std::set<DataLayout> same_layout_set;
    auto dtype_size = Tensor::dtype_size(node->inputs()[0]->type());
    if (dtype_size == 1)
    {
        same_layout_set = {
            DataLayout_NHWC,
            DataLayout_NCHWC32,
            DataLayout_NDHWC,
            DataLayout_NDCHWC32,
        };
    }
    else if (dtype_size == 2)
    {
        same_layout_set = {
            DataLayout_NHWC,
            DataLayout_NCHWC16,
            DataLayout_NDHWC,
            DataLayout_NDCHWC16,
        };
    }
    return same_layout_set;
};

static auto get_z3_same_layout_set(NodePtr node)
{
    std::set<DataLayout> same_layout_set;
    auto dtype_size = Tensor::dtype_size(node->inputs()[0]->type());
    if (dtype_size == 1)
    {
        same_layout_set = {
            DataLayout_NHWC,
            DataLayout_NCHWC32,
            DataLayout_NCHWC32T8,
            DataLayout_NDHWC,
            DataLayout_NDCHWC32,
            DataLayout_NDCHWC32T8,
            DataLayout_NHWCSparse,
            DataLayout_NCHWC32Sparse,
            DataLayout_NCHWC32T8Sparse,
        };
    }
    else if (dtype_size == 2)
    {
        same_layout_set = {
            DataLayout_NHWC,
            DataLayout_NCHWC16,
            DataLayout_NCHWC16T8,
            DataLayout_NDHWC,
            DataLayout_NDCHWC16,
            DataLayout_NDCHWC16T8,
            DataLayout_NHWCSparse,
            DataLayout_NCHWC16Sparse,
            DataLayout_NCHWC16T8Sparse,
        };
    }
    return same_layout_set;
};

static auto get_x3_same_layout_set(NodePtr node)
{
    std::set<DataLayout> same_layout_set;
    auto dtype_size = Tensor::dtype_size(node->inputs()[0]->type());
    if (dtype_size == 1)
    {
        same_layout_set = {
            DataLayout_NHWC,
            DataLayout_NCHWC32,
            DataLayout_NCHWC32T8,
            DataLayout_NCHWC32T1,
            DataLayout_NDHWC,
            DataLayout_NDCHWC32,
            DataLayout_NDCHWC32T8,
            DataLayout_NDCHWC32T1,
            DataLayout_NHWCSparse,
            DataLayout_NCHWC32Sparse,
            DataLayout_NCHWC32T8Sparse,
            DataLayout_NCHWC32T1Sparse,
        };
    }
    else if (dtype_size == 2)
    {
        same_layout_set = {
            DataLayout_NHWC,
            DataLayout_NCHWC16,
            DataLayout_NCHWC16T8,
            DataLayout_NCHWC16T1,
            DataLayout_NDHWC,
            DataLayout_NDCHWC16,
            DataLayout_NDCHWC16T8,
            DataLayout_NDCHWC16T1,
            DataLayout_NHWCSparse,
            DataLayout_NCHWC16Sparse,
            DataLayout_NCHWC16T8Sparse,
            DataLayout_NCHWC16T1Sparse,
        };
    }
    return same_layout_set;
};

// check is the node can be remaped, if yes, remap, else do nothing
bool is_a_remapping_convertor(NodePtr node, bool remap = true)
{
    auto set_out_tensor_offset = [](NodePtr node) {
        node->outputs()[0]->memory_offset()->set_base_offset(node->inputs()[0]->memory_offset());
    };

    DataLayout from = string_to_datalayout(node->params()["from"]);
    DataLayout to = string_to_datalayout(node->params()["to"]);
    auto inp = node->inputs()[0]->shape();
    auto out = node->outputs()[0]->shape();
    if (from == to || from == DataLayout_Unkown || to == DataLayout_Unkown || inp.layout() != from || out.layout() != to)
    {
        if (remap)
        {
            set_out_tensor_offset(node);
        }
        return true;
    }

    if ((inp.h() == 1 && inp.w() == 1) || ((from == DataLayout_NHWC) && (out.c() == 1)) //NHWC->NCHWC16, NHWC->NCHWC32
        || ((to == DataLayout_NHWC) && (inp.c() == 1)))                                 //NCHWC16->NHWC, NCHWC32->NHWC
    {
        if (remap)
        {
            set_out_tensor_offset(node);
        }
        return true;
    }
    auto target = get_target();
    auto arch = target->isainfo().type;
    if (arch == aipubt::ISA_ZYv2)
    {
        if (are_same_layouts(node, get_z2_same_layout_set(node)))
        {
            if (remap)
            {
                set_out_tensor_offset(node);
            }
            return true;
        }
    }
    else if (arch >= aipubt::ISA_ZYv6)
    {
        if (are_same_layouts(node, get_x3_same_layout_set(node)))
        {
            if (remap)
            {
                set_out_tensor_offset(node);
            }
            return true;
        }
    }
    else if (arch >= aipubt::ISA_ZYv3)
    {
        if (are_same_layouts(node, get_z3_same_layout_set(node)))
        {
            if (remap)
            {
                set_out_tensor_offset(node);
            }
            return true;
        }
    }
    return false;
}

RO_GENERATOR(OpType_LayoutConvertor)
{
    if (is_a_remapping_convertor(node))
    {
        //x2 shouldn't generate unuse ro because no text is generated
        //if this node share tcb with other nodes, this ro will not be skiped therefore cause bug
        return;
    }
    uint32_t flag = 0;
    if (BuilderConfig::get<bool>(Constants::enable_ds) && node->is_dynamic())
    {
        flag = flag | GBPlugin_DS;
    }
    auto plugin = BuilderOptimizer::get_single_op_plugin(OpType_LayoutConvertor, flag);
    if (plugin)
    {
        BaseNodePtrList m;
        auto target = get_target();
        m.push_back((BaseNodePtr)node);
        bool res = plugin->check_params(m);
        res &= plugin->set_target(target->name().c_str());
        if (res)
        {
            auto sgnode = node;
            INFO() << "Building " << sgnode->name() << "...";
            auto params = plugin->generate_params(sgnode, m);
            std::string error_msg;
            if (get_target()->isainfo().type >= ISA_ZYv5 &&
                !check_params(params, error_msg))
            {
                ERROR() << "Failed to check params: " << error_msg << "\n"
                        << "Node: " << sgnode->name();
            }
            ro.insert(ro.end(), params.begin(), params.end());
            // DEBUG() << "Using layoutconverter plugin";
            return;
        }
    }
    ERROR() << "Not support the layout convertor " << node->name() << ", please double check the params.";
}
RO_GENERATOR_END(OpType_LayoutConvertor);

TEXT_GENERATOR(OpType_LayoutConvertor)
{
    if (is_a_remapping_convertor(node))
    {
        return "";
    }
    uint32_t flag = 0;
    if (BuilderConfig::get<bool>(Constants::enable_ds) && node->is_dynamic())
    {
        flag = flag | GBPlugin_DS;
    }
    auto plugin = BuilderOptimizer::get_single_op_plugin(OpType_LayoutConvertor, flag);
    if (plugin)
    {
        BaseNodePtrList m;
        auto target = get_target();
        m.push_back(node);
        bool res = plugin->check_params(m);
        res &= plugin->set_target(target->name().c_str());
        if (res)
        {
            //use fake sgnode
            auto sgnode = node;
            BaseNodePtrList mm = m;
            // DEBUG() << "Using layoutconverter plugin";
            std::string ret = plugin->generate_code_name(sgnode, mm);
            bool enable_perf = aipubt::AIPUBuilder::get_profiler()->enable_profile();
            if (enable_perf && !ret.empty())
            {
                ret += "\n";
                ret += Constants::profiler_code;
                node->set_attr(Constants::node_perf, true);
            }
            return ret;
        }
    }
    ERROR() << "Not support the layout convertor " << node->name() << ", please double check the params.";
    return "";
}
TEXT_GENERATOR_END(OpType_LayoutConvertor);

KERNEL_GENERATOR(OpType_LayoutConvertor)
{
    if (is_a_remapping_convertor(node))
    {
        return "";
    }
    uint32_t flag = 0;
    if (BuilderConfig::get<bool>(Constants::enable_ds) && node->is_dynamic())
    {
        flag = flag | GBPlugin_DS;
    }
    auto plugin = BuilderOptimizer::get_single_op_plugin(OpType_LayoutConvertor, flag);
    if (plugin)
    {
        BaseNodePtrList m;
        m.push_back(node);
        bool res = plugin->check_params(m);
        if (res)
        {
            //use fake sgnode
            auto sgnode = node;
            BaseNodePtrList mm = m;
            // DEBUG() << "Using layoutconverter plugin";
            std::string kernel = plugin->generate_kernel(sgnode, mm);
            bool enable_perf = aipubt::AIPUBuilder::get_profiler()->enable_profile();
            if (enable_perf && !kernel.empty())
            {
                kernel += "\n";
                kernel += Constants::profiler_kernel;
            }
            auto codetree = node->get_attr<CodeTreePtr>(Constants::codetree);
            for (auto &s : utils::split_string(kernel, "\n"))
            {
                if (is_valid_kernel(s))
                {
                    codetree->push_back(s);
                }
            }
            return kernel;
        }
    }
    return "";
}
KERNEL_GENERATOR_END(OpType_LayoutConvertor);

MEMORY_REMAPPING_OP(OpType_LayoutConvertor)
{
    is_a_remapping_convertor(node);
}
MEMORY_REMAPPING_OP_END(OpType_LayoutConvertor);

}; // namespace aipubt

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
 * @Author: Kyle Huang
 * @Date: 2020-01-29 14:27:33
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2023-04-12 17:00:35
 */

#include "load_balancer.h"
#include "builder/lowering/datalayout_schedule.h"
#include "builder/lowering/hash_constant.h"
#include "builder/lowering/lowering.h"
#include "builder/ops/ops.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/ops/ops.h"
#include "core/pass_manager.h"
#include "lowering/tiling.h"
#include "statistic/perf_estimator.h"
#include "transform/passes.h"
#include "transform/split_node_output_c.h"
#include "utils/utils.h"
#include "x2_SM_opt.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stack>
#include <vector>
namespace aipubt
{
inline size_t aligned(size_t s, size_t align = 256)
{
    return (s + align - 1) & ~(align - 1);
}

static int get_cal_intense(NodePtr node)
{
    int cal_intense = 0;
    int total_constant_size = node->get_constant_size();
    int ops = node->get_attr<Expr>("ops_count")->value();
    cal_intense = total_constant_size ? ops / total_constant_size : 0;
    return cal_intense;
}

static bool is_memory_bound(NodePtr node, int parallel_cal_num)
{
    uint32_t cal_intense = get_cal_intense(node);
    uint32_t max_cal = Target::get_the_target()->ff_mac_num() * 1.25;
    uint32_t max_bw = Target::get_the_target()->axi_bandwidth() / 8 / parallel_cal_num;
    DEBUG() << "cal intense for " << node << " " << cal_intense << " max_cal " << max_cal << " max_bw " << max_bw;
    if (cal_intense > (max_cal / max_bw))
    {
        return false;
    }
    return true;
}

static bool show_sg_name(NodePtr node, int32_t depth)
{
    std::string prefix;
    for (int32_t i = 0; i < depth; ++i)
    {
        prefix = prefix + "-*-";
    }
    auto sgn = SubGraphNode::from_node(node);
    if (sgn)
    {
        DEBUG() << prefix << node->name();
        depth++;
        auto sg = sgn->subgraph();
        auto nodes = sg->nodes();
        for (auto sg_node : nodes)
        {
            show_sg_name(sg_node, depth);
        }
    }
    else
    {
        DEBUG() << prefix << "[real]" << node->name();
    }

    // for (auto inp : node->parents())
    // {
    //     DEBUG() << prefix << "[parents]" << inp->name();
    // }

    // for (auto outp : node->children())
    // {
    //     DEBUG() << prefix << "[children]" << outp->name();
    // }

    return true;
}

static int64_t get_load_from_model(NodePtr node)
{
    // should use performance model to get load
    int64_t load = -1;
    if (node->attrs().count("count_cycles"))
    {
        load = node->get_attr("count_cycles");
        // DEBUG() << node->name() << " load: " << load;
    }
    if (load == -1)
    {
        // WARN() << "no load for " << node->name() << " type: " << node->type();
        load = 1;
    }

    return load;
}

static bool show_level(std::vector<std::vector<NodePtr>> levels, std::string str, bool show = true)
{
    if (!show)
        return true;
    DEBUG() << "schedule level: " << str;
    for (size_t i = 0; i < levels.size(); ++i)
    {
        DEBUG() << "[level " << i << "]";
        for (auto node : levels[i])
        {
            show_sg_name(node, 0);
            DEBUG() << "[load] " << get_load_from_model(node);
        }
    }
    DEBUG() << "schedule level: " << str << " done";

    return true;
}

static bool check_dependency(std::vector<std::vector<NodePtr>> levels)
{
    for (auto level : levels)
    {
        for (auto n : level)
        {
            if (get_load_from_model(n) == 0)
            {
                continue;
            }
            for (auto p : n->parents())
            {
                if (get_load_from_model(p) == 0)
                {
                    continue;
                }
                if (contains(level, p))
                {
                    // WARN() << "node " << n->name() << " and parent " << p->name() << " are in the same level";
                    return false;
                }
            }
            for (auto c : n->children())
            {
                if (get_load_from_model(c) == 0)
                {
                    continue;
                }
                if (contains(level, c))
                {
                    // WARN() << "node " << n->name() << " and child " << c->name() << " are in the same level";
                    return false;
                }
            }
        }
    }
    return true;
}

/*merge nodes into 1 sg*/
static NodePtr merge_sg(GraphPtr g, std::vector<NodePtr> &nodes)
{
    if (nodes.size() > 1 || nodes[0]->type() != OpType_InternalConstant)
    {
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            auto n = nodes[i];
            if (n->type() == OpType_InternalConstant)
            {
                bool const_has_in_child = false;
                std::vector<NodePtr> const_out_child;
                for (auto c : n->children())
                {
                    if (contains(nodes, c))
                    {
                        const_has_in_child = true;
                    }
                    else
                    {
                        const_out_child.push_back(c);
                    }
                }
                if (const_has_in_child && !const_out_child.size())
                {
                    continue;
                }
                else if (const_has_in_child && const_out_child.size())
                {
                    auto n_clone = n->clone();
                    for (auto c : const_out_child)
                    {
                        c->replace_input_by(n->outputs()[0], n_clone->outputs()[0]);
                    }
                    g->add_node(n_clone);
                }
                else if (!const_has_in_child && const_out_child.size())
                {
                    nodes.erase(nodes.begin() + i);
                    --i;
                }
                else
                {
                    ERROR();
                }
            }

            for (auto p : n->parents())
            {
                if (p->type() == OpType_InternalConstant && !contains(nodes, p))
                {
                    std::vector<NodePtr> const_out_child;
                    for (auto c : n->children())
                    {
                        if (!contains(nodes, c))
                        {
                            const_out_child.push_back(c);
                        }
                    }

                    if (const_out_child.size())
                    {
                        auto n_clone = p->clone();
                        for (auto c : const_out_child)
                        {
                            c->replace_input_by(n->outputs()[0], n_clone->outputs()[0]);
                        }
                        g->add_node(n_clone);
                        nodes.push_back(p);
                    }
                    else
                    {
                        nodes.push_back(p);
                    }
                }
            }
        }
    }

    auto sg_merged = SubGraph::create(g, nodes);
    auto sg_merged_node = sg_merged->as_node(true, true);
    sg_merged_node->set_unified_view(false);
    int64_t load = 0;
    for (auto n : nodes)
    {
        load += get_load_from_model(n);
    }
    sg_merged_node->set_attr("count_cycles", load);
    // DEBUG() << sg_merged_node->name() << " merged load: " << load;
    return sg_merged_node;
}

static DataLayout get_new_layout(DataLayout lo)
{
    DataLayout new_layout;
    if (lo == DataLayout_NCHWC32 ||
        lo == DataLayout_NCHWC16 ||
        lo == DataLayout_NCHWC8 ||
        lo == DataLayout_NCHWC4)
    {
        new_layout = DataLayout::DataLayout_NHWC;
    }
    else if (lo == DataLayout_NDCHWC32 || lo == DataLayout_NDCHWC16)
    {
        new_layout = DataLayout::DataLayout_NDHWC;
    }
    else if (lo == DataLayout_NCHWC32T8 || lo == DataLayout_NCHWC16T8 ||
             lo == DataLayout_NCHWC32T1 || lo == DataLayout_NCHWC16T1)
    {
        new_layout = DataLayout::DataLayout_NHWC;
    }
    else if (lo == DataLayout_NDCHWC32T8 || lo == DataLayout_NDCHWC16T8 ||
             lo == DataLayout_NDCHWC32T1 || lo == DataLayout_NDCHWC16T1)
    {
        new_layout = DataLayout::DataLayout_NDHWC;
    }
    else
    {
        new_layout = lo;
    }
    return new_layout;
}

static void convert_output_internal_layout(GraphPtr g)
{
    auto criteria_outp = [](NodePtr n) -> bool {
        bool ret = false;
        for (auto outp : n->outputs())
        {
            if (outp->shape().layout() == DataLayout_NCHWC32 ||
                outp->shape().layout() == DataLayout_NCHWC16 ||
                outp->shape().layout() == DataLayout_NCHWC8 ||
                outp->shape().layout() == DataLayout_NCHWC4 ||
                outp->shape().layout() == DataLayout_NCHWC32T8 ||
                outp->shape().layout() == DataLayout_NCHWC16T8 ||
                outp->shape().layout() == DataLayout_NCHWC32T1 ||
                outp->shape().layout() == DataLayout_NCHWC16T1 ||
                outp->shape().layout() == DataLayout_NDCHWC32 ||
                outp->shape().layout() == DataLayout_NDCHWC16 ||
                outp->shape().layout() == DataLayout_NDCHWC32T8 ||
                outp->shape().layout() == DataLayout_NDCHWC16T8 ||
                outp->shape().layout() == DataLayout_NDCHWC32T1 ||
                outp->shape().layout() == DataLayout_NDCHWC16T1)
                ret = true;
        }
        return ret;
    };
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (n->type() == OpType_InternalConcatenated ||
            n->type() == OpType_InternalCrop ||
            n->type() == OpType_InternalSlice ||
            n->type() == OpType_InternalSpliter ||
            n->type() == OpType_InternalPad ||
            n->type() == OpType_SubGraphInput ||
            n->type() == OpType_LayoutConvertor)
        {
            n->set_attr(Constants::pre_config_layouts, true);
            LayoutCombinations out_layouts = {{n->outputs()[0]->shape().layout()}};
            n->set_attr(Constants::valid_output_layouts, out_layouts);
            LayoutCombinations in_layouts;
            if (n->type() != OpType_SubGraphInput)
                in_layouts = {{n->inputs()[0]->shape().layout()}};
            n->set_attr(Constants::valid_input_layouts, in_layouts);
        }
    }

    for (auto &&n : nodes)
    {
        if (!criteria_outp(n))
            continue;
        if (n->type() == OpType_InternalConcatenated ||
            n->type() == OpType_InternalCrop ||
            n->type() == OpType_InternalSlice ||
            n->type() == OpType_InternalSpliter ||
            n->type() == OpType_InternalPad ||
            n->type() == OpType_SubGraphInput ||
            n->type() == OpType_LayoutConvertor)
        {
            for (size_t i = 0; i < n->outputs().size(); ++i)
            {
                bool need_convertor = false;
                auto outp = n->outputs()[i];
                auto consumers = outp->consumers();
                DataLayout new_layout = get_new_layout(outp->shape().layout());
                for (auto c : consumers)
                {
                    if (c->type() != OpType_InternalConcatenated &&
                        c->type() != OpType_InternalCrop &&
                        c->type() != OpType_InternalSlice &&
                        c->type() != OpType_InternalSpliter &&
                        c->type() != OpType_InternalPad &&
                        c->type() != OpType_LayoutConvertor)
                    {
                        need_convertor = true;
                    }
                }

                if (need_convertor)
                {
                    auto convert_out = ops::convert_layout(outp, new_layout);
                    auto convert_node = convert_out->node();
                    convert_node->set_attr(Constants::use_out_tensor, true);
                    convert_node->set_attr(Constants::pre_config_layouts, true);
                    LayoutCombinations io_layouts = {{outp->shape().layout()}};
                    convert_node->set_attr(Constants::valid_output_layouts, io_layouts);
                    convert_node->set_attr(Constants::valid_input_layouts, io_layouts);

                    g->add_node(convert_node);

                    for (auto c : consumers)
                    {
                        if (c->type() != OpType_InternalConcatenated &&
                            c->type() != OpType_InternalCrop &&
                            c->type() != OpType_InternalSlice &&
                            c->type() != OpType_InternalSpliter &&
                            c->type() != OpType_InternalPad &&
                            c->type() != OpType_LayoutConvertor)
                        {
                            c->add_input(convert_out, c->remove_input(outp));
                        }
                    }
                }
            }
        }
        else
        {
            for (size_t i = 0; i < n->outputs().size(); ++i)
            {
                auto outp = n->outputs()[i];
                auto consumers = outp->consumers();
                DataLayout new_layout = get_new_layout(outp->shape().layout());
                if (outp->shape().layout() != new_layout)
                {
                    bool need_convertor = false;
                    for (auto c : consumers)
                    {
                        if (c->type() == OpType_InternalConcatenated ||
                            c->type() == OpType_InternalCrop ||
                            c->type() == OpType_InternalSlice ||
                            c->type() == OpType_InternalSpliter ||
                            c->type() == OpType_InternalPad ||
                            c->type() == OpType_LayoutConvertor)
                        {
                            need_convertor = true;
                        }
                    }
                    if (contains(g->get_output_tensors(), outp))
                    {
                        need_convertor = true;
                    }
                    if (need_convertor)
                    {
                        TensorShape new_shape = outp->shape();
                        new_shape.change_layout(new_layout);
                        auto convert_input = std::make_shared<Tensor>(n->name() + "/out_" + std::to_string(i), new_shape, outp->type());
                        convert_input->quantization() = outp->quantization();

                        auto convert_node = Node::create(n->name() + "_layout_convertor_out" + std::to_string(i), OpType_LayoutConvertor);
                        convert_node->params()["from"] = datalayout_to_string(new_layout);
                        convert_node->params()["to"] = datalayout_to_string(outp->shape().layout());
                        convert_node->add_input(convert_input);
                        convert_node->add_output(outp);
                        convert_node->set_attr(Constants::use_in_tensor, true);
                        convert_node->set_attr(Constants::pre_config_layouts, true);
                        LayoutCombinations io_layouts = {{outp->shape().layout()}};
                        convert_node->set_attr(Constants::valid_output_layouts, io_layouts);
                        convert_node->set_attr(Constants::valid_input_layouts, io_layouts);
                        n->add_output(convert_input, n->remove_output(outp));
                        g->add_node(convert_node);

                        for (auto c : consumers)
                        {
                            if (c->type() != OpType_InternalConcatenated &&
                                c->type() != OpType_InternalCrop &&
                                c->type() != OpType_InternalSlice &&
                                c->type() != OpType_InternalSpliter &&
                                c->type() != OpType_InternalPad &&
                                c->type() != OpType_LayoutConvertor)
                            {
                                c->add_input(convert_input, c->remove_input(outp));
                            }
                        }
                    }
                    else
                    {
                        outp->shape().change_layout(new_layout);
                    }
                }
            }
        }
    }
}

static void adjust_internal_layout(GraphPtr g)
{
    convert_output_internal_layout(g);
}

static int get_output_c(NodePtr n)
{
    int output_c = 0;
    switch (n->type())
    {
    case OpType_BatchNorm:
    {
        int axis = n->params()["axis"];
        output_c = n->outputs()[0]->shape()[axis];
        break;
    }
    case OpType_Pooling:
    case OpType_Eltwise:
    case OpType_Convolution:
    case OpType_Activation:
    case OpType_Mul:
    default:
        output_c = n->outputs()[0]->shape().c();
        break;
    }
    return output_c;
}
static bool split_filters(GraphPtr g, int split_num)
{
    auto hwa_criterion = [](NodePtr node, int split_num) -> bool {
        if (node->is_dynamic())
        {
            return false;
        }

        if (node->type() != OpType_Pooling &&
            node->type() != OpType_Eltwise &&
            node->type() != OpType_Convolution &&
            node->type() != OpType_Activation &&
            node->type() != OpType_Mul &&
            node->type() != OpType_BatchNorm)
        {
            return false;
        }

        for (auto inp : node->inputs())
        {
            for (auto c : inp->consumers())
            {
                if (c->tag() & Node::Inplace)
                {
                    return false;
                }
            }
        }

        for (auto outp : node->outputs())
        {
            auto clone_outp = std::make_shared<Tensor>(outp->name() + "_clone", outp->shape(), outp->type());
            clone_outp->shape().reset_strides();
            if (outp->shape().strides() != clone_outp->shape().strides())
            {
                return false;
            }
        }

        if (node->type() == OpType_Mul)
        {
            if (node->inputs()[1]->shape().dim() > 1)
            {
                return false;
            }
        }

        if (node->outputs()[0]->shape().dim() < 3)
        {
            // DEBUG() << "split filter ignore dim < 3 node: " << node->name();
            return false;
        }

        if (node->type() == OpType_BatchNorm)
        {
            auto shape = node->outputs()[0]->shape();
            int axis = node->params()["axis"];
            if ((shape.dim() == 3 && axis != 2) ||
                (shape.dim() == 4 && axis != 3) ||
                (shape.dim() == 5 && axis != 1))
            {
                return false;
            }
        }

        int output_c = get_output_c(node);
        int32_t min_ch = 32 * split_num;
        if (output_c < min_ch)
        {
            return false;
        }

        if (output_c % 32 != 0) // if not aligned to 32 will generate hardware op concat/split
        {
            return false;
        }

        if (node->params().count("group") &&
            (node->params()["group"] > 1) && (node->params()["group"] != output_c))
        {
            return false;
        }

        if (split_num == 1)
        {
            return false;
        }

        // if (node->constants().count("shift")) //filter-wise quant for depthwise
        // {
        //     return false;
        // }

        return true;
    };

    SplitNodeOutputC sn;
    auto m_nodes = g->nodes();
    bool changed = false;
    for (auto n : m_nodes)
    {
        if (!hwa_criterion(n, split_num))
        {
            continue;
        }
        DEBUG() << "spliting " << n->name() << " type: " << n->type();
        changed = true;
        std::vector<int> output_c_vec;
        int output_c = get_output_c(n);
        int split_c = output_c / split_num;
        split_c = aligned(split_c, 32);
        int remain = 0;
        for (int i = 0; i < split_num; ++i)
        {
            remain = output_c - (split_c * i);
            if (i < split_num - 1)
            {
                if (remain > split_c)
                {
                    output_c_vec.push_back(split_c);
                }
                else
                {
                    output_c_vec.push_back(remain);
                }
            }
            else
            {
                if (remain > 0)
                {
                    output_c_vec.push_back(remain);
                }
            }
        }
        if (n->type() == OpType_Convolution)
        {
            split_conv_output_c_(g, n, output_c_vec);
        }
        else
        {
            sn.set_output_c(output_c_vec);
            sn.split_output_c(g, n);
        }
    }

    return changed;
}

static std::set<TensorPtr> collect_input_tensors(GraphPtr g)
{
    std::set<TensorPtr> ret;
    for (auto n : g->nodes())
    {
        if (n->type() == OpType_SubGraphInput)
        {
            ret.insert(n->outputs()[0]);
        }
    }
    return ret;
}

static std::set<TensorPtr> collect_output_tensors(GraphPtr g, NodePtr node)
{
    std::set<TensorPtr> ret;
    for (auto n : g->nodes())
    {
        for (auto outp : n->outputs())
        {
            if (outp->consumers().size() == 0) // TODO: for both internal & output tensors
            {
                ret.insert(outp);
            }
            else
            {
                auto outp_p = std::dynamic_pointer_cast<ProxyTensor>(outp);
                if (outp_p)
                {
                    auto proxied_outp = outp_p->proxied_tensor();
                    for (auto o : node->outputs())
                    {
                        if (o == proxied_outp)
                        {
                            ret.insert(outp);
                        }
                    }
                }
            }
        }
    }
    return ret;
}

static std::vector<NodePtr> init_nodes(GraphPtr g, std::vector<NodePtr> nodes)
{
    std::vector<NodePtr> ret;
    // for single core optimization, we only generate one subgraph
    if ((BuilderConfig::get<bool>(Constants::single_core_opt) ||
         Target::get_the_target()->core_num() == 1 ||
         BuilderConfig::get<uint32_t>(Constants::multi_batch)) &&
        !BuilderConfig::get<bool>("disable_schedule_optimization"))
    {
        auto sgnode = merge_sg(g, nodes);
        ret.push_back(sgnode);
    }
    else
    {
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            auto node = nodes[i];
            std::vector<NodePtr> nodes = {node};
            auto sgnode = merge_sg(g, nodes);
            // copy attributes
            for (auto kv : node->attrs())
            {
                if (kv.first == "count_cycles")
                    continue;
                if (kv.first == Constants::same_w)
                    continue;
                if (kv.first != Constants::plugin_attrs_name)
                    sgnode->set_attr(kv.first, kv.second);
            }
            ret.push_back(sgnode);
            // m_graph->remove_node(node);
        }
    }
    return ret;
}

static bool setup_level(std::vector<NodePtr> input_nodes, std::vector<NodePtr> all_nodes, std::vector<std::vector<NodePtr>> &levels)
{
    levels.clear();
    std::vector<NodePtr> ran_nodes(input_nodes.begin(), input_nodes.end());
    std::vector<NodePtr> current_level(input_nodes.begin(), input_nodes.end());
    std::vector<NodePtr> next_level;

    // std::set<NodePtr> stack;
    auto all_in_set = [](const std::vector<NodePtr> &mset, const std::vector<NodePtr> &checks) -> bool {
        for (auto n : checks)
        {
            if (!contains(mset, n))
            {
                return false;
            }
        }
        return true;
    };
    // each level can only has 1 sgnode
    while (current_level.size())
    {

        levels.push_back(std::vector<NodePtr>(current_level.begin(), current_level.end()));
        for (auto node : current_level)
        {
            for (auto c : node->children())
            {
                if (all_in_set(ran_nodes, c->parents()) && !contains(ran_nodes, c) && !contains(next_level, c) && contains(all_nodes, c))
                {
                    next_level.push_back(c);
                }
            }
        }
        for (auto node : next_level)
        {
            if (!contains(ran_nodes, node))
            {
                ran_nodes.push_back(node);
            }
        }
        current_level = std::move(next_level);
        next_level = std::vector<NodePtr>();
    }

    if (ran_nodes.size() != all_nodes.size())
    {
        WARN() << "setup level failed";
        return false;
    }

    return true;
}

static void find_all_children(NodePtr node, std::vector<NodePtr> &children)
{

    for (size_t j = 0; j < node->children().size(); ++j)
    {
        auto c = node->children()[j];
        if (!contains(children, c))
        {
            children.push_back(c);
        }
        if (get_load_from_model(c) == 0)
        {
            // total_depd_size += get_dped_size(p);
            find_all_children(c, children);
        }
    }
}

static void find_not_internal_children(NodePtr node, std::vector<NodePtr> &children)
{

    for (size_t j = 0; j < node->children().size(); ++j)
    {
        auto c = node->children()[j];

        if (get_load_from_model(c) == 0)
        {
            // total_depd_size += get_dped_size(p);
            find_not_internal_children(c, children);
        }
        else if (!contains(children, c))
        {
            children.push_back(c);
        }
    }
}

static void find_all_parents(NodePtr node, std::vector<NodePtr> &parents)
{

    for (size_t j = 0; j < node->parents().size(); ++j)
    {
        auto p = node->parents()[j];
        if (!contains(parents, p))
        {
            parents.push_back(p);
        }
        if (get_load_from_model(p) == 0)
        {
            // total_depd_size += get_dped_size(p);
            find_all_parents(p, parents);
        }
    }
}

static void find_not_internal_parents(NodePtr node, std::vector<NodePtr> &parents)
{

    for (size_t j = 0; j < node->parents().size(); ++j)
    {
        auto p = node->parents()[j];

        if (get_load_from_model(p) == 0)
        {
            // total_depd_size += get_dped_size(p);
            find_not_internal_parents(p, parents);
        }
        else if (!contains(parents, p))
        {
            parents.push_back(p);
        }
    }
}

static NodePtr get_child_parent_same_node(NodePtr n)
{
    std::vector<NodePtr> all_parents;
    find_not_internal_parents(n, all_parents);
    std::vector<NodePtr> all_children;
    find_not_internal_children(n, all_children);
    for (auto p : all_parents)
    {
        if (contains(all_children, p))
        {
            return p;
        }
    }
    return nullptr;
}

/*check if the two level can be merged*/
static bool level_merge_condition(const std::vector<NodePtr> &level, const std::vector<NodePtr> &next_level, int32_t connection_num)
{
    // if a node in level has multiple child in next_level then false
    for (auto node : level)
    {
        if (get_load_from_model(node) == 0)
        {
            continue;
        }
        std::vector<NodePtr> real_children;
        int child_count = 0;
        find_not_internal_children(node, real_children);
        for (auto child : real_children)
        {
            if (count(next_level.begin(), next_level.end(), child))
            {
                child_count++;
            }
        }
        if (child_count > connection_num)
        {
            return false;
        }
        if (node->inputs().size() == 0)
        {
            return false;
        }
    }

    // if a node in next_level has multiple parent in level then false
    for (auto node : next_level)
    {
        if (get_load_from_model(node) == 0)
        {
            continue;
        }
        std::vector<NodePtr> real_parents;
        int parent_count = 0;
        find_not_internal_parents(node, real_parents);
        for (auto parent : real_parents)
        {
            if (count(level.begin(), level.end(), parent))
            {
                parent_count++;
            }
        }
        if (parent_count > connection_num)
        {
            return false;
        }
    }
    return true;
}

/*remove least load node in vector*/
static NodePtr remove_least_load(std::vector<NodePtr> &loads)
{
    int least_load_idx = 0;
    int64_t least_load = get_load_from_model(loads[0]);
    for (size_t i = 1; i < loads.size(); ++i)
    {
        if (least_load > get_load_from_model(loads[i]))
        {
            least_load = get_load_from_model(loads[i]);
            least_load_idx = i;
        }
    }
    NodePtr node = loads[least_load_idx];
    loads.erase(loads.begin() + least_load_idx);
    return node;
}

static std::vector<NodePtr> get_dped(NodePtr node, const std::map<NodePtr, std::set<NodePtr>> &dped_map)
{
    std::vector<NodePtr> dped, merged_dped;

    if (get_load_from_model(node) == 0)
    {
        return merged_dped;
    }

    find_not_internal_parents(node, dped);
    bool is_dped = true;
    for (size_t i = 0; i < dped.size(); ++i)
    {
        is_dped = true;
        for (size_t j = 0; j < dped.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            if (contains(dped_map.at(dped[j]), dped[i]))
            {
                is_dped = false;
                break;
            }
        }
        if (is_dped)
        {
            merged_dped.push_back(dped[i]);
        }
    }
    return merged_dped;
};

static std::vector<NodePtrList> get_balanced_groups(const NodePtrList &nodes, size_t core_num)
{
    using LoadList = std::vector<std::pair<NodePtr, int64_t>>;

    auto cmp = [](std::pair<NodePtr, int64_t> n0, std::pair<NodePtr, int64_t> n1) -> bool {
        if (n0.second == n1.second)
        {
            return (n0.first->name() > n1.first->name());
        }
        else
        {
            return (n0.second > n1.second);
        }
    };
    auto get_sort_nodes = [cmp](const std::map<NodePtr, int64_t> &nodes) -> LoadList {
        LoadList sort_nodes(nodes.begin(), nodes.end());
        std::stable_sort(sort_nodes.begin(), sort_nodes.end(), cmp);
        return sort_nodes;
    };
    auto get_exp_load = [](const LoadList sort_nodes, size_t core_num) -> double {
        double exp_load = 0;
        int64_t sum_load = 0;
        for (auto node : sort_nodes)
        {
            sum_load += node.second;
        }
        exp_load = sum_load * 1.0 / core_num;
        return exp_load;
    };
    auto find_nearest_node = [](LoadList sort_nodes, double val) -> uint32_t {
        uint32_t best_id = 0;
        double best_diff = std::abs(sort_nodes[0].second - val);
        for (size_t i = 1; i < sort_nodes.size(); ++i)
        {
            double diff = std::abs(sort_nodes[i].second - val);
            if (diff < best_diff)
            {
                best_id = i;
                best_diff = diff;
            }
        }
        return best_id;
    };
    // CP-24606: make convs with the same parent through "split_conv_output_c_" in the same group when tiling
    auto get_balanced_split_conv_groups = [](const LoadList &sort_nodes, size_t core_num) -> std::vector<NodePtrList> {
        auto criteria = [](NodePtr n) {
            if (SubGraphNode::is_subgraph_node(n))
            {
                SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(n);
                auto first_node = sgn->get_real_node()[0];
                // check pass
                if (first_node->type() == OpType_Convolution && first_node->loc() && first_node->loc()->through_pass("split_conv_output_c_"))
                {
                    return true;
                }
            }
            return false;
        };
        // check
        if (sort_nodes.size() % core_num != 0)
        {
            return {};
        }
        NodePtrList unmark_nodes;
        std::vector<NodePtrList> groups;
        std::map<NodePtr, NodePtrList> conv_groups;
        for (size_t i = 0; i < sort_nodes.size(); i += 1)
        {
            auto n = sort_nodes[i].first;
            if (criteria(n))
            {
                conv_groups[n->parents()[0]].push_back(n);
            }
            else
            {
                unmark_nodes.push_back(n);
            }
        }
        // check tiling
        if (conv_groups.size() != core_num)
        {
            return {};
        }
        // generate groups of conv
        for (auto &conv_group : conv_groups)
        {
            NodePtrList group;
            for (auto conv : conv_group.second)
            {
                group.push_back(conv);
            }
            groups.push_back(group);
        }
        // generate groups of other nodes
        for (size_t cid = 0; cid < core_num; ++cid)
        {
            NodePtrList group;
            for (size_t base = 0; base < unmark_nodes.size(); base += core_num)
            {
                group.push_back(unmark_nodes[base + cid]);
            }
            groups[cid].insert(groups[cid].end(), group.begin(), group.end());
        }
        return groups;
    };
    auto get_balanced_groups_fast = [](const LoadList &sort_nodes, size_t core_num) -> std::vector<NodePtrList> {
        std::vector<NodePtrList> groups;
        if (sort_nodes.size() % core_num != 0)
        {
            return {};
        }
        // check
        for (size_t base = 0; base < sort_nodes.size(); base += core_num)
        {
            auto load_core0 = sort_nodes[base].second;
            for (size_t cid = 0; cid < core_num; ++cid)
            {
                auto load = sort_nodes[base + cid].second;
                if (load != load_core0)
                {
                    return {};
                }
            }
        }
        // generate groups
        for (size_t cid = 0; cid < core_num; ++cid)
        {
            NodePtrList group;
            for (size_t base = 0; base < sort_nodes.size(); base += core_num)
            {
                group.push_back(sort_nodes[base + cid].first);
            }
            groups.push_back(group);
        }
        return groups;
    };
    auto get_balanced_groups_normal = [get_exp_load, find_nearest_node](const LoadList &sort_nodes, size_t core_num) -> std::vector<NodePtrList> {
        LoadList nodes;
        for (auto pair : sort_nodes)
        {
            nodes.push_back(pair);
        }
        auto exp_load = get_exp_load(nodes, core_num);
        std::vector<NodePtrList> groups;
        for (size_t cid = 0; cid < core_num - 1 && nodes.size() > 0; ++cid)
        {
            NodePtrList group = {nodes[0].first};
            int64_t sum_load = nodes[0].second;
            // DEBUG() << "nodes.size(): " << nodes.size() << ", nodes.begin(): " << nodes.begin()->first->name();
            nodes.erase(nodes.begin());
            while (nodes.size())
            {
                auto best_id = find_nearest_node(nodes, exp_load - sum_load);
                auto best_val = nodes[best_id].second;
                double old_diff = std::abs(exp_load - sum_load);
                double new_diff = std::abs(exp_load - (sum_load + best_val));
                if (new_diff < old_diff)
                {
                    auto best_node = nodes[best_id];
                    // DEBUG() << "\t best_node: " << best_node.first << ", load: " << best_node.second << ", id: " << best_id << ", sort_nodes.size()";
                    group.push_back(best_node.first);
                    sum_load += best_node.second;
                    nodes.erase(nodes.begin() + best_id);
                }
                else
                {
                    break;
                }
                if (nodes.size() == 0)
                {
                    break;
                }
            }
            if (group.size() > 0)
            {
                groups.push_back(group);
            }
        }
        // last core
        NodePtrList last_group;
        for (auto node : nodes)
        {
            last_group.push_back(node.first);
        }
        if (last_group.size() > 0)
        {
            groups.push_back(last_group);
        }
        return groups;
    };
    // std::map<std::string, int> load_map =
    // {
    //     {"n1", 1},
    //     {"n2", 2},
    //     {"n3", 3},
    //     {"n4", 4},
    //     {"n5", 5},
    //     {"n6", 6},
    //     {"n7", 7},
    //     {"n8", 8},
    //     {"n9", 9},
    //     {"n10", 10},
    // };
    std::map<NodePtr, int64_t> load_map;
    std::set<OpType> internals = {
        OpType_InternalConcatenated,
        OpType_InternalCrop,
        OpType_InternalSlice,
        OpType_InternalSpliter,
        OpType_InternalPad,
        OpType_Reshape,
        OpType_Dummy,
    };
    std::vector<NodePtrList> rest_groups;
    for (auto node : nodes)
    {
        auto load = get_load_from_model(node);
        if (internals.count(get_real_op_type(node)) || load == 0)
        {
            rest_groups.push_back({node});
        }
        else
        {
            load_map[node] = load;
        }
    }
    if (load_map.size() <= core_num)
    {
        std::vector<NodePtrList> groups;
        for (auto node : nodes)
        {
            groups.push_back({node});
        }
        return groups;
    }
    auto sort_nodes = get_sort_nodes(load_map);
    // DEBUG();
    // DEBUG() << "sorted";
    // for (auto node: sort_nodes)
    // {
    //     DEBUG() << node.first->name() << ", load: " << node.second;
    // }
    // auto exp_load = get_exp_load(sort_nodes, core_num);
    // DEBUG() << "exp_load: " << exp_load;

    std::vector<NodePtrList> groups;
    if (BuilderConfig::get<bool>(Constants::tile_split_output_c_conv))
    {
        groups = get_balanced_split_conv_groups(sort_nodes, core_num);
    }
    if (groups.size() == 0)
    {
        groups = get_balanced_groups_fast(sort_nodes, core_num);
    }
    if (groups.size() == 0)
    {
        groups = get_balanced_groups_normal(sort_nodes, core_num);
    }
    for (size_t i = 0; i < groups.size(); ++i)
    {
        auto group = groups[i];
        int64_t sum_load = 0;
        for (size_t j = 0; j < group.size(); ++j)
        {
            sum_load += load_map[group[j]];
        }
        // DEBUG() << "core" << i << ", load: " << sum_load;
    }

    for (auto rest : rest_groups)
    {
        groups.push_back(rest);
    }

    return groups;
}

/*remove most load node in vector*/
static NodePtr remove_most_load(std::vector<NodePtr> &loads)
{
    int most_load_idx = 0;
    int64_t most_load = get_load_from_model(loads[0]);
    for (size_t i = 1; i < loads.size(); ++i)
    {
        if (most_load < get_load_from_model(loads[i]))
        {
            most_load = get_load_from_model(loads[i]);
            most_load_idx = i;
        }
    }
    NodePtr node = loads[most_load_idx];
    loads.erase(loads.begin() + most_load_idx);
    return node;
}

// static NodePtr remove_zero_load(std::vector<NodePtr> &loads)
// {
//     int zero_load_idx = -1;
//     for (size_t i = 1; i < loads.size(); ++i)
//     {
//         if (get_load_from_model(loads[i]) == 0)
//         {
//             zero_load_idx = i;
//             break;
//         }
//     }
//     if (zero_load_idx == -1)
//     {
//         return nullptr;
//     }

//     NodePtr node;
//     node = loads[zero_load_idx];
//     loads.erase(loads.begin() + zero_load_idx);

//     return node;
// }

/* if loads > core num, merge loads until loads = core num */
static bool merge_load(GraphPtr g, size_t core_num, int32_t idx, std::vector<std::vector<NodePtr>> &levels, uint32_t strategy = 0)
{
    std::vector<NodePtr> zero_sgs;
    // NodePtr sg;
    // sg = remove_zero_load(levels[idx]);
    // while (sg)
    // {
    //     zero_sgs.push_back(sg);
    //     sg = remove_zero_load(levels[idx]);
    // }
    // auto nodes = levels[idx];
    for (int i = 0; i < (int)levels[idx].size(); ++i)
    {
        auto node = levels[idx][i];
        if (get_load_from_model(node) == 0)
        {
            levels[idx].erase(levels[idx].begin() + i);
            zero_sgs.push_back(node);
            i--;
        }
    }

    if (strategy == 0)
    {
        if (levels[idx].size() > core_num)
        {
            // INFO() << "DEBUG level: " << idx << " before balanced group " << levels[idx].size();
            // if(idx == 3)
            // {
            //     for (auto n : levels[idx])
            //     {
            //         INFO() << n->name();
            //     }
            // }

            auto sgs = get_balanced_groups(levels[idx], core_num);
            levels[idx] = {};
            // INFO() << "DEBUG level: " << idx << " after balanced group";

            for (auto sg : sgs)
            {
                // if(idx == 3)
                // {
                //     INFO() << "new sg";
                //     int cnt = 0;
                //     for (auto n : sg)
                //     {
                //         INFO() << cnt << " " << n->name();
                //         auto sgn = SubGraphNode::from_node(n);
                //         if (sgn)
                //         {
                //             auto nodes = sgn->subgraph()->nodes();
                //             for (auto node : nodes)
                //             {
                //                 INFO() << "--" << node->name();
                //             }
                //         }
                //         cnt++;
                //     }
                // }
                auto sg_merged_node = merge_sg(g, sg);
                levels[idx].push_back(sg_merged_node);
            }
        }
    }
    else
    {
        while (levels[idx].size() > core_num)
        {
            auto sg1 = remove_least_load(levels[idx]);
            auto sg2 = remove_least_load(levels[idx]);
            std::vector<NodePtr> nodes = {sg1, sg2};
            auto sg_merged_node = merge_sg(g, nodes);
            levels[idx].push_back(sg_merged_node);
        }
    }

    if (levels[idx].size())
    {
        for (auto sg : zero_sgs)
        {
            levels[idx].push_back(sg);
        }
    }
    else
    {
        levels[idx] = zero_sgs;
    }

    // DEBUG() << "level: " << idx << " sg num: " << levels[idx].size();
    return true;
}

/*merge m_level[idx] and m_level[idx+1]*/
bool merge_next_level(size_t idx, GraphPtr g, std::vector<std::vector<NodePtr>> &levels, size_t core_num)
{
    if (idx < levels.size() - 1)
    {
        if (level_merge_condition(levels[idx], levels[idx + 1], 1))
        {
            // merge nodes by dependency
            auto nodes = levels[idx];
            for (size_t i = 0; i < nodes.size(); ++i)
            {
                std::vector<NodePtr> all_children;
                auto node = nodes[i];
                if (get_load_from_model(node) == 0)
                {
                    continue;
                }
                find_all_children(node, all_children);
                std::vector<NodePtr> nodes_to_merge;
                std::vector<NodePtr> zero_nodes;
                for (auto child : all_children)
                {
                    auto it = find(levels[idx + 1].begin(), levels[idx + 1].end(), child);
                    if (it != levels[idx + 1].end())
                    {
                        levels[idx + 1].erase(it);
                        if (get_load_from_model(child) == 0)
                        {
                            zero_nodes.push_back(child);
                            continue;
                        }
                        else
                        {
                            nodes_to_merge.push_back(child);
                        }

                        std::vector<NodePtr> all_parents;
                        find_all_parents(child, all_parents);
                        for (auto p : all_parents)
                        {
                            auto it2 = find(levels[idx].begin(), levels[idx].end(), p);
                            if (it2 != levels[idx].end())
                            {
                                if (p == node)
                                {
                                    nodes_to_merge.insert(nodes_to_merge.begin(), p);
                                    levels[idx].erase(it2);
                                }
                                else if (get_load_from_model(p) == 0)
                                {
                                    zero_nodes.push_back(p);
                                    // nodes_to_merge.insert(nodes_to_merge.begin(), p);
                                    levels[idx].erase(it2);
                                }
                                else
                                {
                                    ERROR();
                                }
                            }
                        }
                    }
                }
                if (nodes_to_merge.size())
                {
                    auto sg_merged_node = merge_sg(g, nodes_to_merge);
                    levels[idx].push_back(sg_merged_node);
                }

                for (auto n : zero_nodes)
                {
                    auto p_c_node = get_child_parent_same_node(n);
                    if (p_c_node)
                    {
                        auto it = std::find(levels[idx].begin(), levels[idx].end(), p_c_node);
                        if (it != levels[idx].end())
                        {
                            levels[idx].erase(it);
                            std::vector<NodePtr> nodes = {p_c_node, n};
                            auto sg_merged_node = merge_sg(g, nodes);
                            levels[idx].push_back(sg_merged_node);
                        }
                        else
                        {
                            ERROR();
                        }
                    }
                    else
                    {
                        levels[idx].push_back(n);
                    }
                }
            }
            // put all nodes to last level and do load balance
            for (auto node : levels[idx + 1])
            {
                levels[idx].push_back(node);
            }
            // merge_load(g, core_num, idx, levels);
            // remove the merged level
            levels[idx + 1].clear();
            levels.erase(levels.begin() + idx + 1);
        }
    }

    return true;
}

/*merge level to reduce level size*/
bool reduce_level(GraphPtr g, std::vector<std::vector<NodePtr>> &levels, size_t core_num, const std::vector<int> &skip_levels)
{
    for (int32_t i = levels.size() - 2; i >= 0; --i)
    {
        if (contains(skip_levels, i + 1) || contains(skip_levels, i))
        {
            // DEBUG() << "reduce skip level " << i;
            continue;
        }
        merge_next_level(i, g, levels, core_num);
    }
    return true;
}

bool split_level(std::vector<std::vector<NodePtr>> &levels)
{
    for (size_t i = 0; i < levels.size(); ++i)
    {
        auto level = levels[i];
        if (level.size() > 3 && level.size() % 3 == 0)
        {
            levels.erase(levels.begin() + i);
            for (size_t j = 0; j < level.size() / 3; ++j)
            {
                std::vector<NodePtr> new_level;
                new_level.push_back(level[j]);
                new_level.push_back(level[j + level.size() / 3]);
                new_level.push_back(level[j + level.size() / 3 * 2]);
                levels.insert(levels.begin() + i, new_level);
                i++;
            }
            i--;
        }
    }
    return true;
}

static int64_t performance_eval(const std::vector<std::vector<NodePtr>> &levels, uint32_t core, bool show = 0)
{
    if (show)
    {
        DEBUG() << "----------------------------------------------";
        DEBUG() << "Scheduler Optimization Performance Evaluation:";
    }
    int64_t total_cycle = 0;
    for (size_t i = 0; i < levels.size(); ++i)
    {
        int64_t cycle = 0;
        std::vector<float> utils;
        auto nodes = levels[i];
        cycle = get_load_from_model(nodes[0]);
        for (size_t j = 1; j < nodes.size(); ++j)
        {
            cycle = std::max(cycle, get_load_from_model(nodes[j]));
        }
        if (show)
        {
            // if (nodes.size() > core)
            // {
            //     WARN() << "sg size(" << nodes.size() << ") > core number(" << core << ")";
            // }
            // DEBUG() << "level: " << i << " "
            //        << " cycle: " << cycle;
            // for (size_t j = 0; j < nodes.size(); ++j)
            // {
            //     DEBUG() << nodes[j]->name() << ": " << get_load_from_model(nodes[j]);
            // }
            for (size_t j = 0; j < nodes.size(); ++j)
            {
                auto node_cycle = get_load_from_model(nodes[j]);
                if (node_cycle > 0)
                {
                    float util = (float)node_cycle / cycle;
                    utils.push_back(util);
                }
                else
                {
                    continue;
                }
            }
            if (utils.size() > core)
            {
                WARN() << "valid sg size(" << utils.size() << ") > core number(" << core << ")";
            }
            DEBUG() << "level: " << i << " cycles: " << cycle << " utils: " << utils::join_string(" ", utils);
        }
        total_cycle += cycle;
    }
    if (show)
    {
        DEBUG() << "total cycles: " << total_cycle;
        DEBUG() << "----------------------------------------------";
    }
    return total_cycle;
}

static bool connect_same_w_node(std::vector<NodePtr> nodes)
{
    std::map<std::string, NodePtr> hex_map;
    bool all_same_w = true;
    for (auto n : nodes)
    {
        if (n->get_constant_size() == 0)
        {
            all_same_w = false;
            break;
        }
        std::string total_hex;
        for (auto kv : n->constants())
        {
            // avoid nullptr
            if (!kv.second || !kv.second->data<void>())
            {
                continue;
            }
            auto hash = utils::sha256_hex(kv.second->data<void>(), kv.second->mem_size());
            total_hex = total_hex + hash;
        }
        auto it = hex_map.find(total_hex);
        if (it != hex_map.end())
        {
            // auto first_node = it->second;
            // if (first_node->attrs().count(Constants::same_w))
            // {
            //     auto next_node = first_node->get_attr<NodeRef>(Constants::same_w).lock();
            //     n->set_attr(Constants::same_w, NodeRef(next_node));
            // }
            // else
            // {
            //     n->set_attr(Constants::same_w, NodeRef(first_node));
            // }
            // first_node->set_attr(Constants::same_w, n);
            // DEBUG() << "same node: " << n;
            continue;
        }
        else
        {
            if (!hex_map.empty())
            {
                all_same_w = false;
            }
            hex_map.insert(std::make_pair(total_hex, n));
        }
    }
    return all_same_w;
}

static void skip_levels_by_tiling(std::vector<int> &top_skip_levels, const std::vector<std::vector<NodePtr>> &m_levels)
{
    bool cur_iso = false, next_iso = false;
    for (size_t i = 0; i < m_levels.size() - 1; ++i)
    {
        // INFO() << "level " << i;
        for (auto node : m_levels[i])
        {
            if (node->has_attr_deep("keep_isolate_when_scheduling"))
            {
                if (node->get_attr_deep<bool>("keep_isolate_when_scheduling"))
                {
                    next_iso = true;
                }
                else
                {
                    if (next_iso == true)
                    {
                        ERROR() << "different iso flag in same level!";
                    }
                }
            }
            else
            {
                next_iso = false;
            }
        }

        if (cur_iso == true && next_iso == true)
        {
            top_skip_levels.push_back(i - 1);
            INFO() << "skip opt level " << i - 1;
        }

        cur_iso = next_iso;
    }
}

static std::vector<int> generate_skip_levels(const std::vector<std::vector<NodePtr>> &levels)
{
    std::vector<int> skip_levels;
    for (size_t i = 0; i < levels.size(); ++i)
    {
        auto level = levels[i];
        if (connect_same_w_node(level) && level.size() > 1)
        {
            // DEBUG() << "all same node for level " << i;
            // for (auto n : level)
            // {
            //     DEBUG() << n;
            // }
            skip_levels.push_back(i);
        }
    }
    return skip_levels;
}

static void passes_before_apply_plugin(GraphPtr g, int strategy)
{
    static std::map<int, PassTuplePtr> passes = {
        {
            0, // strategy 0
            make_pass_tuple<
                // FPass_fuse_concat_split,
                FPass_fuse_concat_splits,
                FPass_fuse_concat_reshape_splits,
                FPass_split_to_crops,
                FPass_hwa_pooling_c_adaptor,
                FPass_hwa_conv_outc_adaptor,
                FPass_hwa_depthwise_conv_c_adaptor,
                FPass_hwa_eltwise_c_adaptor,
                FPass_hwa_add_mul_sub_c_adaptor,
                FPass_hwa_relu_c_adaptor,
                FPass_adapt_arithmetic_op,
                FPass_offset_crop_pad,
                FPass_eliminate_crop,
                FPass_eliminate_pad,
                FPass_fuse_reshape,
                FPass_eliminate_reshape>(),
        },
        {
            2, // strategy 2
            make_pass_tuple<
                FPass_fuse_concat_splits>(),
        },
    };

    if (passes.count(strategy))
    {
        passes[strategy]->run_on_graph(g);
    }
}

static void passes_after_apply_plugin(GraphPtr g)
{
    static auto passes = make_pass_tuple<
        FPass_calculate_node_ops_count,
        DataLayoutSchedule,
        FPass_remove_split,
        FPass_remove_concat,
        FPass_adapt_concat_strides,
        FPass_max_batch_adaptor,
        FPass_virtual_node_memory_remapping,
        FPass_add_address_to_dot>();

    passes->run_on_graph(g);
}

// static bool recover_bak_nodes(GraphPtr g, std::vector<NodePtr> &nodes_bak, std::vector<NodePtr> &nodes)
// {
//     auto merged_node = merge_sg(g, nodes);
//     for (auto node_bak : nodes_bak)
//     {
//         auto sgn_bak = SubGraphNode::from_node(node_bak);
//         auto sg_bak = sgn_bak->subgraph();
//         //update matchlist
//         for (auto n : sg_bak->nodes())
//         {
//             if (n->attrs().count("_match_list"))
//             {
//                 auto sgn_n = SubGraphNode::from_node(n);
//                 auto sg_n = sgn_n->subgraph();
//                 std::vector<NodePtr> m;
//                 std::vector<NodePtr> m_ori = n->get_attr("_match_list");
//                 bool found_node = false;
//                 for (auto mn : m_ori)
//                 {
//                     found_node = false;
//                     for (auto nn : sg_n->nodes())
//                     {
//                         if (mn->name() == nn->name())
//                         {
//                             found_node = true;
//                             m.push_back(nn);
//                             break;
//                         }
//                     }
//                     if (!found_node)
//                     {
//                         ERROR() << "cannot found " << mn->name();
//                     }
//                 }
//                 n->set_attr("_match_list", m);
//             }
//         }
//         //remove desc chain
//         for (auto n : sg_bak->nodes())
//         {
//             if (n->attrs().count(Constants::next_descriptor_length))
//             {
//                 n->erase_attr(Constants::next_descriptor_length);
//             }
//             if (n->attrs().count(Constants::next_descriptor_offset))
//             {
//                 n->erase_attr(Constants::next_descriptor_offset);
//             }
//             if (n->attrs().count(Constants::next_descriptor_eof))
//             {
//                 n->erase_attr(Constants::next_descriptor_eof);
//             }
//         }
//         // make inputs and outputs of merged_node the same as node_bak
//         std::map<TensorPtr, TensorPtr> input_map;
//         std::map<TensorPtr, TensorPtr> output_map;
//         for (auto t_raw : node_bak->inputs())
//         {
//             for (auto t : merged_node->inputs())
//             {
//                 if (t->name() == t_raw->name())
//                 {
//                     input_map[t_raw] = t;
//                     break;
//                 }
//             }
//         }
//         for (auto t_raw : node_bak->outputs())
//         {
//             for (auto t : merged_node->outputs())
//             {
//                 if (t->name() == t_raw->name())
//                 {
//                     output_map[t_raw] = t;
//                     break;
//                 }
//             }
//         }
//         for (auto it = input_map.begin(); it != input_map.end(); ++it)
//         {
//             node_bak->replace_input_by(it->first, it->second);
//         }
//         for (auto it = output_map.begin(); it != output_map.end(); ++it)
//         {
//             int idx = node_bak->remove_output(it->first);
//             while (idx != INT32_MAX)
//             {
//                 node_bak->add_output(it->second, idx);
//                 idx = node_bak->remove_output(it->first);
//             }
//         }
//         g->add_node(node_bak);
//     }
//     merged_node->clear_inputs();
//     merged_node->clear_outputs();
//     g->remove_node(merged_node, false);
//     return true;
// }

std::vector<NodePtr> split_sg(GraphPtr g, NodePtr sg_node, size_t split_num, int split_strategy = 0)
{
    auto sgn = SubGraphNode::from_node(sg_node);
    auto sg = sgn->subgraph();

    // sg->to_svg("before_split.svg");

    auto sg_input_tensors = collect_input_tensors(sg);
    auto sg_output_tensors = collect_output_tensors(sg, sg_node);

    std::vector<TensorPtr> sg_inputs;
    for (auto i : sg_input_tensors)
    {
        sg_inputs.push_back(i);
    }
    std::vector<TensorPtr> sg_outputs;
    for (auto o : sg_output_tensors)
    {
        sg_outputs.push_back(o);
    }

    sg->record_input_tensors(sg_inputs);
    sg->record_output_tensors(sg_outputs);

    auto nodes = sg->nodes();
    for (auto n : nodes)
    {
        if (n->type() == OpType_Loop || n->type() == OpType_If)
        {
            continue;
        }
        auto sgn_tmp = SubGraphNode::from_node(n);
        if (sgn_tmp)
        {
            sgn_tmp->unfold();
        }
    }

    adjust_internal_layout(sg);
    bool is_changed = false;
    if (split_strategy == 0)
    {
        // split filters
        is_changed = split_filters(sg, split_num);
    }
    else if (split_strategy == 1)
    {
        // split act
        // split_act(sg, split_num);
        tile_graph_by_size(sg, 1, split_num, -1);
    }
    else if (split_strategy == 2)
    {
        // split batch
        // cut_batches(sg);
        ERROR() << "split strategy 2 is not available now";
        is_changed = true;
    }
    else
    {
        WARN() << "no such split strategy " << split_strategy;
    }
    if (is_changed)
    {
        passes_before_apply_plugin(sg, split_strategy);
    }

    infer_symbol_shape(sg);
    apply_plugins(sg);

    passes_after_apply_plugin(sg);

    apply_internal_plugins(sg);

    // sg->to_svg("after_split.svg");
    auto sg_new_input_tensors = collect_input_tensors(sg);
    auto sg_new_output_tensors = collect_output_tensors(sg, sg_node);

    if (sg_input_tensors != sg_new_input_tensors)
    {
        ERROR() << "check input tensor failed for load balance spliting";
    }
    if (sg_output_tensors != sg_new_output_tensors)
    {
        ERROR() << "check output tensor failed for load balance spliting";
    }

    PerfEstimator perf;
    perf(sg);

    std::vector<NodePtr> unfolded_nodes;
    unfolded_nodes = sgn->unfold();

    return unfolded_nodes;
}

/*subgraph merge its child nodes*/
bool sg_merge_child(NodePtr sg, std::vector<NodePtr> &merged_node, const std::map<NodePtr, int> &node_to_level, const std::vector<int> &skip_levels)
{
    if (sg->children().size() == 1)
    {
        auto child = sg->children()[0];
        size_t merged_parent_count = 0;
        for (auto parent : child->parents())
        {
            if (std::count(merged_node.begin(), merged_node.end(), parent))
            {
                merged_parent_count++;
            }
        }
        if (merged_parent_count == child->parents().size()) // all its parent nodes are in merged_node, then it can be merged
        {
            auto child_level = node_to_level.at(child);
            if (!contains(skip_levels, child_level))
            {
                merged_node.push_back(child);
                sg_merge_child(child, merged_node, node_to_level, skip_levels);
            }
        }
    }

    return true;
}

// static bool internal_constant_merge(GraphPtr g, std::vector<std::vector<NodePtr>> &levels)
// {
//     std::vector<NodePtr> const_nodes;
//     for (size_t i = 0; i < levels.size(); ++i)
//     {
//         auto &level = levels[i];
//         for (size_t j = 0; j < level.size(); ++j)
//         {
//             auto node = level[j];
//             if (is_internal_const(node))
//             {
//                 if (node->children().size() > 1)
//                 {
//                     WARN() << "constant node has more than 1 child";
//                 }
//                 const_nodes.push_back(node);
//                 level.erase(level.begin() + j);
//                 j--;
//             }

//             for (auto n : node->parents())
//             {
//                 auto it = std::find(const_nodes.begin(), const_nodes.end(), n);
//                 if (it != const_nodes.end())
//                 {
//                     const_nodes.erase(it);
//                     level.insert(level.begin(), n);
//                     j++;
//                 }
//             }
//         }
//     }

//     if (const_nodes.size())
//     {
//         ERROR();
//     }
//     return true;
// }

/*if a node has only 1 child and the child has only 1 parent then they can be merged*/
static bool vertical_merge(GraphPtr g, std::vector<std::vector<NodePtr>> &levels, const std::vector<int> &skip_levels)
{
    std::vector<NodePtr> total_merged_node, merged_node;
    // for (auto &level : levels)
    std::map<NodePtr, int> node_to_level;
    for (size_t j = 0; j < levels.size(); ++j)
    {
        auto &level = levels[j];
        for (size_t i = 0; i < level.size(); ++i)
        {
            auto node = level[i];
            node_to_level[node] = j;
        }
    }
    for (size_t j = 0; j < levels.size(); ++j)
    {
        auto &level = levels[j];
        if (contains(skip_levels, j))
        {
            continue;
        }
        for (size_t i = 0; i < level.size(); ++i)
        {
            if (level[i]->parents().size() == 0)
            {
                continue;
            }
            if (std::count(total_merged_node.begin(), total_merged_node.end(), level[i]))
            {
                level.erase(level.begin() + i);
                --i;
            }
            else
            {
                auto sg = level[i];
                if (sg->inputs().size() == 0)
                {
                    continue;
                }
                merged_node.clear();
                merged_node.push_back(sg);
                sg_merge_child(sg, merged_node, node_to_level, skip_levels);
                if (merged_node.size() > 1) // has more than 1 node to be merged
                {
                    auto sg_merged_node = merge_sg(g, merged_node);
                    level[i] = sg_merged_node;
                    node_to_level[sg_merged_node] = j;
                    total_merged_node.insert(total_merged_node.end(), merged_node.begin(), merged_node.end());
                }
            }
        }
    }
    return true;
}

std::vector<std::vector<NodePtr>> nodes_to_levels(GraphPtr g, const std::vector<NodePtr> &inited_nodes)
{
    std::vector<std::vector<NodePtr>> sub_levels;

    std::vector<NodePtr> input_nodes;
    bool is_input_node;
    for (auto n : inited_nodes)
    {
        is_input_node = false;
        if (n->parents().size() == 0)
        {
            input_nodes.push_back(n);
            continue;
        }
        for (auto p : n->parents())
        {
            if (!contains(inited_nodes, p))
            {
                is_input_node = true;
                break;
            }
        }
        if (is_input_node)
        {
            input_nodes.push_back(n);
        }
    }

    std::vector<NodePtr> inited_nodes_with_parents(inited_nodes.begin(), inited_nodes.end());
    std::vector<NodePtr> new_input_nodes;
    for (auto n : input_nodes)
    {
        if (n->parents().size())
        {
            for (auto p : n->parents())
            {
                if (!contains(inited_nodes_with_parents, p))
                {
                    new_input_nodes.push_back(p);
                    inited_nodes_with_parents.push_back(p);
                }
            }
        }
        else
        {
            new_input_nodes.push_back(n);
        }
    }

    if (!setup_level(new_input_nodes, inited_nodes_with_parents, sub_levels))
    {
        ERROR() << "graph is invalid";
    }

    for (size_t i = 0; i < sub_levels.size(); ++i)
    {
        auto &level = sub_levels[i];
        for (size_t j = 0; j < level.size(); ++j)
        {
            auto node = level[j];
            if (!contains(inited_nodes, node))
            {
                level.erase(sub_levels[i].begin() + j);
                j--;
            }
        }
        if (!sub_levels[i].size())
        {
            sub_levels.erase(sub_levels.begin() + i);
            i--;
        }
    }

    return sub_levels;
}

static bool recover_bak_nodes(GraphPtr g, NodePtr node_bak, std::vector<NodePtr> &nodes)
{
    auto merged_node = merge_sg(g, nodes);

    auto sgn_bak = SubGraphNode::from_node(node_bak);
    auto sg_bak = sgn_bak->subgraph();
    // update matchlist
    for (auto n : sg_bak->nodes())
    {
        if (n->attrs().count("_match_list"))
        {
            auto sgn_n = SubGraphNode::from_node(n);
            auto sg_n = sgn_n->subgraph();
            std::vector<NodePtr> m;
            std::vector<NodePtr> m_ori = n->get_attr("_match_list");
            bool found_node = false;
            for (auto mn : m_ori)
            {
                found_node = false;
                for (auto nn : sg_n->nodes())
                {
                    if (mn->name() == nn->name())
                    {
                        found_node = true;
                        m.push_back(nn);
                        break;
                    }
                }
                if (!found_node)
                {
                    ERROR() << "cannot found " << mn->name();
                }
            }
            n->set_attr("_match_list", m);
        }
    }
    // remove desc chain
    for (auto n : sg_bak->nodes())
    {
        if (n->attrs().count(Constants::next_descriptor_length))
        {
            n->erase_attr(Constants::next_descriptor_length);
        }
        if (n->attrs().count(Constants::next_descriptor_offset))
        {
            n->erase_attr(Constants::next_descriptor_offset);
        }
        if (n->attrs().count(Constants::next_descriptor_eof))
        {
            n->erase_attr(Constants::next_descriptor_eof);
        }
    }
    // make inputs and outputs of merged_node the same as node_bak
    // make inputs and outputs of merged_node the same as node_bak
    std::vector<TensorPtr> inputs, outputs;
    for (auto t_raw : node_bak->inputs())
    {
        for (auto t : merged_node->inputs())
        {
            if (t->name() == t_raw->name())
            {
                inputs.push_back(t);
                break;
            }
        }
    }
    for (auto t_raw : node_bak->outputs())
    {
        for (auto t : merged_node->outputs())
        {
            if (t->name() == t_raw->name())
            {
                outputs.push_back(t);
                break;
            }
        }
    }

    merged_node->clear_inputs();
    merged_node->clear_outputs();
    for (auto t : inputs)
    {
        merged_node->add_input(t);
    }
    for (auto t : outputs)
    {
        merged_node->add_output(t);
    }
    g->replace_node_by(merged_node, node_bak);
    return true;
}

/* if loads < core num, split loads until loads = core num */
uint32_t split_load(GraphPtr g, size_t core_num, int32_t idx, uint32_t split_load, uint32_t strategy, bool gm_enhance, std::vector<std::vector<NodePtr>> &levels, std::vector<int> &top_skip_levels)
{
    uint32_t level_num = 0;
    if (strategy == 0)
    {
        // do nothing
        level_num = 0;
    }
    else
    {
        if (levels[idx].size() == 1)
        {
            DEBUG() << "spliting idx " << idx;
            auto sg_node = remove_most_load(levels[idx]);
            int64_t ori_cycle = get_load_from_model(sg_node);
            if (ori_cycle)
            {
                // std::vector<NodePtr> nodes_bak;
                // auto sub_levels = split_sg(g, SubGraphNode::from_node(sg), core_num, gm_enhance, skip_levels, split_load);
                auto sgn = SubGraphNode::from_node(sg_node);
                auto sg = sgn->subgraph();
                if (sgn)
                {
                    sgn->flat();
                }
                auto node_bak = sg_node->clone();
                // nodes_bak.push_back(node_bak);
                auto nodes = split_sg(g, sg_node, core_num);
                auto inited_nodes = init_nodes(g, nodes);
                auto new_levels = nodes_to_levels(g, inited_nodes);
                // std::vector<int> skip_levels;
                // reduce_level(g, new_levels, core_num, skip_levels);
                auto new_cycle = performance_eval(new_levels, core_num, 0);
                if (new_cycle > ori_cycle)
                {
                    DEBUG() << "using original node(" << ori_cycle << ") to replace the optimized node(" << new_cycle << ")";
                    std::vector<NodePtr> all_nodes;
                    for (auto level : new_levels)
                    {
                        for (auto n : level)
                        {
                            all_nodes.push_back(n);
                        }
                    }
                    recover_bak_nodes(g, node_bak, all_nodes);
                    levels[idx].push_back(node_bak);
                }
                else
                {
                    levels.erase(levels.begin() + idx);
                    for (size_t i = 0; i < new_levels.size(); ++i)
                    {
                        levels.insert(levels.begin() + idx + i, new_levels[i]);
                    }
                    level_num = new_levels.size() - 1;
                }
            }
            else
            {
                levels[idx].push_back(sg_node);
            }
        }
        // else // split the larger load
        // {
        //     auto sg = remove_most_load(m_levels[idx]);
        //     auto split_num = core_num - m_levels[idx].size();
        //     std::vector<NodePtr> new_level;
        //     new_level.push_back(sg);
        //     m_levels.insert(m_levels.begin() + idx, new_level);
        //     level_num = split_load(g, split_num, idx + 1, strategy);
        // }
    }
    // DEBUG() << "level num: " << level_num;
    return level_num;
}

/*subgroup num in the same level must be < core num*/
static bool load_balance_merge(GraphPtr g, std::vector<std::vector<NodePtr>> &levels, size_t core_num, uint32_t merge_strategy)
{
    auto sgs = g->nodes();
    std::vector<NodePtr> load_compare_node;
    if (merge_strategy == 2)
    {
        for (size_t i = 0; i < levels.size(); ++i)
        {
            auto level = levels[i];
            auto sg_merged_node = merge_sg(g, level);
        }
        std::vector<int> skip_levels;
        reduce_level(g, levels, 1, skip_levels);
    }
    else
    {
        for (size_t i = 0; i < levels.size(); ++i)
        {
            if (levels[i].size() > core_num)
            {
                merge_load(g, core_num, i, levels, merge_strategy);
            }
        }
    }
    return true;
}
static bool load_balance_split(GraphPtr g, std::vector<std::vector<NodePtr>> &levels, size_t core_num, uint32_t split_load_strategy, uint32_t split_strategy, bool gm_enhance, std::vector<int> &top_skip_levels)
{
    auto sgs = g->nodes();
    std::vector<NodePtr> load_compare_node;
    for (size_t i = 0; i < levels.size(); ++i)
    {
        if (levels[i].size() < core_num)
        {
            uint32_t extra_level = split_load(g, core_num, i, split_load_strategy, split_strategy, gm_enhance, levels, top_skip_levels);
            i += extra_level;
        }
    }
    return true;
}

// backtrack algo
static void backtrack_algo(const std::vector<int64_t> &weight, std::vector<int> &group, int64_t target, int index, std::vector<int64_t> &group_w, int64_t &max_v, std::vector<int> &result, int &count)
{
    if (index == (int)group.size())
    {
        // DEBUG() << "try " << count << " index " << index << " " << utils::join_string(" ", group);
        count++;
        std::vector<int64_t> group_v(group_w.size(), 0);
        for (size_t i = 0; i < group.size(); ++i)
        {
            group_v[group[i]] += weight[i];
        }

        int64_t v = max_v + 1;
        auto max_iter = std::max_element(group_v.begin(), group_v.end());
        if (max_iter != group_v.end())
        {
            v = *max_iter;
        }
        else
        {
            WARN() << "the vector is empty";
        }
        if (v <= max_v)
        {
            max_v = v;
            result = group;
        }
        return;
    }
    for (int i = index; i < (int)weight.size(); i++)
    {
        for (int g = 0; g < (int)group_w.size(); g++)
        {
            if (group[i] == -1 && group_w[g] + weight[i] <= target)
            {
                group[i] = g;
                index++;
                group_w[g] += weight[i];
                backtrack_algo(weight, group, target, index, group_w, max_v, result, count);
                group[i] = -1;
                index--;
                group_w[g] -= weight[i];
            }
        }
        if (group[i] == -1) // cannot push to any group
        {
            return;
        }
    }
}

static std::vector<std::vector<int>> get_balanced_backtrack(const std::vector<int64_t> &group, size_t group_num)
{
    std::vector<std::vector<int>> result(group_num);
    if (group.size() == 0)
    {
        return result;
    }
    int64_t total_v = 0;
    for (auto v : group)
    {
        total_v += v;
    }
    int64_t max_v = total_v;

    std::vector<int64_t> order;
    std::vector<int64_t> weight = group;
    for (int i = 0; i < (int)weight.size(); i++)
    {
        order.push_back(i);
    }
    for (int i = 0; i < (int)weight.size(); i++)
    {
        for (int j = 0; j < (int)weight.size() - i - 1; j++)
        {
            if (weight[j] < weight[j + 1])
            {
                std::swap(weight[j], weight[j + 1]);
                std::swap(order[j], order[j + 1]);
            }
        }
    }

    std::vector<int64_t> group_v(group_num, 0);
    for (auto v : weight)
    {
        auto it = std::min_element(group_v.begin(), group_v.end());
        group_v[it - group_v.begin()] += v;
    }
    int64_t target_v = 0;
    auto max_iter = std::max_element(group_v.begin(), group_v.end());
    if (max_iter != group_v.end())
    {
        target_v = *max_iter;
    }
    else
    {
        WARN() << "the vector is empty";
    }

    DEBUG() << "target v " << target_v;
    std::vector<int> status(weight.size(), -1);
    std::vector<int> target_group(weight.size(), -1);
    std::vector<int64_t> group_w(group_num, 0);
    int count = 0;
    status[0] = 0;
    group_w[0] += weight[0];
    backtrack_algo(weight, status, target_v + 1, 1, group_w, max_v, target_group, count);
    // DEBUG() << "max_v: " << max_v << " count: " << count << " size: " << weight.size();

    for (int i = 0; i < (int)target_group.size(); ++i)
    {
        result[target_group[i]].push_back(order[i]);
    }
    return result;
}

// greedy algo
static std::vector<std::vector<int>> get_balanced_greedy(const std::vector<int64_t> &group, size_t group_num)
{
    auto get_least_value = [group](std::vector<std::vector<int>> &balanced_groups) -> int {
        int index = 0;
        int64_t total_value = 0;
        int64_t last_total_value = 0;
        for (size_t i = 0; i < balanced_groups.size(); ++i)
        {
            total_value = 0;
            for (auto v : balanced_groups[i])
            {
                total_value += group[v];
            }
            if (total_value < last_total_value || i == 0)
            {
                last_total_value = total_value;
                index = i;
            }
        }

        return index;
    };

    std::vector<std::vector<int>> balanced_groups;
    for (size_t i = 0; i < group_num; ++i)
    {
        std::vector<int> balanced_group;
        balanced_groups.push_back(balanced_group);
    }

    std::map<int, int64_t> index_value;
    for (size_t i = 0; i < group.size(); ++i)
    {
        index_value.insert(std::make_pair(i, group[i]));
    }

    for (auto it = index_value.begin(); it != index_value.end(); ++it)
    {
        auto index = get_least_value(balanced_groups);
        balanced_groups[index].push_back(it->first);
    }
    return balanced_groups;
}

static std::vector<std::vector<int>> balance_algo(const std::vector<int64_t> &group, size_t group_num)
{
    std::vector<std::vector<int>> ret;
    // if (contains(enable_passes, "tnode_to_sg"))
    if (group.size() < 20)
    {
        ret = get_balanced_backtrack(group, group_num);
    }
    else
    {
        ret = get_balanced_greedy(group, group_num);
    }
    return ret;
}

static int64_t get_perf(const std::vector<NodePtr> &nodes)
{
    int64_t cycle = 0;
    if (nodes.size())
    {
        cycle = get_load_from_model(nodes[0]);
        for (size_t j = 1; j < nodes.size(); ++j)
        {
            cycle = std::max(cycle, get_load_from_model(nodes[j]));
        }
    }
    else
    {
        cycle = 0;
    }
    return cycle;
}

static int64_t balance_in_level(GraphPtr g, std::vector<std::vector<NodePtr>> &levels, size_t core_num, const std::vector<NodePtr> &next_ref_level, const std::vector<NodePtr> &next_next_ref_level)
{
    std::vector<NodePtr> next_level;
    std::vector<NodePtr> next_next_level;

    auto closer_opt_cond = [&next_level, &next_next_level](const std::vector<std::vector<NodePtr>> &node_groups) -> bool {
        if (node_groups.size() == 1)
        {
            return false;
        }

        for (auto group : node_groups)
        {
            for (auto n : group)
            {
                bool has_child = false;
                std::vector<NodePtr> real_children;
                find_not_internal_children(n, real_children);
                for (auto c : real_children)
                {
                    if (contains(next_level, c) || contains(next_next_level, c))
                    {
                        if (has_child)
                        {
                            return false;
                        }
                        has_child = true;
                    }
                }
            }
        }
        return true;
    };

    auto distance = [](NodePtr node_a, NodePtr node_b, std::set<NodePtr> &valid_children) -> int {
        NodePtr child_a0;
        NodePtr child_a1;
        NodePtr child_a2;
        for (auto c : node_a->children())
        {
            if (contains(valid_children, c))
            {
                child_a0 = c;
                break;
            }
        }

        if (child_a0)
        {
            for (auto c : child_a0->children())
            {
                if (contains(valid_children, c))
                {
                    child_a1 = c;
                    break;
                }
            }
        }

        if (child_a1)
        {
            for (auto c : child_a1->children())
            {
                if (contains(valid_children, c))
                {
                    child_a2 = c;
                    break;
                }
            }
        }

        NodePtr child_b0;
        NodePtr child_b1;
        NodePtr child_b2;
        for (auto c : node_b->children())
        {
            if (contains(valid_children, c))
            {
                child_b0 = c;
                break;
            }
        }

        if (child_b0)
        {
            for (auto c : child_b0->children())
            {
                if (contains(valid_children, c))
                {
                    child_b1 = c;
                    break;
                }
            }
        }

        if (child_b1)
        {
            for (auto c : child_b1->children())
            {
                if (contains(valid_children, c))
                {
                    child_b2 = c;
                    break;
                }
            }
        }

        if (child_a0 && child_b0)
        {
            if (child_a0 == child_b0)
            {
                return 2;
            }
            if (child_a1 && child_a1 == child_b0)
            {
                return 3;
            }
            if (child_b1 && child_b1 == child_a0)
            {
                return 3;
            }
            if (child_a1 && child_b1 && child_a1 == child_b1)
            {
                return 4;
            }
            if (child_a2 && child_a2 == child_b0)
            {
                return 4;
            }
            if (child_b2 && child_b2 == child_a0)
            {
                return 4;
            }
            if (child_a2 && child_b1 && child_a2 == child_b1)
            {
                return 5;
            }
            if (child_b2 && child_a1 && child_b2 == child_a1)
            {
                return 5;
            }
            if (child_a2 && child_b2 && child_a2 == child_b2)
            {
                return 6;
            }
        }
        return 100;
    };

    auto similar_sg = [](NodePtr a, NodePtr b) -> bool {
        auto sgn_a = SubGraphNode::from_node(a);
        if (sgn_a)
        {
            auto sg_a = sgn_a->subgraph();
            auto sgn_b = SubGraphNode::from_node(b);
            if (sgn_b)
            {
                auto sg_b = sgn_b->subgraph();
                if (sg_a->nodes().size() != sg_b->nodes().size())
                {
                    return false;
                }
                for (size_t i = 0; i < sg_a->nodes().size(); ++i)
                {
                    auto node_a = sg_a->nodes()[i];
                    auto node_b = sg_b->nodes()[i];
                    if (node_a->type() == OpType_SubGraphInput)
                    {
                        continue;
                    }
                    if (!node_a->attrs().count(Constants::plugin_attrs_name) || !node_b->attrs().count(Constants::plugin_attrs_name))
                    {
                        return false;
                    }
                    BuilderOpPluginInfo *plugin_a = node_a->get_attr(Constants::plugin_attrs_name);
                    BuilderOpPluginInfo *plugin_b = node_b->get_attr(Constants::plugin_attrs_name);
                    if (plugin_a->name != plugin_b->name)
                    {
                        return false;
                    }
                }
                // INFO() << a << " and " << b << " is similar";
                return true;
            }
        }
        return false;
    };

    // auto get_total_cycle = [](std::vector<NodePtr> & group) -> int{
    //     int total = 0;
    //     for (auto n : group)
    //     {
    //         // INFO() << n << " " << get_load_from_model(n);
    //         total += get_load_from_model(n);
    //     }
    //     return total;
    // };

    auto find_closer_node = [distance, similar_sg](NodePtr node_a, NodePtr node, std::vector<std::vector<NodePtr>> &groups, std::set<NodePtr> &valid_children) -> NodePtr {
        // find a node in group_b that
        // load is similar to node
        // is closer to all nodes in group_a
        // float min_diff = 0.3;
        int a = distance(node_a, node, valid_children);
        std::vector<NodePtr> candidates;
        std::vector<NodePtr> similar_candidates;
        NodePtr ret;
        for (auto group_b : groups)
        {
            for (size_t i = 0; i < group_b.size(); ++i)
            {
                auto n = group_b[i];
                int b = distance(node_a, n, valid_children);
                if (b < a)
                {
                    if (similar_sg(node, n))
                    {
                        similar_candidates.push_back(n);
                    }
                    candidates.push_back(n);
                }
            }
        }

        if (similar_candidates.size())
        {
            float min_diff = 1.0;
            for (auto n : similar_candidates)
            {
                float diff = (float)std::abs(get_load_from_model(node) - get_load_from_model(n)) / (float)get_load_from_model(node);
                if (diff < min_diff)
                {
                    min_diff = diff;
                    ret = n;
                }
            }
        }
        else if (candidates.size())
        {
            float min_diff = 1.0;
            for (auto n : candidates)
            {
                float diff = (float)std::abs(get_load_from_model(node) - get_load_from_model(n)) / (float)get_load_from_model(node);
                if (diff < min_diff)
                {
                    min_diff = diff;
                    ret = n;
                }
            }
        }

        return ret;
    };

    int64_t total_perf = 0;
    next_level = next_ref_level;
    next_next_level = next_next_ref_level;
    for (int i = levels.size() - 1; i >= 0; --i)
    {
        auto &level = levels[i];
        uint32_t count_node = 0;
        for (auto n : level)
        {
            if (get_load_from_model(n) > 0)
            {
                count_node++;
            }
        }

        if (count_node <= core_num)
        {
            return get_perf(level);
        }

        std::vector<NodePtr> zero_sgs;
        for (int i = 0; i < (int)level.size(); ++i)
        {
            auto node = level[i];
            if (get_load_from_model(node) == 0)
            {
                level.erase(level.begin() + i);
                zero_sgs.push_back(node);
                i--;
            }
        }

        std::vector<int64_t> value_groups;
        int64_t perf = 0;
        for (auto n : level)
        {
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                sgn->flat();
            }
            value_groups.push_back(get_load_from_model(n));
        }

        auto balanced_groups = balance_algo(value_groups, core_num);

        for (size_t i = 0; i < balanced_groups.size(); ++i)
        {
            auto &group = balanced_groups[i];
            if (group.size() == 0)
            {
                balanced_groups.erase(balanced_groups.begin() + i);
                i--;
            }
        }

        std::vector<std::vector<NodePtr>> node_groups(balanced_groups.size());
        for (size_t j = 0; j < balanced_groups.size(); ++j)
        {
            auto group = balanced_groups[j];
            if (group.size() == 0)
            {
                continue;
            }
            auto &node_vec = node_groups[j];
            for (size_t k = 0; k < group.size(); ++k)
            {
                auto idx = group[k];
                node_vec.push_back(level[idx]);
            }
        }

        // group closer node together

        if (BuilderConfig::get<bool>(Constants::closer_sg_opt) && closer_opt_cond(node_groups))
        {
            // INFO();
            // for(auto node : next_ref_level)
            // {
            //     INFO() << "next level " << node;
            // }
            // for(auto node : next_next_ref_level)
            // {
            //     INFO() << "next next level " << node;
            // }
            // INFO() << "find closer node for nodes";
            // for (auto group : node_groups)
            // {
            //     for (auto n : group)
            //     {
            //         INFO() << n;
            //         for(auto c : n->children())
            //         {
            //             INFO() << "child " << c;
            //         }
            //     }
            // }

            std::map<NodePtr, int> idx_map;
            std::map<NodePtr, int> grp_map;
            for (size_t g = 0; g < node_groups.size(); ++g)
            {
                auto group = node_groups[g];
                for (size_t j = 0; j < group.size(); ++j)
                {
                    grp_map[group[j]] = g;
                    idx_map[group[j]] = j;
                }
            }

            std::set<NodePtr> valid_children(zero_sgs.begin(), zero_sgs.end());
            valid_children.insert(next_level.begin(), next_level.end());
            valid_children.insert(next_next_level.begin(), next_next_level.end());

            for (size_t g = 0; g < node_groups.size() - 1; ++g)
            {
                auto &group_0 = node_groups[g];
                // INFO() << "working on group " << g << " group_0[0] " << group_0[0];
                for (size_t j = 1; j < group_0.size(); ++j)
                {
                    // INFO() << "checking " << group_0[j];
                    std::vector<std::vector<NodePtr>> groups;
                    if (j < group_0.size() - 1)
                    {
                        std::vector<NodePtr> group;
                        for (size_t l = j + 1; l < group_0.size(); ++l)
                        {
                            group.push_back(group_0[l]);
                        }
                        if (group.size())
                        {
                            groups.push_back(group);
                        }
                    }

                    for (size_t k = g + 1; k < node_groups.size(); ++k)
                    {
                        groups.push_back(node_groups[k]);
                    }

                    if (groups.size())
                    {

                        auto node = find_closer_node(group_0[0], group_0[j], groups, valid_children);
                        if (node)
                        {
                            auto grp_idx = grp_map[node];
                            auto &group = node_groups[grp_idx];
                            auto idx = idx_map[node];

                            auto tmp1 = group_0[j];
                            auto tmp2 = group[idx];
                            group_0.erase(group_0.begin() + j);
                            group_0.insert(group_0.begin() + j, tmp2);
                            group.erase(group.begin() + idx);
                            group.insert(group.begin() + idx, tmp1);
                            grp_map[tmp1] = grp_idx;
                            idx_map[tmp1] = idx;
                            grp_map[tmp2] = g;
                            idx_map[tmp2] = j;
                            // INFO() << "switch " << g << " " << tmp1 << " and " << grp_idx << " " << tmp2;
                        }
                    }
                }
            }
            // INFO() << "find closer node for nodes done";
            // for (auto group : node_groups)
            // {
            //     INFO() << "new group!";
            //     for (auto n : group)
            //     {
            //         INFO() << n;
            //         for(auto c : n->children())
            //         {
            //             INFO() << "child " << c->name();
            //         }
            //     }
            // }
        }

        std::vector<NodePtr> new_level;
        for (size_t j = 0; j < node_groups.size(); ++j)
        {
            auto &node_vec = node_groups[j];
            auto merged_node = merge_sg(g, node_vec);
            new_level.push_back(merged_node);
        }

        level = new_level;
        perf = get_perf(level);
        total_perf += perf;
        if (perf != 0)
        {
            next_next_level = next_level;
            next_level = level;
        }

        for (auto sg : zero_sgs)
        {
            level.push_back(sg);
        }
    }

    return total_perf;
}

std::vector<std::vector<NodePtr>> no_split(GraphPtr g, const std::vector<NodePtr> &sub_level, size_t core_num)
{
    return {sub_level};
}

std::vector<std::vector<NodePtr>> lb_split(GraphPtr g, const std::vector<NodePtr> &sub_level, size_t core_num, int split_way, int split_node_sel)
{
    auto get_biggest_index = [](const std::vector<NodePtr> &nodes) -> int {
        int64_t last_biggest_perf = 0;
        int last_biggest_index = 0;
        for (size_t i = 0; i < nodes.size(); ++i)
        {
            int64_t perf = get_load_from_model(nodes[i]);
            if (perf > last_biggest_perf)
            {
                last_biggest_perf = perf;
                last_biggest_index = i;
            }
        }
        return last_biggest_index;
    };

    auto level = sub_level;

    NodePtr split_node;
    std::vector<NodePtr> res_nodes;
    size_t cut_num = core_num;
    int index = 0;
    if (split_node_sel == 0)
    {
        index = get_biggest_index(level);
        auto node = level[index];
        auto node_perf = get_load_from_model(node);

        int64_t res_perf = 0;
        for (auto n : level)
        {
            if (n != node)
            {
                res_nodes.push_back(n);
                res_perf += get_load_from_model(n);
            }
        }
        // cut_num = (res_perf >= node_perf * 0.4) ? 2 : 3;
        cut_num = core_num;
        for (int i = core_num; i > 0; --i)
        {
            if (0.8 * node_perf / i < res_perf)
            {
                cut_num = i;
            }
        }
        if (cut_num == 1)
        {
            return {level};
        }

        if (res_perf >= node_perf * 0.8)
        {
            return {level};
        }
        auto sgn = SubGraphNode::from_node(node);
        auto sg = sgn->subgraph();
        if (sgn)
        {
            sgn->flat();
        }

        split_node = node;
    }
    else
    {
        cut_num = core_num;
        auto merged_node = merge_sg(g, level);
        split_node = merged_node;
    }

    auto nodes = split_sg(g, split_node, cut_num, split_way);
    auto inited_nodes = init_nodes(g, nodes);
    // std::vector<NodePtr> all_nodes = inited_nodes;
    for (auto n : res_nodes)
    {
        inited_nodes.push_back(n);
    }
    auto new_levels = nodes_to_levels(g, inited_nodes);
    reduce_level(g, new_levels, core_num, std::vector<int>());

    return new_levels;
}

static bool check_level(const std::vector<NodePtr> &nodes, const std::vector<std::vector<NodePtr>> &levels)
{
    bool is_in_level = false;
    bool ret = true;
    for (auto node : nodes)
    {
        is_in_level = false;
        for (auto level : levels)
        {
            if (std::count(level.begin(), level.end(), node))
            {
                is_in_level = true;
                break;
            }
        }
        if (!is_in_level)
        {
            WARN() << "cannot find " << node->name() << " in levels";
            ret = false;
        }
    }
    return ret;
}

std::vector<std::vector<NodePtr>> max_split_c(GraphPtr g, const std::vector<NodePtr> &sub_level, size_t core_num)
{
    auto ret = lb_split(g, sub_level, core_num, 0, 0);
    return ret;
}

std::vector<std::vector<NodePtr>> max_split_h(GraphPtr g, const std::vector<NodePtr> &sub_level, size_t core_num)
{
    auto ret = lb_split(g, sub_level, core_num, 1, 0);
    return ret;
}

std::vector<std::vector<NodePtr>> avg_split_c(GraphPtr g, const std::vector<NodePtr> &sub_level, size_t core_num)
{
    auto ret = lb_split(g, sub_level, core_num, 0, 1);
    return ret;
}

std::vector<std::vector<NodePtr>> avg_split_h(GraphPtr g, const std::vector<NodePtr> &sub_level, size_t core_num)
{
    return {sub_level};
}

std::vector<NodePtr> copy_level(GraphPtr g, const std::vector<NodePtr> &level)
{
    std::vector<NodePtr> res;
    for (auto n : level)
    {
        auto node = n->clone();
        g->add_node(node);
        res.push_back(node);
    }
    return res;
}

static bool apply_load_balance_algo(GraphPtr g, std::vector<std::vector<NodePtr>> &levels, size_t core_num, uint32_t strategy)
{
    typedef std::vector<std::vector<NodePtr>> (*func_ptr)(GraphPtr g, const std::vector<NodePtr> &sub_levels, size_t core_num);

    for (int i = levels.size() - 1; i >= 0; --i)
    {
        if (get_perf(levels[i]) == 0)
        {
            continue;
        }

        // auto last_best_perf = performance_eval(sub_levels, core_num, true);
        int64_t last_best_perf = -1;

        std::vector<func_ptr> funcs;

        if (strategy == 0)
        {
            funcs = {no_split};
        }
        else if (strategy == 1)
        {
            funcs = {no_split, max_split_c};
        }
        else if (strategy == 2)
        {
            funcs = {no_split, max_split_c, max_split_h};
        }
        else if (strategy == 3)
        {
            funcs = {no_split, max_split_c, max_split_h, avg_split_c, avg_split_h};
        }
        else
        {
            WARN() << "doesn't support the strategy!";
        }

        int idx = -1;

        if (strategy == 0)
        {
            idx = 0;
        }
        else
        {
            for (size_t j = 0; j < funcs.size(); ++j)
            {
                DEBUG() << "level " << i << " try func" << j;

                GraphPtr g_copy = Graph::create();
                auto level_copy = copy_level(g_copy, levels[i]);
                auto opt_levels = funcs[j](g_copy, level_copy, core_num);

                auto perf = balance_in_level(g_copy, opt_levels, core_num, std::vector<NodePtr>(), std::vector<NodePtr>());

                DEBUG() << "level " << i << " try func" << j << " perf: " << perf << " last best perf: " << last_best_perf;
                if ((perf < last_best_perf && perf > 0) || (last_best_perf <= 0))
                {
                    last_best_perf = perf;
                    idx = j;
                }
                // levels.erase(levels.begin() + i);
                // levels.insert(levels.begin() + i, opt_levels[0]);
            }
        }

        if (idx >= 0)
        {
            // DEBUG() << "level " << i << " apply func" << idx;
            auto opt_levels = funcs[idx](g, levels[i], core_num);
            std::vector<NodePtr> next_ref_level;
            std::vector<NodePtr> next_next_ref_level;
            bool get_next_level = false;
            for (size_t k = i + 1; k < levels.size(); ++k)
            {
                if (get_perf(levels[k]) != 0)
                {
                    if (!get_next_level)
                    {
                        next_ref_level = levels[k];
                        get_next_level = true;
                    }
                    else
                    {
                        next_next_ref_level = levels[k];
                        break;
                    }
                }
            }
            balance_in_level(g, opt_levels, core_num, next_ref_level, next_next_ref_level);
            levels.erase(levels.begin() + i);
            for (size_t j = 0; j < opt_levels.size(); ++j)
            {
                levels.insert(levels.begin() + i + j, opt_levels[j]);
            }
        }
        else
        {
            DEBUG() << "level " << i << " no change is best change";
        }
    }
    return true;
}

/*remove extra level*/
// static bool remove_extra_level(GraphPtr g, std::vector<std::vector<NodePtr>> &levels, size_t core_num)
// {
//     size_t total_node_num;
//     for (int32_t i = levels.size() - 2; i >= 0; --i)
//     {
//         total_node_num = levels[i].size() + levels[i + 1].size();
//         if (((total_node_num < core_num) || levels[i + 1].size() == 0) && level_merge_condition(levels[i], levels[i + 1], 0))
//         {
//             for (auto node : levels[i + 1])
//             {
//                 levels[i].push_back(node);
//             }
//             levels.erase(levels.begin() + i + 1);
//         }
//     }
//     return true;
// }

static bool balance_level_condition(const std::vector<NodePtr> &src_level, const std::vector<NodePtr> &dest_level, int32_t direction, uint32_t core_num)
{
    if (dest_level.size() > core_num - 1)
    {
        return false; // cannot larger than core num
    }
    int64_t src_load = 0;
    int64_t new_src_load = 0;
    for (auto node : src_level)
    {
        src_load = std::max(get_load_from_model(node), src_load);
    }
    auto new_src_level = src_level;
    auto most_load_node = remove_most_load(new_src_level);

    if (direction == 0)
    {
        for (auto c : most_load_node->children())
        {
            for (auto n : new_src_level)
            {
                if (c == n)
                {
                    return false; // cannot break dependency
                }
            }
            for (auto n : dest_level)
            {
                if (c == n)
                {
                    return false; // cannot break dependency
                }
            }
        }
    }
    else // direction == 1
    {
        for (auto p : most_load_node->parents())
        {
            for (auto n : new_src_level)
            {
                if (p == n)
                {
                    return false; // cannot break dependency
                }
            }
            for (auto n : dest_level)
            {
                if (p == n)
                {
                    return false; // cannot break dependency
                }
            }
        }
    }

    for (auto node : new_src_level)
    {
        new_src_load = std::max(get_load_from_model(node), new_src_load);
    }

    int64_t dest_load = 0;
    int64_t new_dest_load = 0;
    for (auto node : dest_level)
    {
        dest_load = std::max(get_load_from_model(node), src_load);
    }
    new_dest_load = std::max(dest_load, get_load_from_model(most_load_node));

    int64_t total_load = src_load + dest_load;
    int64_t new_total_load = new_src_load + new_dest_load;
    if (new_total_load >= total_load)
    {
        return false; // no improvement
    }

    return true;
}

static bool balance_next_level(size_t idx, GraphPtr g, size_t core_num, std::vector<std::vector<NodePtr>> &levels)
{
    if (idx < levels.size() - 1)
    {
        if (balance_level_condition(levels[idx], levels[idx + 1], 0, core_num))
        {
            auto node = remove_most_load(levels[idx]);
            levels[idx + 1].insert(levels[idx + 1].begin(), node);
        }
        else if (balance_level_condition(levels[idx + 1], levels[idx], 1, core_num))
        {
            auto node = remove_most_load(levels[idx + 1]);
            levels[idx].push_back(node);
        }
    }
    return true;
}

static bool level_balance(GraphPtr g, std::vector<std::vector<NodePtr>> &levels, size_t core_num)
{
    if (!core_num)
    {
        return true;
    }

    for (int32_t i = levels.size() - 2; i >= 0; --i)
    {
        balance_next_level(i, g, core_num, levels);
    }
    return true;
}

static bool remove_empty_level(std::vector<std::vector<NodePtr>> &levels)
{
    for (size_t i = 0; i < levels.size(); ++i)
    {
        if (levels[i].size() == 0)
        {
            levels.erase(levels.begin() + i);
            i--;
        }
    }
    return true;
}

static NodePtr get_node(GraphPtr g, std::string name)
{
    NodePtr node;
    for (auto n : g->nodes())
    {
        if (n->name() == name)
        {
            node = n;
            break;
        }
    }
    if (!node)
    {
        ERROR() << "cannot find " << name;
    }
    return node;
}

static std::vector<NodePtr> split_into_sub_levels(NodePtr node, uint64_t max_split_size, uint64_t max_size, bool is_wt)
{
    auto g = node->graph();
    auto sgn_tmp = SubGraphNode::from_node(node);
    auto sg = sgn_tmp->subgraph();
    std::vector<NodePtr> unfolded_nodes;
    if (sgn_tmp)
    {
        sgn_tmp->flat();
        unfolded_nodes = sgn_tmp->unfold();
    }

    auto new_levels = nodes_to_levels(g, unfolded_nodes);

    uint64_t total_size = 0;
    std::vector<NodePtr> current_group, new_group;
    for (auto level : new_levels)
    {
        for (auto n : level)
        {
            uint64_t data_size = is_wt ? n->get_constant_size() : n->get_output_size();
            if (data_size > max_size)
            {
                ERROR() << "node: " << n->name() << " has size: " << data_size << " larger than max size: " << max_size;
                if (total_size <= max_split_size)
                {
                    current_group.push_back(n);
                }
                else
                {
                    new_group.push_back(n);
                }
                total_size += data_size;
            }
            else
            {
                total_size += data_size;
                if (total_size <= max_split_size)
                {
                    current_group.push_back(n);
                }
                else
                {
                    new_group.push_back(n);
                }
            }
        }
    }

    NodePtr current_node;
    NodePtr new_node;

    if (!current_group.size())
    {
        current_node = merge_sg(g, new_group);
        WARN() << "cannot divide node " << node;
    }
    else if (!new_group.size())
    {
        current_node = merge_sg(g, current_group);
        WARN() << "cannot divide node " << node;
    }
    else
    {
        current_node = merge_sg(g, current_group);
        new_node = merge_sg(g, new_group);
    }

    return {current_node, new_node};
}

static void merge_tuned_nodes(GraphPtr g)
{
    std::string load_file = BuilderConfig::get<std::string>(Constants::load_tune_strategy);
    if (load_file != "")
    {
        std::map<int, std::vector<NodePtr>> sg_group;
        for (auto n : g->nodes())
        {
            if (SubGraphNode::is_subgraph_node(n))
            {
                SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(n);
                auto nodes = sgn->subgraph()->nodes();
                for (auto real_n : nodes)
                {
                    if (real_n->attrs().count("merge_sg_num"))
                    {
                        int merge_sg_num = real_n->get_attr<int>("merge_sg_num");
                        sg_group[merge_sg_num].push_back(n);
                        INFO() << "merge sg num " << merge_sg_num << " for " << n;
                        break;
                    }
                }
            }
            else
            {
                if (n->attrs().count("merge_sg_num"))
                {
                    int merge_sg_num = n->get_attr<int>("merge_sg_num");
                    sg_group[merge_sg_num].push_back(n);
                    INFO() << "merge sg num " << merge_sg_num << " for " << n;
                }
            }
        }
        for (auto pair : sg_group)
        {
            auto vec = pair.second;
            merge_sg(g, vec);
        }

        std::map<NodePtr, std::vector<NodePtr>> lc_group;
        for (auto n : g->nodes()) // for layout convertor
        {
            if (n->type() == OpType_LayoutConvertor && n->parents().size() && n->children().size() && n->parents()[0] == n->children()[0])
            {
                lc_group[n->parents()[0]].push_back(n);
            }
        }
        for (auto pair : lc_group)
        {
            auto node = pair.first;
            auto vec = pair.second;
            vec.push_back(node);
            merge_sg(g, vec);
        }
    }
}

bool LoadBalancer::schedule()
{
    if (!m_graph)
    {
        return false;
    }
    // graph construct: node -> subgraphnode
    // DEBUG() << "before graph init";
    // m_graph->show_graph();
    // DEBUG() << "before graph init done";
    auto enable_passes = BuilderConfig::get<std::vector<std::string>>("enable-pass");

    if (contains(enable_passes, "tnode_to_sg") &&
        !aipubt::BuilderConfig::get<bool>("disable_schedule_optimization") &&
        !(aipubt::BuilderConfig::get<bool>(Constants::enable_gm_enhance) &&
          aipubt::BuilderConfig::get<uint32_t>(Constants::gm_enhance_strategy) == 1))
    {
        tnode_to_sg(m_graph);
    }
    if (contains(enable_passes, "PFAnode_to_sg"))
    {
        PFAnode_to_sg(m_graph);
    }

    merge_tuned_nodes(m_graph);

    // connect_same_w_node(m_graph->nodes());

    init_nodes(m_graph, m_graph->nodes());

    // DEBUG() << "after graph init";
    // m_graph->show_graph();
    // DEBUG() << "after graph init done";

    // level setup
    auto inputs = m_graph->get_input_nodes();
    if (!setup_level(inputs, m_graph->nodes(), m_levels))
    {
        ERROR() << "graph is invalid";
    }
    // show_level("before opt");
    // m_graph->show_graph();
    if (!BuilderConfig::get<bool>("disable_schedule_optimization") &&
        !BuilderConfig::get<uint32_t>(Constants::multi_batch) &&
        !BuilderConfig::get<bool>(Constants::single_core_opt))
    {
        // INFO() << "enable multicore schedule optimization for load balance strategy " << aipubt::BuilderConfig::get<int32_t>(Constants::load_balance_strategy) << " it may degrade performance on single core targets.";
        std::vector<int> top_skip_levels;
        auto strategy = aipubt::BuilderConfig::get<int32_t>(Constants::load_balance_strategy);

        top_skip_levels.push_back(0);

        if (contains(enable_passes, "tnode_to_sg"))
        {
            skip_levels_by_tiling(top_skip_levels, m_levels);
        }

        // remove_extra_level(m_graph, m_levels, Target::get_the_target()->core_num());
        // show_level("after remove level");
        if (aipubt::BuilderConfig::get<bool>(Constants::enable_gm_enhance))
        {
            split_level(m_levels);
            top_skip_levels = generate_skip_levels(m_levels);
            // for (auto l : top_skip_levels)
            // {
            //     DEBUG() << "skip 1 " << l;
            // }
        }

        // show_level(m_levels, "before vertical");
        vertical_merge(m_graph, m_levels, top_skip_levels);
        // show_level(m_levels, "before reduce");
        reduce_level(m_graph, m_levels, Target::get_the_target()->core_num(), top_skip_levels);
        // show_level(m_levels, "after reduce");
        if (strategy != -1)
        {
            if (contains(enable_passes, "tnode_to_sg"))
            {
                auto merge_strategy = m_load_balance_strategy[aipubt::BuilderConfig::get<int32_t>(Constants::load_balance_strategy)][0];
                load_balance_merge(m_graph, m_levels, Target::get_the_target()->core_num(), merge_strategy);
                auto split_strategy = m_load_balance_strategy[aipubt::BuilderConfig::get<int32_t>(Constants::load_balance_strategy)][1];
                auto split_load_strategy = m_load_balance_strategy[aipubt::BuilderConfig::get<int32_t>(Constants::load_balance_strategy)][2];
                load_balance_split(m_graph, m_levels, Target::get_the_target()->core_num(), split_load_strategy, split_strategy, aipubt::BuilderConfig::get<bool>(Constants::enable_gm_enhance), top_skip_levels);
            }
            else
            {

                apply_load_balance_algo(m_graph, m_levels, Target::get_the_target()->core_num(), strategy);
            }

            // show_level(m_levels,"after load split");

            if (aipubt::BuilderConfig::get<bool>(Constants::enable_gm_enhance))
            {
                std::vector<int> candidates;
                std::map<std::string, std::string> share_w_nodes;
                if (BuilderConfig::get<std::string>(Constants::load_gm_enhance_cfg) != "")
                {
                    utils::JsonObject cfg_graph_info;
                    std::stringstream buffer;
                    std::ifstream ifile(BuilderConfig::get<std::string>(Constants::load_gm_enhance_cfg));
                    buffer << ifile.rdbuf();
                    cfg_graph_info = utils::JsonObject::parse(buffer.str());
                    auto obj = cfg_graph_info.obj();
                    ifile.close();

                    for (auto cfg_node : obj)
                    {
                        // DEBUG() << "cfg node: " << cfg_node.first;
                        auto cfg_info = cfg_node.second;
                        bool cfg_gm_candidate = cfg_info["gm_candidate"];
                        std::string cfg_share_weight = cfg_info["share_weight"];
                        if (cfg_gm_candidate && cfg_share_weight != "")
                        {
                            INFO() << "cfg node: " << cfg_node.first << " is shared weight";
                            share_w_nodes[cfg_node.first] = cfg_share_weight;
                        }
                    }

                    for (size_t i = 0; i < m_levels.size(); ++i)
                    {
                        auto level = m_levels[i];
                        bool should_skip = true;
                        for (auto node : level)
                        {
                            auto sgn = SubGraphNode::from_node(node);
                            if (sgn)
                            {

                                auto sg_nodes = sgn->subgraph()->nodes();
                                for (auto sg_n : sg_nodes)
                                {
                                    if (sg_n->type() == OpType_SubGraphInput)
                                    {
                                        continue;
                                    }
                                    if (share_w_nodes.find(sg_n->name()) == share_w_nodes.end())
                                    {
                                        should_skip = false;
                                    }
                                }
                            }
                        }
                        if (should_skip)
                        {
                            candidates.push_back(i);
                        }
                    }
                }
                else if (aipubt::BuilderConfig::get<uint32_t>(Constants::gm_enhance_strategy) == 1)
                {
                    candidates = generate_skip_levels(m_levels);
                }
                else if (aipubt::BuilderConfig::get<uint32_t>(Constants::gm_enhance_strategy) == 0)
                {
                    top_skip_levels = generate_skip_levels(m_levels);
                    auto gm_size = BuilderConfig::get<uint32_t>(Constants::gm_size);
                    for (auto idx : top_skip_levels)
                    {
                        auto level = m_levels[idx];
                        bool should_merge = false;
                        for (auto node : level)
                        {
                            if (!is_memory_bound(node, level.size()) || node->get_constant_size() < 1024 || node->get_constant_size() > gm_size)
                            {
                                should_merge = true;
                                break;
                            }
                        }
                        if (!should_merge)
                        {
                            candidates.push_back(idx);
                        }
                    }
                }
                top_skip_levels = candidates;
            }

            reduce_level(m_graph, m_levels, Target::get_the_target()->core_num(), top_skip_levels);
            // show_level(m_levels, "after opt level");

            for (size_t i = 0; i < m_levels.size(); ++i)
            {
                merge_load(m_graph, Target::get_the_target()->core_num(), i, m_levels);
            }
        }

        // load_balance_merge(m_graph, m_levels, Target::get_the_target()->core_num());

        // show_level("before level_balance");
        level_balance(m_graph, m_levels, 0);
        // show_level("after level_balance");
        remove_empty_level(m_levels);

        performance_eval(m_levels, Target::get_the_target()->core_num(), true);
    }
    else
    {
        performance_eval(m_levels, Target::get_the_target()->core_num(), false);
    }

    for (auto n : m_graph->nodes())
    {
        auto sgn = SubGraphNode::from_node(n);
        if (sgn)
        {
            sgn->flat();
        }
    }

    if (BuilderConfig::get<std::string>(Constants::load_schedule_cfg) != "")
    {
        for (size_t i = 0; i < m_levels.size(); ++i)
        {
            auto level = m_levels[i];
            for (auto node : level)
            {
                DEBUG() << node;
                auto sgn = SubGraphNode::from_node(node);
                if (sgn)
                {
                    auto unfold_nodes = sgn->unfold();
                }
            }
        }

        std::stringstream buffer;
        std::ifstream ifile(BuilderConfig::get<std::string>(Constants::load_schedule_cfg));
        buffer << ifile.rdbuf();
        utils::JsonObject cfg_graph_info = utils::JsonObject::parse(buffer.str());
        auto level_obj = cfg_graph_info.obj();
        ifile.close();

        std::vector<std::vector<NodePtr>> new_levels;
        for (auto l_obj : level_obj)
        {
            auto sg_obj = l_obj.second.obj();
            std::vector<NodePtr> new_level;
            for (auto s_obj : sg_obj)
            {
                auto node_obj = s_obj.second.obj();
                std::vector<NodePtr> node_vec;
                for (auto n_obj : node_obj)
                {
                    NodePtr node = get_node(m_graph, n_obj.first);
                    if (node)
                    {
                        node_vec.push_back(node);
                    }
                    else
                    {
                        WARN() << "cannot find " << n_obj.first;
                    }
                }
                auto sg_node = merge_sg(m_graph, node_vec);
                new_level.push_back(sg_node);
            }
            new_levels.push_back(new_level);
        }

        m_levels = new_levels;
    }
    else if (BuilderConfig::get<bool>(Constants::dump_schedule_cfg))
    {
        utils::JsonObject level_info;
        for (size_t i = 0; i < m_levels.size(); ++i)
        {
            utils::JsonObject sg_node_info;
            auto level = m_levels[i];
            for (size_t j = 0; j < level.size(); ++j)
            {
                utils::JsonObject node_info;
                auto node = level[j];
                SubGraphNodePtr sgnode = std::static_pointer_cast<SubGraphNode>(node);
                if (sgnode)
                {
                    for (auto n : *sgnode->subgraph())
                    {
                        if (n->type() == OpType_SubGraphInput)
                        {
                            continue;
                        }
                        int cycle = get_load_from_model(n);
                        node_info[n->name()] = cycle;
                        // node_info.push_back(n->name());
                    }
                }

                sg_node_info["sg" + std::to_string(j)] = node_info;
            }
            level_info["level" + std::to_string(i)] = sg_node_info;
        }
        std::ofstream os("schedule.json");
        if (os.is_open())
        {
            os << level_info;
            os << std::endl;
            os.close();
        }
    }

    for (size_t i = 0; i < m_levels.size(); ++i)
    {
        auto level = m_levels[i];
        int core_id = level.size() > 1 ? 0 : -1;
        for (size_t j = 0; j < level.size(); ++j)
        {
            auto node = level[j];
            if (get_load_from_model(node) > 0)
            {
                if (BuilderConfig::get<bool>(Constants::enable_separate_gm))
                {
                    node->set_attr(Constants::virtual_core_id, core_id);
                    core_id++;
                }
                else
                {
                    node->set_attr(Constants::virtual_core_id, -1);
                }
            }
            else
            {
                node->set_attr(Constants::virtual_core_id, -1);
            }
        }
    }

    if (Target::get_the_target()->isainfo().type >= ISA_ZYv6 && !aipubt::BuilderConfig::get<bool>(Constants::disable_dependency))
    {
        std::map<NodePtr, std::set<NodePtr>> dped_map;
        for (auto n : m_graph->nodes())
        {
            for (auto p : n->parents())
            {
                auto dped_p = dped_map[p];
                dped_map[n].insert(p);
                dped_map[n].insert(dped_p.begin(), dped_p.end());
            }
        }

        const uint32_t max_dep_size = std::min((uint32_t)4, Target::get_the_target()->core_num());
        for (size_t i = 1; i < m_levels.size(); ++i)
        {
            bool remove_level = true;
            for (auto node : m_levels[i])
            {
                auto dped_vec = get_dped(node, dped_map);
                // DEBUG() << "dped size for " << node->name() << " is " << dped_vec.size();
                // for (auto n : dped_vec)
                // {
                //     DEBUG() << n->name();
                // }
                node->set_attr("dped_nodes", dped_vec);
                if (dped_vec.size() > max_dep_size)
                {
                    remove_level = false;
                    break;
                }
            }
            if (remove_level)
            {
                for (auto node : m_levels[i])
                {
                    m_levels[i - 1].push_back(node);
                }
                m_levels.erase(m_levels.begin() + i);
                i--;
            }
            else
            {
                for (auto node : m_levels[i])
                {
                    node->set_attr("dped_nodes", std::vector<NodePtr>());
                }
            }
        }
    }

    auto max_wt_size = aipubt::BuilderConfig::get<uint64_t>(Constants::max_wt_size);
    auto max_act_size = aipubt::BuilderConfig::get<uint64_t>(Constants::max_act_size);
    if (max_wt_size || max_act_size) // make sure total constant for each level < max_wt_size
    {
        auto update_dped = [](std::vector<NodePtr> level, NodePtr split_node, NodePtr splited_node) -> void {
            for (auto n : level)
            {
                std::vector<NodePtr> dped_nodes = n->get_attr("dped_nodes");
                std::vector<NodePtr> new_dped_nodes;
                for (auto dped : dped_nodes)
                {
                    if (dped == split_node)
                    {
                        new_dped_nodes.push_back(splited_node);
                    }
                    else
                    {
                        if (contains(level, dped))
                        {
                            new_dped_nodes.push_back(dped);
                        }
                    }
                }
                n->set_attr("dped_nodes", new_dped_nodes);
            }
        };

        auto get_sg_size = [](NodePtr node) -> uint64_t {
            uint64_t sg_size = 0;
            auto sgn = SubGraphNode::from_node(node);
            if (sgn)
            {
                auto sg_nodes = sgn->subgraph()->nodes();
                for (auto n : sg_nodes)
                {
                    if (n->type() == OpType_InternalConcatenated ||
                        n->type() == OpType_InternalCrop ||
                        n->type() == OpType_InternalSlice ||
                        n->type() == OpType_InternalSpliter ||
                        n->type() == OpType_InternalPad ||
                        n->type() == OpType_InternalConstant ||
                        (n->type() == OpType_LayoutConvertor &&
                         !is_a_remapping_convertor(n)))
                    {
                        continue;
                    }
                    else
                    {
                        sg_size += n->get_output_size();
                    }
                }
            }
            else
            {
                // sg_size += node->get_output_size();
                ERROR();
            }
            return sg_size;
        };

        uint64_t max_size = max_wt_size ? max_wt_size : max_act_size;

        for (size_t i = 0; i < m_levels.size(); ++i)
        {
            std::vector<NodePtr> new_level;
            int64_t total_size = 0;
            NodePtr split_node;
            NodePtr splited_node;
            for (size_t j = 0; j < m_levels[i].size(); ++j)
            {
                if (new_level.size())
                {
                    new_level.push_back(m_levels[i][j]);
                    m_levels[i].erase(m_levels[i].begin() + j);
                    j--;
                    continue;
                }
                // INFO() << "total constant " << total_constant << " node " << m_levels[i][j] << " const " << m_levels[i][j]->get_constant_size();
                uint64_t data_size = max_wt_size ? m_levels[i][j]->get_constant_size() : get_sg_size(m_levels[i][j]);
                if (total_size + data_size > max_size)
                {
                    int64_t max_split_size = (int64_t)max_size - total_size;
                    DEBUG() << "node " << m_levels[i][j] << " " << data_size << " larger than max_split_size " << max_split_size << " will be divided";
                    std::vector<NodePtr> dped_nodes = m_levels[i][j]->get_attr("dped_nodes");
                    bool split_wt = max_wt_size ? true : false;
                    std::vector<NodePtr> sub_levels = split_into_sub_levels(m_levels[i][j], max_split_size, max_size, split_wt);
                    if (sub_levels[0] && sub_levels[1])
                    {
                        split_node = m_levels[i][j];
                        int split_vid = split_node->get_attr<int>(Constants::virtual_core_id);
                        splited_node = sub_levels[1];

                        sub_levels[0]->set_attr("dped_nodes", dped_nodes);
                        sub_levels[0]->set_attr(Constants::virtual_core_id, split_vid);
                        m_levels[i].erase(m_levels[i].begin() + j);
                        m_levels[i].insert(m_levels[i].begin() + j, sub_levels[0]);
                        uint64_t data_0_size = max_wt_size ? sub_levels[0]->get_constant_size() : get_sg_size(sub_levels[0]);
                        total_size += data_0_size;
                        DEBUG() << "splited into " << sub_levels[0] << " " << data_0_size;

                        sub_levels[1]->set_attr("dped_nodes", dped_nodes);
                        sub_levels[1]->set_attr(Constants::virtual_core_id, split_vid);
                        new_level.push_back(sub_levels[1]);
                        uint64_t data_1_size = max_wt_size ? sub_levels[1]->get_constant_size() : get_sg_size(sub_levels[1]);
                        DEBUG() << "splited into " << sub_levels[1] << " " << data_1_size;
                    }
                    else
                    {
                        split_node = m_levels[i][j];
                        int split_vid = split_node->get_attr<int>(Constants::virtual_core_id);
                        splited_node = sub_levels[0];
                        sub_levels[0]->set_attr(Constants::virtual_core_id, split_vid);
                        new_level.push_back(sub_levels[0]);
                        m_levels[i].erase(m_levels[i].begin() + j);
                        j--;
                    }
                }
                else
                {
                    total_size += data_size;
                }
            }
            if (new_level.size())
            {
                update_dped(new_level, split_node, splited_node);
                m_levels.insert(m_levels.begin() + i + 1, new_level);
            }
        }

        // internal_constant_merge(m_graph, m_levels);

        // for (size_t i = 0; i < m_levels.size(); ++i)
        // {
        //     for (size_t j = 0; j < m_levels[i].size(); ++j)
        //     {
        //         auto sgn = SubGraphNode::from_node(m_levels[i][j]);
        //         auto sg = sgn->subgraph();
        //         for (auto n : sg->nodes())
        //         {
        //             INFO() << "level " << i << " node " << n << " is estimated weight size 0x" << std::hex << get_total_constant(n);
        //         }
        //         if (get_total_constant(m_levels[i][j]) > max_wt_size)
        //         {
        //             ERROR() << "node " << m_levels[i][j] << " is estimated weight size " << get_total_constant(m_levels[i][j]) << " which is larger than max_wt_size " << max_wt_size;
        //         }
        //     }
        // }
    }

    if (!check_dependency(m_levels) && Target::get_the_target()->isainfo().type < ISA_ZYv6)
    {
        ERROR() << "check dependency failed";
    }

    if (!m_graph->check_sg())
    {
        ERROR() << "check sg failed";
    }

    if (!m_graph->check_valid_io())
    {
        ERROR() << "check valid io failed";
    }

    if (!check_level(m_graph->nodes(), m_levels))
    {
        ERROR() << "check level failed";
    }
    // DEBUG() << "show graph before flat";
    // m_graph->show_graph();
    m_graph->topological_sort();
    m_graph->reset_layer_id();
    if (!m_graph->is_topological_sorted())
    {
        WARN() << "it is not topological......";
    }

    // DEBUG() << "before final memory remap";

    virtual_node_memory_remapping(m_graph);

    // remove subgraph input node from graph

    // if (aipubt::BuilderConfig::get<bool>(Constants::enable_x2_sm_opt))
    // {
    //     uint32_t cnt = 0;
    //     uint32_t gm_debug_num = 3;
    //     for (auto n : m_graph->nodes())
    //     {
    //         if(gm_debug_num > cnt)
    //             x2_sm_opt(SubGraphNode::from_node(n));
    //         cnt++;
    //     }
    // }

    show_level(m_levels, "done", false);
    // m_graph->show_graph();
    // collect level & gid
    uint32_t gid = 0;
    for (uint i = 0; i < m_levels.size(); ++i)
    {
        for (auto node : m_levels[i])
        {
            auto sgn = SubGraphNode::from_node(node);
            if (sgn)
            {
                auto real_nodes = sgn->get_real_node();
                for (auto real_node : real_nodes)
                {
                    // use attributes for schedule_level and id
                    real_node->set_attr(Constants::schedule_level, i);
                    real_node->set_attr(Constants::schedule_gid, gid);
                }
                gid++;
            }
            else
            {
                ERROR() << "this is not a sgn";
            }
        }
    }

    return true;
}

const std::vector<std::vector<NodePtr>> &LoadBalancer::get_levels()
{
    return m_levels;
}

void LoadBalancer::set_graph(GraphPtr g)
{
    m_graph = g;
}

}; // namespace aipubt

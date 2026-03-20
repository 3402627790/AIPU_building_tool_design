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
 * @Date: 2020-06-17 14:57:41
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-13 20:21:24
 */

#include "datalayout_schedule.h"
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/pass_manager.h"
#include "utils/utils.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <stdint.h>
#include <vector>

namespace aipubt
{

typedef std::vector<DataLayout> LayoutList;
typedef std::vector<LayoutList> LayoutCombinations;
typedef std::map<LayoutList, LayoutCombinations> LayoutCombMap;
typedef std::map<NodePtr, LayoutCombinations> NodeCombMap;
struct BackupOutLayout
{
    TensorPtr tensor;
    NodePtr node;
    DataLayout layout;
};

namespace
{
    template <typename T>
    bool in_object(std::vector<T> &objects, T &obj)
    {
        auto it = std::find(objects.begin(), objects.end(), obj);
        return (it != objects.end());
    }
    bool satisfy_input_node(NodePtr header_node,
                            std::set<TensorPtr> &set_layout_tensors)
    {
        for (auto input : header_node->inputs())
        {
            if (!set_layout_tensors.count(input))
            {
                return false;
            }
        }
        return true;
    }

    bool is_keeping_layout(NodePtr node)
    {
        bool is_keep = false;
        if (node->attrs().count(Constants::keeping_layout))
        {
            is_keep = node->get_attr<bool>(Constants::keeping_layout);
        }
        return is_keep;
    }

    bool check_layout_support(TensorPtr parent_output,
                              NodePtr n,
                              DataLayout layout,
                              std::set<NodePtr> &change_node,
                              std::set<NodePtr> &convertor_node)
    {
        if (n->type() == OpType_LayoutConvertor)
        {
            convertor_node.insert(n);
            return (n->outputs()[0]->shape().layout() == layout);
        }
        change_node.insert(n);
        bool keeping_layout = n->get_attr(Constants::keeping_layout);
        if (!keeping_layout)
        {
            return false;
        }
        auto valid_input_layouts = n->get_attr<LayoutCombinations>(Constants::valid_input_layouts);
        auto valid_output_layouts = n->get_attr<LayoutCombinations>(Constants::valid_output_layouts);
        auto deny_layouts = n->get_attr<LayoutCombMap>(Constants::deny_layouts_comb);
        if (valid_input_layouts.size() > 0)
        {
            if (valid_input_layouts[0].size() == 0)
                return false;
            bool input_valid = false;
            if (valid_input_layouts[0].size() != n->inputs().size())
            {
                return false;
            }
            for (auto valid_input_layout : valid_input_layouts)
            {

                size_t input_id = 0;
                for (size_t i = 0; i < n->inputs().size(); ++i)
                {
                    if (n->inputs()[i] == parent_output)
                    {
                        input_id = i;
                        break;
                    }
                }
                if (valid_input_layout[input_id] == layout)
                {
                    input_valid = true;
                    break;
                }
            }
            if (!input_valid)
            {
                return false;
            }
        }
        if (valid_output_layouts.size() > 0)
        {
            if (valid_output_layouts[0].size() != 1)
                return false;
            bool output_valid = false;
            for (auto valid_output_layout : valid_output_layouts)
            {
                if (valid_output_layout[0] == layout)
                {
                    output_valid = true;
                    break;
                }
            }
            if (!output_valid)
            {
                return false;
            }
        }
        if (deny_layouts.size() > 0 && deny_layouts.count({layout}))
        {
            auto deny_layout = deny_layouts[{layout}];
            if (deny_layout.size() > 0 && deny_layout[0].size() != 1)
                return false;
            for (auto deny : deny_layout)
            {
                if (deny[0] == layout)
                {
                    return false;
                }
            }
        }

        TensorShape output_shape = n->outputs()[0]->shape();
        if (!output_shape.change_layout(layout))
        {
            return false;
        }

        for (auto c : n->outputs()[0]->consumers())
        {
            return check_layout_support(n->outputs()[0], c, layout, change_node, convertor_node);
        }
        return true;
    };
} // namespace

namespace pass
{
    bool merge_layerconvertor_layerconvertor(GraphPtr g)
    {
        size_t node_size = g->nodes().size();
        for (size_t i = 0; i < node_size;)
        {
            auto &nodes = g->nodes();
            auto &node = nodes[i];
            if (node->type() != OpType_LayoutConvertor ||
                node->children().size() != 1 ||
                node->children()[0]->type() != OpType_LayoutConvertor)
            {
                ++i;
                continue;
            }
            size_t node_size_origin = g->nodes().size();
            auto children = node->children()[0];
            DataLayout nd_from = node->inputs()[0]->shape().layout();
            DataLayout ch_to = children->outputs()[0]->shape().layout();
            if (nd_from == ch_to)
            {
                if (!node->outputs()[0]->is_io_tensor() && !children->outputs()[0]->is_io_tensor())
                {
                    // eliminate node and children
                    g->eliminate_node(children);
                    g->eliminate_node(node);
                }
                else if (node->outputs()[0]->is_io_tensor())
                {
                    auto in = node->inputs()[0];
                    auto out = children->outputs()[0];
                    for (auto c : out->consumers())
                    {
                        c->replace_input_by(out, in);
                    }
                    children->clear_outputs();
                    g->remove_node(children);
                }
                else if (children->outputs()[0]->is_io_tensor())
                {
                    auto in = node->inputs()[0];
                    auto inp_n = in->node();
                    auto out = children->outputs()[0];
                    for (auto c : in->consumers())
                    {
                        c->replace_input_by(in, out);
                    }
                    inp_n->clear_outputs();
                    inp_n->add_output(out);
                    g->remove_node(children);
                    g->remove_node(node);
                }
            }
            else
            {
                if (!node->outputs()[0]->is_io_tensor())
                {
                    // node change params to 'ch_to' /output tensor layout,
                    // and remove children
                    node->add_output(children->outputs()[0], node->remove_output(node->outputs()[0]));
                    node->params()["to"] = datalayout_to_string(ch_to);
                    g->remove_node(children);
                }
            }
            node_size = g->nodes().size();
            // Make sure don't get into an infinite loop
            if (node_size_origin == node_size)
            {
                ++i;
            }
        }
        return true;
    }

    bool reuse_layerconvertor(GraphPtr g)
    {
        auto creteria = [](NodePtr n)
        {
            if (n->type() != OpType_LayoutConvertor)
            {
                return false;
            }
            auto inp = n->inputs()[0];
            if (inp->consumers().size() < 2)
                return false;
            DataLayout to_layout = n->outputs()[0]->shape().layout();
            std::set<NodePtr> reuse_change_node;
            std::set<NodePtr> reuse_convertor_node;
            for (size_t i = 0; i < inp->consumers().size(); ++i)
            {
                if (inp->consumers()[i] == n)
                {
                    continue;
                }
                if (inp->consumers()[i]->inputs().size() > 1)
                {
                    return false;
                }
                std::set<NodePtr> change_node;
                std::set<NodePtr> convertor_node;
                if (check_layout_support(inp, inp->consumers()[i], to_layout, change_node, convertor_node))
                {
                    reuse_change_node.insert(change_node.begin(), change_node.end());
                    reuse_convertor_node.insert(convertor_node.begin(), convertor_node.end());
                }
            }
            if (reuse_convertor_node.size() < 1)
                return false;
            std::vector<TensorPtr> all_input_tensor;
            for (auto node : reuse_change_node)
            {
                for (auto input : node->inputs())
                {
                    all_input_tensor.push_back(input);
                }
            }
            std::vector<TensorPtr> all_output_tensor;
            all_output_tensor.push_back(inp);
            for (auto node : reuse_change_node)
            {
                for (auto output : node->outputs())
                {
                    all_output_tensor.push_back(output);
                }
            }
            bool need_resue = true;
            for (auto input_tensor : all_input_tensor)
            {
                if (std::find(all_output_tensor.begin(), all_output_tensor.end(), input_tensor) == all_output_tensor.end())
                {
                    need_resue = false;
                    break;
                }
            }
            return need_resue;
        };

        auto m_nodes = g->nodes();
        for (auto &n : m_nodes)
        {
            if (!creteria(n))
            {
                continue;
            }
            DataLayout to_layout = n->outputs()[0]->shape().layout();
            auto inp = n->inputs()[0];

            std::vector<size_t> reuse_id;
            std::set<NodePtr> reuse_change_node;
            std::set<NodePtr> reuse_convertor_node;
            for (size_t i = 0; i < inp->consumers().size(); ++i)
            {
                if (inp->consumers()[i] == n)
                {
                    continue;
                }
                std::set<NodePtr> change_node;
                std::set<NodePtr> convertor_node;
                if (check_layout_support(inp, inp->consumers()[i], to_layout, change_node, convertor_node))
                {
                    reuse_id.push_back(i);
                    reuse_change_node.insert(change_node.begin(), change_node.end());
                    reuse_convertor_node.insert(convertor_node.begin(), convertor_node.end());
                }
            }

            // 1 first level edge
            std::vector<NodePtr> reuse_nodes;
            for (auto id : reuse_id)
            {
                reuse_nodes.push_back(inp->consumers()[id]);
            }
            // 2. Process the first level node
            for (auto reuse_node : reuse_nodes)
            {
                auto out = n->outputs()[0];
                if (reuse_node->type() == OpType_LayoutConvertor)
                {
                    for (auto c : reuse_node->outputs()[0]->consumers())
                    {
                        c->replace_input_by(reuse_node->outputs()[0], out);
                    }
                    reuse_convertor_node.erase(reuse_node);
                    g->eliminate_node(reuse_node);
                }
                else
                {
                    reuse_node->replace_input_by(reuse_node->inputs()[0], out);
                }
            }

            // 3. change reuse node output layout
            for (auto change_node : reuse_change_node)
            {
                change_node->outputs()[0]->shape().change_layout(to_layout);
            }

            // 4. eliminate other layout convertor node
            for (auto convertor : reuse_convertor_node)
            {
                for (auto c : convertor->outputs()[0]->consumers())
                {
                    c->replace_input_by(convertor->outputs()[0], convertor->inputs()[0]);
                }
                // delete node,
                g->eliminate_node(convertor);
            }
        }
        return true;
    }

    bool merge_node_layerconvertor(GraphPtr g)
    {
        size_t node_size = g->nodes().size();
        for (size_t i = 0; i < node_size;)
        {
            auto &nodes = g->nodes();
            auto &node = nodes[i];
            if (node->type() != OpType_LayoutConvertor)
            {
                ++i;
                continue;
            }
            if (node->parents().size() == 1 &&
                node->parents()[0]->children().size() == 1 &&
                node->parents()[0]->type() != OpType_LayoutConvertor)
            {
                auto p = node->parents()[0];
                auto t = node->outputs()[0];
                auto layouts = p->get_attr<LayoutCombinations>(Constants::valid_output_layouts);
                bool p_keep = p->get_attr(Constants::keeping_layout);
                if (p_keep)
                {
                    ++i;
                    continue;
                }
                auto deny_layouts = p->get_attr<LayoutCombMap>(Constants::deny_layouts_comb);
                if (deny_layouts.size())
                {
                    LayoutList input_layouts;
                    for (auto inp : p->inputs())
                    {
                        input_layouts.push_back(inp->shape().layout());
                    }
                    auto &forbidden = deny_layouts[input_layouts];
                    for (auto &not_allow : forbidden)
                    {
                        auto it = std::find(layouts.begin(), layouts.end(), not_allow);
                        if (it != layouts.end())
                        {
                            layouts.erase(it);
                        }
                    }
                }

                bool remove = false;
                for (auto los : layouts)
                {
                    if (los.size() == 1)
                    {
                        if (t->shape().layout() == los[0])
                        {
                            remove = true;
                            break;
                        }
                    }
                    else if (los.size() > 1)
                    {
                        auto v = p->outputs();
                        auto it = std::find(v.begin(), v.end(), node->inputs()[0]);
                        if (it != v.end())
                        {
                            int idx = it - v.begin();
                            if (t->shape().layout() == los[idx])
                            {
                                remove = true;
                                break;
                            }
                        }
                    }
                }
                if (remove && !node->inputs()[0]->is_io_tensor())
                {
                    if (!node->outputs()[0]->is_io_tensor())
                    {
                        node->inputs()[0]->shape().change_layout(node->outputs()[0]->shape().layout());
                        g->eliminate_node(node);
                    }
                    else
                    {
                        auto parent = node->parents()[0];
                        parent->clear_outputs();
                        parent->add_output(node->outputs()[0]);
                        node->clear_outputs();
                        g->remove_node(node);
                    }
                }
                node_size = g->nodes().size();
            }
            ++i;
        }
        return true;
    }

} // namespace pass

class DataLayoutScheduler
{
public:
    DataLayoutScheduler(GraphPtr g) : m_graph(g)
    {
        m_enable_sparse_opt = BuilderConfig::get<bool>(Constants::enable_sparse_opt);
        m_debug = BuilderConfig::get<bool>("datalayout_schedule_debug");
    }
    void schedule();
    void do_schedule();

private:
    bool insert_convertor_before_node(NodePtr node, TensorPtr inp, DataLayout convert_to);
    bool insert_convertor_after_node(NodePtr node,
                                     TensorPtr out,
                                     DataLayout convert_to,
                                     bool is_inserted_outside_edge = false);
    void insert_convertor_before_node_s(NodePtr node, TensorPtr inp, DataLayout convert_to);
    void insert_convertor_after_node_s(NodePtr node,
                                       TensorPtr inp,
                                       DataLayout convert_to,
                                       bool is_inserted_outside_edge = false);
    void preprocessor_flat_layout();
    void set_layout_attr(NodePtr n);
    void setup_deny_layouts(NodePtr n);
    std::vector<BackupOutLayout> backup_out_layout(std::vector<TensorPtr> &out_tensors);
    void set_valid_layout_attr();

    inline uint32_t is_layout_compatible(NodePtr node);
    uint32_t check_insert_layout_converter(NodePtr node);
    uint32_t compute_datalayout_loss();
    uint32_t insert_layout_converter();
    LayoutList export_valid_input_layouts(TensorPtr output,
                                          NodePtr c,
                                          std::set<TensorPtr> &set_layout_tensors);
    void intersection_layout(LayoutList &output_layout_list,
                             LayoutList &c_input_layout_list,
                             std::map<DataLayout, int> &output_layout_count);
    bool can_be_spreaded(TensorPtr output,
                         NodePtr &c,
                         LayoutList &output_layout_list,
                         std::vector<NodePtr> &objects,
                         std::set<TensorPtr> &set_layout_tensors);
    DataLayout get_best_layout(std::map<DataLayout, int> &output_layout_count,
                               LayoutList &original_output_layout_list);
    void child_layout_spread(NodePtr &header_node,
                             std::vector<NodePtr> &objects,
                             std::set<TensorPtr> &set_layout_tensors,
                             LayoutList &original_output_layout_list,
                             LayoutList output_layout_list,
                             std::map<DataLayout, int> &output_layout_count,
                             std::vector<NodePtr> &set_node);
    NodePtrList layout_spread_empty_layout(NodePtr &header_node,
                                           NodePtrList &objects,
                                           std::set<TensorPtr> &set_layout_tensors);
    NodePtrList layout_spread_single_layout(NodePtr &header_node,
                                            NodePtrList &objects,
                                            std::set<TensorPtr> &set_layout_tensors);
    NodePtrList layout_spread_multi_layout(NodePtr &header_node,
                                           NodePtrList &objects,
                                           std::set<TensorPtr> &set_layout_tensors);
    std::vector<NodePtr> layout_spread(NodePtr &header_node,
                                       std::vector<NodePtr> &objects,
                                       std::set<TensorPtr> &set_layout_tensors);
    void layout_locally_spread(std::vector<NodePtr> &objects,
                               std::set<TensorPtr> &set_layout_tensors);
    bool input_can_change_layout(TensorPtr t, DataLayout layout);
    bool all_input_can_change_layout(NodePtr n, DataLayout layout);
    bool is_internal_op_keep_layout(NodePtr n);
    bool adjust_convertor_params();

    GraphPtr m_graph = nullptr;
    std::map<NodePtr, LayoutCombinations> m_inlayouts_map;
    std::map<NodePtr, LayoutCombinations> m_outlayouts_map;
    std::map<NodePtr, std::map<LayoutList, LayoutCombinations>> m_deny_outlayouts_map;
    std::map<TensorPtr, std::vector<std::pair<DataLayout, NodePtr>>> m_inserted_convertor;
    std::chrono::time_point<std::chrono::steady_clock> m_start_timestamp;
    const std::set<OpType> m_internal_op_list = {
        OpType_InternalCrop,
        OpType_InternalSlice,
        OpType_InternalConcatenated,
        OpType_InternalSpliter,
        OpType_InternalPad,
    };
    bool m_enable_sparse_opt = false;
    bool m_debug = false;
};

bool DataLayoutScheduler::is_internal_op_keep_layout(NodePtr n)
{
    if (!m_internal_op_list.count(n->type()))
        return false;

    auto in_lo = n->get_attr<LayoutCombinations>(Constants::valid_input_layouts);
    auto out_lo = n->get_attr<LayoutCombinations>(Constants::valid_output_layouts);
    if (in_lo.size() == 0 || out_lo.size() == 0)
        return true;
    return false;
}

void DataLayoutScheduler::preprocessor_flat_layout()
{
    for (auto n : m_graph->nodes())
    {
        for (auto t : n->outputs())
        {
            if (t->shape().layout() != DataLayout_Flat)
            {
                continue;
            }
            if (t->shape().dim() == 4)
            {
                t->shape().set_layout(DataLayout_NHWC);
            }
        }
    }
    return;
}

void DataLayoutScheduler::set_layout_attr(NodePtr n)
{
    const std::set<OpType> no_modify_layout = {
        OpType_Reshape,
    };
    std::map<OpType, LayoutCombinations> builtin_in_layouts =
        {
            {OpType_Input, {{}}},
        };

    if (n->attrs().count(Constants::pre_config_layouts))
    {
        if (!n->attrs().count(Constants::valid_output_layouts) || !n->attrs().count(Constants::valid_input_layouts))
        {
            WARN() << "no valid layouts for pre config layout";
        }
    }
    else if (m_internal_op_list.count(n->type()))
    {
        LayoutCombinations valid_layouts;
        if (n->inputs()[0]->shape().n() != n->outputs()[0]->shape().n())
        {
            valid_layouts = {
                {DataLayout_NCHWC32},
                {DataLayout_NCHWC16},
                {DataLayout_NHWC},
                {DataLayout_NCHW},
                {DataLayout_NDHWC},
                {DataLayout_NDCHWC32},
                {DataLayout_NDCHWC16},
                {DataLayout_NCDHWC32},
                {DataLayout_NCDHWC16},
                {DataLayout_Flat},
            };
        }
        else
        {
            valid_layouts = {
                {DataLayout_NHWC},
                {DataLayout_NDHWC},
                {DataLayout_Flat},
            };
        }
        n->set_attr(Constants::keeping_layout, true);
        if (!n->attrs().count(Constants::valid_input_layouts))
        {
            n->set_attr(Constants::valid_input_layouts, valid_layouts);
        }
        if (!n->attrs().count(Constants::valid_output_layouts))
        {
            n->set_attr(Constants::valid_output_layouts, valid_layouts);
        }
    }
    if (n->type() == OpType_Input)
    {
        LayoutList o_layout;
        if (n->params().count(Constants::layout))
        {
            std::string layout_name = n->params()[Constants::layout];
            DataLayout layout = string_to_datalayout(layout_name);
            if (layout != DataLayout_UnSupport)
            {
                o_layout = {layout};
            }
        }
        n->set_attr(Constants::keeping_layout, false);
        LayoutCombinations o_valid = {o_layout};
        n->set_attr(Constants::valid_output_layouts, o_valid);
    }
    else if (no_modify_layout.count(n->type()))
    {
        if (n->inputs().size())
        {
            LayoutCombinations input_layout = {{n->inputs()[0]->shape().layout()}};
            n->set_attr(Constants::valid_input_layouts, input_layout);
        }
        LayoutCombinations output_layout = {{n->outputs()[0]->shape().layout()}};
        n->set_attr(Constants::valid_output_layouts, output_layout);
    }

    if (builtin_in_layouts.count(n->type()))
    {
        n->set_attr(Constants::valid_input_layouts, builtin_in_layouts[n->type()]);
    }
    auto valid_input_layout_size = n->get_attr<LayoutCombinations>(Constants::valid_input_layouts).size();
    auto valid_output_layout_size = n->get_attr<LayoutCombinations>(Constants::valid_output_layouts).size();

    if (valid_input_layout_size == 1 && valid_output_layout_size == 1 && n->type() != OpType_Input && n->type() != OpType_LayoutConvertor)
    {
        auto i_lo = n->get_attr<LayoutCombinations>(Constants::valid_input_layouts)[0];
        auto o_lo = n->get_attr<LayoutCombinations>(Constants::valid_output_layouts)[0];
        if (i_lo == o_lo)
        {
            // WARN() << n->name() << ", only one valid layout (" << datalayout_to_string(i_lo[0]) << ") for in/out. So force setting keeping_layout true";
            n->set_attr(Constants::keeping_layout, true);
        }
    }
    return;
}

// setup deny layouts
void DataLayoutScheduler::setup_deny_layouts(NodePtr n)
{
    m_deny_outlayouts_map[n] = n->get_attr<LayoutCombMap>(Constants::deny_layouts_comb);
    // update forbidden layout for layout size==1
    for (auto &kv : m_deny_outlayouts_map[n])
    {
        LayoutList need_append;
        for (auto &outlayouts : kv.second)
        {
            if (outlayouts.size() > 1 && std::equal(outlayouts.begin() + 1, outlayouts.end(), outlayouts.begin()))
            {
                need_append.push_back(outlayouts[0]);
            }
        }
        for (auto lo : need_append)
        {
            kv.second.push_back({lo});
        }
    }
}

void DataLayoutScheduler::set_valid_layout_attr()
{
    auto check_lo_compatible = [](NodePtr node, TensorPtr inp, DataLayout lo) -> bool
    {
        auto shape = inp->shape();
        if (shape.layout() != lo)
        {
            shape.change_layout(lo);
            if (shape.layout() != lo)
            {
                return false;
            }
        }
        std::set<DataLayout> layout_c16 = {
            DataLayout_NCHWC16,
            DataLayout_NDCHWC16,
            DataLayout_NCDHWC16,
            DataLayout_NCHWC16T8,
            DataLayout_NCHWC16T1,
            DataLayout_NCHWC16Sparse,
            DataLayout_NCHWC16T8Sparse,
            DataLayout_NCHWC16T1Sparse,
            DataLayout_NDCHWC16T8,
            DataLayout_NDCHWC16T1,
        };
        std::set<DataLayout> layout_c32 = {
            DataLayout_NCHWC32,
            DataLayout_NDCHWC32,
            DataLayout_NCDHWC32,
            DataLayout_NCHWC32T8,
            DataLayout_NCHWC32T1,
            DataLayout_NCHWC32Sparse,
            DataLayout_NCHWC32T8Sparse,
            DataLayout_NCHWC32T1Sparse,
            DataLayout_NDCHWC32T8,
            DataLayout_NDCHWC32T1,
        };
        if (((inp->type() == Tensor::Dtype::UINT8) || (inp->type() == Tensor::Dtype::INT8)) && (layout_c16.count(shape.layout()) == 1))
        {
            return false;
        }
        else if (((inp->type() == Tensor::Dtype::UINT16) || (inp->type() == Tensor::Dtype::INT16)) && (layout_c32.count(shape.layout()) == 1))
        {
            return false;
        }
        else if (((inp->type() == Tensor::Dtype::UINT32) ||
                  (inp->type() == Tensor::Dtype::INT32) ||
                  (inp->type() == Tensor::Dtype::FP32)) &&
                 ((layout_c16.count(shape.layout()) == 1) ||
                  (layout_c32.count(shape.layout()) == 1)))
        {
            return false;
        };
        return true;
    };
    auto check_valid_list = [check_lo_compatible](NodePtr node, const std::vector<TensorPtr> &inputs, LayoutCombinations &in_lo)
    {
        LayoutCombinations new_in_lo;
        for (auto lolist : in_lo)
        {
            bool remove = false;
            if (lolist.size() == 1)
            {
                auto lo = lolist[0];
                for (auto &inp : inputs)
                {
                    if (!check_lo_compatible(node, inp, lo))
                    {
                        remove = true;
                        break;
                    }
                }
            }
            else if (lolist.size() != inputs.size())
            {
                // todo
            }
            else
            {
                for (size_t i = 0; i < lolist.size(); ++i)
                {
                    if (!check_lo_compatible(node, inputs[i], lolist[i]))
                    {
                        remove = true;
                        break;
                    }
                }
            }

            if (!remove)
            {
                new_in_lo.push_back(lolist);
            }
        }
        return new_in_lo;
    };

    auto remove_sparse_layout = [](LayoutCombinations &in_lo)
    {
        LayoutCombinations new_in_lo;
        for (auto &lolist : in_lo)
        {
            bool no_sparse_layout = true;
            for (size_t i = 0; i < lolist.size(); ++i)
            {
                if (lolist[i] > 32)
                {
                    no_sparse_layout = false;
                    break;
                }
            }
            if (no_sparse_layout)
            {
                new_in_lo.push_back(lolist);
            }
        }
        return new_in_lo;
    };

    for (auto node : m_graph->nodes())
    {
        bool is_keep = node->get_attr<bool>(Constants::keeping_layout);
        auto in_lo = node->get_attr<LayoutCombinations>(Constants::valid_input_layouts);
        auto out_lo = node->get_attr<LayoutCombinations>(Constants::valid_output_layouts);
        LayoutCombinations new_in_lo = check_valid_list(node, node->inputs(), in_lo);
        LayoutCombinations new_out_lo = check_valid_list(node, node->outputs(), out_lo);
        if (is_keep)
        {
            LayoutCombinations temp_input_layout_comb;
            LayoutCombinations temp_output_layout_comb;
            for (auto lo : new_in_lo)
            {
                if (new_out_lo.size() < 1)
                    continue;
                if ((lo.size() > 1) && new_out_lo[0].size() == 1)
                {
                    // input Layout comb like this {{nhwc,nhwc,},...},
                    // output Layout comb like this  {{nhwc,},...},
                    // for example concat
                    bool is_all_input_layout_same = true;
                    auto input_layout = lo[0];
                    for (auto l : lo)
                    {
                        if (l != input_layout)
                        {
                            is_all_input_layout_same = false;
                            break;
                        }
                    }
                    if (is_all_input_layout_same)
                    {
                        bool is_in_output_layout_comb = false;
                        for (auto new_out_l : new_out_lo)
                        {
                            if (input_layout == new_out_l[0])
                            {
                                is_in_output_layout_comb = true;
                                temp_output_layout_comb.push_back(new_out_l);
                                break;
                            }
                        }
                        if (is_in_output_layout_comb)
                        {
                            temp_input_layout_comb.push_back(lo);
                        }
                    }
                }
                else if ((lo.size() == 1) && (new_out_lo[0].size() > 1))
                {
                    // input Layout comb like this {{nhwc,},...},
                    // output Layout comb like this  {{nhwc,nhwc},...},
                    // for example split
                    for (auto new_out_l : new_out_lo)
                    {
                        bool is_all_output_layout_same = true;
                        auto output_layout = new_out_l[0];
                        for (auto l : new_out_l)
                        {
                            if (l != output_layout)
                            {
                                is_all_output_layout_same = false;
                                break;
                            }
                        }
                        if (is_all_output_layout_same)
                        {
                            if (output_layout == lo[0])
                            {
                                temp_input_layout_comb.push_back(lo);
                                temp_output_layout_comb.push_back(new_out_l);
                                break;
                            }
                        }
                    }
                }
                else
                {
                    if (contains(new_out_lo, lo))
                    {
                        temp_input_layout_comb.push_back(lo);
                        temp_output_layout_comb.push_back(lo);
                    }
                }
            }
            if (!temp_input_layout_comb.empty() && !temp_output_layout_comb.empty())
            {
                new_in_lo = temp_input_layout_comb;
                new_out_lo = temp_output_layout_comb;
            }
            else
            {
                // TODO check
            }
        }
        if (!m_enable_sparse_opt)
        {
            // remove parse layout
            new_in_lo = remove_sparse_layout(new_in_lo);
            new_out_lo = remove_sparse_layout(new_out_lo);
        }

        if (new_in_lo.size() == 0 && node->inputs().size())
        {
            WARN() << node->name() << " does NOT have valid input layouts, we will treats it accept any layout!!!!";
        }
        if (new_out_lo.size() == 0)
        {
            WARN() << node->name() << " does NOT have valid output layouts, we will treats it accept any layout!!!!";
        }
        node->set_attr(Constants::valid_input_layouts, new_in_lo);
        node->set_attr(Constants::valid_output_layouts, new_out_lo);
    }
    return;
}

std::vector<BackupOutLayout> DataLayoutScheduler::backup_out_layout(std::vector<TensorPtr> &out_tensors)
{
    std::vector<BackupOutLayout> tensors_need_to_final_layout;

    // we using the data layout as a tag for virtual subgraph
    if (m_graph->layout() == DataLayout_Unkown)
        return tensors_need_to_final_layout;

    for (auto n : m_graph->nodes())
    {
        for (size_t i = 0; i < n->outputs().size(); ++i)
        {
            auto output = n->outputs()[i];
            bool is_output_tensor = true;
            for (auto c : n->children())
            {
                if (contains(c->inputs(), output))
                {
                    is_output_tensor = false;
                }
            }
            is_output_tensor |= contains(out_tensors, output);
            if (!is_output_tensor)
                continue;
            // set all output nodes valid output layout
            auto valid_input_layouts = n->get_attr<LayoutCombinations>(Constants::valid_input_layouts);
            auto valid_output_layouts = n->get_attr<LayoutCombinations>(Constants::valid_output_layouts);
            bool can_remove_other_valid_layout = true;
            auto target_layout = output->shape().layout();
            auto deny_layout = m_deny_outlayouts_map[n];
            for (auto &kv : deny_layout)
            {
                for (auto los : kv.second)
                {
                    auto deny_lo = los[0];
                    if (i > 0 && los.size() > i)
                    {
                        deny_lo = los[i];
                    }
                    if (deny_lo == target_layout)
                    {
                        // the target layout in not allow list, so, we can not remove other valid layout to avoid not valid layout.
                        can_remove_other_valid_layout = false;
                        break;
                    }
                }
                if (!can_remove_other_valid_layout)
                {
                    break;
                }
            }

            bool is_need = true;
            if (can_remove_other_valid_layout)
            {
                if (valid_input_layouts.size() > 0 &&
                    valid_input_layouts[0].size() == 1 &&
                    valid_output_layouts.size() > 0 &&
                    valid_output_layouts[0].size() == 1)
                {
                    for (auto output_layout : valid_output_layouts)
                    {
                        auto valid_lo = output_layout[0];
                        if (i > 0 && output_layout.size() > i)
                        {
                            valid_lo = output_layout[i];
                        }
                        if (target_layout == valid_lo)
                        {
                            LayoutCombinations mlo = {output_layout};
                            if (n->get_attr<bool>(Constants::keeping_layout))
                            {
                                n->set_attr(Constants::keeping_layout, false);
                                n->set_attr(Constants::valid_input_layouts, mlo);
                            }
                            n->set_attr(Constants::valid_output_layouts, mlo);
                            is_need = false;
                            break;
                        }
                    }
                }
            }
            if (is_need)
            {
                tensors_need_to_final_layout.push_back({output, n, output->shape().layout()});
            }
        }
    }
    return tensors_need_to_final_layout;
}

// insert convertor before node
bool DataLayoutScheduler::insert_convertor_before_node(NodePtr node, TensorPtr inp, DataLayout convert_to)
{
    NodePtr parent = inp->node();
    bool has_inserted = false;
    NodePtr convertor = nullptr;
    TensorPtr out = nullptr;
    if (m_inserted_convertor.count(inp))
    {
        for (auto ins : m_inserted_convertor[inp])
        {
            if (ins.first == convert_to)
            {
                has_inserted = true;
                convertor = ins.second;
                out = convertor->outputs()[0];
                break;
            }
        }
    }
    if (!has_inserted)
    {
        out = ops::convert_layout(inp, convert_to);
        if (!out)
        {
            WARN() << "convert layout Failed.";
            return false;
        }
        ADD_LOCATION_BY_TENSOR(out, node);
        convertor = out->node();
        m_inserted_convertor[inp].push_back({convert_to, convertor});
    }
    // replace input
    node->add_input(out, node->remove_input(inp));
    return convert_to == out->shape().layout();
}

// insert convertor after node
bool DataLayoutScheduler::insert_convertor_after_node(NodePtr node,
                                                      TensorPtr out,
                                                      DataLayout convert_to,
                                                      bool is_inserted_outside_edge)
{
    auto inpshape = out->shape();
    bool has_inserted = false;
    NodePtr convertor = nullptr;
    if (m_inserted_convertor.count(out))
    {
        for (auto ins : m_inserted_convertor[out])
        {
            if (ins.first == convert_to)
            {
                has_inserted = true;
                convertor = ins.second;
                break;
            }
        }
    }

    TensorPtr inp = std::make_shared<Tensor>(*out);
    inp->set_io_tensor(false);
    auto base_name = out->name() + "/layout/after/" + datalayout_to_string(convert_to);
    inp->set_name(node->graph()->get_valid_tensor_name(base_name));

    // replace the output bu inp
    node->add_output(inp, node->remove_output(out));
    if (is_inserted_outside_edge)
    {
        for (auto c : out->consumers())
        {
            c->add_input(inp, c->remove_input(out));
        }
    }
    if (!out->shape().change_layout(convert_to))
    {
        return false;
    }
    TensorPtr convert_out = nullptr;
    if (!has_inserted)
    {
        convert_out = ops::convert_layout(inp, convert_to);
        if (!convert_out)
        {
            WARN() << "convert layout Failed.";
            return false;
        }
        ADD_LOCATION_BY_TENSOR(convert_out, node);
        convertor = convert_out->node();
        m_inserted_convertor[out].push_back({convert_to, convertor});
    }
    else
    {
        convert_out = convertor->outputs()[0];
        for (auto c : convert_out->consumers())
        {
            c->add_input(out, c->remove_input(convert_out));
        }
    }
    convertor->add_output(out, convertor->remove_output(convert_out));
    return convert_to == out->shape().layout();
}

void DataLayoutScheduler::insert_convertor_before_node_s(NodePtr node,
                                                         TensorPtr inp,
                                                         DataLayout convert_to)
{
    if (!insert_convertor_before_node(node, inp, convert_to))
    {
        ERROR() << "Nonsupport  convert node:" << node << "'s tensor " << inp->name() << " with shape " << inp->shape() << " to layout " << convert_to;
    }
}

void DataLayoutScheduler::insert_convertor_after_node_s(NodePtr node,
                                                        TensorPtr inp,
                                                        DataLayout convert_to,
                                                        bool is_inserted_outside_edge)
{
    if (!insert_convertor_after_node(node, inp, convert_to, is_inserted_outside_edge))
    {
        ERROR() << "Unsupport  convert node:" << node << "'s tensor "
                << inp->name() << " with shape "
                << inp->shape() << " to layout " << convert_to;
    }
}

void DataLayoutScheduler::schedule()
{
    auto start_t = std::chrono::steady_clock::now();
    std::vector<TensorPtr> out_tensors = m_graph->get_output_tensors();
    // preprocessor Flat layout
    preprocessor_flat_layout();

    // set layout attr
    for (auto n : m_graph->nodes())
    {
        set_layout_attr(n);
        setup_deny_layouts(n);
    }

    auto tensors_need_to_final_layout = backup_out_layout(out_tensors);
    m_graph->topological_sort();
    set_valid_layout_attr();
    do_schedule();
    adjust_convertor_params();
    for (auto t : tensors_need_to_final_layout)
    {
        auto shape = t.tensor->shape();
        if (shape.change_layout(t.layout))
        {
            // we only insert compatible layout to NHWC
            insert_convertor_after_node_s(t.node, t.tensor, t.layout, true);
        }
    }
    m_graph->topological_sort();
    std::chrono::duration<double, std::milli> used_seconds = std::chrono::steady_clock::now() - start_t;
    INFO() << "layout schedule costs: " << used_seconds.count() << "ms";
    return;
}

uint32_t DataLayoutScheduler::is_layout_compatible(NodePtr node)
{
    uint32_t score = UINT32_MAX;
    if (node->parents().size() == 0)
    {
        // no parents~  we support the in layout always meets the requirement.
        return 0;
    }
    auto &tensors = node->inputs();
    if (is_keeping_layout(node))
    {
        int max = -1;
        LayoutList input_layouts;
        if (m_inlayouts_map[node].size() == 0)
        {
            input_layouts.push_back(tensors[0]->shape().layout());
            max = tensors.size();
        }
        else
        {
            std::vector<int> counter(DataLayout_MaxCount, 0);
            for (auto &t : tensors)
            {
                counter[t->shape().layout()]++;
            }
            int max_counter = *std::max_element(counter.begin(), counter.end());
            bool be_match_output_layout = false;
            for (const auto &layouts : m_inlayouts_map[node])
            {
                if (layouts.size() == 0)
                    continue;
                for (auto layout : layouts)
                {
                    if (counter[layout] > max ||
                        (counter[layout] == max && layout == node->outputs()[0]->shape().layout()))
                    {
                        max = counter[layout];
                        input_layouts = layouts;
                    }
                    if (layout == node->outputs()[0]->shape().layout())
                    {
                        be_match_output_layout = true;
                    }
                }
                if (max == max_counter && be_match_output_layout)
                    break;
            }
            if (max < 0)
            {
                ERROR() << node->name() << ", Invalid Plugin. Please check 'valid_input_layouts' in it's plugin";
            }
        }

        // Calculate the real number of layout convertor
        if ((input_layouts.size() > 1) && (input_layouts.size() == tensors.size()))
        {
            uint32_t insert_num = 0;
            for (size_t i = 0; i < tensors.size(); ++i)
            {
                if ((tensors[i]->shape().layout() != input_layouts[i]) && (DataLayout_Flat != tensors[i]->shape().layout()))
                    ++insert_num;
            }
            return insert_num;
        }
        else
        {
            return tensors.size() - max;
        }
    }
    else
    {
        int best = -1;
        const auto &valid_layout = m_inlayouts_map[node];
        for (uint32_t k = 0; k < valid_layout.size(); ++k)
        {
            auto &layout = valid_layout[k];
            uint32_t ms = 0;
            uint32_t num_sparse = 0;
            if (layout.size() == 1)
            {
                for (auto t : tensors)
                {
                    if (t->shape().layout() != layout[0])
                    {
                        ms++;
                    }
                    else if (t->shape().layout() >= 32)
                    {
                        num_sparse++;
                    }
                }
            }
            else if ((layout.size() > 1) && layout.size() == tensors.size())
            {
                for (uint32_t i = 0; i < layout.size(); ++i)
                {
                    if (tensors[i]->shape().layout() != layout[i])
                    {
                        ms++;
                    }
                    else if (tensors[i]->shape().layout() >= 32)
                    {
                        num_sparse++;
                    }
                }
            }
            // using Sparse layout first
            if (ms == score && num_sparse > 0)
            {
                score = ms;
                best = k;
                if (score == 0)
                    break;
            }
            else if (ms < score)
            {
                score = ms;
                best = k;
            }
        }
        if (best == -1)
        {
            return 0;
        }
        return score;
    }
}

bool DataLayoutScheduler::input_can_change_layout(TensorPtr t, DataLayout layout)
{
    TensorShape shape = t->shape();
    if (shape.layout() != DataLayout_Flat)
    {
        if (shape.layout() != layout && !shape.change_layout(layout))
        {
            return false;
        }
    }
    return true;
}

bool DataLayoutScheduler::all_input_can_change_layout(NodePtr n, DataLayout layout)
{
    for (auto input : n->inputs())
    {
        if (!input_can_change_layout(input, layout))
        {
            return false;
        }
    }
    return true;
}

uint32_t DataLayoutScheduler::check_insert_layout_converter(NodePtr node)
{
    uint32_t score = 0;
    // 1. no parents~  we support the in layout always meets the requirement.
    // 2. intertnal OP and No layout information,we infer that it needs to keep the original layout.
    if (node->parents().size() == 0 || is_internal_op_keep_layout(node))
    {
        return 0;
    }
    auto &inputs = node->inputs();
    auto inlayouts = m_inlayouts_map[node];

    if (is_keeping_layout(node))
    {
        if (inlayouts.size() == 0)
        {
            WARN() << "inlayouts is empty!";
        }
        DataLayout dl = node->outputs()[0]->shape().layout();
        for (auto input : inputs)
        {
            if (input->shape().layout() != dl &&
                DataLayout_Flat != input->shape().layout() &&
                input_can_change_layout(input, dl))
            {
                insert_convertor_before_node_s(node, input, dl);
                ++score;
            }
        }
    }
    else
    {
        if (inlayouts.size() == 0)
        {
            WARN() << node->name() << "inlayouts is empty!";
            return 0;
        }
        if (inlayouts[0].size() == 1)
        {
            LayoutList input_layout_list;
            std::map<DataLayout, int> input_layout_count;
            LayoutList node_in_layout_list;
            for (auto inlayout : inlayouts)
            {
                node_in_layout_list.push_back(inlayout[0]);
            }
            for (auto input : inputs)
            {
                if (input_layout_count.count(input->shape().layout()))
                {
                    input_layout_count[input->shape().layout()] += 1;
                }
                else
                {
                    input_layout_count[input->shape().layout()] = 1;
                    input_layout_list.push_back(input->shape().layout());
                }
            }
            // Find the datalayout that best satisfies inlayouts
            int best_layout_count = 0;
            DataLayout best_input_dl = input_layout_count.begin()->first;

            std::map<DataLayout, size_t> layout_order;
            for (int i = 0; i < static_cast<int>(input_layout_list.size()); ++i)
            {
                layout_order[input_layout_list[i]] = i;
            }
            for (auto input_layout : input_layout_count)
            {
                if ((input_layout.second > best_layout_count) ||
                    (input_layout.second == best_layout_count &&
                     layout_order[input_layout.first] < layout_order[best_input_dl]))
                {
                    DataLayout dl = input_layout.first;
                    if (in_object(node_in_layout_list, dl) && all_input_can_change_layout(node, dl))
                    {
                        best_layout_count = input_layout.second;
                        best_input_dl = input_layout.first;
                    }
                }
            }

            DEBUG() << node->name() << " best_input_dl:" << datalayout_to_string(best_input_dl)
                    << ", best_layout_count: " << best_layout_count;
            // No match on, find minimum insert
            int minimum_insert = INT_MAX;
            if (best_layout_count == 0)
            {
                for (auto dl : node_in_layout_list)
                {
                    if (all_input_can_change_layout(node, dl))
                    {
                        int cur_insert_nul = 0;
                        for (auto input : node->inputs())
                        {
                            TensorShape shape = input->shape();
                            if (shape.layout() != dl)
                            {
                                ++cur_insert_nul;
                            }
                        }
                        if (minimum_insert > cur_insert_nul)
                        {
                            best_input_dl = dl;
                            minimum_insert = cur_insert_nul;
                        }
                    }
                }
            }
            for (auto input : inputs)
            {
                if (input->shape().layout() != best_input_dl &&
                    DataLayout_Flat != input->shape().layout() &&
                    DataLayout_Flat != best_input_dl)
                {
                    insert_convertor_before_node_s(node, input, best_input_dl);
                    ++score;
                }
            }
        }
        else if (inlayouts[0].size() == inputs.size())
        {
            LayoutList input_layout_list;
            LayoutList node_in_layout_list;
            for (auto input : inputs)
            {
                input_layout_list.push_back(input->shape().layout());
            }
            LayoutList best_data_list = inlayouts[0];
            int best_count = 0;
            for (auto inlayout : inlayouts)
            {
                int count = 0;
                if (input_layout_list.size() != inlayout.size())
                {
                    continue;
                }
                for (size_t i = 0; i < input_layout_list.size(); ++i)
                {
                    if (input_layout_list[i] == inlayout[i])
                    {
                        count += 2 ^ (input_layout_list.size() - 1 - i);
                    }
                }
                if (best_count < count)
                {
                    best_count = count;
                    best_data_list = inlayout;
                }
            }
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                if (inputs[i]->shape().layout() != best_data_list[i] &&
                    DataLayout_Flat != inputs[i]->shape().layout() &&
                    DataLayout_Flat != best_data_list[i])
                {
                    insert_convertor_before_node_s(node, inputs[i], best_data_list[i]);
                    ++score;
                }
            };
        }
        else
        {
            WARN() << node->name() << "Invalid inlayouts[0] size=" << inlayouts[0].size();
        }
    }
    return score;
}

uint32_t DataLayoutScheduler::compute_datalayout_loss()
{
    uint32_t score = 0;
    auto nodes = m_graph->nodes();
    for (auto n : nodes)
    {
        auto s = is_layout_compatible(n);
        score += s;
    }
    return score;
}

uint32_t DataLayoutScheduler::insert_layout_converter()
{
    uint32_t score = 0;
    auto nodes = m_graph->nodes();
    for (auto n : nodes)
    {
        auto s = check_insert_layout_converter(n);
        score += s;
    }
    return score;
}

// Export the valid input layout of the consumer node
LayoutList DataLayoutScheduler::export_valid_input_layouts(TensorPtr output,
                                                           NodePtr c,
                                                           std::set<TensorPtr> &set_layout_tensors)
{
    // Check if c has been spreeded
    bool be_spreeded = false;
    LayoutList c_input_layout_list;
    LayoutCombinations c_input_layout_comb = m_inlayouts_map[c];
    for (auto inp : c->inputs())
    {
        if (set_layout_tensors.count(inp))
        {
            be_spreeded = true;
            break;
        }
    }
    if (c_input_layout_comb.size() != 0)
    {
        if (be_spreeded)
        {
            if (c_input_layout_comb[0].size() == 1)
            {
                for (auto c_input_layouts : c_input_layout_comb)
                {
                    if (is_keeping_layout(c))
                    {
                        if (all_input_can_change_layout(c, c_input_layouts[0]))
                        {
                            c_input_layout_list.push_back(c_input_layouts[0]);
                        }
                    }
                    else
                    {
                        c_input_layout_list.push_back(c_input_layouts[0]);
                    }
                }
            }
            else
            {
                // Like {{NHWC32, NHWC},...}
                // Check this input id
                size_t id = 0;
                std::map<size_t, DataLayout> sub_layout;
                for (size_t i = 0; i < c->inputs().size(); ++i)
                {
                    if (output.get() == c->inputs()[i].get())
                    {
                        id = i;
                    }
                    if (set_layout_tensors.count(c->inputs()[i]))
                    {
                        sub_layout[i] = c->inputs()[i]->shape().layout();
                    }
                }
                for (auto c_input_layouts : c_input_layout_comb)
                {
                    bool is_valid = true;
                    for (auto inp_layout : sub_layout)
                    {
                        if (inp_layout.first >= c_input_layouts.size())
                        {
                            is_valid = false;
                            break;
                        }
                        if (c_input_layouts[inp_layout.first] != inp_layout.second)
                        {
                            is_valid = false;
                            break;
                        }
                    }
                    if (is_valid)
                        c_input_layout_list.push_back(c_input_layouts[id]);
                }
                if (c_input_layout_list.size() == 0)
                {
                    for (auto c_input_layouts : c_input_layout_comb)
                    {
                        c_input_layout_list.push_back(c_input_layouts[id]);
                    }
                }
            }
        }
        else
        {
            if (c_input_layout_comb[0].size() == 1)
            {
                for (auto c_input_layouts : c_input_layout_comb)
                {
                    if (is_keeping_layout(c))
                    {
                        if (all_input_can_change_layout(c, c_input_layouts[0]))
                        {
                            c_input_layout_list.push_back(c_input_layouts[0]);
                        }
                    }
                    else
                    {
                        c_input_layout_list.push_back(c_input_layouts[0]);
                    }
                }
            }
            else
            {
                size_t id = 0;
                for (size_t i = 0; c->inputs().size(); ++i)
                {
                    if (output == c->inputs()[i])
                    {
                        id = i;
                        break;
                    }
                }
                for (auto c_input_layouts : c_input_layout_comb)
                {
                    if (id < c_input_layouts.size())
                        c_input_layout_list.push_back(c_input_layouts[id]);
                }
            }
        }
    }
    std::sort(c_input_layout_list.begin(), c_input_layout_list.end());
    c_input_layout_list.erase(std::unique(c_input_layout_list.begin(), c_input_layout_list.end()), c_input_layout_list.end());
    return c_input_layout_list;
}

void DataLayoutScheduler::intersection_layout(LayoutList &output_layout_list,
                                              LayoutList &c_input_layout_list,
                                              std::map<DataLayout, int> &output_layout_count)
{
    // 1. output_layout_list && c_input_layout_list
    for (auto it = output_layout_list.begin(); it != output_layout_list.end();)
    {
        if (in_object(c_input_layout_list, *it))
        {
            output_layout_count[*it] += 1;
            ++it;
        }
        else
        {
            it = output_layout_list.erase(it);
        }
    }
    return;
}

bool DataLayoutScheduler::can_be_spreaded(TensorPtr output,
                                          NodePtr &c,
                                          LayoutList &output_layout_list,
                                          std::vector<NodePtr> &objects,
                                          std::set<TensorPtr> &set_layout_tensors)
{
    if (!is_keeping_layout(c))
    {
        return false;
    }
    if (!in_object(objects, c))
    {
        return false;
    }
    LayoutList c_input_layout_list = export_valid_input_layouts(output, c, set_layout_tensors);
    if (c_input_layout_list.size() == 0)
        return true;
    for (auto output_layout : output_layout_list)
    {
        if (in_object(c_input_layout_list, output_layout))
        {
            return true;
        }
    }
    return false;
}

DataLayout DataLayoutScheduler::get_best_layout(std::map<DataLayout, int> &output_layout_count,
                                                LayoutList &original_output_layout_list)
{
    int best_layout_count = 0;
    if (output_layout_count.size() == 0)
        return DataLayout_NHWC;
    DataLayout best_layout = output_layout_count.begin()->first;
    std::map<DataLayout, size_t> layout_order;
    for (int i = 0; i < static_cast<int>(original_output_layout_list.size()); ++i)
    {
        layout_order[original_output_layout_list[i]] = i;
    }
    for (auto output_layout : output_layout_count)
    {
        if ((output_layout.second > best_layout_count) ||
            (output_layout.second == best_layout_count &&
             layout_order[output_layout.first] < layout_order[best_layout]))
        {
            best_layout_count = output_layout.second;
            best_layout = output_layout.first;
        }
    }
    return best_layout;
}

// Child conditions met: keep_layout=true
void DataLayoutScheduler::child_layout_spread(NodePtr &header_node,
                                              std::vector<NodePtr> &objects,
                                              std::set<TensorPtr> &set_layout_tensors,
                                              LayoutList &original_output_layout_list,
                                              LayoutList output_layout_list,
                                              std::map<DataLayout, int> &output_layout_count,
                                              std::vector<NodePtr> &set_node)
{
    set_node.push_back(header_node);
    // 1. Find All consumer input LayoutCombinations intersection
    for (auto output : header_node->outputs())
    {
        for (auto c : output->consumers())
        {
            LayoutList current_output_layout_list = output_layout_list;
            LayoutList c_input_layout_list = export_valid_input_layouts(output, c, set_layout_tensors);
            if (c_input_layout_list.size() != 0)
            {
                intersection_layout(current_output_layout_list,
                                    c_input_layout_list,
                                    output_layout_count);
            }
            if (current_output_layout_list.size() == 0)
            {
                continue;
            }
            // 2. If keep layout, need to continue spreading.
            if (can_be_spreaded(output, c, current_output_layout_list, objects, set_layout_tensors))
            {
                child_layout_spread(c, objects, set_layout_tensors, original_output_layout_list, current_output_layout_list, output_layout_count, set_node);
            }
        }
    }
    return;
}

// if output_layout_comb  is empty,
// indicates that it supports any datalayout.
NodePtrList DataLayoutScheduler::layout_spread_empty_layout(NodePtr &header_node,
                                                            NodePtrList &objects,
                                                            std::set<TensorPtr> &set_layout_tensors)
{
    NodePtrList set_node = {};
    LayoutList input_layout_list;
    for (auto input : header_node->inputs())
    {
        input_layout_list.push_back(input->shape().layout());
    }

    for (auto output : header_node->outputs())
    {
        for (auto layout : input_layout_list)
        {
            if (input_can_change_layout(output, layout) && is_keeping_layout(header_node))
            {
                output->shape().change_layout(layout);
                DEBUG() << header_node->name() << ", best_layout:" << datalayout_to_string(layout);
            }
        }
        set_layout_tensors.insert(output);
    }
    set_node.push_back(header_node);
    return set_node;
}

// All output use one layout list
NodePtrList DataLayoutScheduler::layout_spread_single_layout(NodePtr &header_node,
                                                             NodePtrList &objects,
                                                             std::set<TensorPtr> &set_layout_tensors)
{
    NodePtrList set_node = {};
    LayoutCombinations output_layout_comb = m_outlayouts_map[header_node];
    // 1. Convert LayoutCombinations to List
    LayoutList output_layout_list;
    // DataLayout
    std::map<DataLayout, int> output_layout_count;
    std::map<DataLayout, int> input_layout_count;
    LayoutList input_layout_list;
    std::vector<NodePtr> set_node_x = {};
    bool is_keep = is_keeping_layout(header_node);
    for (auto input : header_node->inputs())
    {
        input_layout_list.push_back(input->shape().layout());
    }
    for (auto output_layout : output_layout_comb)
    {
        if (is_keep)
        {
            if (all_input_can_change_layout(header_node, output_layout[0]))
            {
                output_layout_list.push_back(output_layout[0]);
                output_layout_count[output_layout[0]] = 1;
            }
        }
        else
        {
            output_layout_list.push_back(output_layout[0]);
            output_layout_count[output_layout[0]] = 1;
        }
    }
    if (is_keep)
    {
        bool all_input_is_set = true;
        bool all_input_is_same = true;
        DataLayout pre_best_layout = DataLayout_Flat;
        auto input0_layout = header_node->inputs()[0]->shape().layout();
        for (auto input : header_node->inputs())
        {
            auto input_layout = input->shape().layout();
            if ((header_node->parents().size() == 1) && (header_node->parents()[0]->type() == OpType_LayoutConvertor))
            {
                auto parent = header_node->parents()[0];
                input_layout = parent->inputs()[0]->shape().layout();
            }
            all_input_is_set = all_input_is_set && (set_layout_tensors.count(input));
            all_input_is_same = all_input_is_same && (input0_layout == input_layout);
            if (input_layout == DataLayout_Flat)
                continue;

            if (output_layout_count.count(input_layout) == 1)
            {
                output_layout_count[input_layout] += 1;
                pre_best_layout = input_layout;
            }
        }
        // all input have set, select best
        if (all_input_is_set && all_input_is_same && pre_best_layout != DataLayout_Flat)
        {
            output_layout_list = {pre_best_layout};
            output_layout_count.clear();
            output_layout_count[pre_best_layout] = 1;
        }
    }

    // remove deny layout
    if (m_deny_outlayouts_map[header_node].size() &&
        m_deny_outlayouts_map[header_node].count(input_layout_list))
    {
        auto &forbidden = m_deny_outlayouts_map[header_node][input_layout_list];
        for (auto fb : forbidden)
        {
            if (output_layout_count.count(fb[0]))
            {
                for (auto it = output_layout_list.begin(); it != output_layout_list.end(); ++it)
                {
                    if (fb[0] == *it)
                    {
                        output_layout_list.erase(it);
                        break;
                    }
                }
                output_layout_count.erase(fb[0]);
            }
        }
    }

    LayoutList original_output_layout_list = output_layout_list;
    child_layout_spread(header_node, objects, set_layout_tensors, original_output_layout_list,
                        output_layout_list, output_layout_count, set_node_x);

    if (m_debug)
    {
        for (auto output_layout : output_layout_count)
        {
            DEBUG() << datalayout_to_string(output_layout.first) << ":" << output_layout.second;
        }
    }
    // set tensor layout
    DataLayout best_layout = get_best_layout(output_layout_count, original_output_layout_list);
    DEBUG() << header_node->name() << ", best_layout:" << datalayout_to_string(best_layout);
    std::vector<NodePtr> temp_set_nodes;
    temp_set_nodes.push_back(header_node);
    while (temp_set_nodes.size() > 0)
    {
        std::vector<NodePtr> next_set_nodes;
        for (auto n : temp_set_nodes)
        {
            LayoutCombinations n_output_layout_comb = m_outlayouts_map[n];
            LayoutList n_output_layout_list;
            for (auto output_layout : n_output_layout_comb)
            {
                n_output_layout_list.push_back(output_layout[0]);
            }
            if (std::find(n_output_layout_list.begin(), n_output_layout_list.end(), best_layout) == n_output_layout_list.end())
            {
                continue;
            }
            if (is_keeping_layout(n))
            {
                if (!all_input_can_change_layout(n, best_layout))
                {
                    continue;
                }
            }
            set_node.push_back(n);
            for (auto output : n->outputs())
            {
                output->shape().change_layout(best_layout);
                set_layout_tensors.insert(output);
            }

            for (auto output : n->outputs())
            {
                for (auto c : output->consumers())
                {
                    if (in_object(set_node_x, c))
                    {
                        next_set_nodes.push_back(c);
                    }
                }
            }
        }
        temp_set_nodes = next_set_nodes;
    }
    return set_node;
}

// Multiple outputs and multiple layouts,for example
// {{NHWC, FLAT},
//  {NHWC32, FLAT},
//    ...}
// First output spread priority
NodePtrList DataLayoutScheduler::layout_spread_multi_layout(NodePtr &header_node,
                                                            std::vector<NodePtr> &objects,
                                                            std::set<TensorPtr> &set_layout_tensors)
{
    NodePtrList set_node = {};
    LayoutCombinations output_layout_comb = m_outlayouts_map[header_node];
    LayoutCombinations input_layout_comb = m_inlayouts_map[header_node];
    auto match_layout = [](LayoutList output_layout_list, LayoutList best_layouts) -> bool
    {
        for (size_t i = 0; i < best_layouts.size(); ++i)
        {
            if (best_layouts[i] != output_layout_list[i])
                return false;
        }
        return true;
    };
    LayoutList best_layouts;
    set_node.push_back(header_node);
    for (size_t i = 0; i < header_node->outputs().size(); ++i)
    {
        auto output = header_node->outputs()[i];
        std::vector<NodePtr> set_node_x = {};
        LayoutList output_layout_list;
        std::map<DataLayout, int> output_layout_count;

        for (auto output_layout : output_layout_comb)
        {
            if (!match_layout(output_layout, best_layouts))
                continue;
            output_layout_list.push_back(output_layout[i]);
            output_layout_count[output_layout[i]] = 1;
        }
        LayoutList original_output_layout_list = output_layout_list;
        // 1. Find All consumer input LayoutCombinations intersection
        for (auto c : output->consumers())
        {
            LayoutList c_input_layout_list = export_valid_input_layouts(output, c, set_layout_tensors);
            if (c_input_layout_list.size() != 0)
            {
                intersection_layout(output_layout_list,
                                    c_input_layout_list,
                                    output_layout_count);
            }
        }
        if (output_layout_list.size() == 0)
        {
            output_layout_list.push_back(get_best_layout(output_layout_count, original_output_layout_list));
        }

        // 2. If keep layout, need to continue spreading.
        for (auto c : output->consumers())
        {
            if (can_be_spreaded(output, c, output_layout_list, objects, set_layout_tensors))
            {
                child_layout_spread(c, objects, set_layout_tensors, original_output_layout_list,
                                    output_layout_list, output_layout_count, set_node_x);
            }
        }
        DataLayout best_layout = get_best_layout(output_layout_count, original_output_layout_list);
        output->shape().change_layout(best_layout);
        set_layout_tensors.insert(output);

        std::vector<NodePtr> temp_set_nodes;
        for (auto c : output->consumers())
        {
            if (in_object(set_node_x, c))
            {
                temp_set_nodes.push_back(c);
            }
        }
        while (temp_set_nodes.size() > 0)
        {
            std::vector<NodePtr> next_set_nodes;
            for (auto n : temp_set_nodes)
            {
                LayoutCombinations n_output_layout_comb = m_outlayouts_map[n];
                LayoutCombinations n_input_layout_comb = m_inlayouts_map[n];
                LayoutList n_output_layout_list;
                for (auto output_layout : n_output_layout_comb)
                {
                    n_output_layout_list.push_back(output_layout[0]);
                }
                if (std::find(n_output_layout_list.begin(), n_output_layout_list.end(), best_layout) == n_output_layout_list.end())
                {
                    continue;
                }
                if (is_keeping_layout(n))
                {
                    if (std::find(n_output_layout_list.begin(), n_output_layout_list.end(), best_layout) == n_output_layout_list.end())
                    {
                        continue;
                    }
                    if (!all_input_can_change_layout(n, best_layout))
                    {
                        continue;
                    }
                }

                set_node.push_back(n);
                for (auto output : n->outputs())
                {
                    output->shape().change_layout(best_layout);
                    set_layout_tensors.insert(output);
                }

                for (auto output : n->outputs())
                {
                    for (auto c : output->consumers())
                    {
                        if (in_object(set_node_x, c))
                        {
                            next_set_nodes.push_back(c);
                        }
                    }
                }
            }
            temp_set_nodes = next_set_nodes;
        }
        best_layouts.push_back(best_layout);
    }
    return set_node;
}

std::vector<NodePtr> DataLayoutScheduler::layout_spread(NodePtr &header_node,
                                                        std::vector<NodePtr> &objects,
                                                        std::set<TensorPtr> &set_layout_tensors)
{
    LayoutCombinations output_layout_comb = m_outlayouts_map[header_node];
    std::vector<NodePtr> set_node = {};

    if (output_layout_comb.size() == 0)
    {
        set_node = layout_spread_empty_layout(header_node, objects, set_layout_tensors);
    }
    else if (output_layout_comb[0].size() == 1)
    {
        set_node = layout_spread_single_layout(header_node, objects, set_layout_tensors);
    }
    else
    {
        set_node = layout_spread_multi_layout(header_node, objects, set_layout_tensors);
    }

    if (set_node.empty())
    {
        set_node.push_back(header_node);
    }
    return set_node;
}

void DataLayoutScheduler::layout_locally_spread(std::vector<NodePtr> &objects,
                                                std::set<TensorPtr> &set_layout_tensors)
{
    while (objects.size())
    {
        NodePtr header_node = nullptr;
        for (size_t i = 0; i < objects.size(); ++i)
        {
            if (satisfy_input_node(objects[i], set_layout_tensors))
            {
                header_node = objects[i];
                break;
            }
        }

        if (header_node == nullptr)
        {
            WARN() << "header_node == nullptr, break.";
            break;
        }
        // do layout change()
        auto set_objs = layout_spread(header_node, objects, set_layout_tensors);
        for (auto obj : set_objs)
        {
            auto it = std::find(objects.begin(), objects.end(), obj);
            if (it != objects.end())
            {
                objects.erase(it);
            }
        }
    }
    return;
}

void DataLayoutScheduler::do_schedule()
{
    std::vector<NodePtr> objects;
    std::set<TensorPtr> set_layout_tensors;

    for (auto node : m_graph->nodes())
    {
        m_inlayouts_map[node] = node->get_attr<LayoutCombinations>(Constants::valid_input_layouts);
        auto out_layouts = node->get_attr<LayoutCombinations>(Constants::valid_output_layouts);
        m_outlayouts_map[node] = out_layouts;

        // only the nodes which has out_layouts>1 , and not OpType_Input
        if (node->type() == OpType_Constant || node->type() == OpType_Input)
        {
            if (node->params().count(Constants::layout))
            {
                std::string layout_name = node->params()[Constants::layout];
                DataLayout layout = string_to_datalayout(layout_name);
                if (layout != DataLayout_UnSupport)
                {
                    for (auto &t : node->outputs())
                    {
                        t->shape().change_layout(layout);
                    }
                }
            }
            for (auto &t : node->outputs())
            {
                set_layout_tensors.insert(t);
            }
            continue;
        }
        else if (node->type() == OpType_InternalConstant)
        {
            if (node->params().count(Constants::layout))
            {
                std::string layout_name = node->params()[Constants::layout];
                DataLayout layout = string_to_datalayout(layout_name);
                if (layout != DataLayout_UnSupport)
                {
                    node->outputs()[0]->shape().change_layout(layout);
                }
                set_layout_tensors.insert(node->outputs()[0]);
                continue;
            }
        }
        else if (node->type() == OpType_LayoutConvertor)
        {
            set_layout_tensors.insert(node->outputs()[0]);
            continue;
        }

        // when the output only have one layout, direct set the layout
        if (out_layouts.size() == 1)
        {
            auto layouts = out_layouts[0];
            if (layouts.size() == 1)
            {
                for (auto &t : node->outputs())
                {
                    t->shape().change_layout(layouts[0]);
                    set_layout_tensors.insert(t);
                }
            }
            else if (layouts.size() == node->outputs().size())
            {
                for (size_t i = 0; i < node->outputs().size(); ++i)
                {
                    node->outputs()[i]->shape().change_layout(layouts[i]);
                    set_layout_tensors.insert(node->outputs()[i]);
                }
            }
            else
            {
                objects.push_back(node);
            }
        }
        else
        {
            if (is_internal_op_keep_layout(node))
            {
                for (auto &t : node->outputs())
                {
                    set_layout_tensors.insert(t);
                }
            }
            else
            {
                objects.push_back(node);
            }
        }
    }

    if (m_debug)
    {
        for (auto node : m_graph->nodes())
        {
            auto valid_input_layouts = node->get_attr<LayoutCombinations>(Constants::valid_input_layouts);
            auto valid_output_layouts = node->get_attr<LayoutCombinations>(Constants::valid_output_layouts);
            bool is_keeping = is_keeping_layout(node);
            DEBUG() << "node:" << node->name();
            DEBUG() << "is_keeping:" << int(is_keeping);
            std::string in_layouts = "in_layouts [";
            for (auto valid_input_layout : valid_input_layouts)
            {
                in_layouts += "{";
                for (auto l : valid_input_layout)
                {
                    in_layouts += datalayout_to_string(l) + ",";
                }
                in_layouts += "},";
            }
            in_layouts += "]";
            std::string out_layouts = "out_layouts [";
            for (auto valid_output_layout : valid_output_layouts)
            {
                out_layouts += "{";
                for (auto l : valid_output_layout)
                {
                    out_layouts += datalayout_to_string(l) + ",";
                }
                out_layouts += "},";
            }
            out_layouts += "]";
            DEBUG() << in_layouts;
            DEBUG() << out_layouts;
        }
    }
    uint32_t loss = compute_datalayout_loss();
    INFO() << "Layout loss: " << loss;
    INFO() << "Layout scheduling ...";
    layout_locally_spread(objects, set_layout_tensors);
    auto score = insert_layout_converter();
    INFO() << "The layout loss for graph " << m_graph->name() << ": " << score;
}

bool DataLayoutScheduler::adjust_convertor_params()
{
    for (auto node : m_graph->nodes())
    {
        if (node->type() == OpType_LayoutConvertor)
        {
            std::string in_layout = datalayout_to_string(node->inputs()[0]->shape().layout());
            std::string param_layout = node->params()["from"];
            if (in_layout != param_layout)
            {
                // INFO() << node << " need adjust params!";
                node->params()["from"] = in_layout;
            }
        }
    }
    return true;
}

bool datalayout_schedule(GraphPtr g)
{
    if (get_target()->isainfo().type < ISA_ZYv2)
    {
        return false;
    }
    DataLayoutScheduler scheduler(g);
    scheduler.schedule();
    return true;
}

DataLayoutSchedule::DataLayoutSchedule()
{
    m_name = "DataLayoutSchedule";

    add("datalayout_schedule", datalayout_schedule);
    add("merge_layerconvertor_layerconvertor", pass::merge_layerconvertor_layerconvertor);
    add("reuse_layerconvertor", pass::reuse_layerconvertor);
    add("merge_node_layerconvertor", pass::merge_node_layerconvertor);
    add("remove_layout_convertor", remove_layout_convertor);
}
} // namespace aipubt

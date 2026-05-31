//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Xiaolong Zhang
 * @Date: 2022-12-09 11:19:07
 * @Last Modified by: Xiaolong Zhang
 * @Last Modified time: 2022-07-22 17:08:22
 */
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>
#include <math.h>

namespace aipubt
{

static auto update_output_shape = [](NodePtr n) {
    auto o_tensorPtr = n->outputs()[0];
    TensorShape NHWC_shape;
    if (o_tensorPtr->shape().dim() == 4)
    {
        NHWC_shape = {o_tensorPtr->shape()[0],
                      o_tensorPtr->shape()[2],
                      o_tensorPtr->shape()[3],
                      o_tensorPtr->shape()[1]};
    }
    else if (o_tensorPtr->shape().dim() == 3)
    {
        NHWC_shape = {
            o_tensorPtr->shape()[0],
            o_tensorPtr->shape()[2],
            o_tensorPtr->shape()[1],
        };
    }
    // else
    // {
    //     ERROR() << "[GSIM] layout transpose only support 2/3 dims for now.";
    // }

    n->outputs()[0]->shape().reshape(NHWC_shape);
};
static bool Input(NodePtr n)
{
    return true;
}
static bool Transpose(NodePtr n)
{

    return true;
}
static bool Convolution(NodePtr n)
{

    return true;
}
static bool ConvTranspose(NodePtr n)
{

    return true;
}
static bool Pooling(NodePtr n)
{

    return true;
}
static bool Concat(NodePtr n)
{
    auto o_tensor_ptr = n->outputs()[0];
    if (o_tensor_ptr->shape().dim() == 3)
    {
        n->params()["axis"] = 1;
    }
    else if (o_tensor_ptr->shape().dim() == 4)
    {
        n->params()["axis"] = 3;
    }
    else
    {
        // WARN() << "[GSIM] pls check eliminate_global_transpose pass[concat function], may missmatch from here.";
        return false;
    }

    update_output_shape(n);

    return true;
}
static bool Constant(NodePtr n)
{
    auto w0 = ComputeTensor::create(n->constants()["weights"]);
    auto new_w = w0.permute({0, 2, 3, 1});
    n->constants()["weights"] = new_w;

    update_output_shape(n);

    return true;
}
static bool Eltwise(NodePtr n)
{
    update_output_shape(n);
    return true;
}
static bool ArgMinMax(NodePtr n)
{
    n->params()["axis"] = 3;
    update_output_shape(n);
    return true;
}
static bool Cast(NodePtr n)
{
    update_output_shape(n);
    return true;
}
static bool Crop(NodePtr n)
{
    auto origin_crops = n->params()["crops"];
    // std::vector<std::pair<int32_t, int32_t>> crops;
    n->params()["crops"] = {origin_crops[0],
                            origin_crops[2],
                            origin_crops[3],
                            origin_crops[1]};
    update_output_shape(n);
    return true;
}
static bool Tile(NodePtr n)
{
    auto origin_tiles = n->params()["repeats"];
    // std::vector<std::pair<int32_t, int32_t>> crops;
    n->params()["repeats"] = {origin_tiles[0],
                              origin_tiles[2],
                              origin_tiles[3],
                              origin_tiles[1]};
    update_output_shape(n);
    return true;
}
static bool Resize(NodePtr n)
{

    return true;
}
static bool DepthToSpace(NodePtr n)
{

    return true;
}
static bool Reshape(NodePtr n)
{
    auto o_tensor_ptr = n->outputs()[0];
    if (o_tensor_ptr->shape().dim() == 4)
    {
        n->params()["shape"] = {
            o_tensor_ptr->shape()[0],
            o_tensor_ptr->shape()[2],
            o_tensor_ptr->shape()[3],
            o_tensor_ptr->shape()[1],
        };
    }
    else if (o_tensor_ptr->shape().dim() == 3)
    {
        // flat output
        if (n->inputs()[0]->shape().dim() != 4)
        {
            // WARN() << "[GSIM] eniminate layout transpose pass, may not consider this situation, pls debug.";
            return false;
        }

        n->params()["shape"] = {
            o_tensor_ptr->shape()[0],
            o_tensor_ptr->shape()[2],
            o_tensor_ptr->shape()[1],
        };
    }

    update_output_shape(n);

    return true;
}
static bool Negative(NodePtr n)
{
    update_output_shape(n);
    return true;
}
static bool Pad(NodePtr n)
{
    auto origin_crops = n->params()["pads"];
    // std::vector<std::pair<int32_t, int32_t>> crops;
    n->params()["pads"] = {origin_crops[0],
                           origin_crops[2],
                           origin_crops[3],
                           origin_crops[1]};
    update_output_shape(n);
    return true;
}
static bool Activation(NodePtr n)
{
    update_output_shape(n);
    return true;
}
static bool Softmax(NodePtr n)
{
    auto o_tensorPtr = n->outputs()[0];
    int axis = n->params()["axis"];
    if (o_tensorPtr->shape().dim() == 3 && axis != 2)
    {
        n->params()["axis"] = 2;
    }
    // else
    // {
    //     ERROR() << "softmax only support 3-dims([batch, m, line])";
    // }

    update_output_shape(n);
    return true;
}

// EGT: eliminate global transpose
using egt_func = std::function<bool(NodePtr n)>;
static std::map<OpType, egt_func> REFRESH_TABLE = {
    {OpType_Input, Input},
    {OpType_Transpose, Transpose},
    {OpType_Convolution, Convolution},
    {OpType_ConvTranspose, ConvTranspose},
    {OpType_Pooling, Pooling},
    {OpType_Concat, Concat},
    {OpType_InternalConcatenated, Concat},
    {OpType_Constant, Constant},
    {OpType_Eltwise, Eltwise},
    {OpType_ArgMinMax, ArgMinMax},
    {OpType_Cast, Cast},
    {OpType_Crop, Crop},
    {OpType_Tile, Tile},
    {OpType_Resize, Resize},
    {OpType_DepthToSpace, DepthToSpace},
    {OpType_Reshape, Reshape},
    {OpType_Negative, Negative},
    {OpType_Pad, Pad},
    {OpType_Activation, Activation},
    {OpType_Softmax, Softmax},

};

class EliminateGlobalTranspose
{
public:
    /**
     * @brief Spanning Tree based algorithm for global transpose eliminate.
     * 
     * 
     */
    enum Layout
    {
        UNKNOW = 0,
        CONFLICT_LAYOUT,
        NCHW,
        NHWC,
        NHWC2NCHW,
        NCHW2NHWC,
    };

    EliminateGlobalTranspose(GraphPtr g) : m_graph(g) {}
    ~EliminateGlobalTranspose() {}

    bool check_layout(void);
    bool check_ops(void);
    void dispatch_dfs(NodePtr cur_n, Layout cur_l);
    bool execute(void);
    bool refresh(void);

    static bool is_layout_convert(NodePtr n);
    static bool is_nhwc2nchw_transpose(NodePtr n);
    static bool is_nchw2nhwc_transpose(NodePtr n);
    static bool is_d2s_transpose(NodePtr n);

    // Utils
    int LAYER_ID(const NodePtr n) const;

private:
    GraphPtr m_graph;
    // std::vector<Layout>  m_all_layout;
    std::map<int, std::vector<Layout>> m_all_layout;
    std::vector<NodePtr> m_d2s_trans;    // the transpose nodes can be converted to d2s.
    std::vector<NodePtr> m_layout_trans; // can be removed transpose node.
    std::vector<NodePtr> m_unknow_trans; // unsupported layers for now.
};                                       // namespace aipubt

bool EliminateGlobalTranspose::check_ops()
{
    bool is_ops_supported = true;
    for (auto n : m_graph->nodes())
    {
        if (REFRESH_TABLE.count(n->type()) == 0)
        {
            // WARN() << "node type: [" << n->type() << "] not supported by GSIM PASS(eliminate global transpose.)";
            // return false;
            is_ops_supported = false;
        }
    }

    return is_ops_supported;
}

int EliminateGlobalTranspose::LAYER_ID(const NodePtr n) const
{
    // int ret = INT32_MAX;
    // auto it = std::find(m_inputs.begin(), m_inputs.end(), t);
    // if (it != m_inputs.end())
    // {
    //     ret = it - m_inputs.begin();
    // }
    // return ret

    return n->get_attr<int32_t>("layer_id");
}
void EliminateGlobalTranspose::dispatch_dfs(NodePtr cur_n, Layout cur_l)
{
    if (cur_n->type() == OpType_Transpose && is_layout_convert(cur_n))
    {
        return;
    }

    if (!contains(m_graph->nodes(), cur_n))
    {
        return;
    }

    int32_t layer_id = LAYER_ID(cur_n);

    if (m_all_layout.count(layer_id)) // already have layout.
    {
        if (m_all_layout[layer_id][0] != cur_l)
        {
            for (auto &l : m_all_layout[layer_id])
            {
                l = Layout::CONFLICT_LAYOUT;
            }

            // WARN() << "duplicated assign layout. layer_id="
            //         << layer_id
            //         << "origin layout:"
            //         << static_cast<int>(m_all_layout[layer_id][0])
            //         << "to new layout:" << static_cast<int>(cur_l);
        }

        return;
    }
    else
    {
        m_all_layout[layer_id] = std::vector<Layout>(cur_n->outputs().size(), cur_l);
    }

    for (auto p : cur_n->parents())
    {
        dispatch_dfs(p, cur_l);
    }

    for (auto c : cur_n->children())
    {
        dispatch_dfs(c, cur_l);
    }
}
bool EliminateGlobalTranspose::check_layout(void)
{
    /**
     * @brief dispatch the default layout for the graph.
     */
    for (auto n : m_graph->nodes())
    {
        if (n->type() != OpType_Transpose)
        {
            continue;
        }

        for (auto o_tensorPtr : n->outputs())
        {
            if (o_tensorPtr->shape().dim() != 4)
            {
                return false;
            }
        }

        if (is_layout_convert(n))
        {
            // exclude case transpose+concat(axis!=3)
            for (auto cn : n->children())
            {
                if (cn->type() == OpType_Concat && cn->params()["axis"] != 3)
                {
                    return false;
                }
            }

            auto layer_id = LAYER_ID(n);

            if (is_nchw2nhwc_transpose(n))
            {
                m_all_layout[layer_id] = std::vector<Layout>(n->outputs().size(), Layout::NCHW2NHWC);

                dispatch_dfs(n->parents()[0], Layout::NCHW);

                for (auto cn : n->children())
                {
                    dispatch_dfs(cn, Layout::NHWC);
                }
            }
            else if (is_nhwc2nchw_transpose(n))
            {
                m_all_layout[layer_id] = std::vector<Layout>(n->outputs().size(), Layout::NHWC2NCHW);

                dispatch_dfs(n->parents()[0], Layout::NHWC);

                for (auto cn : n->children())
                {
                    dispatch_dfs(cn, Layout::NCHW);
                }
            }

            // if io_tensor(input/output node) just
            if (n->parents()[0]->inputs().size())
            {
                m_layout_trans.push_back(n);
            }
        }
        else if (is_d2s_transpose(n))
        {
            m_d2s_trans.push_back(n);
        }
        // else
        // {
        //     WARN() << "unsupported transpose node, layer_id=" << LAYER_ID(n) << ", treat it as normal node.";
        // }
    }

    if (m_graph->nodes().size() != m_all_layout.size())
    {
        return false;
    }

    for (auto l : m_all_layout)
    {
        if (l.second[0] == Layout::CONFLICT_LAYOUT || l.second[0] == Layout::UNKNOW)
        {
            return false;
        }
    }

    int32_t nchw2nhwc_nums = 0;
    int32_t nhwc2nchw_nums = 0;
    for (auto &l : m_all_layout)
    {
        if (l.second[0] == Layout::NCHW2NHWC)
        {
            nchw2nhwc_nums++;
        }

        if (l.second[0] == Layout::NHWC2NCHW)
        {
            nhwc2nchw_nums++;
        }
    }
    if (nchw2nhwc_nums != nhwc2nchw_nums || nchw2nhwc_nums < 2 || nhwc2nchw_nums < 2)
    {
        return false;
    }

    return true;
}
bool EliminateGlobalTranspose::execute(void)
{
    if (m_d2s_trans.size())
    {
        // convert transpose to D2S.
        ;
    }

    if (m_layout_trans.size())
    {
        for (auto n : m_layout_trans)
        {
            auto inp_tensor = n->inputs()[0];
            auto out_tensor = n->outputs()[0];

            if (out_tensor->is_io_tensor())
            {
                continue;
            }

            for (auto cn : n->children())
            {
                for (auto inp : cn->inputs())
                {
                    auto idx = cn->remove_input(out_tensor);
                    if (idx == INT32_MAX)
                    {
                        continue;
                    }
                    cn->add_input(inp_tensor, idx);
                }
            }
            n->remove_input(inp_tensor);
            n->remove_output(n->outputs()[0]);
            m_graph->remove_node(n);
        }
    }

    return true;
}
bool EliminateGlobalTranspose::refresh(void)
{

    auto add_transpose = [](TensorPtr input_tensor, NodePtr origin_node) {
        auto node_name = input_tensor->name() + "_layout_transpose";
        // NOTE: OpType_Transpose4D is deprecated
        auto transpose_node = Node::create(node_name, OpType_Transpose);
        transpose_node->add_input(input_tensor);

        TensorShape output_shape(input_tensor->shape());
        std::swap(output_shape.w(), output_shape.c());
        std::swap(output_shape.h(), output_shape.w()); // from NHWC to NCHW;
        auto output_type = input_tensor->type();
        output_shape.set_layout(DataLayout_NHWC);
        auto output_tensor = std::make_shared<Tensor>(node_name + "/output", output_shape, output_type);
        output_tensor->quantization() = input_tensor->quantization();
        transpose_node->add_output(output_tensor);

        transpose_node->params()["perm"].push_back(0);
        transpose_node->params()["perm"].push_back(3);
        transpose_node->params()["perm"].push_back(1);
        transpose_node->params()["perm"].push_back(2);

        // transpose_node->set_attr(Constants::origin_node, origin_node->get_attr(Constants::origin_node));
        return transpose_node;
    };

    // auto convert_transpose_to_D2s = [m_graph](NodePtr n){};

    std::vector<TensorShape> origin_output_shape;
    for (auto t : m_graph->get_output_tensors())
    {
        origin_output_shape.push_back(t->shape());
    }

    std::vector<TensorPtr> graph_output_tensors;

    auto output_nodes = m_graph->get_output_nodes();
    // update params and shape.
    for (auto n : m_graph->nodes())
    {
        auto layer_id = static_cast<unsigned int>(LAYER_ID(n));
        if (m_all_layout.count(layer_id) == 0)
        {
            // replaced node: such as: transpose -> D2S.
            continue;
        }

        if (m_all_layout[layer_id][0] == Layout::NCHW)
        {
            // do params update.
            if (REFRESH_TABLE.count(n->type()) == 0)
            {
                // WARN() << "node type: " << n->type() << "not supported by GSIM PASS(eliminate global transpose.)";
                return false;
            }

            REFRESH_TABLE[n->type()](n);
        }
        else
        {
            continue;
        }

        if (contains(output_nodes, n) &&
            !contains(origin_output_shape, n->outputs()[0]->shape()))
        {
            // for the last node.
            auto end_trans = add_transpose(n->outputs()[0], n);
            graph_output_tensors.push_back(end_trans->outputs()[0]);
            m_graph->add_node(end_trans);
        }
    }

    for (auto t : m_graph->get_output_tensors())
    {
        if (!contains(graph_output_tensors, t))
        {
            graph_output_tensors.push_back(t);
        }
    }

    m_graph->record_output_tensors(graph_output_tensors);

    return true;
}
bool EliminateGlobalTranspose::is_layout_convert(NodePtr n)
{
    return is_nhwc2nchw_transpose(n) || is_nchw2nhwc_transpose(n);
}
bool EliminateGlobalTranspose::is_nhwc2nchw_transpose(NodePtr n)
{
    std::vector<int> perm = n->params()["perm"];
    if (perm.size() == 3)
    {
        std::vector<int> perm_nchw2nhwc_3 = {0, 1, 2};
        return perm == perm_nchw2nhwc_3;
    }
    else if (perm.size() == 4)
    {
        std::vector<int> perm_nchw2nhwc_4 = {0, 3, 1, 2};
        return perm == perm_nchw2nhwc_4;
    }

    return false;
}
bool EliminateGlobalTranspose::is_nchw2nhwc_transpose(NodePtr n)
{
    std::vector<int> perm = n->params()["perm"];
    if (perm.size() == 3)
    {
        std::vector<int> perm_nchw2nhwc_3 = {0, 2, 1};
        return perm == perm_nchw2nhwc_3;
    }
    else if (perm.size() == 4)
    {
        std::vector<int> perm_nchw2nhwc_4 = {0, 2, 3, 1};
        return perm == perm_nchw2nhwc_4;
    }

    return false;
}
bool EliminateGlobalTranspose::is_d2s_transpose(NodePtr n)
{
    std::vector<int> perm = n->params()["perm"];
    std::vector<int> perm_s2d = {2, 0, 3, 1};

    if (perm.size() == 4)
    {
        return perm == perm_s2d;
    }

    return false;
}

static inline bool criteria(EliminateGlobalTranspose &e)
{
    /*
     * @brief support float and quant IR.
     *        dispatch layout.
     */
    if (e.check_layout())
    {
        if (e.check_ops())
        {
            return true;
        }
    }

    return false;
}

bool eliminate_global_transpose(GraphPtr g)
{
    g->topological_sort();
    // g->to_svg("./gsim_orgin.svg");

    bool is_changed = true;
    auto egl = EliminateGlobalTranspose(g);

    if (!criteria(egl))
    {
        return false;
    }

    // remove transpose node.
    is_changed = egl.execute() && is_changed;

    // update params and shape
    is_changed = egl.refresh() && is_changed;

    return is_changed;
}
}; // namespace aipubt

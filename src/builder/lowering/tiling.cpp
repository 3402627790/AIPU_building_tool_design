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
 * @Date: 2020-12-30 15:18:29
 * @Last Modified by: Ryan Lai
 * @Last Modified time: 2024-09-09 20:28:00
 */
#include "tiling.h"
#include "builder/lowering/lowering.h"
#include "builder/scheduler.h"
#include "builder/target.h"
#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "transform/passes.h"
#include "transform/split_node_output_c.h"
#include "utils/utils.h"
#include <checker/checker.h>
#include <fstream>
#include <iostream>
#include <sstream>
namespace aipubt
{
#define TINY_TENSOR_THRES (1024 * 1024)

struct InputTSlice
{
    uint32_t offset;
    uint32_t size;
    uint32_t share = 0;
    uint32_t pad_left = 0;
    uint32_t pad_right = 0;
};
using InputTileInfo = std::vector<InputTSlice>;
struct WorkSpaceTileInfo
{
    uint32_t tile_num;
    uint32_t ws_size_original;
    uint32_t ws_size_tiled;
};

// one subgraph may have more than one worst_point
struct WorstInfo
{
    TensorPtr in_tensor = nullptr;
    uint32_t in_tile_size;
};
using WorstTileMap = std::map<NodePtr, WorstInfo>;

static std::vector<OpType> SUPPORT_TILE_OPS =
    {
        OpType_Convolution,
        OpType_DepthwiseConv,
        OpType_ConvTranspose,
        OpType_Eltwise,
        OpType_DepthToSpace,
        OpType_SpaceToDepth,
        OpType_Pooling,
        OpType_Activation,
        OpType_Cast,
        OpType_Concat,
        OpType_Resize,
        OpType_Add,
        OpType_Sub,
        OpType_Mul,
        OpType_SquaredDifference,
        OpType_BatchNorm,
        OpType_Softmax,
        OpType_RMSNorm,
        OpType_LayerNorm,
        OpType_MatMul,
        OpType_Where,
        OpType_Quantize,
        OpType_DeQuantize,
        OpType_Transpose,
        OpType_ArgMinMax,
        OpType_Crop,
        OpType_Pad
};

static std::set<OpType> SUPPORT_W_TILE_OPS =
    {
        OpType_Eltwise,
        OpType_Add,
        OpType_Softmax,
        OpType_MatMul,
        OpType_Transpose
};

static std::set<OpType> SUPPORT_C_TILE_OPS =
    {
        OpType_Convolution,
        OpType_ConvTranspose,
        OpType_FullyConnected,
        OpType_Eltwise,
        OpType_Pooling,
        OpType_Pad,
        OpType_Activation,
        OpType_Resize,
        OpType_Transpose,
        OpType_Reduce,
        OpType_Tile,
        OpType_BatchToSpace
};

static uint32_t g_num_nodes = 0;
static bool g_auto_mode = true; // to be removed

std::set<OpType> &get_disable_tiling_ops()
{
    static std::set<OpType> disable_tiling_ops;
    return disable_tiling_ops;
}

void tiling_set_constant(NodePtr node, NodePtr subnode, std::string name, int start, int size, int axis)
{
    if (!node->constants().count(name))
    {
        return;
    }

    ComputeTensor t = node->constants()[name];
    auto shape = t->shape();
    std::vector<ComputeTensor::Slice> slices;
    for (int i = 0; i < shape.dim(); ++i)
    {
        if (i == axis)
        {
            ComputeTensor::Slice slice(start, start + size);
            slices.push_back(slice);
        }
        else
        {
            ComputeTensor::Slice slice(0, shape[i]);
            slices.push_back(slice);
        }
    }

    ComputeTensor t_new = t[slices];
    TensorShape t_shape = t_new->shape();
    for (int i = 0; i < shape.dim(); ++i)
    {
        if (i == axis)
        {
            t_shape = change_shape_by_axis(t_shape, axis, size);
        }
    }
    t_new->shape() = t_shape;
    t_new->monopolize_data();
    subnode->constants()[name] = t_new;
}

void tiling_update_params(NodePtr n, std::string param, uint32_t axis, int32_t size_0, int32_t size_1)
{
    if (n && n->params().count(param) &&
        (uint32_t)n->outputs()[0]->shape().dim() > axis)
    {
        if (param == "shape")
        {
            n->params()["shape"][axis] = size_0;
        }
        else if (param == "crops")
        {
            n->params()["crops"][axis][0] = size_0;
            n->params()["crops"][axis][1] = size_1;
            if (n->params().count("ds_crops"))
            {
                std::vector<std::vector<Expr>> ds_crops = n->params()["ds_crops"];
                ds_crops[axis] = {tongue::Expr(size_0), tongue::Expr(size_1)};
                n->params()["ds_crops"] = ds_crops;
            }
        }
        else if (param == "repeats")
        {
            n->params()["repeats"][axis] = n->outputs()[0]->shape()[axis] / n->inputs()[0]->shape()[axis];
        }
    }
}

/**
 * @brief change_shape_by_axis
 *
 * for CP-23385: lib or profiler need using ds shape,
 * this function is a util to change both dynamic and statis shape.
 *
 * @param old_shape
 * @param axis
 * @param new_size
 * @return new_shape
 *
 * NOTE: is unsafe if shape contains real symbol! refine me after tiling support ds.
 */
TensorShape change_shape_by_axis(TensorShape shape, int axis, int new_size)
{
    // change shape
    shape[axis] = new_size;

    // change symbol shape
    std::vector<tongue::Expr> symbol = shape.symbol();
    if (axis < (int)symbol.size())
    {
        symbol[axis] = tongue::Expr(new_size);
        shape.symbol() = SymbolShape(symbol, shape.layout(), shape.indexing());
    }

    return shape;
}

void set_node_nums(GraphPtr g)
{
    g_num_nodes = g->nodes().size();
    DEBUG() << "num_nodes: " << g_num_nodes;
}

bool is_special_eltwise(const NodePtr &node)
{
    std::set<OpType> special_eltwise = {
        OpType_Add,
        OpType_Sub,
        OpType_Mul,
        OpType_SquaredDifference};
    if (special_eltwise.count(node->type()))
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool keep_isolate_when_scheduling(const NodePtr &n)
{
    return n->get_attr_deep(Constants::keep_isolate_when_scheduling);
}

static int get_broadcast_tensor_id(const NodePtr &node, int axis)
{
    if (!is_special_eltwise(node))
    {
        return -1;
    }

    auto shape_0 = node->inputs()[0]->shape();
    auto shape_1 = node->inputs()[1]->shape();
    auto out_shape = node->outputs()[0]->shape();

    /* no broadcasting */
    if (shape_0 == shape_1)
    {
        return -1;
    }

    int in0_axis = shape_0.dim() - out_shape.dim() + axis;
    int in1_axis = shape_1.dim() - out_shape.dim() + axis;

    if (in0_axis < 0 || shape_0[in0_axis] == 1)
    {
        return 0;
    }
    else if (in1_axis < 0 || shape_1[in1_axis] == 1)
    {
        return 1;
    }
    else
    {
        return -1;
    }
}

void convert_infos_remove_share(InputTileInfo &infos)
{
    int overlap = 0;
    for (auto &info : infos)
    {
        info.offset -= overlap;
        info.size += overlap;
        overlap = info.share;
        info.share = 0;
    }
}

void convert_infos_compute_share(InputTileInfo &infos)
{
    for (size_t i = 1; i < infos.size(); i++)
    {
        // overlap = this end - last begin;
        int32_t overlap = infos[i - 1].offset + infos[i - 1].size - infos[i].offset;
        // overlap may larger than size
        overlap = std::min(int32_t(infos[i].size), overlap);
        // overlap could be small than 0 if stride exist
        if (overlap > 1)
        {
            infos[i - 1].share = overlap;
            infos[i].size -= overlap;
            infos[i].offset += overlap;
            infos[i].share = 0;
        }
    }
    // remove empty slice
    while (!infos.empty() && infos.back().size == 0)
        infos.pop_back();
}

static bool get_real_ratio_float(NodePtr n, std::string key, double &ratio)
{
    auto input_shape = n->inputs()[0]->shape();
    auto output_shape = n->outputs()[0]->shape();
    auto refine_ratio = [](int i, int o, double &r) -> void {
        if (std::floor(i * r) != (double)o)
        {
            r = (double)(o + 0.1f) / i;
        }
    };
    if (key == "ratio_x")
    {
        ratio = (double)output_shape.w() / input_shape.w();
        refine_ratio(input_shape.w(), output_shape.w(), ratio);
    }
    else if (key == "ratio_y")
    {
        ratio = (double)output_shape.h() / input_shape.h();
        refine_ratio(input_shape.h(), output_shape.h(), ratio);
    }

    if (n->params().count(key))
    {
        ratio = n->params()[key];
        if (n->params()[key].t == NodeParamValue::NodeParamType_int || n->params()[key].t == NodeParamValue::NodeParamType_uint)
        {
            int _tmp_ratio = n->params()[key];
            ratio = (double)_tmp_ratio;
        }
    }

    if ((std::floor(ratio * input_shape.h()) != output_shape.h()) &&
        (std::floor(ratio * (input_shape.h() - 1)) != output_shape.h()))
    {
        return false;
    }
    return true;
}

static int get_real_ratio_int(NodePtr n, std::string key)
{
    int ratio = n->outputs()[0]->shape().h() / n->inputs()[0]->shape().h();
    if (n->params().count(key))
    {
        ratio = n->params()[key];
        if (n->params()[key].t == NodeParamValue::NodeParamType_float)
        {
            double _tmp_ratio = n->params()[key];
            ratio = (int)_tmp_ratio;
        }
    }
    return ratio;
}

static bool valid_scale_down(const NodePtr &n, int axis)
{
    if (n->type() != OpType_Resize)
    {
        return false;
    }
    if (n->inputs()[0]->shape()[axis] == 1 || n->outputs()[0]->shape()[axis] == 1)
    {
        return false;
    }
    auto i_size = n->inputs()[0]->shape()[axis];
    auto o_size = n->outputs()[0]->shape()[axis];
    // scale down
    if (i_size >= o_size && i_size % o_size == 0)
    {
        return true;
    }
    return false;
}

static bool valid_scale_up_nearest(const NodePtr &n, int axis)
{
    if (n->inputs()[0]->shape()[axis] == 1 || n->outputs()[0]->shape()[axis] == 1)
    {
        return false;
    }
    auto i_size = n->inputs()[0]->shape()[axis];
    auto o_size = n->outputs()[0]->shape()[axis];
    // scale up
    if (i_size < o_size)
    {
        std::string method = n->params()["method"];
        transform(method.begin(), method.end(), method.begin(), ::toupper);
        if (method != "NEAREST")
        {
            return false;
        }
        std::string nearest_mode = n->params()["nearest_mode"];
        transform(nearest_mode.begin(), nearest_mode.end(), nearest_mode.begin(), ::toupper);
        if (nearest_mode != "FLOOR")
        {
            return false;
        }
        if (o_size / i_size == 2 && o_size % i_size == 0)
        {
            return true;
        }
    }
    return false;
}

static bool valid_scale_up_bilinear(const NodePtr &n, int axis)
{
    if (n->inputs()[0]->shape()[axis] == 1 || n->outputs()[0]->shape()[axis] == 1)
    {
        return false;
    }
    auto i_size = n->inputs()[0]->shape()[axis];
    auto o_size = n->outputs()[0]->shape()[axis];
    // scale up
    if (i_size < o_size)
    {
        std::string method = n->params()["method"];
        transform(method.begin(), method.end(), method.begin(), ::toupper);
        if (method != "BILINEAR")
        {
            return false;
        }
        std::string mode = n->params()["mode"];
        transform(mode.begin(), mode.end(), mode.begin(), ::toupper);
        if (mode != "ASYMMETRIC")
        {
            return false;
        }
        double ratio_y = 0.f;
        double ratio_x = 0.f;
        if (!get_real_ratio_float(n, "ratio_y", ratio_y))
        {
            return false;
        }
        if (!get_real_ratio_float(n, "ratio_x", ratio_x))
        {
            return false;
        }
        // (ratio_y, ratio_x)
        static std::set<std::pair<double, double>> spec_ratio = {
            {2.f, 2.f},
            {4.f, 4.f},
        };
        for (auto &sr : spec_ratio)
        {
            if ((ratio_y == sr.first) && (ratio_x == sr.second))
            {
                return true;
            }
        }
        return false;
    }
    else
    {
        return false;
    }
}

static bool valid_scale_up(const NodePtr &n, int axis)
{
    auto valid_nearest = valid_scale_up_nearest(n, axis);
    auto valid_bilinear = valid_scale_up_bilinear(n, axis);
    return (valid_nearest || valid_bilinear);
}

static bool is_tile_actg_crop(const NodePtr &n)
{
    OpType real_type = get_real_op_type(n);
    if (OpType_Crop == real_type || OpType_InternalCrop == real_type)
    {
        // the corp may remove by pass [remove_crop]
        // so we check the output tenser here
        if (n->outputs()[0]->get_attr<bool>("ACTG_SHARE"))
            return true;
    }
    return false;
}

static bool is_tile_node(const NodePtr &n)
{
    auto is_tiled = [](const NodePtr &n) -> bool {
        return (n->loc() && (n->loc()->through_pass("tiling_node_along_axis") ||
                             n->loc()->through_pass("apply_split")));
    };

    auto sgn = SubGraphNode::from_node(n);
    // If all real node is tile node, the sgn is tile node
    if (sgn)
    {
        for (auto real_n : sgn->get_real_node())
        {
            if (!is_tile_node(real_n))
                return false;
        }
        return true;
    }
    else if (n->type() == OpType_LayoutConvertor)
    {
        bool all_parents_tiled = n->parents().size() > 0;
        for (auto p : n->parents())
        {
            if (!is_tiled(p))
            {
                all_parents_tiled = false;
            }
        }
        bool all_children_tiled = n->children().size() > 0;
        for (auto c : n->children())
        {
            if (!is_tiled(c))
            {
                all_children_tiled = false;
            }
        }
        if (all_parents_tiled || all_children_tiled)
        {
            return true;
        }
    }
    else if (n->type() == OpType_Reshape)
    {
        // CP-22242: if the reshape fused by this passes, it's not longer the tiled reshape
        if (n->loc() && n->loc()->through_pass("fuse_reshape") && n->inputs()[0]->is_io_tensor())
            return false;
    }
    else if (is_tile_actg_crop(n))
    {
        return true;
    }
    else
    {
        return is_tiled(n);
    }
    return false;
}

void append_schedule_flag(const TileSubGraphMap &tile_map)
{
    auto iter = tile_map.begin();
    while (iter != tile_map.end())
    {
        NodePtr merge_point = iter->first;
        merge_point->attrs()[Constants::keep_isolate_when_scheduling] = true;
        iter++;
    }
}

static int get_in_axis(const NodePtr &n, int o_axis, uint32_t tensor_id = 0)
{
    int i_axis = o_axis;
    if (n->type() == OpType_Transpose)
    {
        std::vector<int> perm = n->params()["perm"];
        i_axis = perm[o_axis];
    }
    else if (is_special_eltwise(n) && tensor_id <= 1)
    {
        auto out_shape = n->outputs()[0]->shape();
        i_axis = n->inputs()[tensor_id]->shape().dim() - out_shape.dim() + o_axis;
    }

    return i_axis;
}

static int get_out_axis(const NodePtr &n, int i_axis)
{
    int o_axis = i_axis;
    if (n->type() == OpType_Transpose)
    {
        std::vector<int> perm = n->params()["perm"];
        for (size_t i = 0; i < perm.size(); i++)
        {
            if (perm[i] == i_axis)
            {
                o_axis = i;
            }
        }
    }
    return o_axis;
}

static bool is_supported_op_manual(const NodePtr &n, int axis)
{
    auto op_type = get_real_op_type(n);

    if (axis == 0 && !check_manual_tile_n(n))
    {
        return false;
    }

    if ((axis == 1) && !contains(SUPPORT_TILE_OPS, op_type))
    {
        return false;
    }

    if ((axis == 2) && !contains(SUPPORT_W_TILE_OPS, op_type))
    {
        return false;
    }

    if (axis == 3 && !contains(SUPPORT_C_TILE_OPS, op_type))
        return false;

    if (axis == 3 && n->attrs().count("is_dwconv") && n->get_attr<bool>("is_dwconv") == true)
    {
        return false;
    }

    if ((axis == 1 || axis == 2) && op_type == OpType_Crop)
    {
        auto input_shape = n->inputs()[0]->shape();
        auto output_shape = n->outputs()[0]->shape();
        if (input_shape.dim() != 4) //nhwc
        {
            return false;
        }
        for (int i = 0; i < input_shape.dim() - 1; ++i) //crop c
        {
            if (input_shape[i] != output_shape[i])
            {
                return false;
            }
        }
    }

    if (op_type == OpType_Resize)
    {
        bool valid_down = valid_scale_down(n, axis);
        bool valid_up = valid_scale_up(n, axis);
        if (!valid_down && !valid_up)
        {
            return false;
        }
    }
    return true;
}

static bool is_supported_op_channel(const NodePtr &n)
{
    if (!contains(SUPPORT_C_TILE_OPS, n->type()))
    {
        return false;
    }
    auto out_size = n->outputs()[0]->shape().size();
    auto inp_size = n->inputs()[0]->shape().size();
    auto out_c = n->outputs()[0]->shape()[-1];
    auto inp_c = n->inputs()[0]->shape()[-1];
    auto fmap_size = out_size / out_c;
    int32_t core_num = Target::get_the_target()->core_num();
    int32_t pe_align = 256 / Tensor::dtype_bits(n->inputs()[0]->type());
    // auto gm_size = BuilderConfig::get<uint32_t>(Constants::gm_size);

    if (is_tile_node(n))
    {
        return false;
    }
    if (out_size == 0)
    {
        return false;
    }
    if (n->type() == OpType_Eltwise || n->type() == OpType_Pooling)
    {
        for (auto p : n->parents())
        {
            if (p->type() != OpType_Convolution)
            {
                return false;
            }
        }
    }

    // DEBUG() << n->name() << " " << inp_size << " " << out_size;
    if (n->type() == OpType_FullyConnected)
    {
        //        if (out_size < 128*128 || inp_size < 128*128) // check param later
        //        {
        //            return false;
        //        }
        // disable check for mobilebert fc
        if (g_num_nodes > 2000 && (out_c / fmap_size) < 2)
        {
            return false;
        }
    }
    if (n->type() == OpType_Convolution || n->type() == OpType_ConvTranspose)
    {
        if (fmap_size < 50)
        {
            return false;
        }
        if (out_size < 400 * 200 || inp_size <= 256 * 160)
        {
            if (out_c < 256 || inp_c < 256)
            {
                return false;
            }
        }
        // disable check for mobilebert fc
        if (g_num_nodes > 2000 && (out_c / fmap_size) < 2)
        {
            return false;
        }
    }

    if (n->type() == OpType_Pad)
    {
        std::vector<std::vector<int>> pads = n->params()["pads"];
        int pad_left = pads.back()[0];
        int pad_right = pads.back()[1];
        // if pad channel, don't split channel
        // TODO: it could be tiled but need cal pad value
        if (pad_left != 0 || pad_right != 0)
            return false;
    }

    if (out_size < core_num * pe_align) // c32 for int8, c16 for int16, c8 for int32, assign to each core
    {
        return false;
    }
    auto output_type = n->outputs()[0]->type();
    uint32_t alignment = ((output_type == Tensor::INT8) || (output_type == Tensor::UINT8)) ? 32 : 16;
    auto num = out_c / alignment;
    return num > 1;
}

/**
 * @brief is_supported_op
 * @param n
 * @param axis -> in axis
 * @return
 */
bool is_supported_op(const NodePtr &n, int32_t axis, bool is_supported_op)
{
    auto is_valid_conv = [](const NodePtr &n, int32_t axis) -> bool {
        uint32_t i_size = 0;
        uint32_t o_size = 0;
        for (auto t : n->inputs())
        {
            i_size += t->mem_size();
        }
        for (auto t : n->outputs())
        {
            o_size += t->mem_size();
        }
        if (i_size < 512 * 1024 && o_size < 512 * 1024)
        {
            if (n->outputs().size() > 0 && n->outputs()[0]->shape()[axis] < 20)
            {
                return false;
            }
        }
        return true;
    };

    if (get_disable_tiling_ops().count(n->type()) || n->get_attr<int>("disable_tiling") == 1)
    {
        return false;
    }

    if (n->is_dynamic())
    {
        SymbolShape shape = n->outputs()[0]->shape().symbol();
        if (n->type() == OpType_FullyConnected && axis == 3)
        {
            if (!shape.c()->is_constant())
            {
                return false;
            }
        }
        else if(axis >= shape.dim() || !shape[axis]->is_constant())
        {
            return false;
        }
    }

    if (!is_supported_op_manual(n, axis))
    {
        return false;
    }

    /* to be removed */
    if (!g_auto_mode || is_supported_op)
    {
        return true;
    }

    /* Transpose
     * auto tile:   support tile without trans axis
     * manual tile: support all(on going)
     */
    if (n->type() == OpType_Transpose)
    {
        auto arch = aipubt::get_target()->isainfo().type;
        // before X3, transpose not split is faster: e.g. codeformer
        if (arch < aipubt::ISA_ZYv6)
        {
            return false;
        }

        if (axis == 2 && n->get_attr<int>("w_tiling"))
        {
            return true;
        }

        int out_axis = get_out_axis(n, axis);
        if (out_axis != axis)
        {
            return false;
        }
    }

    if (axis == 2 && !is_tile_node(n))
    {
        return false;
    }

    if (3 == axis)
    {
        return is_supported_op_channel(n);
    }

    if (axis == 0 && !check_auto_tile_n(n))
    {
        return false;
    }

    if (g_num_nodes > 2000)
    {
        return false;
    }
    // CSM-1020: remove this restriction for LLM
    // if (g_num_nodes > 1000)
    // {
    //     auto out_dim = n->outputs()[0]->shape().dim();
    //     if (out_dim != 4)
    //     {
    //         return false;
    //     }
    // }
    uint32_t o_size = 0;
    for (auto t : n->outputs())
    {
        o_size += t->mem_size();
    }
    if (o_size < 15 * 1024)
    {
        return false;
    }
    if (n->type() == OpType_Pooling || n->type() == OpType_Concat)
    {
        if (o_size < 300 * 1024 && n->outputs()[0]->shape()[axis] < 20)
        {
            return false;
        }
    }
    for (auto t : n->outputs())
    {
        if (t->shape().dim() <= 1 || t->shape().dim() > 5 || t->shape().dim() <= axis || t->shape()[axis] == 1)
        {
            return false;
        }
    }
    if (n->type() == OpType_Convolution || n->type() == OpType_DepthwiseConv)
    {
        auto valid_conv = is_valid_conv(n, axis);
        if (!valid_conv)
        {
            return false;
        }
    }
    if (n->type() == OpType_Eltwise)
    {
        for (auto p : n->parents())
        {
            if (p->type() == OpType_Convolution)
            {
                auto valid_conv = is_valid_conv(p, axis);
                if (!valid_conv)
                {
                    return false;
                }
            }
        }

        if (n->children().size() == 1 && n->children()[0]->type() == OpType_FullyConnected)
        {
            return false;
        }
    }
    if (n->type() == OpType_Mul || n->type() == OpType_Eltwise)
    {
        for (auto p : n->parents())
        {
            if (p->type() == OpType_Activation)
            {
                if (p->inputs()[0]->shape()[0] > 256 || is_tile_node(p) || is_tile_node(n))
                {
                    p->set_attr("disable_tiling", 1);
                    return false;
                }
            }
        }
    }
    if (n->type() == OpType_Activation && n->children().size() == 1)
    {
        auto c = n->children()[0];
        auto c_type = c->type();
        if (c_type == OpType_Mul || c_type == OpType_Eltwise)
        {
            if (n->inputs()[0]->shape()[0] > 256 || is_tile_node(n) || is_tile_node(c))
            {
                c->set_attr("disable_tiling", 1);
                return false;
            }
        }
    }
    if (n->type() == OpType_Add && n->outputs()[0]->consumers().size() == 2)
    {
        if ((n->outputs()[0]->consumers()[0]->type() == OpType_Add && n->outputs()[0]->consumers()[1]->type() == OpType_RMSNorm) ||
            (n->outputs()[0]->consumers()[0]->type() == OpType_RMSNorm && n->outputs()[0]->consumers()[1]->type() == OpType_Add))
        {
            return false;
        }
    }
    if (n->type() == OpType_Cast)
    {
        for (auto t : n->outputs())
        {
            if (t->shape().dim() != 4)
            {
                return false;
            }
        }
    }
    // remove me if crop support fp32/64
    if (n->type() == OpType_Cast)
    {
        for (auto t : n->inputs())
        {
            if (t->type() == Tensor::FP32 || t->type() == Tensor::FP64)
            {
                return false;
            }
        }
    }
    if (n->type() == OpType_DepthToSpace)
    {
        for (auto t : n->outputs())
        {
            if (t->shape().dim() != 4)
            {
                return false;
            }
            // optimize for AIB mv3_depth
            if (t->shape().n() > 1)
            {
                return false;
            }
        }
    }
    if (n->type() == OpType_ConvTranspose)
    {
        std::string postfix = "_y";
        if (axis == 2)
        {
            postfix = "_x";
        }
        int default_p[3] = {1, 1, 1};
        std::vector<std::string> default_keys = {"kernel", "stride", "dilation"};
        for (unsigned int i = 0; i < default_keys.size(); ++i)
        {
            if (n->params().count(default_keys[i] + postfix))
            {
                default_p[i] = n->params()[default_keys[i] + postfix];
            }
        }
        auto kernel = default_p[0];
        auto stride = default_p[1];
        if (kernel != stride)
        {
            return false;
        }
    }
    if (n->type() == OpType_Concat || n->type() == OpType_InternalConcatenated)
    {
        if (n->params().count("axis"))
        {
            int32_t axis_concat = n->params()["axis"];
            if (axis_concat == axis)
            {
                return false;
            }
            // optimize for AIB mv3_depth
            // if (axis_concat == 0 && n->parents().size() == 1 && n->parents()[0]->type() == OpType_Split)
            else if (axis_concat == 0)
            {
                return false;
            }
        }
    }
    if (n->type() == OpType_Split && n->params().count("axis"))
    {
        int32_t axis_op = n->params()["axis"];
        if (axis_op == axis)
        {
            return false;
        }
    }

    if (n->type() == OpType_Crop && n->params().count("crops"))
    {
        std::vector<std::vector<int32_t>> crops = n->params()["crops"];
        int crop_dim = crops.size();
        auto i_shape = n->inputs()[0]->shape();

        if (axis < crop_dim && (crops[axis][0] != 0 || crops[axis][1] != i_shape[axis]))
        {
            return false;
        }
    }

    if (n->type() == OpType_Pad && n->params().count("pads"))
    {
        std::vector<std::vector<int32_t>> pads = n->params()["pads"];

        if(pads[axis][0] != 0)
            return false;
        if(pads[axis][1] != 0)
            return false;
    }

    if (n->type() == OpType_Resize)
    {
        bool valid_down = valid_scale_down(n, axis);
        bool valid_up = valid_scale_up(n, axis);
        if (!valid_down && !valid_up)
        {
            return false;
        }
    }

    auto is_RGB_CHW_Conv = [](const NodePtr &trans, const NodePtr &conv) -> bool {
        if (trans->type() != OpType_Transpose || conv->type() != OpType_Convolution)
        {
            return false;
        }
        bool is_target_trans = false;
        bool is_target_conv = false;
        if (trans->outputs()[0]->shape().dim() == 4)
        {
            std::vector<int> perm = trans->params()["perm"];
            if ((perm[0] == 0) && (perm[1] == 2) && (perm[2] == 3) && (perm[3] == 1))
            {
                is_target_trans = true;
            }
            if (trans->outputs()[0]->shape()[3] <= 4)
            {
                is_target_conv = true;
            }
        }
        return (is_target_trans && is_target_conv);
    };

    if (n->type() == OpType_Pooling)
    {
        if (n->parents().size() > 0 && n->parents()[0]->parents().size() > 0)
        {
            if (is_RGB_CHW_Conv(n->parents()[0]->parents()[0], n->parents()[0]))
            {
                return false;
            }
        }
    }

    if (n->type() == OpType_Convolution)
    {
        if (n->parents().size() > 0)
        {
            if (is_RGB_CHW_Conv(n->parents()[0], n))
            {
                return false;
            }
        }
    }

    //softmax & bn
    if (n->type() == OpType_Softmax || n->type() == OpType_BatchNorm || n->type() == OpType_ArgMinMax)
    {
        if (n->params().count("axis"))
        {
            int32_t local_axis = n->params()["axis"];
            if (local_axis <= axis) // 0 1 2 3 vs 1
            {
                return false;
            }
        }
    }

    // rmsnorm or layernorm
    if (n->type() == OpType_RMSNorm || n->type() == OpType_LayerNorm)
    {
        if (n->params().count("axis"))
        {
            std::vector<int> axis_rms = n->params()["axis"];
            for (auto axis_rms_local : axis_rms)
            {
                if (axis_rms_local <= axis) // 0 1 2 3 vs 1
                {
                    return false;
                }
            }
        }
    }

    if (n->type() == OpType_Where)
    {
        if (n->parents().size() != 3)
        {
            return false;
        }
    }

    if (n->type() == OpType_Reshape)
    {
        // skip tile axis == reshape axis
        if (0 == axis && n->inputs()[0]->shape()[0] != n->outputs()[0]->shape()[0])
            return false;
    }

    if (n->type() == OpType_Tile)
    {
        auto i_shape = n->inputs()[0]->shape();
        auto o_shape = n->outputs()[0]->shape();

        if (axis >= i_shape.size())
        {
            return false;
        }

        if (i_shape[axis] != o_shape[axis] && i_shape[axis] != 1)
        {
            return false;
        }
    }

    if (n->type() == OpType_TopK)
    {
        if (n->params().count("axis") && n->params()["axis"] == axis)
        {
            return false;
        }
    }
    if (n->type() == OpType_Gather)
    {
        if (n->params().count("axis") && n->params()["axis"] != axis && n->inputs()[1]->shape().dim() != 1)
        {
            return false;
        }
    }

    if (n->type() == OpType_MatMul)
    {
        auto dims = n->outputs()[0]->shape().dim();
        if (dims < 3 || (dims == 3 && axis != 0))
        {
            return false;
        }
    }

    return true;
}

static uint32_t get_shared_size(const InputTSlice &a, const InputTSlice &b)
{
    uint32_t inter_size;
    auto start = std::max(a.offset, b.offset);
    auto end = std::min(a.offset + a.size, b.offset + b.size);
    inter_size = end > start ? (end - start) : 0;
    return inter_size;
};

// find the info with max shared part
static uint32_t get_best_info_id(const InputTSlice &info, const InputTileInfo &union_infos)
{
    uint32_t best_id = 0;
    uint32_t best_size = 0;
    for (size_t j = 0; j < union_infos.size(); ++j)
    {
        auto union_info = union_infos[j];
        auto size = get_shared_size(info, union_info);
        if (size > best_size)
        {
            best_id = j;
            best_size = size;
            if (best_size == info.size)
            {
                break;
            }
        }
    }
    return best_id;
}

/**
 * @brief get_init_output_infos
 * @param n
 * @param axis
 * @param out_tile_size
 * @return TileInfo: size & offset
 *
 * if actg disable:
 * the overlap of left & right is grow slow than center
 * e.g. 1
 * conv 3x3         24   24   24
 * conv 3x3         23   22   23
 * conv 3x3         22   20   22 <--- this func cal these nums
 *                                    In this case
 *                                    22+20+22 better than 22+22+20
 *
 *
 */
static TileInfo get_init_output_infos(NodePtr n, int axis, uint32_t out_tile_size, bool actg)
{
    uint32_t out_size = n->outputs()[0]->shape()[axis];
    uint32_t tile_num = (out_size + out_tile_size - 1) / out_tile_size;
    // uint32_t out_tile_size = get_ceil(out_size, global_tile_num);

    TileInfo out_infos(tile_num);
    for (size_t i = 0; i < tile_num; ++i)
    {
        out_infos[i].size = out_tile_size;
    }

    if (actg)
    {
        // make first smaller
        out_infos[0].size = out_size - out_tile_size * (tile_num - 1);
    }
    else
    {
        out_infos[tile_num - 1].size = out_size - out_tile_size * (tile_num - 1);
        // swap big num to right
        // 22+22+20 -> 22+20+22
        // 12+12+10+10+10+10 -> 12+10+10+10+10+12
        if (tile_num - 1 > 1 && out_infos[tile_num - 1].size < out_infos[1].size)
            std::swap(out_infos[tile_num - 1].size, out_infos[1].size);
    }

    // fill offset
    uint32_t offset = 0;
    for (size_t i = 0; i < tile_num; ++i)
    {
        out_infos[i].offset = offset;
        offset += out_infos[i].size;
        // init output share nothing
        out_infos[i].share = 0;
    }

    return out_infos;
}

/**
 * @brief get_input_infos
 * @param n
 * @param axis
 * @param out_infos
 * @param actg: if enable, move overlap to inputslice->share
 * @return
 */
static InputTileInfo get_input_infos(NodePtr n, int axis, const TileInfo &out_infos, bool actg)
{
    size_t tile_num = out_infos.size();
    InputTileInfo in_infos;

    // if axis == 0
    std::string pad_key_0 = "pad_batch";
    std::string pad_key_1 = "pad_batch";
    std::string postfix = "_b";

    if (axis == 1)
    {
        pad_key_0 = "pad_top";
        pad_key_1 = "pad_bottom";
        postfix = "_y";
    }
    else if (axis == 2)
    {
        pad_key_0 = "pad_left";
        pad_key_1 = "pad_right";
        postfix = "_x";
    }
    int default_p[4] = {1, 1, 1, 1};

    double ratio = 1.0;
    std::vector<std::string> default_keys = {"kernel", "stride", "dilation", "block_size"};
    for (unsigned int i = 0; i < default_keys.size(); ++i)
    {
        if (0 == axis)
        {
            default_p[i] = 1;
        }
        else if (n->params().count(default_keys[i] + postfix))
        {
            default_p[i] = n->params()[default_keys[i] + postfix];
        }
    }
    if (n->type() == OpType_Resize)
    {
        std::string key = "ratio" + postfix;
        if (n->params().count(key))
        {
            if (n->params()[key].t == NodeParamValue::NodeParamType_float)
            {
                ratio = n->params()[key];
            }
            else if (n->params()[key].t == NodeParamValue::NodeParamType_int ||
                     n->params()[key].t == NodeParamValue::NodeParamType_uint)
            {
                int ratio_int = n->params()[key];
                ratio = (double)ratio_int;
            }
        }
        else
        {
            ratio = n->outputs()[0]->shape()[axis] * 1.0 / n->inputs()[0]->shape()[axis];
        }
    }
    int pad_0 = 0; //, pad_1 = 0;
    if (n->params().count(pad_key_0))
    {
        pad_0 = n->params()[pad_key_0];
        // pad_1 = n->params()[pad_key_1];
    }
    // conv/dwconv/eltwise/pooling
    // auto out_size_compute = [](int size, int k, int stride, int pad, int dilated = 1) -> int {
    //     return (size + pad - (k - 1) * dilated - 1) / stride + 1;
    // };
    auto input_size_compute = [](int size, int k, int stride, int pad, int dilated = 1) -> int {
        int dk = (k - 1) * (dilated - 1) + k; // dk = 1 + (k - 1) * dilated;
        return (size - 1) * stride - pad + dk;
    };

    // deconv
    // auto out_size_compute = [](int size, int k, int stride, int pad, int dilated = 1) -> int {
    //     return (size - 1) * stride - pad + dk;
    // };
    auto input_size_compute_deconv = [](int size, int k, int stride, int pad, int dilated = 1) -> int {
        int dk = (k - 1) * (dilated - 1) + k; // dk = 1 + (k - 1) * dilated;
        return (size + pad - dk) / stride + 1;
    };

    // d2s
    // auto out_size_compute = [](int size, int block_size) -> int {
    //     return (size * block_size);
    // };
    auto input_size_compute_d2s = [](int size, int block_size) -> int {
        return (size / block_size);
    };

    auto input_size_compute_s2d = [](int size, int block_size) -> int {
        return (size * block_size);
    };

    // resize
    auto input_size_compute_resize = [](int size, double ratio) -> int {
        return (int)(size / ratio);
    };

    for (size_t i = 0; i < tile_num; ++i)
    {
        InputTSlice in_info;
        // offset

        int32_t offset;
        if (n->type() == OpType_ConvTranspose)
        {
            offset = (out_infos[i].offset + pad_0) / default_p[1];
        }
        else if (n->type() == OpType_DepthToSpace)
        {
            offset = out_infos[i].offset / default_p[3];
        }
        else if (n->type() == OpType_SpaceToDepth)
        {
            offset = out_infos[i].offset * default_p[3];
        }
        else if (n->type() == OpType_Resize)
        {
            offset = (int32_t)(out_infos[i].offset / ratio);
        }
        else
        {
            offset = out_infos[i].offset * default_p[1] - pad_0;
        }

        if (offset < 0)
        {
            in_info.pad_left = 0 - offset;
            offset = 0;
        }

        in_info.offset = offset;

        // size
        int32_t size;
        if (0 == axis)
        {
            // batch split no need any pad.
            size = out_infos[i].size;
        }
        else if (n->type() == OpType_ConvTranspose)
        {
            size = input_size_compute_deconv(out_infos[i].size, default_p[0], default_p[1], in_info.pad_left, default_p[2]);
        }
        else if (n->type() == OpType_DepthToSpace)
        {
            size = input_size_compute_d2s(out_infos[i].size, default_p[3]);
        }
        else if (n->type() == OpType_SpaceToDepth)
        {
            size = input_size_compute_s2d(out_infos[i].size, default_p[3]);
        }
        else if (n->type() == OpType_Resize)
        {
            size = input_size_compute_resize(out_infos[i].size, ratio);
            auto valid_bilinear = valid_scale_up_bilinear(n, axis);
            if (valid_bilinear && i < (tile_num - 1))
            {
                size += 1;
            }
        }
        else
        {
            size = input_size_compute(out_infos[i].size,
                                      default_p[0], default_p[1], in_info.pad_left, default_p[2]);
        }

        auto old_in_size = is_special_eltwise(n) ? n->outputs()[0]->shape()[axis] : n->inputs()[0]->shape()[get_in_axis(n, axis)];
        int32_t max_size = old_in_size - in_info.offset;
        if (size > max_size)
        {
            in_info.pad_right = size - max_size;
            size = max_size;
        }

        in_info.size = size;

        if (size > 0) // TODO: assert here
            in_infos.push_back(in_info);
        // else:  check size_compute_xxx;
    }

    if (actg) // calculate share part of slice
    {
        convert_infos_compute_share(in_infos);
    }

    return in_infos;
}

static void config_tiled_graph(GraphPtr tiled_graph)
{
    for (auto tnode : tiled_graph->nodes())
    {
        if (tnode->type() == OpType_Resize)
        {
            double ratio_y = 0.f;
            double ratio_x = 0.f;

            get_real_ratio_float(tnode, "ratio_y", ratio_y);
            tnode->params()["ratio_y"] = ratio_y;
            get_real_ratio_float(tnode, "ratio_x", ratio_x);
            tnode->params()["ratio_x"] = ratio_x;

            if (tnode->params().count("size"))
            {
                tnode->params().erase("size");
            }
        }
    }
}

static void set_ds_crops(GraphPtr g, NodePtr crop, int axis, TensorPtr input, TensorPtr output)
{
    if (g->is_dynamic())
    {
        std::vector<std::vector<Expr>> ds_crops;
        for (int i = 0; i < input->shape().dim(); ++i)
        {
            if (i == axis)
            {
                ds_crops.push_back({tongue::Expr(static_cast<uint32_t>(crop->params()["crops"][i][0])),
                                    tongue::Expr(static_cast<uint32_t>(crop->params()["crops"][i][1]))});
            }
            else
            {
                ds_crops.push_back({0, input->shape().symbol()[i]});
            }
        }
        crop->params()["ds_crops"] = ds_crops;
        crop->infer_shape(true);
    }
}

/**
 * @brief tiling_node_along_axis
 * @param g
 * @param n
 * @param axis -> out axis
 * @param out_infos
 * @param target_in_infos_list
 * 		should be empty if not branch of input
 * @param actg
 * 			try to enable actg, some node ignore this flag
 * 			e.g. input
 * @return
 */
static NodePtrList tiling_node_along_axis(GraphPtr g, NodePtr n, int axis,
                                          TileInfo &out_infos,
                                          std::vector<InputTileInfo> target_in_infos_list,
                                          bool actg)
{
    NodePtrList tile_nodes;
    if (out_infos.size() <= 1)
    {
        return tile_nodes;
    }

    // if tile in n:
    // * not crop
    // * not overlap
    if (axis == 0 && n->type() != OpType_Transpose)
    {
        std::vector<int> batchs(out_infos.size());
        for (size_t i = 0; i < out_infos.size(); i++)
        {
            batchs[i] = out_infos[i].size;
            // not need offest for split batch
        }
        tile_nodes = apply_tile_n(n, batchs);
        return tile_nodes;
    }

    // if tile c
    if (axis >= 3)
    {
        return tile_nodes;
    }

    std::string tsg_name = n->get_attr(Constants::tsg_name);
    std::string pad_key_0 = "pad_top";
    std::string pad_key_1 = "pad_bottom";
    std::string postfix = "_y";
    if (axis == 2)
    {
        pad_key_0 = "pad_left";
        pad_key_1 = "pad_right";
        postfix = "_x";
    }
    auto inps = n->inputs();
    auto out = n->outputs()[0];
    size_t tile_num = out_infos.size();

    /* get in_infos first and update out_infos if necessary */
    auto in_infos = get_input_infos(n, axis, out_infos, actg);
    if (n->type() == OpType_Resize)
    {
        double ratio_y = 0.;
        get_real_ratio_float(n, "ratio_y", ratio_y);

        /* fix overlaps */
        for (size_t i = 0; i < in_infos.size(); i++)
        {
            out_infos[i].size = (int)(in_infos[i].size * ratio_y);
        }
    }

    auto bc_tid = get_broadcast_tensor_id(n, axis);

    for (size_t i = 0; i < target_in_infos_list.size(); ++i)
    {
        if (is_special_eltwise(n) && bc_tid == (int)i)
        {
            InputTileInfo in_infos_tmp;
            InputTSlice slice_tmp;
            slice_tmp.offset = 0;
            slice_tmp.size = 0;
            for (size_t j = 0; j < tile_num; ++j)
            {
                in_infos_tmp.push_back(slice_tmp);
            }
            target_in_infos_list[i] = in_infos_tmp;
        }
        else if (target_in_infos_list[i].size() > tile_num)
        {
            WARN() << "Invalid tile_num."
                   << " in_tile_num: " << target_in_infos_list[i].size()
                   << ", out_tile_num: " << tile_num
                   << ", expected in_tile_num <= out_tile_num";
            target_in_infos_list[i] = in_infos;
        }
        else if (target_in_infos_list[i].size() == 0)
        {
            if (n->type() == OpType_MatMul)
            {
                if ((axis == 1 && inps[i]->shape()[1] == 1) ||
                    (axis == 2 && i == 1))
                {
                    break;
                }
            }
            target_in_infos_list[i] = in_infos;
        }
    }

    //                n-parent/concat
    //                       |
    //  in_crop_share*   in_crop_pre
    //            \          |
    //             \       in_crop
    //              \        |
    //               `-- in_concat*
    //                       |
    //                   this_node
    //                       |
    //                    out_crop
    //                       |
    //                    \  |   /
    //                     concat
    //
    // Note:
    // * node (in_crop_share and in concat) exist when enable actg
    // * in_crop_pre: should merged with parent concat, shape depends on parent slice
    // * in_crop    : shape depends on this slice
    // * out_crop   : shape depends on this slice
    //

    std::vector<TensorPtrList> crop_pre_outs_list;
    for (size_t j = 0; j < target_in_infos_list.size(); ++j)
    {
        auto target_in_infos = target_in_infos_list[j];
        auto target_tile_num = target_in_infos.size();
        auto inpt = inps[j];
        auto in_axis = get_in_axis(n, axis, j);

        if (inpt->is_io_tensor()) // disable actg for input tensor force
            actg = false;

        // force close actg for some node
        if (!actg)
        {
            convert_infos_remove_share(target_in_infos);
            convert_infos_remove_share(in_infos);
        }

        TensorPtrList crop_pre_outs;
        for (size_t i = 0; i < target_tile_num; ++i) // each input
        {
            // input id + tile slice num id
            std::string name_id = std::to_string(j) + "_" + std::to_string(i);
            auto target_in_info = target_in_infos[i];

            NodePtr in_crop_pre = Node::create(n->name() + "/tile/in_crop_pre/" + name_id, OpType_Crop);
            ADD_LOCATION_BY_NODE(in_crop_pre, n);
            g->add_node(in_crop_pre);
            auto in_crop_pre_shape = inpt->shape();

            if (target_in_info.size > 0)
            {
                in_crop_pre_shape[in_axis] = target_in_info.size;
            }
            // else: e.g. esrgan & srgan

            TensorPtr in_crop_pre_out = std::make_shared<Tensor>(n->name() + "/tile/in_crop_pre/out/" + name_id,
                                                                 in_crop_pre_shape, inpt->type());
            in_crop_pre_out->quantization() = inpt->quantization();

            in_crop_pre->set_attr(Constants::tsg_name, tsg_name);
            in_crop_pre->set_attr("tile_id", i);

            // param of in_crop_pre
            for (int k = 0; k < in_crop_pre_shape.dim(); ++k)
            {
                in_crop_pre->params()["crops"].push_back({0, in_crop_pre_shape[k]});
            }

            if (target_in_info.size > 0)
            {
                in_crop_pre->params()["crops"][in_axis][0] = target_in_info.offset;
                in_crop_pre->params()["crops"][in_axis][1] = target_in_info.offset + target_in_info.size;
            }

            // else  e.g. esrgan & srgan
            // DEBUG() << target_in_info.offset << "-" << target_in_info.offset + target_in_info.size;
            in_crop_pre->add_input(inpt);
            in_crop_pre->add_output(in_crop_pre_out);
            set_ds_crops(g, in_crop_pre, in_axis, inpt, in_crop_pre_out);

            crop_pre_outs.push_back(in_crop_pre_out);
        }
        crop_pre_outs_list.push_back(crop_pre_outs);
    }

    NodePtrList out_crop_nodes;
    for (size_t i = 0; i < tile_num; ++i)
    {
        TensorShape tnode_shape = change_shape_by_axis(out->shape(), axis, out_infos[i].size);
        auto out_crop_shape = out->shape();
        tnode_shape.reset_strides();

        // TODO: explan not using out_infos[i].size:
        if (i == tile_num - 1)
        {
            out_crop_shape[axis] = out->shape()[axis] - out_infos[i].offset;
        }
        else
        {
            out_crop_shape[axis] = out_infos[i + 1].offset - out_infos[i].offset;
        }

        // in_crop
        TensorPtrList tnode_in;
        for (size_t j = 0; j < inps.size(); ++j)
        {
            TensorPtrList tnode_in_candidate;
            // input_id + tile_num_id
            std::string name_id = std::to_string(j) + "_" + std::to_string(i);

            auto inpt = inps[j];

            auto crop_pre_outs = crop_pre_outs_list[j];

            InputTSlice in_info = {0};

            // common parts of in_crop
            if (is_special_eltwise(n) && bc_tid == (int)j)
            {
                in_info = target_in_infos_list[j][i];
            }
            else if (n->type() == OpType_MatMul &&
                     ((axis == 1 && inpt->shape()[1] == 1) || (axis == 2 && j == 1)))
            {
                tnode_in.push_back(inpt);
                break;
            }
            else
            {
                in_info = in_infos[i];
            }

            // find the correct in_crop_pre
            auto target_in_infos = target_in_infos_list[j];
            auto best_id = get_best_info_id(in_info, target_in_infos);
            // assume that target_tile_num == tile_num
            best_id = i;
            auto target_in_info = target_in_infos[best_id];
            auto in_axis = get_in_axis(n, axis, j);

            // in crop share
            // share part from tail of last slice
            // share exist when:
            //		1. overlap between slice
            //		2. shift between inputs
            //
            // share part could be larger if children more than 1
            int in_share = 0;
            if (i >= 1)
                in_share = in_infos[i - 1].share;
            // if need share more, when in share not eq out share
            if (in_info.offset < target_in_info.offset)
            {
                in_share += target_in_info.offset - in_info.offset;
            }

            if (actg && in_share > 0)
            {
                NodePtr in_crop_share = Node::create(n->name() + "/tile/in_crop_share/" + name_id, OpType_Crop);
                ADD_LOCATION_BY_NODE(in_crop_share, n);
                g->add_node(in_crop_share);

                auto last_slice_in_info = in_infos[i - 1];
                if (is_special_eltwise(n) && bc_tid == (int)j)
                {
                    last_slice_in_info = target_in_infos_list[j][i - 1];
                }

                TensorShape in_crop_share_shape = inpt->shape();
                in_crop_share_shape[in_axis] = in_share;
                TensorPtr in_crop_share_out = std::make_shared<Tensor>(n->name() + "/tile/in_crop_share/out/" + name_id,
                                                                       in_crop_share_shape, inpt->type());
                in_crop_share_out->quantization() = inpt->quantization();

                int crop_begin = target_in_info.offset - in_share;
                int crop_end = target_in_info.offset;

                // param of in_crop_pre
                for (int k = 0; k < in_crop_share_shape.dim(); ++k)
                {
                    in_crop_share->params()["crops"].push_back({0, in_crop_share_shape[k]});
                }
                in_crop_share->params()["crops"][in_axis][0] = crop_begin;
                in_crop_share->params()["crops"][in_axis][1] = crop_end;

                in_crop_share->add_input(inpt);
                in_crop_share->add_output(in_crop_share_out);
                set_ds_crops(g, in_crop_share, in_axis, inpt, in_crop_share_out);

                tnode_in_candidate.push_back(in_crop_share_out);

                // flag
                in_crop_share_out->set_attr("ACTG_SHARE", true);
            }

            auto in_crop_shape = inpt->shape();

            if (in_info.size > 0)
            {
                // if shared more on last slice, make this slice smaller
                if (actg && in_info.offset < target_in_info.offset)
                {
                    in_info.size -= target_in_info.offset - in_info.offset;
                }
                in_crop_shape = change_shape_by_axis(in_crop_shape, in_axis, in_info.size);
            }

            TensorPtr in_crop_out = std::make_shared<Tensor>(n->name() + "/tile/in_crop/out/" + name_id, in_crop_shape, inpt->type());
            in_crop_out->quantization() = inpt->quantization();
            NodePtr in_crop = Node::create(n->name() + "/tile/in_crop/" + name_id, OpType_Crop);
            ADD_LOCATION_BY_NODE(in_crop, n);

            in_crop->set_attr(Constants::tsg_name, tsg_name);
            in_crop->set_attr("tile_id", i);

            // in_crop_pre + in_crop
            // auto in_crop_pre_shape = inps[j]->shape();
            // in_crop_pre_shape[axis] = target_in_infos[i].size;
            // TensorPtr in_crop_pre_out = std::make_shared<Tensor>(n->name() + "/tile/in_crop_pre/out/" + crop_id, in_crop_pre_shape, inps[j]->type());
            // NodePtr in_crop_pre = Node::create(n->name() + "/tile/in_crop_pre/" + crop_id, OpType_Crop);

            auto crop_pre_out = crop_pre_outs[best_id];

            // param of in_crop
            for (int k = 0; k < in_crop_shape.dim(); ++k)
            {
                in_crop->params()["crops"].push_back({0, in_crop_shape[k]});
            }
            if (in_info.size > 0)
            {
                if (actg && in_info.offset < target_in_info.offset)
                {
                    // have share part
                    target_in_info.offset = in_info.offset;
                }

                in_crop->params()["crops"][in_axis][0] = in_info.offset - target_in_info.offset;
                in_crop->params()["crops"][in_axis][1] = in_info.offset - target_in_info.offset + in_info.size;
            }

            in_crop->add_input(crop_pre_out);
            in_crop->add_output(in_crop_out);
            set_ds_crops(g, in_crop, in_axis, crop_pre_out, in_crop_out);

            g->add_node(in_crop);

            tnode_in_candidate.push_back(in_crop_out);

            if (0 == tnode_in_candidate.size())
            {
                ERROR() << "No any input"; // Never here
            }
            else if (1 == tnode_in_candidate.size()) // input from share or parent
            {
                tnode_in.push_back(tnode_in_candidate[0]);
            }
            else if (2 == tnode_in_candidate.size()) // need concat (actg)
            {
                // need a concat
                // TODO: need a flag for lib to match this "actg concat" ?
                NodePtr concat = Node::create(n->name() + "/tile/share_concat/" + name_id, OpType_Concat);
                ADD_LOCATION_BY_NODE(concat, n);
                g->add_node(concat);
                concat->params()["axis"] = in_axis;

                if (g->is_dynamic())
                {
                    concat->infer_shape(true);
                }

                TensorShape in_concat_out_shape = inpt->shape();
                in_concat_out_shape[in_axis] = 0;

                for (TensorPtr &t : tnode_in_candidate)
                {
                    concat->add_input(t);
                    in_concat_out_shape[in_axis] += t->shape()[in_axis];
                }

                // Output add the share shape
                TensorPtr in_concat_out = std::make_shared<Tensor>(n->name() + "/tile/in_concat/out/" + name_id, in_concat_out_shape, inpt->type());
                in_concat_out->quantization() = inpt->quantization();
                concat->add_output(in_concat_out);
                tnode_in.push_back(in_concat_out);
            }
        }

        std::string tile_id = std::to_string(i);
        // tnode
        NodePtr tnode = Node::create(n->name() + "/tile/" + tile_id, n->type());
        g->add_node(tnode);
        ADD_LOCATION_BY_NODE(tnode, n);

        tnode->set_attr(Constants::tsg_name, tsg_name);
        tnode->set_attr("tile_id", i);
        auto original_id = n->get_layer_id() >= 0 ? n->get_layer_id() : n->get_attr<int>("original_id");
        tnode->set_attr("original_id", original_id);

        // copy gsize
        if (n->attrs().count(Constants::gsize))
            tnode->attrs()[Constants::gsize] = n->attrs()[Constants::gsize];

        // copy flag
        if (n->attrs().count(Constants::keep_isolate_when_scheduling))
            tnode->attrs()[Constants::keep_isolate_when_scheduling] = n->attrs()[Constants::keep_isolate_when_scheduling];

        //copy params and weights
        tnode->params() = n->params();
        tnode->constants() = n->constants();

        if (tnode->params().count("method") &&
            tnode->params()["method"] == "PRELU" &&
            tnode->constants().count("negative_slope"))
        {
            ComputeTensor negative_slope = n->constants()["negative_slope"];
            std::vector<ComputeTensor::Slice> slices;
            int diff = tnode_in[0]->shape().dim() - negative_slope->shape().dim();
            int negative_slope_axis = axis - diff;
            if (diff >= 2)
            {
                WARN() << "unsupported negative_slope shape";
            }

            for (int s = 0; s < negative_slope->shape().dim(); ++s)
            {
                if (s == negative_slope_axis && negative_slope->shape()[s] != 1)
                {
                    ComputeTensor::Slice slice(out_infos[i].offset, out_infos[i].offset + out_infos[i].size);
                    slices.push_back(slice);
                }
                else
                {
                    ComputeTensor::Slice slice(0, negative_slope->shape()[s]);
                    slices.push_back(slice);
                }
            }

            ComputeTensor t_negative_slope = negative_slope[slices];
            t_negative_slope->monopolize_data();
            tnode->constants()["negative_slope"] = t_negative_slope;
        }

        if (tnode->type() == OpType_Crop)
        {
            std::vector<std::vector<int>> new_crop = tnode->params()["crops"];
            // change max size
            new_crop[axis][1] = out_infos[i].size;
            tnode->params()["crops"] = new_crop;
            if (n->params().count("ds_crops"))
            {
                std::vector<std::vector<Expr>> ds_crops = n->params()["ds_crops"];
                ds_crops[axis][1] = out_infos[i].size;
                n->params()["ds_crops"] = ds_crops;
            }
        }

        tnode->add_input(tnode_in);
        TensorPtr tnode_out = std::make_shared<Tensor>(n->name() + "/tile/out/" + tile_id, tnode_shape, out->type());
        tnode_out->quantization() = out->quantization();
        tnode->add_output(tnode_out);

        if (g->is_dynamic())
        {
            tnode->infer_shape(true);
        }

        // out_crop
        // usually, this crop will be removed
        TensorPtr out_crop_out = std::make_shared<Tensor>(n->name() + "/tile/out_crop/out/" + tile_id, out_crop_shape, out->type());
        out_crop_out->quantization() = out->quantization();
        NodePtr out_crop = Node::create(n->name() + "/tile/out_crop/" + tile_id, OpType_Crop);
        ADD_LOCATION_BY_NODE(out_crop, n);

        out_crop->set_attr("tile_id", i);
        g->add_node(out_crop);
        out_crop->add_input(tnode_out);
        out_crop->add_output(out_crop_out);
        for (int j = 0; j < out_crop->outputs()[0]->shape().dim(); ++j)
        {
            out_crop->params()["crops"].push_back({0, out_crop->outputs()[0]->shape()[j]});
        }
        out_crop->params()["crops"][axis][0] = 0;
        out_crop->params()["crops"][axis][1] = out_crop_shape[axis];

        set_ds_crops(g, out_crop, axis, tnode_out, out_crop_out);

        if (tnode->params().count(pad_key_0))
        {
            tnode->params()[pad_key_0] = in_infos[i].pad_left;
            tnode->params()[pad_key_1] = in_infos[i].pad_right;
        }
        tile_nodes.push_back(tnode);
        out_crop_nodes.push_back(out_crop);
    }
    // concat
    // std::string tile_cfg = BuilderConfig::get<std::string>(Constants::tile_cfg);
    // OpType concat_type;
    // if (tile_cfg == "footprint")
    // {
    //     concat_type = OpType_InternalConcatenated;
    // }
    // else // balance
    // {
    //     concat_type = OpType_Concat;
    // }
    OpType concat_type = OpType_Concat;
    NodePtr concat = Node::create(n->name() + "/tile/concat", concat_type);
    // ADD_LOCATION_BY_NODE(concat, n);

    concat->set_attr(Constants::tsg_name, tsg_name);
    concat->set_attr("tile_id", -1);

    if (g->is_dynamic())
    {
        concat->infer_shape(true);
    }

    g->add_node(concat);
    concat->params()["axis"] = axis;

    for (auto out_crop_node : out_crop_nodes)
    {
        concat->add_input(out_crop_node->outputs()[0]);
        concat->params()["scale_value"].push_back(NodeParamValue(1));
        concat->params()["scale_type"].push_back(Tensor::Dtype::UINT8);
        concat->params()["shift_value"].push_back(NodeParamValue(0));
        concat->params()["shift_type"].push_back(Tensor::Dtype::UINT8);
        concat->set_attr("TILE_MERGE_POINT", true);
    }
    concat->add_output(out);
    n->clear_inputs();
    n->clear_outputs();
    g->remove_node(n);

    return tile_nodes;
}

static TileInfo convert_infos_in2out(const InputTileInfo &in_infos)
{
    TileInfo out_infos;
    for (auto in_info : in_infos)
    {
        TSlice out_info;
        out_info.offset = in_info.offset;
        out_info.size = in_info.size;
        out_info.share = in_info.share;
        out_infos.push_back(out_info);
    }
    return out_infos;
}

static InputTileInfo convert_infos_out2in(const TileInfo &out_infos)
{
    InputTileInfo in_infos;
    for (auto out_info : out_infos)
    {
        InputTSlice in_info;
        in_info.offset = out_info.offset;
        in_info.size = out_info.size;
        in_info.share = out_info.share;
        in_info.pad_left = 0;
        in_info.pad_right = 0;
        in_infos.push_back(in_info);
    }
    return in_infos;
}

static void show_out_infos(const NodePtr &node, const TileInfo &slices)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < slices.size(); ++i)
    {
        TSlice slice = slices[i];
        oss << slice.size;
        if (slice.share > 0)
        {
            oss << "(" << slice.share << ")";
        }
        if (i < slices.size() - 1)
        {
            oss << ",";
        }
    }
    oss << "]";

    // std::ostringstream oss_offset;
    // oss_offset << "[";
    // for (size_t i = 0; i < slices.size(); ++i)
    // {
    //     auto slice = slices[i];
    //     oss_offset << slice.offset;
    //     if (i < slices.size() - 1)
    //     {
    //         oss_offset << ",";
    //     }
    // }
    // oss_offset << "]";
    DEBUG() << node->name() << ", out_slice_sizes[" << slices.size() << "]: " << oss.str();
    //    << ", out_slice_offsets[" << slices.size() << "]: " << oss_offset.str();
    //if (keep_isolate_when_scheduling(node))
    //    DEBUG() << node->name() << " would not merge with children ";
}

static void show_tile_plan(const TilePlan &out_tile_infos)
{
    DEBUG();
    DEBUG() << "show_tile_plan";
    NodePtrList node_list;
    for (auto info_pair : out_tile_infos)
    {
        node_list.push_back(info_pair.first);
    }
    auto cmp = [](NodePtr n0, NodePtr n1) -> bool {
        int id_0 = n0->get_attr<int>("layer_id");
        int id_1 = n1->get_attr<int>("layer_id");
        return (id_0 < id_1);
    };
    std::sort(node_list.begin(), node_list.end(), cmp);
    for (auto node : node_list)
    {
        show_out_infos(node, out_tile_infos.at(node));
    }
}

void update_tile_plan(const NodePtrList &merge_points,
                      const NodePtrList &subgraph_nodes,
                      NodePtrList &ready_points,
                      std::map<NodePtr, int> &res_child_num,
                      TilePlan &out_tile_infos,
                      NodePtr &n,
                      // uint32_t global_tile_num,
                      uint32_t max_out_size,
                      int32_t axis,
                      bool actg,
                      int32_t consumer_id)
{
    if (!contains(subgraph_nodes, n))
    {
        return;
    }
    bool need_update_parent = false;
    // first time
    if (out_tile_infos.count(n) == 0)
    {
        // merge points
        if (contains(merge_points, n))
        {
            auto infos = get_init_output_infos(n, axis, max_out_size, actg);
            out_tile_infos.insert(std::make_pair(n, infos));
            ready_points.push_back(n);
            need_update_parent = true;
        }
        // unsupported points
        else if (!is_supported_op(n, axis))
        {
            return;
        }
        // branch points
        else if (n->outputs()[0]->consumers().size() > 1)
        {
            auto child = n->outputs()[0]->consumers()[consumer_id];
            if (contains(ready_points, child))
            {
                auto in_infos = get_input_infos(child, axis, out_tile_infos[child], actg);
                auto out_infos = convert_infos_in2out(in_infos);
                out_tile_infos.insert(std::make_pair(n, out_infos));
                res_child_num[n]--;
            }
        }
        // normal points
        else if (n->outputs().size() == 1 && n->outputs()[0]->consumers().size() == 1)
        {
            auto child = n->outputs()[0]->consumers()[0];
            auto in_infos = get_input_infos(child, axis, out_tile_infos[child], actg);
            auto out_infos = convert_infos_in2out(in_infos);
            out_tile_infos.insert(std::make_pair(n, out_infos));
            ready_points.push_back(n);
            need_update_parent = true;
        }
    }
    // branch points (not first time)
    else if (n->outputs()[0]->consumers().size() > 1 && res_child_num[n] > 0)
    {
        auto child = n->outputs()[0]->consumers()[consumer_id];
        if (contains(ready_points, child))
        {
            // merge infos
            auto new_infos = get_input_infos(child, axis, out_tile_infos[child], actg);
            auto old_out_infos = out_tile_infos[n];
            auto old_infos = convert_infos_out2in(old_out_infos);

            // don't need care about share when merging infos
            convert_infos_remove_share(new_infos);
            convert_infos_remove_share(old_infos);

            // long_infos mean it has less elements, but each length of element may be longer
            auto get_avg_slice_size = [](const InputTileInfo &slices) -> double {
                if (slices.size() == 0)
                {
                    return 0;
                }
                uint32_t sum = 0;
                for (auto slice : slices)
                {
                    sum += slice.size;
                }
                double avg = sum * 1.0 / (slices.size());
                return avg;
            };
            auto long_infos = get_avg_slice_size(new_infos) >= get_avg_slice_size(old_infos) ? new_infos : old_infos;
            auto short_infos = get_avg_slice_size(new_infos) >= get_avg_slice_size(old_infos) ? old_infos : new_infos;

            auto get_union_info = [](const InputTSlice &a, const InputTSlice &b) -> InputTSlice {
                InputTSlice ret;
                auto start = std::min(a.offset, b.offset);
                auto end = std::max(a.offset + a.size, b.offset + b.size);
                ret.offset = start;
                ret.size = end - start;
                return ret;
            };

            auto union_infos = long_infos;
            for (size_t i = 0; i < short_infos.size(); ++i)
            {
                auto short_info = short_infos[i];
                auto best_id = get_best_info_id(short_info, union_infos);
                union_infos[best_id] = get_union_info(union_infos[best_id], short_info);
            }

            if (actg)
                convert_infos_compute_share(union_infos);

            auto union_out_infos = convert_infos_in2out(union_infos);

            out_tile_infos[n] = union_out_infos;
            res_child_num[n]--;
            if (res_child_num[n] == 0)
            {
                ready_points.push_back(n);
                need_update_parent = true;
            }
        }
    }

    // next
    if (need_update_parent)
    {
        for (auto inpt : n->inputs())
        {
            auto parent = inpt->node();
            // consumer id for next
            auto consumers = inpt->consumers();
            int c_id = INT32_MAX;
            auto it = std::find(consumers.begin(), consumers.end(), n);
            if (it != consumers.end())
            {
                c_id = it - consumers.begin();
            }
            update_tile_plan(merge_points, subgraph_nodes, ready_points, res_child_num, out_tile_infos, parent, max_out_size, axis, actg, c_id);
        }
    }
}

static NodePtrList get_root_points(GraphPtr g, int32_t axis, bool is_tile_by_num = false)
{
    NodePtrList root_points;
    auto nodes = g->nodes();
    auto in_nodes = g->get_input_nodes();
    auto out_nodes = g->get_output_nodes();

    for (auto n : out_nodes)
    {
        root_points.push_back(n);
    }

    for (auto n : nodes)
    {
        if (contains(in_nodes, n))
        {
            continue;
        }

        if (n->type() == OpType_ConvTranspose || n->type() == OpType_Resize ||
            !is_supported_op(n, axis, is_tile_by_num))
        {
            root_points.push_back(n);
        }
    }
    return root_points;
}

static NodePtrList roots_to_merge_points(const NodePtrList &roots, int32_t axis, bool is_tile_by_num = false)
{
    NodePtrList root_points(roots);
    NodePtrList merge_points;
    while (root_points.size() > 0)
    {
        auto n = root_points[0];
        if (is_supported_op(n, axis, is_tile_by_num))
        {
            if (!contains(merge_points, n))
            {
                merge_points.push_back(n);
            }
        }
        else
        {
            for (auto inpt : n->inputs())
            {
                auto parent = inpt->node();
                if (!contains(root_points, parent))
                {
                    root_points.push_back(parent);
                }
            }
        }
        root_points.erase(root_points.begin());
    }
    return merge_points;
}

static NodePtrList get_merge_points(const TileSubGraphMap &mp_bp_map)
{
    NodePtrList merge_points;
    for (auto iter : mp_bp_map)
    {
        auto mp = iter.first;
        merge_points.push_back(mp);
    }
    return merge_points;
}

static void show_mp_bp(const TileSubGraphMap &mp_bp_map, TileNumMap &max_num_map)
{
    for (auto iter = mp_bp_map.begin(); iter != mp_bp_map.end(); ++iter)
    {
        auto mp = iter->first;
        auto bps = iter->second;
        DEBUG();
        for (auto bp : bps)
        {
            DEBUG() << "BP: " << bp->name();
        }
        DEBUG() << "MP: " << mp->name();
        DEBUG() << "max_tile_num: " << max_num_map[mp];
    }
}

/**
 * @brief get_subgraph
 * @param merge_point
 * @param bottom_points
 * @param check: If input merge point is not the actually merge point(s) of bottom points,
 *               return empty node list and raise error
 * @return
 */
static NodePtrList get_subgraph(const NodePtr &merge_point, const NodePtrList &bottom_points, bool check = false)
{
    NodePtrList subgraph;
    NodePtrList root_points;
    std::set<NodePtr> merge_points;

    for (auto bp : bottom_points)
    {
        root_points.push_back(bp);
    }
    while (root_points.size() > 0)
    {
        auto n = root_points[0];
        if (!contains(subgraph, n))
        {
            // cur
            subgraph.push_back(n);
            // next
            if (n != merge_point)
            {
                auto t = n->outputs()[0];
                auto comsumers = t->consumers();
                for (auto c : comsumers)
                {
                    if (!contains(root_points, c))
                    {
                        root_points.push_back(c);
                    }
                }
                // if no next node, is should be a unexcept merge point
                if (comsumers.empty())
                    merge_points.insert(n);
            }
            else
            {
                // arrive the input merge point
                merge_points.insert(n);
            }
        }
        root_points.erase(root_points.begin());
    }

    if (check)
    {
        for (auto it = merge_points.begin(); it != merge_points.end(); it++)
        {
            if (*it == merge_point)
                ; // It's the input merge point, skip
            else
            {
                ERROR() << "Fond unexcept merge point (" << (*it)->name() << ")";
                return NodePtrList();
            }
        }
    }

    return subgraph;
}

using NodeTensorInfoMap = std::map<NodePtr, std::map<TensorPtr, uint32_t>>;

//FIX ME: if this subgraph has more than one deconv
uint32_t get_mp_align(const NodePtrList &nodes, int32_t axis)
{
    uint32_t stride = 1;

    if (axis != 1 && axis != 2)
    {
        return stride;
    }

    for (auto n : nodes)
    {
        std::string key;
        uint32_t value = 0;
        if (n->type() == OpType_ConvTranspose)
        {
            key = (axis == 1 ? "stride_y" : "stride_x");
            value = n->params()[key];
        }
        else if (n->type() == OpType_DepthToSpace)
        {
            key = (axis == 1 ? "block_size_y" : "block_size_x");
            if (n->params().count(key))
            {
                uint32_t align = n->params()[key];
                if (n->parents().size() > 0 &&
                    (n->parents()[0]->type() == OpType_Convolution || n->parents()[0]->type() == OpType_DepthwiseConv))
                {
                    align *= 2;
                }
                value = align;
            }
        }
        else if (n->type() == OpType_Resize)
        {
            key = (axis == 1 ? "ratio_y" : "ratio_x");
            value = get_real_ratio_int(n, key);
        }
        if (value > stride) // FIXME
        {
            stride = value;
        }
    }
    return stride;
}

static void mark_tiled_sg(const NodePtrList &sg, const std::string mp_name)
{
    for (auto n : sg)
    {
        n->set_attr(Constants::tsg_name, mp_name + "_tiled_SG");
    }
}

static double get_overlap_rate(const TileInfo &tile_infos, uint32_t original_size)
{
    double overlap_rate;
    uint32_t new_size = 0;
    for (auto tile_info : tile_infos)
    {
        new_size += tile_info.size;
        new_size += tile_info.share;
    }
    overlap_rate = (new_size - original_size) * 1.0 / original_size;
    return overlap_rate;
}

static int get_transpose_out_axis(const NodePtr &n, int o_axis)
{
    if (n->type() == OpType_Transpose)
    {
        auto i_shape = n->inputs()[0]->shape();
        std::vector<int> perm = n->params()["perm"];
        auto in_axis = perm[o_axis];
        size_t size_after_in_axis = 1;
        if (in_axis < i_shape.dim() - 1)
        {
            for (int i = in_axis + 1; i < i_shape.dim(); ++i)
            {
                size_after_in_axis *= i_shape[i];
            }
        }
        if (size_after_in_axis < 256)
        {
            o_axis += 1;
        }
    }
    return o_axis;
}

std::map<NodePtr, int> init_res_child_num(const NodePtrList &nodes, int32_t axis)
{
    std::map<NodePtr, int> res_child_num;
    for (auto n : nodes)
    {
        if (is_supported_op(n, axis))
        {
            auto child_num = n->outputs()[0]->consumers().size();
            if (child_num > 1)
            {
                res_child_num.insert(std::make_pair(n, child_num));
            }
        }
    }
    return res_child_num;
}

static bool check_plan(const TilePlan &plan, int32_t axis, bool actg)
{
    auto check_duplicate_begin_end = [](const TileInfo &info) -> bool {
        std::set<uint32_t> begin_set;
        std::set<uint32_t> end_set;
        for (size_t i = 0; i < info.size(); ++i)
        {
            auto slice = info[i];
            auto begin = slice.offset;
            auto end = slice.offset + slice.size;
            if (begin_set.count(begin) || end_set.count(end))
            {
                return false;
            }
            begin_set.insert(begin);
            end_set.insert(end);
        }
        return true;
    };
    for (auto iter = plan.begin(); iter != plan.end(); ++iter)
    {
        auto n = iter->first;
        auto out_infos = iter->second;

        //check duplicate begin/end
        if (g_auto_mode && !check_duplicate_begin_end(out_infos))
        {
            return false;
        }
        if (n->type() == OpType_DepthToSpace)
        {
            std::string key = (axis == 1 ? "block_size_y" : "block_size_x");
            if (key != "" && n->params().count(key))
            {
                uint32_t align = n->params()[key];
                if (n->parents().size() > 0 &&
                    (n->parents()[0]->type() == OpType_Convolution || n->parents()[0]->type() == OpType_DepthwiseConv))
                {
                    align *= 2;
                }
                for (auto slice : out_infos)
                {
                    auto size = slice.size;
                    if (size % align != 0)
                    {
                        // DEBUG() << n->name() << ", size: " << size << ", ratio: " << ratio;
                        return false;
                    }
                }
            }
        }
        if (n->type() == OpType_Resize && (axis == 1 || axis == 2))
        {
            std::string key = (axis == 1 ? "ratio_y" : "ratio_x");
            if (key != "" && n->params().count(key))
            {
                // uint32_t ratio = n->params()[key];
                auto ratio = get_real_ratio_int(n, key);
                if (ratio > 0)
                {
                    for (auto slice : out_infos)
                    {
                        auto size = slice.size;
                        if (size % ratio != 0)
                        {
                            // DEBUG() << "Not align resize: " << n->name() << ", size: " << size << ", ratio: " << ratio;
                            return false;
                        }
                    }
                }
            }
        }
        if (n->type() == OpType_Convolution || n->type() == OpType_DepthwiseConv)
        {
            auto in_infos = get_input_infos(n, axis, out_infos, actg);
            for (auto info : in_infos)
            {
                auto size = info.size;
                if (size > INT32_MAX)
                {
                    return false;
                }
            }
        }
    }
    return true;
};

static TilePlan make_plan(const GraphPtr g, const TileSubGraphMap &mp_bp_map,
                          const TileNumMap &max_num_map, int32_t axis, bool actg)
{
    auto merge_points = get_merge_points(mp_bp_map);
    // auto core_num = Target::get_the_target()->core_num();

    // tile_plan: map the n and it's expected out_infos (assume that n has only one output)
    // TODO: if n has more than one outputs, it should be std::map<NodePtr, std::vector<TileInfo>>
    auto get_out_size = [](uint32_t max_num, uint32_t total_size) -> uint32_t {
        uint32_t tile_size = (total_size + max_num - 1) / max_num;
        return tile_size;
    };
    auto copy_plan = [](const TilePlan &src, TilePlan &dst) -> void {
        std::copy(src.begin(), src.end(), std::inserter(dst, dst.begin()));
    };
    auto copy_nodes = [](const NodePtrList &src, NodePtrList &dst) -> void {
        std::copy(src.begin(), src.end(), std::inserter(dst, dst.begin()));
    };
    auto copy_child_num = [](const std::map<NodePtr, int> &src, std::map<NodePtr, int> &dst) -> void {
        std::copy(src.begin(), src.end(), std::inserter(dst, dst.begin()));
    };
    TilePlan tile_plan;
    NodePtrList ready_points;
    auto res_child_num = init_res_child_num(g->nodes(), axis);
    for (auto mp_tile_num : max_num_map)
    {
        auto mp = mp_tile_num.first;
        auto tile_num = mp_tile_num.second;
        auto axis_ = get_transpose_out_axis(mp, axis);
        auto out_size_min = get_out_size(mp_tile_num.second, mp->outputs()[0]->shape()[axis_]);
        auto out_size_max = out_size_min;
        auto subgraph = get_subgraph(mp, mp_bp_map.at(mp));
        auto mp_align = get_mp_align(subgraph, axis_);
        if (mp->type() != OpType_DepthToSpace || tile_num <= 30)
        {
            out_size_max = ((out_size_max + mp_align - 1) / mp_align) * mp_align;
        }

        for (uint32_t max_out = out_size_max; max_out >= out_size_min; max_out--)
        {
            TilePlan tile_plan_tmp;
            copy_plan(tile_plan, tile_plan_tmp); // tile_plan --> tile_plan_tmp

            NodePtrList ready_points_tmp;
            copy_nodes(ready_points, ready_points_tmp);

            std::map<NodePtr, int> res_child_num_tmp;
            copy_child_num(res_child_num, res_child_num_tmp);

            update_tile_plan(merge_points, subgraph, ready_points_tmp, res_child_num_tmp, tile_plan_tmp, mp, max_out, axis_, actg, 0);
            auto is_valid_plan = check_plan(tile_plan_tmp, axis_, actg);
            if (is_valid_plan)
            {
                mark_tiled_sg(subgraph, mp->name());
                tile_plan.clear();
                ready_points.clear();
                res_child_num.clear();
                copy_plan(tile_plan_tmp, tile_plan); // tile_plan_tmp --> tile_plan
                copy_nodes(ready_points_tmp, ready_points);
                copy_child_num(res_child_num_tmp, res_child_num);
                break;
            }
        }
    }

    return tile_plan;
}

static void merge_slots(MemoryObjectPtr &root_slot, MemoryObjectPtr &slot)
{
    if (root_slot->size % slot->align != 0)
    {
        // align the slot
        root_slot->size += (slot->align - root_slot->size % slot->align);
    }
    slot->relative_offset = root_slot->size;
    slot->set_base_offset(root_slot);
    root_slot->size += slot->size;
    root_slot->foot_print += slot->foot_print;
}

static NodePtrList get_special_nodes(const NodePtrList &nodes)
{
    NodePtrList special_nodes;
    for (auto n : nodes)
    {
        if (n->type() == OpType_Convolution)
        {
            if (n->children().size() == 1 && n->children()[0]->type() == OpType_Eltwise)
            {
                special_nodes.push_back(n);
            }
            else if (n->children().size() == 1 && n->children()[0]->type() == OpType_DepthToSpace)
            {
                auto d2s = n->children()[0];
                auto d2s_i_shape = d2s->inputs()[0]->shape();
                auto d2s_o_shape = d2s->outputs()[0]->shape();
                if (d2s_i_shape[1] % 2 != 0)
                {
                    continue;
                }
                bool case0 = (d2s_i_shape[3] <= 32);
                bool case1 = (d2s_i_shape[3] > 32 && d2s_i_shape[3] % 32 == 0 && d2s_o_shape[3] % 32 == 0);
                if (case0 || case1)
                {
                    special_nodes.push_back(n);
                }
            }
        }
    }
    return special_nodes;
};

static uint32_t get_workspace_size(GraphPtr g, const NodePtr &mp, const NodePtrList &bps, const NodePtrList &_nodes)
{
    auto clear_base_offset = [](const NodePtrList &nodes) -> void {
        for (auto n : nodes)
        {
            for (auto t : n->outputs())
            {
                t->memory_offset()->reset();
            }
        }
    };
    auto nodes_tmp = g->topological_sort(_nodes);
    NodePtrList nodes;
    // work around bug of topological_sort
    for (auto n : nodes_tmp)
    {
        if (contains(_nodes, n) && !contains(nodes, n))
        {
            nodes.push_back(n);
        }
    }
    clear_base_offset(nodes);
    std::set<TensorPtr> final_tensors;
    std::set<MemoryObjectPtr> keep;
    std::map<MemoryObjectPtr, int> sg_consumer_cnt;
    Scheduler scheduler;
    MemoryAllocator *allocator = get_mem_allocator();
    std::map<MemoryObjectPtr, size_t> memory_size;
    std::set<MemoryObjectPtr> allocated;
    std::set<TensorPtr> skip_tensors;

    for (auto bp : bps)
    {
        for (auto t : bp->inputs())
        {
            skip_tensors.insert(t);
        }
    }

    for (auto n : nodes)
    {
        for (auto t : n->outputs())
        {
            if (skip_tensors.count(t))
            {
                skip_tensors.erase(t);
            }
        }
    }

    auto special_nodes = get_special_nodes(nodes);
    for (auto n : special_nodes)
    {
        for (auto t : n->outputs())
        {
            skip_tensors.insert(t);
        }
    }

    allocator->enable_soc_sram(false);
    scheduler.graph_memory(memory_size, nodes);
    scheduler.graph_consumer(final_tensors, keep, sg_consumer_cnt, nodes);
    scheduler.graph_memory_allocate(allocator, nodes, keep, final_tensors, sg_consumer_cnt, memory_size, allocated, skip_tensors);
    allocator->enable_soc_sram(true);
    auto slots = allocator->slots();
    auto ws_slot = std::make_shared<MemoryObject>(0);
    for (auto slot : slots)
    {
        merge_slots(ws_slot, slot);
    }
    clear_base_offset(nodes);
    // scheduler.print_slots_size(slots);
    // DEBUG() << "ws_size: " << ws_slot->size;
    if (allocator != nullptr)
    {
        delete allocator;
    }
    return ws_slot->size;
}

static WorkSpaceTileInfo tile_workspace(GraphPtr g,
                                        const NodePtr &mp,
                                        const NodePtrList &bps,
                                        uint32_t gm_size,
                                        uint32_t core_num,
                                        uint32_t axis)
{
    auto get_clone_nodes = [](GraphPtr g, const NodePtrList &nodes) -> NodePtrList {
        NodePtrList nodes_new;
        std::set<std::string> node_names;
        for (auto n : nodes)
        {
            node_names.insert(n->name());
        }
        for (auto n : g->nodes())
        {
            if (node_names.count(n->name()))
            {
                nodes_new.push_back(n);
            }
        }
        return nodes_new;
    };
    auto get_max_size = [](const TileInfo &tile_info) -> uint32_t {
        if (tile_info.size() == 0)
        {
            return 0;
        }
        else if (tile_info.size() == 1)
        {
            return tile_info[0].size;
        }
        else if (tile_info.size() == 2)
        {
            return std::max(tile_info[0].size, tile_info[1].size);
        }
        else
        {
            uint32_t mid = tile_info.size() / 2;
            return tile_info[mid].size;
        }
    };
    auto get_tensor_old_sizes = [](const NodePtrList &nodes, const TilePlan &tile_plan, uint32_t axis) -> std::map<TensorPtr, uint32_t> {
        std::map<TensorPtr, uint32_t> tensor_size_map;
        for (auto n : nodes)
        {
            if (tile_plan.count(n))
            {
                for (auto t : n->outputs())
                {
                    tensor_size_map[t] = t->shape()[axis];
                }
            }
        }
        return tensor_size_map;
    };
    auto get_tensor_new_sizes = [get_max_size](const NodePtrList &nodes, const TilePlan &tile_plan, uint32_t axis) -> std::map<TensorPtr, uint32_t> {
        std::map<TensorPtr, uint32_t> tensor_size_map;
        for (auto n : nodes)
        {
            if (tile_plan.count(n))
            {
                auto tile_info = tile_plan.at(n);
                auto max_size = get_max_size(tile_info);
                for (auto t : n->outputs())
                {
                    tensor_size_map[t] = max_size;
                }
            }
        }
        return tensor_size_map;
    };
    auto update_tensor_sizes = [](const std::map<TensorPtr, uint32_t> new_size_map, uint32_t axis) -> void {
        for (auto iter = new_size_map.begin(); iter != new_size_map.end(); ++iter)
        {
            auto t = iter->first;
            auto size = iter->second;
            t->shape()[axis] = size;
        }
    };
    auto revert_tensor_sizes = [](const std::map<TensorPtr, uint32_t> old_size_map, uint32_t axis) -> void {
        for (auto iter = old_size_map.begin(); iter != old_size_map.end(); ++iter)
        {
            auto t = iter->first;
            auto size = iter->second;
            t->shape()[axis] = size;
        }
    };
    auto need_insert_crop = [](const NodePtr &n) -> bool {
        // optimize for AIB imdn conv+crop+d2s
        if (n->type() == OpType_Convolution)
        {
            if (n->children().size() == 1 &&
                n->children()[0]->type() == OpType_DepthToSpace &&
                n->outputs().size() == 1 &&
                n->outputs()[0]->shape()[3] % 8 != 0)
            {
                return true;
            }
        }
        return false;
    };
    auto insert_crop_before_node = [need_insert_crop](GraphPtr g_new, NodePtrList &bps_new, const bool is_bp) -> void {
        // std::set<OpType> ops_cant_fuse_crop = {
        //     // OpType_Eltwise,
        //     // OpType_Activation,
        // };
        NodePtrList removed_bps;
        NodePtrList added_bps;
        for (auto bp : bps_new)
        {
            // if (ops_cant_fuse_crop.count(bp->type()))
            if (need_insert_crop(bp))
            {
                auto inputs = bp->inputs();
                TensorPtrList crop_outs;
                for (size_t i = 0; i < inputs.size(); ++i)
                {
                    auto in = inputs[i];
                    Quantization quant(1.0, 0);
                    auto crop_out = ops::cast(in, in->type(), false, ops::ClipMode::Saturation, quant); // FIXME
                    auto crop_node = crop_out->node();
                    g_new->add_node(crop_node);
                    added_bps.push_back(crop_node);
                    crop_outs.push_back(crop_out);
                }
                bp->clear_inputs();
                for (auto crop_out : crop_outs)
                {
                    bp->add_input(crop_out);
                }
                removed_bps.push_back(bp);
            }
        }
        if (is_bp)
        {
            for (auto bp : removed_bps)
            {
                if (contains(bps_new, bp))
                {
                    auto iter = std::remove(bps_new.begin(), bps_new.end(), bp);
                    bps_new.erase(iter, bps_new.end());
                }
            }
            for (auto bp : added_bps)
            {
                if (!contains(bps_new, bp))
                {
                    bps_new.push_back(bp);
                }
            }
        }
    };
    auto update_tile_num = [](int last_tile_num, int core_num) -> int {
        /* a temp simple way */
        if (core_num == 1)
        {
            return last_tile_num + 1;
        }
        else
        {
            return last_tile_num * 2;
        }
    };
    WorkSpaceTileInfo ws_info;
    GraphPtr g_new = g->clone();
    auto nodes = get_subgraph(mp, bps);

    auto bps_new = get_clone_nodes(g_new, bps);
    auto mps_new = get_clone_nodes(g_new, {mp});
    bool is_mp_in_bps = contains(bps_new, mps_new[0]);
    insert_crop_before_node(g_new, bps_new, true);
    if (!is_mp_in_bps)
    {
        insert_crop_before_node(g_new, mps_new, false);
    }
    auto mp_new = mps_new[0];

    auto nodes_new = get_subgraph(mp_new, bps_new);
    auto ws_size = get_workspace_size(g_new, mp_new, bps_new, nodes_new);

    ws_info.tile_num = core_num;
    ws_info.ws_size_original = ws_size;
    ws_info.ws_size_tiled = ws_size / core_num;

    if (ws_size <= gm_size / core_num)
    {
        if (is_tile_node(mp_new) || (nodes.size() == 1 && ws_size < 512 * 1024))
        {
            ws_info.tile_num = 1;
            ws_info.ws_size_tiled = ws_size;
        }
        return ws_info;
    }

    auto special_nodes = get_special_nodes(nodes);
    bool has_special_nodes = special_nodes.size() > 0;
    int max_num = ((ws_size + gm_size - 1) / gm_size) * core_num * 3;
    if (max_num > mp->outputs()[0]->shape()[axis])
    {
        max_num = mp->outputs()[0]->shape()[axis];
    }
    for (int tile_num = core_num; tile_num <= mp->outputs()[0]->shape()[axis]; tile_num = update_tile_num(tile_num, core_num))
    {
        TileSubGraphMap mp_bp_map_tmp = {std::pair<NodePtr, NodePtrList>(mp_new, bps_new)};
        TileNumMap max_num_map_tmp = {std::pair<NodePtr, uint32_t>(mp_new, tile_num)};
        auto tile_plan = make_plan(g_new, mp_bp_map_tmp, max_num_map_tmp, axis, false);
        if (tile_plan.empty())
        {
            continue;
        }

        auto tensor_old_sizes = get_tensor_old_sizes(nodes_new, tile_plan, axis);
        auto tensor_new_sizes = get_tensor_new_sizes(nodes_new, tile_plan, axis);
        update_tensor_sizes(tensor_new_sizes, axis);
        auto ws_size_tmp = get_workspace_size(g_new, mp_new, bps_new, nodes_new);
        revert_tensor_sizes(tensor_old_sizes, axis);

        ws_info.tile_num = tile_num;
        ws_info.ws_size_original = ws_size;
        ws_info.ws_size_tiled = ws_size_tmp;
        uint32_t gm_usage = has_special_nodes ? 90 : 100;
        if (ws_size_tmp <= (gm_size / core_num) * gm_usage / 100)
        {
            break;
        }
    }
    return ws_info;
}

static uint32_t get_sram_size(const NodePtrList &base_mps, int axis)
{
    uint32_t sram_size = 0;

    auto get_size_string = [](uint64_t size, uint32_t precision = 2) -> std::string {
        return utils::get_human_readable_size(size, utils::DataUnit::DU_Undefined, precision);
    };

    auto dtcm_size = BuilderConfig::get<uint32_t>(Constants::tcm_size);
    auto gm_size = BuilderConfig::get<uint32_t>(Constants::gm_size);
    auto ocm_size = BuilderConfig::get<uint32_t>(Constants::ocm_size);
    auto arch = aipubt::get_target()->isainfo().type;

    if (arch < aipubt::ISA_ZYv5)
    {
        DEBUG() << "dtcm_size: " << get_size_string(dtcm_size);
        sram_size = dtcm_size;
    }
    else
    {
        DEBUG() << "gm_size: " << get_size_string(gm_size) << ", ocm_size: " << get_size_string(ocm_size);
        auto big = std::max(gm_size, ocm_size);
        auto small = std::min(gm_size, ocm_size);
        if (big >= small * 3)
        {
            sram_size = big;
        }
        else
        {
            sram_size = ((big + small) * 90 / 100);
        }
    }

    if (base_mps.size() > 4 && base_mps[0]->outputs().size() > 0 && base_mps[0]->outputs()[0]->shape().n() > 1)
    {
        sram_size = sram_size * 95 / 100;
    }

    return sram_size;
}

static void get_tsg_map(GraphPtr g, int gm_size, uint32_t core_num, int32_t axis,
                        const NodePtrList &base_mps, TileSubGraphMap &mp_bp_map, TileNumMap &max_num_map,
                        double overlap_threshold, bool is_tile_by_num)
{
    auto subgraph_to_bps = [](const NodePtrList &nodes) -> NodePtrList {
        NodePtrList bps;
        for (auto n : nodes)
        {
            if (n->parents().size() == 0)
            {
                bps.push_back(n);
            }
            else
            {
                for (auto p : n->parents())
                {
                    if (!contains(nodes, p))
                    {
                        bps.push_back(n);
                        break;
                    }
                }
            }
        }
        return bps;
    };
    auto subgraph_to_mps = [](const NodePtrList &nodes) -> NodePtrList {
        NodePtrList mps;
        for (auto n : nodes)
        {
            if (n->children().empty())
            {
                mps.push_back(n);
            }
            else if (n->get_free_outputs().size())
            {
                mps.push_back(n);
            }
            else
            {
                for (auto c : n->children())
                {
                    if (!contains(nodes, c))
                    {
                        mps.push_back(n);
                        break;
                    }
                }
            }
        }
        return mps;
    };
    auto get_overlap = [](const WorkSpaceTileInfo &info) -> double {
        double overlap = 1.0;

        if (info.ws_size_original)
        {
            overlap = info.ws_size_tiled * info.tile_num * 1.0 / info.ws_size_original;
        }
        return overlap;
    };
    auto is_valid_parent = [](const NodePtr &p, int axis = 1, bool is_tile_by_num = false) -> bool {
        if (!is_supported_op(p, axis, is_tile_by_num) || p->type() == OpType_ConvTranspose)
        {
            return false;
        }
        if (p->type() == OpType_Resize && valid_scale_up(p, axis))
        {
            return false;
        }
        // optimize for AIB esrgan
        if (p->children().size() >= 6)
        {
            return false;
        }
        if (axis == 2 && p->children()[0]->type() == OpType_MatMul)
        {
            if (p->outputs()[0]->shape()[axis] != p->children()[0]->outputs()[0]->shape()[axis])
            {
                return false;
            }
        }
        return true;
    };
    auto push_node = [](NodePtrList &nodes, const NodePtr &n) -> void {
        if (!contains(nodes, n))
        {
            nodes.push_back(n);
        }
    };
    auto get_overlap_thr = [&overlap_threshold](const NodePtr &n) {
        std::string tile_cfg = BuilderConfig::get<std::string>(Constants::tile_cfg);

        // optimize for AIB imdn
        if (n->children().size() >= 3)
        {
            return 1.46;
        }

        if (tile_cfg == "footprint")
        {
            return 3.0;
        }
        else
        {
            return overlap_threshold;
        }
    };

    std::set<NodePtr> matched_nodes;
    TileSubGraphMap mp_subgraph_map;
    NodePtrList short_subgraphs;
    NodePtrList rest_mps(base_mps);
    auto add_type_cnt = [](std::map<OpType, uint32_t> &type_cnt, OpType type) -> void {
        if (type_cnt.count(type))
        {
            type_cnt[type]++;
        }
        else
        {
            type_cnt[type] = 1;
        }
    };
    auto get_type_cnt = [](std::map<OpType, uint32_t> &type_cnt, OpType type) -> uint32_t {
        uint32_t cnt = 0;
        if (type_cnt.count(type))
        {
            cnt = type_cnt[type];
        }
        return cnt;
    };

    while (rest_mps.size() > 0)
    {
        auto mp = rest_mps[0];
        TilePlan mp_plan;
        if (matched_nodes.count(mp))
        {
            rest_mps.erase(rest_mps.begin());
            continue;
        }
        NodePtrList subgraph = {mp};

        std::map<OpType, uint32_t> type_cnt;
        add_type_cnt(type_cnt, mp->type());

        uint32_t tile_num = (gm_size == -1) ? core_num : tile_workspace(g, mp, subgraph, gm_size, core_num, axis).tile_num;

        matched_nodes.insert(mp);
        NodePtrList roots;
        for (auto p : mp->parents())
        {
            push_node(roots, p);
        }

        while (roots.size())
        {
            auto n = roots[0];
            if (matched_nodes.count(n) || contains(base_mps, n) || contains(subgraph, n))
            {
                roots.erase(roots.begin());
                continue;
            }

            if (mp->type() == OpType_Transpose && mp->get_attr<int>("w_tiling"))
            {
                tile_num = mp->get_attr<int>("w_tiling");
                break;
            }

            if (!is_valid_parent(n, axis, is_tile_by_num))
            {
                if (is_supported_op(n, axis, is_tile_by_num))
                {
                    push_node(rest_mps, n);
                }
                roots.erase(roots.begin());
                continue;
            }

            // check topology
            NodePtrList sg_tmp(subgraph);
            push_node(sg_tmp, n);
            auto mp_tmp = subgraph_to_mps(sg_tmp);
            if (mp_tmp.size() != 1 || mp_tmp[0]->name() != mp->name())
            {
                roots.erase(roots.begin());
                push_node(rest_mps, n);
                continue;
            }

            // check overlap & tile_num
            auto tile_num_tmp = core_num;
            if (gm_size != -1)
            {
                auto bps_tmp = subgraph_to_bps(sg_tmp);
                auto info = tile_workspace(g, mp_tmp[0], bps_tmp, gm_size, core_num, axis);
                auto overlap = get_overlap(info);

                if (n->type() == OpType_FullyConnected && info.tile_num > (core_num * 2))
                {
                    tile_num_tmp = core_num;
                }
                else
                {
                    tile_num_tmp = info.tile_num;
                    if (overlap > get_overlap_thr(n) || tile_num_tmp == 1)
                    {
                        roots.erase(roots.begin());
                        push_node(rest_mps, n);
                        continue;
                    }
                }
            }

            // optimize for AIB imdn d2s
            if (get_type_cnt(type_cnt, OpType_DepthToSpace) >= 1 &&
                get_type_cnt(type_cnt, OpType_Eltwise) >= 1 &&
                n->type() == OpType_Eltwise)
            {
                roots.erase(roots.begin());
                push_node(rest_mps, n);
                continue;
            }

            push_node(subgraph, n);
            tile_num = tile_num_tmp;
            add_type_cnt(type_cnt, n->type());
            matched_nodes.insert(n);
            if ((n->inputs()[0]->size() * 8) > n->outputs()[0]->size())
            {
                for (auto p : n->parents())
                {
                    push_node(roots, p);
                }
            }
            else
            {
                for (auto p : n->parents())
                {
                    if (is_supported_op(p, axis, is_tile_by_num))
                    {
                        push_node(rest_mps, p);
                    }
                }
            }
            roots.erase(roots.begin());
        }

        rest_mps.erase(rest_mps.begin());
        auto bps = subgraph_to_bps(subgraph);

        /* do not tile single node subgraph of certain type */
        if (bps.size() == 1 && mp == bps[0])
        {
            if (mp->type() == OpType_Constant || mp->type() == OpType_Concat || mp->type() == OpType_Crop)
            {
                tile_num = 1;
            }

            if (mp->type() == OpType_Eltwise)
            {
                for (auto p : mp->parents())
                {
                    if (p->type() == OpType_Convolution)
                    {
                        tile_num = 1;
                        break;
                    }
                }
            }

            if (mp->type() == OpType_FullyConnected && tile_num != 1)
            {
                tile_num = core_num;
            }
        }

        if (axis == 1 &&
            mp->type() == OpType_MatMul &&
            tile_num == (uint32_t)mp->outputs()[0]->shape()[axis] &&
            mp->outputs()[0]->consumers().size() == 1 &&
            mp->outputs()[0]->consumers()[0]->type() == OpType_Transpose &&
            mp->outputs()[0]->consumers()[0]->params()["perm"][0] == 0 &&
            mp->outputs()[0]->consumers()[0]->params()["perm"][1] == 2)
        {
            mp->outputs()[0]->consumers()[0]->set_attr("w_tiling", tile_num);
        }

        if (mp->outputs()[0]->consumers().size() > 1 &&
            subgraph.size() < 4 &&
            (mp->outputs()[0]->shape()[-1] % 32 != 0) &&
            tile_num == core_num)
        {
            if (mp->type() == OpType_Convolution || mp->type() == OpType_DepthwiseConv ||
                mp->type() == OpType_Eltwise)
            {
                tile_num = 1;
            }
        }

        if (tile_num != 1)
        {
            mp_bp_map[mp] = bps;
            max_num_map[mp] = tile_num;
        }
    }
}

static int32_t get_c_tile_num(const NodePtr &n)
{
    static std::set<OpType> NON_INP_SPLIT_OPS =
        {
            OpType_Convolution,
            OpType_ConvTranspose,
            OpType_FullyConnected,
        };
    auto out_c = n->outputs()[0]->shape()[-1];
    auto out_n = n->outputs()[0]->shape().n();
    auto out_type = n->outputs()[0]->type();
    uint32_t alignment = ((out_type == Tensor::INT8) || (out_type == Tensor::UINT8)) ? 32 : 16;
    uint32_t num = out_c / alignment;
    uint32_t max_tile_num = 200;
    auto core_num = Target::get_the_target()->core_num();
    if (!contains(NON_INP_SPLIT_OPS, n->type()))
    {
        max_tile_num = core_num;
    }
    if (num >= core_num * 2)
    {
        /*
         * For llama 2, have some batch fc
         * To avoid weight reload inside op, cut c here
         * However, it's not easily calulate the "best c tile num"
         * So, it just a experience value for make more slices
         *
         * e.g. 128 * 4096 split  = 48 times
         **/
        if (out_n >= 64)
            num *= 6; // batch fc to conv -> tile more

        num = ((num - 1) / 256 + 1) * core_num;

        return num > max_tile_num ? max_tile_num : num;
    }
    else
    {
        return num > core_num ? core_num : num;
    }
}

TilePlan make_plan_axis(const GraphPtr g, TileSubGraphMap &mp_bp_map,
                      TileNumMap &max_num_map, int32_t axis = 1,
                      bool is_tile_by_num = false, bool en_actg = false,
                      uint32_t core_num = 0, int gm_size = 0, double overlap_threshold = 1.6)
{
    auto root_points = get_root_points(g, axis, is_tile_by_num);
    auto base_mps = roots_to_merge_points(root_points, axis, is_tile_by_num);

    /* core_num == 0 means to use the default value */
    if (core_num == 0)
    {
        core_num = Target::get_the_target()->core_num();
    }

    /**
     * gm_size:
     *  == 0 means to use the default value of gm;
     *  == -1 means to tile by core_num, without considering the value of gm;
     */
    if (gm_size == 0)
    {
        gm_size = get_sram_size(base_mps, axis);
    }

    if (g_auto_mode || is_tile_by_num)
    {
        /* to tile automatically or to tile by core_num/gm_size */
        get_tsg_map(g, gm_size, core_num, axis, base_mps, mp_bp_map, max_num_map,
                    overlap_threshold, is_tile_by_num);
    }
    else
    {
        /* to tile manually */
        for (auto m : mp_bp_map)
        {
            auto mp = m.first;
            auto bps = m.second;
            if (max_num_map.count(mp) == 0)
            {
                auto ws_info = tile_workspace(g, mp, bps, gm_size, core_num, axis);
                max_num_map[mp] = ws_info.tile_num <= 200 ? ws_info.tile_num : core_num;
            }
        }
    }

    if (mp_bp_map.size() == 0 || max_num_map.size() == 0)
    {
        return {};
    }

    if (0)
    {
        show_mp_bp(mp_bp_map, max_num_map);
    }
    return make_plan(g, mp_bp_map, max_num_map, axis, en_actg);
}

static void make_plan_c(GraphPtr g, TileSubGraphMap &mp_bp_map, TileNumMap &max_num_map, bool is_auto = true)
{
    if (is_auto)
    {
        for (auto n : g->nodes())
        {
            if (is_supported_op(n, 3))
            {
                NodePtrList bps;
                bps.push_back(n);
                mp_bp_map[n] = bps;
                max_num_map[n] = get_c_tile_num(n);
            }
        }
    }
    else
    {
        for (auto m : mp_bp_map)
        {
            auto mp = m.first;
            if (max_num_map.count(mp) == 0 || max_num_map[mp] < 1)
            {
                max_num_map[mp] = get_c_tile_num(mp);
            }
        }
    }
}

static NodePtrList get_in_crops(NodePtr n)
{
    NodePtrList ret;
    for (size_t i = 0; i < n->inputs().size(); ++i)
    {
        auto crop = n->inputs()[i]->node();
        if (crop->type() == OpType_Crop && is_tile_node(crop) && crop->params().count("crops") > 0)
        {
            ret.push_back(crop);
        }
    }
    auto cmp = [](NodePtr n0, NodePtr n1) -> bool {
        int tile_id_0 = n0->get_attr("tile_id");
        int tile_id_1 = n1->get_attr("tile_id");
        return (tile_id_0 < tile_id_1);
    };
    std::sort(ret.begin(), ret.end(), cmp);
    return ret;
}
static NodePtrList get_out_crops(NodePtr n)
{
    NodePtrList ret;
    auto children = n->outputs()[0]->consumers();
    for (size_t i = 0; i < children.size(); ++i)
    {
        auto crop = children[i];
        if (crop->type() == OpType_Crop && is_tile_node(crop) && crop->params().count("crops") > 0)
        {
            ret.push_back(crop);
        }
    }
    auto cmp = [](NodePtr n0, NodePtr n1) -> bool {
        int tile_id_0 = n0->get_attr("tile_id");
        int tile_id_1 = n1->get_attr("tile_id");
        return (tile_id_0 < tile_id_1);
    };
    std::sort(ret.begin(), ret.end(), cmp);
    return ret;
}

/**
 * @brief lower_crop_contains_crop
 * If one of crop cantains another, lower the smaller one
 * if enable actg, always have these crop
 *
 * Note: It's confilct with "fuse_crop"
 *
 * e.g.
 * [1, 100, 1, 1] - crop 1 -> [1, 0~10, 1, 1]
 *                ` crop 2 -> [1, 0~5, 1, 1]
 *
 * [1, 100, 1, 1] - crop 1 -> [1, 0~10, 1, 1] -> crop 2 -> [1, 0~5, 1, 1]
 */
bool lower_actg_crop(GraphPtr g)
{
    bool is_changed = false;

    for (auto kv : g->tensors())
    {
        TensorPtr tensor = kv.second;

        NodePtrList candidate_crops;
        for (NodePtr c : tensor->consumers())
        {
            if (OpType_Crop == c->type())
                candidate_crops.push_back(c);
        }
        if (candidate_crops.size() < 2)
        {
            continue;
        }

        std::vector<NodePtrList> need_lower;

        // sort by size
        // e.g. [9-10]  < [8-10] < [9-11]  < [0-5]
        sort(candidate_crops.begin(), candidate_crops.end(), [](NodePtr &a, NodePtr &b) {
            int size_a = a->outputs()[0]->shape().size();
            int size_b = b->outputs()[0]->shape().size();

            std::vector<std::vector<int>> crop_a = a->params()["crops"];
            std::vector<std::vector<int>> crop_b = b->params()["crops"];

            if (size_a == size_b)
                return crop_a < crop_b;
            else
                return size_a < size_b;
        });

        // find which crop can be lowered
        std::set<int> visited;
        for (size_t i = 0; i < candidate_crops.size(); i++)
        {
            for (size_t j = i + 1; j < candidate_crops.size(); j++)
            {
                if (visited.count(i) || visited.count(j))
                    continue;
                std::vector<std::vector<int>> small_crop = candidate_crops[i]->params()["crops"];
                std::vector<std::vector<int>> large_crop = candidate_crops[j]->params()["crops"];

                // only lower actg crop
                if (!is_tile_actg_crop(candidate_crops[i]))
                    continue;

                bool ok = true;
                for (size_t d = 0; ok && d < small_crop.size(); ++d)
                {
                    if (small_crop[d][0] < large_crop[d][0])
                        ok = false;
                    if (small_crop[d][1] > large_crop[d][1])
                        ok = false;
                }

                if (ok)
                {
                    need_lower.push_back({candidate_crops[i], candidate_crops[j]});
                    visited.insert(i);
                    // visited.insert(j); -> reuse big crop
                }
            }
        }

        // lower
        for (NodePtrList crops : need_lower)
        {
            // DEBUG() << "lower " << crops[0]->name() << " under " << crops[1]->name();
            is_changed = true;
            NodePtr s_crop = crops[0]; // small
            NodePtr l_crop = crops[1]; // large
            // move small crop behind large crop
            s_crop->clear_inputs();
            s_crop->add_input(l_crop->outputs()[0]);

            std::vector<std::vector<int>> new_crop = s_crop->params()["crops"];
            std::vector<std::vector<int>> large_crop = l_crop->params()["crops"];
            for (size_t i = 0; i < large_crop.size(); i++)
            {
                int shift = large_crop[i][0];
                new_crop[i][0] -= shift;
                new_crop[i][1] -= shift;
            }
            s_crop->params()["crops"] = new_crop;
        }
    }
    return is_changed;
}

bool remove_crop_concat_crop(GraphPtr g)
{
    auto criteria = [](NodePtr n) -> bool {
        // if (!is_tile_node(n))
        // {
        //     return false;
        // }
        if (n->type() == OpType_Concat || n->type() == OpType_InternalConcatenated)
        {
            // check concat
            if (n->params().count("axis") == 0)
            {
                return false;
            }
            // skip io_tensors (CB-5389)
            if (n->outputs().size() > 0 and n->outputs()[0]->is_io_tensor())
            {
                return false;
            }

            int axis = n->params()["axis"];

            auto in_nums = n->inputs().size();
            auto out_nums = n->outputs()[0]->consumers().size();
            if (in_nums != out_nums)
            {
                return false;
            }

            // crop number
            auto in_crops = get_in_crops(n);
            auto out_crops = get_out_crops(n);
            if (in_crops.size() == 0 || in_crops.size() != in_nums ||
                out_crops.size() == 0 || out_crops.size() != out_nums)
            {
                return false;
            }

            // in offset
            // for (auto in_crop : in_crops)
            // {
            //     std::vector<std::vector<int>> crops = in_crop->params()["crops"];
            //    if (crops[axis][0] != 0)
            //    {
            // return false;
            //    }
            //}
            // Note: in offset not always start wiht 0 if enabled actg

            // same offset

            int offset = 0;
            for (size_t i = 0; i < in_crops.size(); ++i)
            {
                std::vector<std::vector<int>> crops = out_crops[i]->params()["crops"];
                if (crops[axis][0] != offset)
                {
                    return false;
                }
                offset += in_crops[i]->outputs()[0]->shape()[axis];
            }

            // size
            for (size_t i = 0; i < out_crops.size(); ++i)
            {
                auto in_shape = in_crops[i]->inputs()[0]->shape();
                auto out_shape = out_crops[i]->outputs()[0]->shape();
                if (in_shape != out_shape)
                {
                    return false;
                }
            }
            // DEBUG() << n->name() << ", in_crop_num: " << in_nums << ", out_crop_num: " << out_nums;
            return true;
        }
        return false;
    };
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto in_crops = get_in_crops(n);
        auto out_crops = get_out_crops(n);
        for (size_t i = 0; i < in_crops.size(); ++i)
        {
            auto in_crop = in_crops[i];
            auto out_crop = out_crops[i];
            auto in_t = in_crop->inputs()[0];
            auto out_t = out_crop->outputs()[0];
            for (auto c : out_t->consumers())
            {
                auto idx = c->remove_input(out_t);
                c->add_input(in_t, idx);
            }
            in_crop->clear_inputs();
            in_crop->clear_outputs();
            out_crop->clear_inputs();
            out_crop->clear_outputs();
        }
        n->clear_inputs();
        n->clear_outputs();
        for (size_t i = 0; i < in_crops.size(); ++i)
        {
            g->remove_node(in_crops[i]);
            g->remove_node(out_crops[i]);
        }
        g->remove_node(n);
    }
    return true;
}

/**
 * @brief get_subnodes
 * @param root_point
 * @return
 * @note:
 * 		when making tile, use get_subgraph()
 */
static NodePtrList get_subnodes(const NodePtr &root_point)
{
    NodePtrList subnodes;
    NodePtrList root_points = {root_point};
    while (root_points.size() > 0)
    {
        auto n = root_points[0];

        // only one merge point per-slice
        if (subnodes.size() > 1 && keep_isolate_when_scheduling(n))
        {
            DEBUG() << "ignore merge point: " << n->name();
        }
        // ignore visited subnodes
        else if (n->loc() && n->loc()->through_pass("tnode_to_sg"))
        {
            DEBUG() << "ignore grouped subnodes: " << n->name();
        }
        else if (!contains(subnodes, n) && is_tile_node(n))
        {
            // this node should be grouping into silce
            subnodes.push_back(n);

            for (auto t : n->inputs())
            {
                auto parent = t->node();
                if (parent == nullptr)
                    continue;

                if (t->get_attr<bool>("ACTG_SHARE"))
                {
                    // skip the tensor: actg crop    -> concat
                    //                 [other slice]   [this slice]
                    // this flag set at: tiling_node_along_axis
                    continue;
                }

                if (!contains(root_points, parent))
                {
                    root_points.push_back(parent);
                }
            }
            // skip all output but actg share crop
            for (auto child : n->children())
            {
                if (child == nullptr)
                    continue;

                if (contains(root_points, child))
                    continue;

                if (is_tile_actg_crop(child))
                    root_points.push_back(child);
            }
        }
        root_points.erase(root_points.begin());
    }
    return subnodes;
}

/**
 * @brief is_perfect_slice
 * @param g
 * @return groups: if is not perfect slice, return empty list
 *
 * pattern:
 * this pattern graph is made by another tools support multi batch):
 *
 * a->b->c->d---|
 * a->b->c->d---+-concat->output
 * a->b->c->d-e-|
 *
 * group nodes to
 * [a,b,c,d],[a,b,c,d],[a,b,c,d,e]
 *
 * let all concat alone
 * let constants alone
 */
std::vector<NodePtrList> is_perfect_slice(GraphPtr g)
{
    std::vector<NodePtrList> groups;
    // criteria start
    // all output must be concat
    NodePtrList concat_list = g->get_output_nodes();
    for(NodePtr concat : concat_list)
        if(!(concat->type() == OpType_Concat || concat->type() == OpType_InternalConcatenated))
            return groups;
    NodePtr concat = concat_list[0];

    int32_t core_num = Target::get_the_target()->core_num();
    if(concat->parents().size() % core_num != 0)
        return groups;
    // creteria end

    // bfs starts from one of concat
    std::vector<std::set<NodePtr>> slices_candidate(concat->inputs().size());
    for(uint32_t i = 0; i < concat->parents().size(); i++)
    {
        std::vector<NodePtr> next;
        next.push_back(concat->parents()[i]);

        while(!next.empty()) {
            NodePtr n = next.back();
            next.pop_back();

            // stop search
            if(slices_candidate[i].count(n)) {
                continue; // visited
            }
            if(contains(concat_list, n)) {
                continue; // output
            }
            if(n->parents().empty()) {
                continue; // input or constant node
            }

            // inside slice
            slices_candidate[i].insert(n);

            // check: prohibited visit other slices
            for(uint32_t j = 0; j < slices_candidate.size(); j++)
            {
                if(i != j)
                {
                    if(slices_candidate[j].count(n))
                        return groups;
                }
            }

            // search parents and children
            for(NodePtr p : n->parents())
                next.push_back(p);
            for(NodePtr c : n->children())
                next.push_back(c);
        }
    }

    // check length
    for(uint32_t i = 1; i < slices_candidate.size(); i++)
    {
        if(slices_candidate[i].size() <= 1)
            return groups;
    }

    // make group
    for(uint32_t i = 0; i < slices_candidate.size(); i++)
        groups.push_back(NodePtrList(slices_candidate[i].begin(), slices_candidate[i].end()));

    return groups;
}

/**
 * @brief tnode_to_sg
 *
 * A pass grouping node by tiling slice
 *
 * e.g.
 *
 * conv1 -> conv2 -> conv3
 *
 * If user configurate tile these 3 nodes into 2 groups:
 *
 * 1: conv1:			tile 3 slices
 * 2: conv2 -> conv3:	tile 3 slices
 *
 * this pass make the graph 6 (not 3) slices:
 *
 *      / [conv1_0](slice0) - [conv2_0 - conv3_0](slice3) \
 * crop - [conv1_1](slice1) - [conv2_1 - conv3_1](slice4) - concat
 *      \ [conv1_2](slice2) - [conv2_2 - conv3_2](slice5) /
 *
 * TODO: CP-24337: new impl
 */
void tnode_to_sg(GraphPtr g)
{
    auto criteria_concat = [](NodePtr n) -> bool {
        auto real_optype = get_real_op_type(n);
        // rules: n is concat, n is not tiled, all its parents are tiled
        if ((real_optype == OpType_Concat || real_optype == OpType_InternalConcatenated))
        {
            if (is_tile_node(n))
                return false;

            auto parents = n->parents();
            for (auto p : parents)
            {
                if (!is_tile_node(p))
                {
                    return false;
                }
            }
            return true;
        }
        return false;
    };
    auto nodes = g->nodes();
    auto get_internal_tensors = [](const NodePtrList &nodes) -> TensorPtrList {
        TensorPtrList tensors;
        for (auto n : nodes)
        {
            for (auto t : n->outputs())
            {
                bool is_internal_tensor = true;
                for (auto c : t->consumers())
                {
                    if (!contains(nodes, c))
                    {
                        is_internal_tensor = false;
                    }
                }
                if (is_internal_tensor && !contains(tensors, t))
                {
                    tensors.push_back(t);
                }
            }
        }
        return tensors;
    };
    auto remove_common_nodes = [](std::vector<NodePtrList> &groups) -> void {
        std::map<NodePtr, uint32_t> map;
        for (auto group : groups)
        {
            for (auto n : group)
            {
                if (map.count(n))
                {
                    map[n] += 1;
                }
                else
                {
                    map[n] = 1;
                }
            }
        }
        NodePtrList common_nodes;
        for (auto iter = map.begin(); iter != map.end(); ++iter)
        {
            auto c = iter->first;
            auto cnt = iter->second;
            if (cnt > 1)
            {
                common_nodes.push_back(c);
            }
        }
        for (auto c : common_nodes)
        {
            for (auto &&group : groups)
            {
                // remove common node from group
                auto iter = std::remove(group.begin(), group.end(), c);
                group.erase(iter, group.end());
            }
        }
    };

    auto group_to_sg = [&get_internal_tensors](const NodePtrList &group,const GraphPtr &g) -> SubGraphNodePtr {

        auto internal_tensors = get_internal_tensors(group);
        for (auto t : internal_tensors)
        {
            auto ptr = t->memory_offset()->get_base_object();
            ptr->set_mem_flag(1);
        }
        auto sg = SubGraph::create(g, group);
        SubGraphNodePtr sgnode = sg->as_node();
        sgnode->set_unified_view(false);

        int64_t load = 0;
        for (auto n : group)
        {
            int64_t n_load = 0;
            if (n->attrs().count("count_cycles"))
            {
                n_load = n->get_attr("count_cycles");
                // INFO() << node->name() << " load: " << load;
            }
            load += n_load;
        }
        sgnode->set_attr("count_cycles", load);
        return sgnode;
    };



    DEBUG() << "tnode_to_sg start";

    std::vector<NodePtrList> groups = is_perfect_slice(g);
    if(!groups.empty())
    {
        for(uint32_t i = 0; i < groups.size(); i++)
        {
            NodePtrList group = groups[i];
            if(!group.empty())
            {
                SubGraphNodePtr sgnode = group_to_sg(group, g);
                sgnode->set_name(g->get_valid_node_name("perfect_slice_" + std::to_string(i)));
                MODIFY_NODE(sgnode);
            }
        }
        // g->to_svg("after_group.svg");
        return;
    }


    for (auto n : nodes)
    {
        std::vector<NodePtrList> groups;
        std::vector<std::string> group_names;
        // start from tile concat
        if (criteria_concat(n))
        {
            // group slice foreach input
            auto inps = n->inputs();
            for (size_t i = 0; i < inps.size(); ++i)
            {
                auto inp = inps[i];
                auto parent = inp->node();
                // get subnodes;
                auto group = get_subnodes(parent);
                // for (auto sub_n : group)
                // {
                // DEBUG() << "\t" << sub_n->name() << ", type: " << get_real_op_type(sub_n);
                // }
                groups.push_back(group);
                group_names.push_back(inps[i]->node()->name() + "_tiled_SG_");
            }
            remove_common_nodes(groups);
        }
        // or start from merge point
        else if (keep_isolate_when_scheduling(n))
        {
            // group slice include this node
            auto group = get_subnodes(n);
            groups.push_back(group);
            group_names.push_back(n->name() + "_tiled_SG_");
        }
        else
        {
            continue;
        }

        DEBUG() << n->name() << "grouping graph into " << groups.size() << "slice(s)";
        for (size_t i = 0; i < groups.size(); ++i)
        {
            DEBUG() << "group " << i << " (" << group_names[i] << ") have " << groups[i].size() << " nodes";
            auto group = groups[i];
            if (!group.empty())
            {
                SubGraphNodePtr sgnode = group_to_sg(group, g);
                sgnode->set_name(g->get_valid_node_name(group_names[i] + std::to_string(i)));
                MODIFY_NODE(sgnode);
            }
        }
    }

    // g->to_svg("after_group.svg");
}

static void simplify_crop(GraphPtr g)
{
    // flat crop to NHWC or C32

    auto get_axis_list = [](const NodePtr &n) -> std::vector<int> {
        std::vector<int> axis_list;
        auto i_shape = n->inputs()[0]->shape();
        auto o_shape = n->outputs()[0]->shape();
        for (int i = 0; i < i_shape.dim(); ++i)
        {
            if (i_shape[i] != o_shape[i])
            {
                axis_list.push_back(i);
            }
        }
        return axis_list;
    };
    // old_shape: [1,96,54,3,4,4]
    // axis: 1
    // new_shape: [1,96,54,48]
    auto get_compact_shape = [](TensorShape &shape, int axis) -> TensorShape {
        uint32_t n = 1;
        for (int i = 0; i < axis; ++i)
        {
            n *= shape[i];
        }
        uint32_t h = shape[axis];

        uint32_t w = shape[axis + 1];
        uint32_t c = 1;
        if (axis < shape.dim() - 2)
        {
            for (int i = axis + 2; i < shape.dim(); ++i)
            {
                c *= shape[i];
            }
        }
        TensorShape compact_shape(n, h, w, c);
        return compact_shape;
    };

    auto criteria = [get_axis_list](NodePtr &n) -> bool {
        bool ret = true;
        if (n->type() != OpType_Crop)
        {
            return false;
        }
        if (n->inputs().empty() || n->outputs().empty())
        {
            return false;
        }
        auto out_shape = n->outputs()[0]->shape();
        if (out_shape.dim() != 6)
        {
            return false;
        }
        auto axis_list = get_axis_list(n);
        // TODO: support more cases
        if (axis_list.size() != 1 || axis_list[0] != 1)
        {
            return false;
        }
        return ret;
    };
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!criteria(n))
            continue;

        DEBUG() << n->name();
        // create reshape_0: [1,96,54,3,4,4] => [1,96,54,48]
        auto axis_list = get_axis_list(n);
        int axis_old = axis_list[0];
        int axis_new = 1; // FIXME
        auto reshape_0_shape = get_compact_shape(n->inputs()[0]->shape(), axis_old);
        reshape_0_shape.set_layout(DataLayout_NHWC);

        auto reshape_0_out = ops::reshape(n->inputs()[0], reshape_0_shape);
        auto reshape_0 = reshape_0_out->node();

        g->add_node(reshape_0);
        // ADD_LOCATION_BY_NODE(reshape_0, n);

        // create crop: [1,96,54,48] => [1,32,54,48]
        TensorPtr original_out = n->outputs()[0];
        std::vector<std::vector<int>> crops_old = n->params()["crops"];
        std::vector<std::pair<int32_t, int32_t>> crops_new;
        for (int i = 0; i < reshape_0_shape.dim(); ++i)
        {
            if (i == axis_new)
            {
                crops_new.push_back({crops_old[axis_old][0], crops_old[axis_old][1]});
            }
            else
            {
                crops_new.push_back({0, reshape_0_shape[i]});
            }
        }
        auto crop_out = ops::crop(reshape_0_out, crops_new);
        auto crop_node = crop_out->node();

        crop_node->set_attr("tile_id", n->get_attr<int>("tile_id"));

        g->add_node(crop_node);
        ADD_LOCATION_BY_NODE(crop_node, n);

        // create reshape_1
        auto reshape_1_out = ops::reshape(crop_out, n->outputs()[0]->shape());
        auto reshape_1_node = reshape_1_out->node();
        reshape_1_node->remove_output(0);
        reshape_1_node->add_output(n->outputs()[0]);

        g->add_node(reshape_1_node);
        ADD_LOCATION_BY_NODE(reshape_1_node, n);
        g->remove_node(n);
    }
}

static void simplify_concat(GraphPtr g)
{
    // flat crop to NHWC or C32

    auto get_axis_list = [](const NodePtr &n) -> std::vector<int> {
        std::vector<int> axis_list;
        auto i_shape = n->inputs()[0]->shape();
        auto o_shape = n->outputs()[0]->shape();
        for (int i = 0; i < i_shape.dim(); ++i)
        {
            if (i_shape[i] != o_shape[i])
            {
                axis_list.push_back(i);
            }
        }
        return axis_list;
    };
    // old_shape: [1,3,32,4,54,4]
    // axis: 2
    // new_shape: [3,32,27,32]
    auto get_compact_shape = [](TensorShape &shape, int axis) -> TensorShape {
        uint32_t n = 1;
        for (int i = 0; i < axis; ++i)
        {
            n *= shape[i];
        }
        uint32_t h = shape[axis];

        uint32_t w = 1;
        if (axis < shape.dim() - 1)
        {
            for (int i = axis + 1; i < shape.dim(); ++i)
            {
                w *= shape[i];
            }
        }
        w /= 32;
        uint32_t c = 32;
        TensorShape compact_shape(n, h, w, c);
        return compact_shape;
    };

    auto criteria = [get_axis_list](NodePtr &n) -> bool {
        bool ret = true;
        if (n->type() != OpType_Concat)
        {
            return false;
        }
        if (n->inputs().empty() || n->outputs().empty())
        {
            return false;
        }
        auto out_shape = n->outputs()[0]->shape();
        if (out_shape.dim() != 6)
        {
            return false;
        }
        auto axis_list = get_axis_list(n);
        // TODO: support more cases
        if (axis_list.size() != 1 || axis_list[0] != 2)
        {
            return false;
        }
        return ret;
    };
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!criteria(n))
            continue;

        DEBUG() << n->name();
        // create reshape_0: [1,3,32,4,54,4] => [3,32,27,32]
        auto axis_list = get_axis_list(n);
        int axis_old = axis_list[0];
        int axis_new = 1; // FIXME
        TensorPtrList reshape_0_outs;
        for (auto inp : n->inputs())
        {
            auto reshape_0_shape = get_compact_shape(inp->shape(), axis_old);
            reshape_0_shape.set_layout(DataLayout_NHWC);

            auto reshape_0_out = ops::reshape(inp, reshape_0_shape);
            reshape_0_outs.push_back(reshape_0_out);
            auto reshape_0 = reshape_0_out->node();

            g->add_node(reshape_0);
            ADD_LOCATION_BY_NODE(reshape_0, inp->node());
        }

        // create concat: [3,32,27,32] => [3,96,27,32]
        TensorPtr original_out = n->outputs()[0];
        auto concat_out = ops::concat(reshape_0_outs, axis_new);
        auto concat = concat_out->node();

        concat->set_attr("tile_id", n->get_attr<int>("tile_id"));
        g->add_node(concat);
        ADD_LOCATION_BY_NODE(concat, n);

        // create reshape_1
        auto reshape_1_out = ops::reshape(concat_out, n->outputs()[0]->shape());
        auto reshape_1_node = reshape_1_out->node();
        reshape_1_node->remove_output(0);
        reshape_1_node->add_output(n->outputs()[0]);

        g->add_node(reshape_1_node);
        ADD_LOCATION_BY_NODE(reshape_1_node, n);
        g->remove_node(n);
    }
}

static TileSubGraphMap apply_tile(GraphPtr g, const TilePlan &out_tile_infos, int32_t axis = 1,
                       bool actg = false)
{
    TileSubGraphMap new_map;

    DEBUG() << "apply tiling plan of axis: " << axis << "...";
    show_tile_plan(out_tile_infos);

    // target_in_infos_list_map: map the n and it's expected in_infos_list
    std::map<NodePtr, std::vector<InputTileInfo>> target_in_infos_list_map;
    for (auto it : out_tile_infos)
    {
        auto n = it.first;
        // node who's parent is branch point
        std::vector<InputTileInfo> target_in_infos_list;
        auto out_axis = get_transpose_out_axis(n, axis);
        auto in_axis = get_in_axis(n, out_axis);
        for (size_t i = 0; i < n->inputs().size(); ++i)
        {
            auto inpt = n->inputs()[i];
            auto parent = inpt->node();
            TileInfo out_infos = {};
            // branch point
            // if (out_tile_infos.count(parent) > 0 && !contains(merge_points, parent))
            if (out_tile_infos.count(parent) > 0)
            {
                auto out_infos_tmp = out_tile_infos.at(parent);
                auto overlap_rate = get_overlap_rate(out_infos_tmp, inpt->shape()[in_axis]);

                if (overlap_rate > 0)
                {
                    out_infos = out_infos_tmp;
                }
            }
            auto in_infos = convert_infos_out2in(out_infos);

            target_in_infos_list.push_back(in_infos);
        }
        target_in_infos_list_map.insert(std::make_pair(n, target_in_infos_list));
    }

    for (auto it : out_tile_infos)
    {
        auto n = it.first;
        auto out_infos = it.second;
        auto target_in_infos_list = target_in_infos_list_map[n];
        auto out_axis = axis; // get_transpose_out_axis(n, axis)
        // TODO: refine get_transpose_out_axis()
        auto tnodes = tiling_node_along_axis(g, n, out_axis, out_infos, target_in_infos_list, actg);

        /**
         * Do NOT dereference node because it has been removed.
         * We use it as a tag to update mp_bp_map/max_num_map for multi-dims cases.
         */
        new_map[n] = tnodes;
    }

    // post tile

    // g->serialize("graph_before_post_tile.txt");
    simplify_crop(g);
    simplify_concat(g);
    CSE_subnode(g);
    // after this pass, multi brahch's pre_crops merged into 1 crop
    // g->serialize("graph_after_CSE_subnode.txt");
    fuse_reshape(g);
    eliminate_reshape(g);
    lower_actg_crop(g);
    // g->serialize("graph_after_lower_crop.txt");

    remove_crop_concat_crop(g);
    // check remove_crop_concat_crop result
    // g->serialize("graph_after_remove_crop_concat_crop.txt");
/*
    for (const NodePtr &node : g->nodes())
    {
        if (node->get_attr<bool>("TILE_MERGE_POINT"))
        {
            for (const NodePtr &child : node->children())
            {
                if (is_tile_node(child))
                    WARN() << "merge point: " << node->name()
                           << " have a tiled child: " << child->name()
                           << " maybe this merge point could be removed";
            }
        }
    }
*/

    eliminate_crop(g);
    fuse_concat_splits(g);
    DEBUG() << "tiling axis " << axis << " done";
    return new_map;
}

static void apply_tile_c(GraphPtr g, const TileSubGraphMap &mp_bp_map2, const TileNumMap &mp_num_map)
{
    DEBUG() << "tiling c...";
    // one of the parents ready
    auto align = [](uint32_t num, uint32_t alignment) -> uint32_t {
        return (((num + alignment - 1) / alignment) * alignment);
    };
    auto get_out_c = [align](const NodePtr &n, uint32_t split_num) -> std::vector<int> {
        auto ori_c = n->outputs()[0]->shape()[-1];
        if (ori_c <= 8 || split_num <= 1)
        {
            return {ori_c};
        }
        auto output_type = n->outputs()[0]->type();
        uint32_t alignment = 16;
        if ((output_type == Tensor::INT8) || (output_type == Tensor::UINT8))
            alignment = 32;

        if (n->type() == OpType_Reduce)
        {
            uint32_t c_index = n->inputs()[0]->shape().dim() - 1;
            std::vector<uint32_t> axis = n->params()["axis"];
            // if [1, 100, 100, 2(1+1)] -> [1, 25, 25, 2(1+1)]  align 1 is ok
            // channel is going to transpose to batch here
            if (!contains(axis, c_index))
            {
                alignment = 1;
            }
        }

        // FIXME
        if (ori_c == 48 && split_num == 3)
        {
            alignment = 16;
        }
        auto max_size = align(ori_c / split_num, alignment);
        auto num_0 = ori_c / max_size;
        auto rest = ori_c - (max_size * num_0);

        std::vector<int> slices;
        for (size_t i = 0; i < num_0; ++i)
        {
            slices.push_back(max_size);
        }
        if (num_0 > 0 && rest > 0)
        {
            slices.push_back(rest);
        }
        return slices;
    };
    auto show_vector = [](std::vector<int> vec) -> std::string {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < vec.size(); ++i)
        {
            oss << vec[i];
            if (i < vec.size() - 1)
            {
                oss << ",";
            }
        }
        oss << "]";
        return oss.str();
    };
    NodePtrList nodes;
    TileNumMap max_num_map;
    for (auto mp_bp : mp_bp_map2)
    {
        auto mp = mp_bp.first;
        auto bps = mp_bp.second;
        auto sg = get_subgraph(mp, bps);
        auto max_num = mp_num_map.at(mp);
        for (auto n : sg)
        {
            if (!contains(g->nodes(), n))
            {
                continue; // skiped removed node, it handled by other method
            }
            if (!contains(nodes, n) && max_num != 1)
            {
                nodes.push_back(n);
                max_num_map[n] = max_num;
            }
        }
    }
    // for (auto n : ready_nodes)
    for (auto n : nodes)
    {
        auto max_tile_num = max_num_map.at(n);
        auto out_c = get_out_c(n, max_tile_num);
        if (out_c.empty())
        {
            WARN() << "Tiling: failed to tile c for node: " << n->name();
            continue;
        }
        if (n->type() == OpType_SubGraphInput)
        {
            continue;
        }
        DEBUG() << n->name() << ", outc:" << show_vector(out_c);
        auto split_c = SplitNodeOutputC();
        split_c.set_output_c(out_c);
        split_c.split_output_c(g, n);
    }

    fuse_concat_split(g);
    config_tiled_graph(g);

    DEBUG() << "tiling c done";
}

static void apply_tile_multi_dims(const GraphPtr g, TileSubGraphMap &mp_bp_map, TileNumMap &max_num_map,
                                  std::vector<MultiDimTile> tiles, uint32_t idx)
{
    if (idx >= tiles.size())
    {
        return;
    }

    auto axis = tiles[idx].axis;
    auto tile_num = tiles[idx].num;
    auto plan = make_plan_axis(g, mp_bp_map, max_num_map, axis);
    auto new_map = apply_tile(g, plan, axis);

    for (auto original : mp_bp_map)
    {
        auto original_mp = original.first;
        auto original_bps = original.second;

        for (uint32_t t = 0; t < tile_num; t++)
        {
            auto new_mp = new_map[original_mp][t];
            NodePtrList new_bps;
            TileSubGraphMap new_mp_bp_map;
            TileNumMap new_max_num_map;

            for (auto n : original_bps)
            {
                new_bps.push_back(new_map[n][t]);
            }

            new_mp_bp_map[new_mp] = new_bps;
            new_max_num_map[new_mp] = (idx + 1) < tiles.size() ? tiles[idx + 1].num : 1;
            apply_tile_multi_dims(g, new_mp_bp_map, new_max_num_map, tiles, idx + 1);
        }
    }
}

static void apply_tile_multi_dims(const GraphPtr g, TileSubGraphMap &mp_bp_map,
                                  TileNumMapMultiDim &max_num_map_dims)
{
    /* tile all subgraphs */
    for (auto m : mp_bp_map)
    {
        auto mp = m.first;
        auto tile_nums = max_num_map_dims[mp];
        TileSubGraphMap mp_bp_map_tmp;
        mp_bp_map_tmp[mp] = mp_bp_map[mp];
        TileNumMap max_num_map;
        max_num_map[mp] = tile_nums[0].num;

        apply_tile_multi_dims(g, mp_bp_map_tmp, max_num_map, tile_nums, 0);
    }
}

static bool tile_graph(GraphPtr g, uint32_t axis, uint32_t tile_num, int gm_size)
{
    TileSubGraphMap mp_bp_map;
    TileNumMap max_num_map;

    if (tile_num == 0 || gm_size == 0)
    {
        WARN() << "invalid core_num or gm_size";
        return false;
    }

    auto oldg = g->clone();

    if (axis == 0 || axis == 1)
    {
        auto tile_plan = make_plan_axis(g, mp_bp_map, max_num_map, axis, true, false, tile_num, gm_size);
        apply_tile(g, tile_plan, axis);
    }
    else if (axis == 3)
    {
        make_plan_c(g, mp_bp_map, max_num_map);
        apply_tile_c(g, mp_bp_map, max_num_map);
    }
    else
    {
        WARN() << "invalid axis: " << axis;
        return false;
    }

    if (!IRChecker::check_graph(g))
    {
        WARN() << "Tiling failed! The original graph is restored.";
        oldg->clone(g);
        return false;
    }

    return true;
}

bool tile_graph_by_size(GraphPtr g, uint32_t axis, uint32_t tile_num, uint32_t gm_size, TileSubGraphMap &mp_bp_map, TileNumMap &max_num_map, double threshold)
{
    if (axis == 0 || axis == 1)
    {
        auto tile_plan = make_plan_axis(g, mp_bp_map, max_num_map, axis, false, false, tile_num, gm_size, threshold);
        apply_tile(g, tile_plan, axis);
    }
    else if (axis == 3)
    {
        make_plan_c(g, mp_bp_map, max_num_map);
        apply_tile_c(g, mp_bp_map, max_num_map);
    }
    else
    {
        WARN() << "invalid axis: " << axis;
        return false;
    }
    return true;
}

bool tile_graph_by_size(GraphPtr g, uint32_t axis, uint32_t core_num, uint32_t gm_size)
{
    return tile_graph(g, axis, core_num, gm_size);
}

static void parse_tile_cfg(GraphPtr g, std::string filename,
                           TileSubGraphMap &mp_bp_map_n,
                           TileSubGraphMap &mp_bp_map_h,
                           TileSubGraphMap &mp_bp_map_w,
                           TileSubGraphMap &mp_bp_map_c,
                           TileSubGraphMap &mp_bp_map_nhw,
                           TileNumMap &max_num_map_n,
                           TileNumMap &max_num_map_h,
                           TileNumMap &max_num_map_w,
                           TileNumMap &max_num_map_c,
                           TileNumMapMultiDim &max_num_map_nhw,
                           bool &en_actg, bool &auto_tile, double &overlap_threshold)
{
    std::map<int, NodePtr> id_map;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (n->type() == OpType_Loop || n->type() == OpType_If)
        {
            INFO() << "Loop node not supprt split for now.";
            continue;
        }

        int id = n->get_attr("layer_id");
        id_map[id] = n;
    }

    std::ifstream json_file(filename);
    std::ostringstream tmp;
    tmp << json_file.rdbuf();
    auto str = tmp.str();
    auto ptr = utils::JsonObject::parse(str);

    /* disable fps tiling of specified op(s) */
    if (ptr.exist("disable_op"))
    {
        auto op_str = ptr["disable_op"].array();
        for (auto s : op_str)
        {
            get_disable_tiling_ops().insert(string_to_optype(s));
        }
        return;
    }

    if (ptr.exist("actg") && ptr["actg"].is_bool())
    {
        en_actg = ptr["actg"];
    }

    if (ptr.exist("auto") && ptr["auto"].is_bool())
    {
        INFO() << "using auto tile mode with more configuration";
        auto_tile = ptr["auto"];
        if (ptr.exist("overlap_threshold"))
        {
            overlap_threshold = ptr["overlap_threshold"];
        }
        return; // no need parser other cfg
    }

    /*
    * set gsize with "--tiling tile.json", an example in json file:
    * {
    *     "lib_cfg": [
    *         {
    *             "NodeID": [6,7],  // node ID list (layer_id in IR)
    *             "gsize": 0        // gsize 0~7
    *         },  // Support adding multiple node group configurations (separated by commas)
    *     ]
    * }
    */
    if (ptr.exist("lib_cfg"))
    {
        auto lib_cfg = ptr["lib_cfg"].array();
        for (auto cfg : lib_cfg)
        {
            auto node_ids = cfg["NodeID"].array();
            uint32_t gsize = cfg["gsize"];
            for (auto id : node_ids)
            {
                if (id_map.count(id) == 0)
                {
                    WARN() << "lib_cfg" << ": Node (" << id << ") no exist";
                    continue;
                }
                id_map[id]->set_attr(Constants::gsize, gsize);
                DEBUG() << "Set node " << id  << "(" << id_map[id]->name() << ") gsize: " << id_map[id]->get_attr<uint32_t>(Constants::gsize);
            }
        }
    }

    auto check_sg_supported = [&](NodePtrList &sg, int axis) -> bool {
        for (NodePtr &node : sg)
        {
            if (!is_supported_op_manual(node, axis))
            {
                ERROR() << "Node (" << node->name() << ")" << " cannot be tiled: axis " << axis;
                return false;
            }
        }
        return true;
    };

    auto get_max_tile_num = [&](std::string tag, utils::JsonObject &cfg, NodePtrList &sg,
                                std::vector<MultiDimTile> &tiles, int axis) -> void {
        MultiDimTile tile;

        if (cfg.exist(tag) && cfg[tag].t == utils::JsonObject::INT &&
            (uint32_t)cfg[tag] > 1 &&
            check_sg_supported(sg, axis))
        {
            tile.axis = axis;
            tile.num = cfg[tag];
            tiles.push_back(tile);
        }
    };

    auto parse_sub_cfg = [&](std::string sub_str,
                             TileSubGraphMap &mp_bp_map,
                             TileNumMap *max_num_map = nullptr) -> void {
        if (ptr.exist(sub_str))
        {
            int axis;
            if ("tile_n" == sub_str)
                axis = 0;
            else if ("tile_h" == sub_str)
                axis = 1;
            else if ("tile_w" == sub_str)
                axis = 2;
            else if ("tile_c" == sub_str)
                axis = 3;
            else if ("tile_nhwc" == sub_str) /* we only support multi-dims nhw now */
                axis = -1;
            else
            {
                DEBUG() << "skip tile cfg field:" << sub_str;
                return;
            }

            auto sub_cfg = ptr[sub_str].array();
            for (auto cfg : sub_cfg)
            {
                int mp_id = cfg["MP"];
                if (id_map.count(mp_id) == 0)
                {
                    ERROR() << sub_str << ": Node MP (" << mp_id << ") no exist";
                    return;
                }
                NodePtr mp = id_map[mp_id];

                NodePtrList bps;
                auto bp_names = cfg["BP"].array();
                for (auto name : bp_names)
                {
                    int bp_id = name;
                    if (id_map.count(bp_id) == 0)
                    {
                        ERROR() << sub_str << ": Node BP (" << bp_id << ") no exist";
                        return;
                    }
                    bps.push_back(id_map[bp_id]);
                }

                NodePtrList sg = get_subgraph(mp, bps, true);
                if (sg.empty())
                {
                    ERROR() << sub_str << ": cfg" << cfg;
                    ERROR() << "cannot build sub graph";
                    return;
                }

                mp_bp_map[mp] = bps;
                if (axis == -1)
                {
                    std::vector<MultiDimTile> tiles;
                    get_max_tile_num("max_tile_num_n", cfg, sg, tiles, 0);
                    get_max_tile_num("max_tile_num_h", cfg, sg, tiles, 1);
                    get_max_tile_num("max_tile_num_w", cfg, sg, tiles, 2);
                    max_num_map_nhw[mp] = tiles;
                }
                else
                {
                    if (check_sg_supported(sg, axis))
                    {
                        if (cfg.exist("max_tile_num") && cfg["max_tile_num"].t == utils::JsonObject::INT)
                        {
                            int num = cfg["max_tile_num"];
                            if (num > 0 && max_num_map)
                            {
                                (*max_num_map)[mp] = cfg["max_tile_num"];
                            }
                        }
                    }
                    else
                    {
                        ERROR() << "check sg failed: mp " << mp->name() << ", axis " << axis;
                    }
                }
            }
        }
    };

    // tile
    parse_sub_cfg("tile_n", mp_bp_map_n, &max_num_map_n);
    parse_sub_cfg("tile_h", mp_bp_map_h, &max_num_map_h);
    parse_sub_cfg("tile_w", mp_bp_map_w, &max_num_map_w);
    parse_sub_cfg("tile_c", mp_bp_map_c, &max_num_map_c);
    parse_sub_cfg("tile_nhwc", mp_bp_map_nhw);

    if (max_num_map_nhw.size())
    {
        DEBUG();
        DEBUG() << "[tile_multi_dims]:";
        for (auto m : max_num_map_nhw)
        {
            auto mp = m.first;
            auto tiles = m.second;

            DEBUG() << "mp: " << mp->name();
            for (auto t : tiles)
            {
                DEBUG() << "-- axis: " << t.axis << ", tile_num: " << t.num;
            }
        }
    }

    return;
}

void dump_cfg_to_json(utils::JsonObject &json,
                      std::string sec_name,
                      const TileSubGraphMap &mp_bp_map,
                      const TileNumMap &max_num_map)
{
    if (mp_bp_map.size() != max_num_map.size() || mp_bp_map.size() == 0)
    {
        return;
    }

    auto get_id = [](NodePtr n) -> int {
        int original_id = n->get_attr<int>("original_id");
        int id = n->get_attr<int>("layer_id");
        return id + original_id;
    };

    auto &tile_h = json[sec_name];
    NodePtrList mp_list;
    for (auto iter = mp_bp_map.begin(); iter != mp_bp_map.end(); ++iter)
    {
        auto mp = iter->first;
        bool is_unique = true;

        for (auto m : mp_list)
        {
            /* exclude nodes splited from the same original node */
            if (get_id(m) == get_id(mp))
            {
                is_unique = false;
                break;
            }
        }

        if (is_unique)
        {
            mp_list.push_back(mp);
        }
    }
    auto cmp = [&](NodePtr n0, NodePtr n1) -> bool {
        int id_0 = get_id(n0);
        int id_1 = get_id(n1);
        return (id_0 < id_1);
    };
    std::sort(mp_list.begin(), mp_list.end(), cmp);
    for (auto mp : mp_list)
    {
        if (max_num_map.at(mp) == 1)
        {
            continue;
        }
        auto bps = mp_bp_map.at(mp);
        utils::JsonObject sg_info;
        auto &BP = sg_info["BP"];
        for (auto bp : bps)
        {
            BP.push_back(get_id(bp));
        }
        sg_info["MP"] = get_id(mp);
        sg_info["max_tile_num"] = max_num_map.at(mp);
        tile_h.push_back(sg_info);
    }
}

static void update_mp_bp_map(TileSubGraphMap& mp_bp_map, TileNumMap& max_num_map, TileSubGraphMap& tiled_map)
{
    TileSubGraphMap original_map = mp_bp_map;

    for (auto iter : original_map)
    {
        auto original_mp = iter.first;
        auto original_bps = iter.second;
        if (tiled_map.count(original_mp))
        {
            auto new_mps = tiled_map[original_mp];
            int tile_num = max_num_map[original_mp];
            mp_bp_map.erase(original_mp);
            max_num_map.erase(original_mp);
            for (uint32_t i = 0; i < new_mps.size(); i++)
            {
                auto new_mp = new_mps[i];
                NodePtrList new_bps;
                for (auto original_bp : original_bps)
                {
                    if (tiled_map.count(original_bp) && tiled_map[original_bp].size() == new_mps.size())
                    {
                        new_bps.push_back(tiled_map[original_bp][i]);
                    }
                }

                if (!new_bps.size())
                {
                    new_bps.push_back(new_mp);
                }

                mp_bp_map[new_mp] = new_bps;
                max_num_map[new_mp] = tile_num;
            }

            tiled_map.erase(original_mp);
        }
    }
}

void tile_graph(GraphPtr g)
{
    auto oldg = g->clone();
    std::string tile_cfg = BuilderConfig::get<std::string>(Constants::tile_cfg);
    set_node_nums(g);

    DEBUG() << "tiling: " << tile_cfg;

    TileSubGraphMap mp_bp_map_n; // tile_n
    TileSubGraphMap mp_bp_map_h; // tile_h
    TileSubGraphMap mp_bp_map_w; // tile_w
    TileSubGraphMap mp_bp_map_c; // tile_c
    TileSubGraphMap mp_bp_map_nhw;
    TileNumMap max_num_map_n;
    TileNumMap max_num_map_h;
    TileNumMap max_num_map_w;
    TileNumMap max_num_map_c;
    TileNumMapMultiDim max_num_map_nhw;
    TilePlan tile_plan_n;
    TilePlan tile_plan_h;
    TilePlan tile_plan_w;
    bool batched_graph = false;
    auto dump_tile_cfg = BuilderConfig::get<bool>(Constants::dump_tile_cfg);
    utils::JsonObject json;
    bool en_actg = false;
    double overlap_threshold = 1.6; // a experience value

    if (utils::endswith(tile_cfg, ".json"))
    {
        if (!utils::file_exists(tile_cfg))
        {
            ERROR() << "Manual tiling failed: the cfg file " << tile_cfg << " does NOT exist";
            return;
        }

        parse_tile_cfg(g, tile_cfg,
                       mp_bp_map_n, mp_bp_map_h, mp_bp_map_w, mp_bp_map_c, mp_bp_map_nhw,
                       max_num_map_n, max_num_map_h, max_num_map_w, max_num_map_c, max_num_map_nhw,
                       en_actg, g_auto_mode, overlap_threshold);

        /* If we find no "disable_op" in cfg.json, enable manual tiling. */
        if (!get_disable_tiling_ops().size())
        {
            g_auto_mode = false;
        }
    }

    if (!g_auto_mode)
    {
        if (en_actg)
        {
            WARN() << "The dev version Activation Gather (ACTG) is enabled.";
        }

        // When manual tiling, let scheduler know something
        append_schedule_flag(mp_bp_map_n);
        append_schedule_flag(mp_bp_map_h);
        append_schedule_flag(mp_bp_map_w);
    }
    else if (check_batched_graph_tile_n(g))
    {
        batched_graph = true;
        g_auto_mode = true;
    }
    else
    {
        INFO() << "Auto tiling now, please wait ...";
        g_auto_mode = true;
    }

    if (batched_graph)
    {
        apply_tile_batched_graph(g, tile_cfg, dump_tile_cfg);
    }
    else
    {
        /* manual + auto */
        /* tile n */
        TileSubGraphMap tiled_map_n;
        TileSubGraphMap tiled_map_h;
        TileSubGraphMap tiled_map_w;

        tile_plan_n = make_plan_axis(g, mp_bp_map_n, max_num_map_n, 0, false, en_actg, 0, 0, overlap_threshold);
        if (!tile_plan_n.empty())
        {
            if (dump_tile_cfg)
            {
                dump_cfg_to_json(json, "tile_n", mp_bp_map_n, max_num_map_n);
            }
            tiled_map_n = apply_tile(g, tile_plan_n, 0, false);
        }

        /* tile h */
        if (!g_auto_mode)
        {
            update_mp_bp_map(mp_bp_map_h, max_num_map_h, tiled_map_n);
        }
        tile_plan_h = make_plan_axis(g, mp_bp_map_h, max_num_map_h, 1, false, en_actg, 0, 0, overlap_threshold);
        if (!tile_plan_h.empty())
        {
            if (dump_tile_cfg)
            {
                dump_cfg_to_json(json, "tile_h", mp_bp_map_h, max_num_map_h);
            }
            tiled_map_h = apply_tile(g, tile_plan_h, 1, en_actg);
        }

        /* tile w */
        if (!g_auto_mode)
        {
            update_mp_bp_map(mp_bp_map_w, max_num_map_w, tiled_map_n);
            update_mp_bp_map(mp_bp_map_w, max_num_map_w, tiled_map_h);
        }
        tile_plan_w = make_plan_axis(g, mp_bp_map_w, max_num_map_w, 2, false, en_actg, 0, 0, overlap_threshold);
        if (!tile_plan_w.empty())
        {
            if (dump_tile_cfg)
            {
                dump_cfg_to_json(json, "tile_w", mp_bp_map_w, max_num_map_w);
            }
            tiled_map_w = apply_tile(g, tile_plan_w, 2, en_actg);
        }

        /* tile c */
        if (tile_cfg == "fps_dev" || !g_auto_mode)
        {
            if (!g_auto_mode)
            {
                update_mp_bp_map(mp_bp_map_c, max_num_map_c, tiled_map_n);
                update_mp_bp_map(mp_bp_map_c, max_num_map_c, tiled_map_h);
                update_mp_bp_map(mp_bp_map_c, max_num_map_c, tiled_map_w);
            }
            make_plan_c(g, mp_bp_map_c, max_num_map_c, g_auto_mode);
        }
        if (!mp_bp_map_c.empty())
        {
            if (dump_tile_cfg)
            {
                dump_cfg_to_json(json, "tile_c", mp_bp_map_c, max_num_map_c);
            }
            apply_tile_c(g, mp_bp_map_c, max_num_map_c);
        }

        if (!mp_bp_map_nhw.empty())
        {
            apply_tile_multi_dims(g, mp_bp_map_nhw, max_num_map_nhw);
        }
    }

    config_tiled_graph(g);

    if (dump_tile_cfg)
    {
        std:: string filename = "tile_cfg_dump.json";
        DEBUG();
        DEBUG() << "dump_tile_cfg ==> " << filename;

        std::ofstream os(filename);
        if (os.is_open())
        {
            os << json;
            os << std::endl;
            os.close();
        }
    }

    g->topological_sort();
    // g->serialize("graph_tiled.txt", "graph_tiled.bin");
    // g->to_svg("graph_tiled.svg");

    auto is_valid_graph = IRChecker::check_graph(g);
    if (!is_valid_graph)
    {
        WARN() << "Tiling failed! The original graph is restored.";
        oldg->clone(g);
    }
}
}; // namespace aipubt

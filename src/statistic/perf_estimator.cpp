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
 * @Date: 2023-03-03 15:30:23
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 15:16:35
 */
#include "perf_estimator.h"
#include "_BIN_RESOURCE_cost_model_x3p.cbin__.h"
#include "_BIN_RESOURCE_cost_model_x3p.txt__.h"
#include "builder/ops/ops.h"
#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/core.h"
#include "engine/engine.h"
#include "profile_data_parser.h"
#include "utils/utils.h"
#include <cmath>
#if USE_OMP
#include <omp.h>
#endif
namespace aipubt
{

static bool convert_shape(std::vector<int> &shape)
{
    if (shape.size() == 5)
    {
        shape = {shape[0], shape[2], shape[3], shape[1] * shape[4]};
    }
    else if (shape.size() == 3)
    {
        shape = {shape[0], shape[1], 1, shape[2]};
    }
    else if (shape.size() == 2)
    {
        shape = {shape[0], 1, 1, shape[1]};
    }
    else if (shape.size() == 1)
    {
        shape = {1, 1, 1, shape[0]};
    }
    else if (shape.size() != 4)
    {
        DEBUG() << "Unsupported shape dim: " << shape.size();
        return false;
    }
    return true;
}

static std::vector<int> get_shape(TensorPtr t)
{
    auto shape = t->shape();
    switch (shape.layout())
    {
    case DataLayout_NCHWC4:
    case DataLayout_NCHWC8:
    case DataLayout_NCHWC16:
    case DataLayout_NCHWC32:
    {
        shape.change_layout(DataLayout_NHWC);
        break;
    }
    case DataLayout_NCDHWC16:
    case DataLayout_NCDHWC32:
    case DataLayout_NDCHWC16:
    case DataLayout_NDCHWC32:
    {
        shape.change_layout(DataLayout_NDHWC);
        break;
    }
    default:
        break;
    }
    return shape;
}

static bool fill_key(const std::string &key, std::vector<float> &res, int &pos)
{
    static std::map<std::string, int> m_optype_keys = {
        {"Activation", 0},
        {"Add", 1},
        {"Mul", 2},
        {"Sub", 3},
        {"BatchToSpace", 4},
        {"SpaceToBatch", 5},
        {"Cast", 6},
        {"Concat", 7},
        {"Convolution", 8},
        {"Crop", 9},
        {"DepthToSpace", 10},
        {"Eltwise", 11},
        {"FullyConnected", 12},
        {"Gather", 13},
        {"MatMul", 14},
        {"Pad", 15},
        {"Pooling", 16},
        {"Resize", 17},
        {"ReverseSequence", 18},
        {"Rsqrt", 19},
        {"SquaredDifference", 20},
        {"Softmax", 21},
        {"SpaceToDepth", 22},
        {"Split", 23},
        {"Tile", 24},
        {"Transpose", 25},
        {"LayoutConvertor", 26},
        {"Normalization", 27},
        {"BatchNorm", 28},
        {"GroupNorm", 29},
        {"InstanceNorm", 30},
        {"LayerNorm", 31},
        {"GridSample", 32},
        {"Slice", 33},
        {"Reduce", 34},
        {"ConvTranspose", 35},
        {"BasicLSTM", 36},
        {"ChannelShuffle", 37},
        {"Sine", 38},
        {"Cosine", 39},
        {"Quantize", 40},
        {"DeQuantize", 41},
        {"RMSNorm", 42},
    };
    static std::map<std::string, int> m_method_keys = {
        {"LEAKYRELU", 0},
        {"PRELU", 1},
        {"RELU", 2},
        {"SIGMOID", 3},
        {"SILU", 4},
        {"TANH", 5},
        {"CLIP", 6},
        {"HARDSWISH", 7},
        {"MAX", 8},
        {"AVG", 9},
        {"ADD", 10},
        {"SUB", 11},
        {"MUL", 12},
        {"MIN", 13},
        {"BILINEAR", 14},
        {"NEAREST", 15},
        {"L1", 16},
        {"L2", 17},
        {"MEAN", 18},
        {"SUM", 19},
        {"GELU", 20},
    };
    static std::map<std::string, int> m_dtype_keys = {
        {"int8", 0},
        {"uint8", 1},
        {"int16", 2},
        {"uint16", 3},
        {"float16", 4},
        {"int32", 5},
        {"uint32", 6},
        {"float32", 7},
    };
    static std::map<std::string, int> m_layout_keys = {
        {"NHWC", 0},
        {"NCHW", 1},
        {"NCHWC16", 2},
        {"NCHWC16T1", 3},
        {"NCHWC16T8", 4},
        {"NCHWC32", 5},
        {"NCHWC32T1", 6},
        {"NCHWC32T8", 7},
        {"NDHWC", 8},
        {"NDCHWC16", 9},
        {"NDCHWC32", 10},
    };
    int val = INT32_MAX;
    for (auto &keys : {&m_optype_keys, &m_method_keys, &m_dtype_keys, &m_layout_keys})
    {
        if (keys->count(key))
        {
            val = keys->at(key);
            break;
        }
    }
    if (val == INT32_MAX)
    {
        DEBUG() << "Unknown key: " << key;
        return false;
    }
    res[pos++] = val;
    return true;
}

template <typename T>
static bool fill_param(NodePtr node, const std::string &param, std::vector<float> &res, int &pos)
{
    if (!node->params().count(param))
    {
        DEBUG() << "Unknown param: " << param;
        return false;
    }
    T val = node->params()[param];
    res[pos++] = val;
    return true;
}

template <typename T>
static bool fill_list(const std::vector<T> &list, std::vector<float> &res, int &pos)
{
    if (list.empty())
    {
        return false;
    }
    for (auto val : list)
    {
        res[pos++] = val;
    }
    return true;
}

static std::vector<float> split_num(float num)
{
    std::vector<float> ret;
    if (num > -1.0 && num < 1.0)
    {
        ret = {num, 0};
    }
    else
    {
        std::stringstream ss;
        ss << std::scientific << num;
        auto strs = utils::split_string(ss.str(), "e");
        ret = {std::stof(strs[0]), std::stof(strs[1])};
    }
    return ret;
}

static void preprocess(std::vector<std::pair<NodePtr, std::vector<bool>>> &perf_nodes,
                       std::vector<float> &inputs,
                       int &batch_size,
                       int input_size = 32)
{
    batch_size = 0;
    uint32_t max_size = 0;
    for (auto &sgn : perf_nodes)
    {
        max_size += sgn.second.size();
    }
    // plus one to avoid index out of range
    std::vector<float> buff((max_size + 1) * input_size, 0);
    int pos = 0;

    for (auto &sgn : perf_nodes)
    {
        std::vector<NodePtr> nodes;
        if (SubGraphNode::is_subgraph_node(sgn.first))
        {
            nodes = SubGraphNode::from_node(sgn.first)->subgraph()->nodes();
        }
        else
        {
            nodes = {sgn.first};
        }
        bool ret = true;
        for (uint32_t i = 0; i < nodes.size(); ++i)
        {
            NodePtr node = nodes[i];
            auto op_type = optype_to_string(node->type());
            switch (node->type())
            {
            case OpType_Activation:
            {
                static std::set<std::string> methods = {
                    "SIGMOID",
                    "LEAKYRELU",
                    "PRELU",
                    "RELU",
                    "TANH",
                    "CLIP",
                    "SILU",
                    "HARDSWISH",
                };
                std::string method = node->params()["method"];
                if (!methods.count(method))
                {                    
                    method = "SIGMOID";
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(method, buff, pos);
                buff[pos++] = node->inputs()[0]->shape().size();
                break;
            }
            case OpType_Add:
            case OpType_Sub:
            case OpType_Mul:
            {
                std::vector<int> in0_shape = node->inputs()[0]->shape();
                std::vector<int> in1_shape = node->inputs()[1]->shape();
                ret &= convert_shape(in0_shape);
                ret &= convert_shape(in1_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in0_shape, buff, pos);
                ret &= fill_list(in1_shape, buff, pos);
                break;
            }
            case OpType_BatchToSpace:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "block_size_x", buff, pos);
                ret &= fill_param<int>(node, "block_size_y", buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_SpaceToBatch:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "block_size_x", buff, pos);
                ret &= fill_param<int>(node, "block_size_y", buff, pos);
                ret &= fill_list(in_shape, buff, pos);
            }
            case OpType_Cast:
            {
                auto from = Tensor::dtype_to_string(node->inputs()[0]->type());
                auto to = Tensor::dtype_to_string(node->params()["to_dtype"]);
                if (from != "int32" && from != "int16")
                {
                    if (from == "uint32")
                    {
                        from = "int32";
                    }
                    else
                    {
                        from = "int16";
                    }
                }
                if (to != "int8")
                {
                    to = "int8";
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(from, buff, pos);
                ret &= fill_key(to, buff, pos);
                buff[pos++] = node->inputs()[0]->shape().size();
                break;
            }
            case OpType_Concat:
            {
                std::vector<std::vector<int>> in_shapes;
                for (auto inp : node->inputs())
                {
                    in_shapes.push_back(get_shape(inp));
                }
                int axis = node->params()["axis"];
                if (axis < 0)
                {
                    axis += in_shapes[0].size();
                }
                std::vector<int> dims;
                for (uint32_t i = 0; i < in_shapes[0].size(); ++i)
                {
                    if (i != static_cast<uint32_t>(axis))
                    {
                        dims.push_back(in_shapes[0][i]);
                    }
                }
                for (uint32_t i = 0; i < in_shapes.size(); ++i)
                {
                    if (i >= 10)
                    {
                        break;
                    }
                    dims.push_back(in_shapes[i][axis]);
                }
                ret &= fill_key(op_type, buff, pos);
                buff[pos++] = axis;
                ret &= fill_list(dims, buff, pos);
                break;
            }
            case OpType_Convolution:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                int kernel_x = 1, kernel_y = 1;
                if (node->params().count("kernel_x"))
                {
                    kernel_x = node->params()["kernel_x"];
                }
                if (node->params().count("kernel_y"))
                {
                    kernel_y = node->params()["kernel_y"];
                }
                ret &= convert_shape(in_shape);
                ret &= convert_shape(out_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "dilation_x", buff, pos);
                ret &= fill_param<int>(node, "dilation_y", buff, pos);
                buff[pos++] = kernel_x;
                buff[pos++] = kernel_y;
                ret &= fill_param<int>(node, "stride_x", buff, pos);
                ret &= fill_param<int>(node, "stride_y", buff, pos);
                ret &= fill_param<int>(node, "group", buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_Crop:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= convert_shape(out_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_DepthToSpace:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "block_size_x", buff, pos);
                ret &= fill_param<int>(node, "block_size_y", buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_Eltwise:
            {
                std::string method = node->params()["method"];
                if (method == "MAX")
                {
                    method = "MIN";
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(method, buff, pos);
                buff[pos++] = node->inputs()[0]->shape().size();
                break;
            }
            case OpType_FullyConnected:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_Gather:
            {
                std::vector<int> in0_shape, in1_shape;
                if (node->inputs().size() == 1 && node->constants().count("lut"))
                {
                    if (node->params().count("idx_lut"))
                    {
                        in0_shape = node->inputs()[0]->shape();
                        in1_shape = node->constants()["lut"]->shape();
                    }
                    else
                    {
                        in0_shape = node->constants()["lut"]->shape();
                        in1_shape = node->inputs()[0]->shape();
                    }
                }
                else
                {
                    in0_shape = node->inputs()[0]->shape();
                    in1_shape = node->inputs()[1]->shape();
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in0_shape, buff, pos);
                ret &= fill_list(in1_shape, buff, pos);
                break;
            }
            case OpType_MatMul:
            {
                std::vector<std::vector<int>> raw_in_shapes = node->params()["origin_bottom_shapes"];
                if (node->params()["trans_a"])
                {
                    std::swap(raw_in_shapes[0][2], raw_in_shapes[0][3]);
                }
                if (node->params()["trans_b"])
                {
                    std::swap(raw_in_shapes[1][2], raw_in_shapes[1][3]);
                }
                ret &= fill_key(op_type, buff, pos);
                buff[pos++] = raw_in_shapes[0][0] * raw_in_shapes[0][1];
                buff[pos++] = raw_in_shapes[0][2];
                buff[pos++] = raw_in_shapes[0][3];
                buff[pos++] = raw_in_shapes[1][3];
                break;
            }
            case OpType_Pad:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= convert_shape(out_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_Pooling:
            {
                std::string method = node->params()["method"];
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                int kernel_x = 1, kernel_y = 1;
                if (node->params().count("kernel_x"))
                {
                    kernel_x = node->params()["kernel_x"];
                }
                if (node->params().count("kernel_y"))
                {
                    kernel_y = node->params()["kernel_y"];
                }
                kernel_x = std::min(19, kernel_x);
                kernel_y = std::min(19, kernel_y);
                ret &= convert_shape(in_shape);
                ret &= convert_shape(out_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(method, buff, pos);
                ret &= fill_param<int>(node, "dilation_x", buff, pos);
                ret &= fill_param<int>(node, "dilation_y", buff, pos);
                buff[pos++] = kernel_x;
                buff[pos++] = kernel_y;
                ret &= fill_param<int>(node, "stride_x", buff, pos);
                ret &= fill_param<int>(node, "stride_y", buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_Resize:
            {
                std::string method = node->params()["method"];
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= convert_shape(out_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(method, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_ReverseSequence:
            {
                std::vector<int> in0_shape = node->inputs()[0]->shape();
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "time_axis", buff, pos);
                ret &= fill_param<int>(node, "batch_axis", buff, pos);
                ret &= fill_list(in0_shape, buff, pos);
                break;
            }
            case OpType_Rsqrt:
            case OpType_SquaredDifference:
            {
                ret &= fill_key(op_type, buff, pos);
                buff[pos++] = node->inputs()[0]->shape().size();
                break;
            }
            case OpType_Softmax:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                int axis = node->params()["axis"];
                if (axis < 0)
                {
                    axis += in_shape.size();
                }
                int dim = in_shape[axis];
                ret &= fill_key(op_type, buff, pos);
                buff[pos++] = axis;
                buff[pos++] = node->inputs()[0]->shape().size() / dim;
                buff[pos++] = dim;
                break;
            }
            case OpType_SpaceToDepth:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "block_size_x", buff, pos);
                ret &= fill_param<int>(node, "block_size_y", buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_Split:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                int axis = node->params()["axis"];
                if (axis < 0)
                {
                    axis += in_shape.size();
                }
                std::vector<std::vector<int>> splits = node->params()["splits"];
                std::vector<int> dims;
                for (uint32_t i = 0; i < splits[0].size(); ++i)
                {
                    if (i >= 10)
                    {
                        break;
                    }
                    dims.push_back(splits[0][i]);
                }
                ret &= fill_key(op_type, buff, pos);
                buff[pos++] = axis;
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(dims, buff, pos);
                break;
            }
            case OpType_Tile:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= convert_shape(out_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_Transpose:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                std::vector<int> perm = node->params()["perm"];
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(perm, buff, pos);
                break;
            }
            case OpType_LayoutConvertor:
            {
                static std::set<std::string> layouts = {
                    "NHWC",
                    "NCHW",
                    "NCHWC16",
                    "NCHWC32",
                };
                if (is_a_remapping_convertor(node, false))
                {
                    node->set_attr(Constants::remapping_node, true);
                    ret = false;
                }
                std::string from = node->params()["from"];
                std::string to = node->params()["to"];
                std::vector<int> in_shape = node->inputs()[0]->shape();
                if (from == "NCHWC32T1" || from == "NCHWC32T8")
                {
                    from = "NCHWC32";
                    in_shape = {in_shape[0], in_shape[3] / 32, in_shape[1], in_shape[2], 32};
                }
                if (from == "NCHWC16T1" || from == "NCHWC16T8")
                {
                    from = "NCHWC16";
                    in_shape = {in_shape[0], in_shape[3] / 16, in_shape[1], in_shape[2], 16};
                }
                if (to == "NCHWC32T1" || to == "NCHWC32T8")
                {
                    to = "NCHWC32";
                }
                if (to == "NCHWC16T1" || to == "NCHWC16T8")
                {
                    to = "NCHWC16";
                }
                if (from == to || !layouts.count(from) || !layouts.count(to))
                {
                    ret = false;
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(from, buff, pos);
                ret &= fill_key(to, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_Normalization:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                std::string method = node->params()["method"];
                std::vector<int> axis = node->params()["axis"];
                for (uint32_t i = 0; i < axis.size(); ++i)
                {
                    if (axis[i] < 0)
                    {
                        axis[i] += in_shape.size();
                    }
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(method, buff, pos);
                ret &= fill_list(axis, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_BatchNorm:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                int axis = node->params()["axis"];
                if (axis < 0)
                {
                    axis += in_shape.size();
                }
                ret &= fill_key(op_type, buff, pos);
                buff[pos++] = axis;
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_GroupNorm:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                int axis = node->params()["axis"];
                if (axis < 0)
                {
                    axis += in_shape.size();
                }
                ret &= fill_key(op_type, buff, pos);
                buff[pos++] = axis;
                ret &= fill_param<int>(node, "group", buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_InstanceNorm:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_LayerNorm:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                std::vector<int> axis = node->params()["axis"];
                for (uint32_t i = 0; i < axis.size(); ++i)
                {
                    if (axis[i] < 0)
                    {
                        axis[i] += in_shape.size();
                    }
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(axis, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_RMSNorm:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                std::vector<int> axis = node->params()["axis"];
                for (uint32_t i = 0; i < axis.size(); ++i)
                {
                    if (axis[i] < 0)
                    {
                        axis[i] += in_shape.size();
                    }
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(axis, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_GridSample:
            {
                std::string method = node->params()["method"];
                if (method != "BILINEAR")
                {
                    method = "BILINEAR";
                }
                auto in0_shape = get_shape(node->inputs()[0]);
                auto in1_shape = get_shape(node->inputs()[1]);
                if (in0_shape[3] == 2 || in1_shape[3] == 2)
                {
                    in0_shape = {in0_shape[0], in0_shape[3], in0_shape[1], in0_shape[2]};
                    in1_shape = {in1_shape[0], in1_shape[3], in1_shape[1], in1_shape[2]};
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(method, buff, pos);
                ret &= fill_list(in0_shape, buff, pos);
                ret &= fill_list(in1_shape, buff, pos);
                break;
            }
            case OpType_Slice:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= convert_shape(out_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_Reduce:
            {
                std::string method = node->params()["method"];
                if (method != "MEAN")
                {
                    method = "MEAN";
                }
                auto in_shape = get_shape(node->inputs()[0]);
                std::vector<int> axis = node->params()["axis"];
                for (uint32_t i = 0; i < axis.size(); ++i)
                {
                    if (axis[i] < 0)
                    {
                        axis[i] += in_shape.size();
                    }
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(method, buff, pos);
                ret &= fill_list(axis, buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                break;
            }
            case OpType_ConvTranspose:
            {
                std::vector<int> in_shape = node->inputs()[0]->shape();
                std::vector<int> out_shape = node->outputs()[0]->shape();
                ret &= convert_shape(in_shape);
                ret &= convert_shape(out_shape);
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "dilation_x", buff, pos);
                ret &= fill_param<int>(node, "dilation_y", buff, pos);
                ret &= fill_param<int>(node, "kernel_x", buff, pos);
                ret &= fill_param<int>(node, "kernel_y", buff, pos);
                ret &= fill_param<int>(node, "stride_x", buff, pos);
                ret &= fill_param<int>(node, "stride_y", buff, pos);
                ret &= fill_param<int>(node, "group", buff, pos);
                ret &= fill_list(in_shape, buff, pos);
                ret &= fill_list(out_shape, buff, pos);
                break;
            }
            case OpType_BasicLSTM:
            {
                std::vector<int> in0_shape = node->inputs()[0]->shape();
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "cell_size", buff, pos);
                ret &= fill_list(in0_shape, buff, pos);
                break;
            }
            case OpType_ChannelShuffle:
            {
                auto in_shape = get_shape(node->inputs()[0]);
                int dim = in_shape[in_shape.size() - 1];
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_param<int>(node, "group", buff, pos);
                buff[pos++] = node->inputs()[0]->shape().size() / dim;
                buff[pos++] = dim;
                break;
            }
            case OpType_Sine:
            case OpType_Cosine:
            {
                ret &= fill_key(op_type, buff, pos);
                buff[pos++] = node->inputs()[0]->shape().size();
                break;
            }
            case OpType_Quantize:
            case OpType_DeQuantize:
            {
                auto from = Tensor::dtype_to_string(node->inputs()[0]->type());
                auto to = Tensor::dtype_to_string(node->outputs()[0]->type());
                if (op_type == "Quantize")
                {
                    if (from != "float16")
                    {
                        from = "float16";
                    }
                    if (to != "int8" && to != "int16")
                    {
                        if (to == "uint8")
                        {
                            to = "int8";
                        }
                        else
                        {
                            to = "int16";
                        }
                    }
                }
                else
                {
                    if (from != "int8" && from != "int16")
                    {
                        if (from == "uint8")
                        {
                            from = "int8";
                        }
                        else
                        {
                            from = "int16";
                        }
                    }
                    if (to != "float16")
                    {
                        to = "float16";
                    }
                }
                ret &= fill_key(op_type, buff, pos);
                ret &= fill_key(from, buff, pos);
                ret &= fill_key(to, buff, pos);
                buff[pos++] = node->inputs()[0]->shape().size();
                break;
            }
            default:
                sgn.second[i] = false; // not supported
                break;
            }
            if (sgn.second[i])
            {
                if (ret && pos <= (batch_size + 1) * input_size)
                {
                    batch_size += 1;
                }
                else
                {
                    for (int j = batch_size * input_size; j < pos; ++j)
                    {
                        buff[j] = 0;
                    }
                    sgn.second[i] = false;
                }
                pos = batch_size * input_size;
            }
        }
    }
    buff.resize(batch_size * input_size);
    inputs.resize(buff.size() * 2, 0);
    for (uint32_t i = 0; i < buff.size(); ++i)
    {
        auto splits = split_num(buff[i]);
        inputs[i * 2] = splits[0];
        inputs[i * 2 + 1] = splits[1];
    }
}

static void postprocess(std::vector<float> &pred,
                        std::vector<std::pair<NodePtr, std::vector<bool>>> &perf_nodes,
                        PerfDataItems &outputs,
                        int output_size = 5)
{
    static std::set<OpType> zero_load_ops = {
        OpType_InternalConcatenated,
        OpType_InternalCrop,
        OpType_InternalSlice,
        OpType_InternalSpliter,
        OpType_InternalPad,
        OpType_Reshape,
        OpType_Input,
        OpType_InternalConstant,
        OpType_SubGraphInput,
        OpType_Dummy,
        OpType_Empty,
    };
    uint32_t batch_index = 0;
    outputs.clear();

    for (auto &sgn : perf_nodes)
    {
        std::vector<NodePtr> nodes;
        if (SubGraphNode::is_subgraph_node(sgn.first))
        {
            nodes = SubGraphNode::from_node(sgn.first)->subgraph()->nodes();
        }
        else
        {
            nodes = {sgn.first};
        }
        std::vector<PerfDataItem> items;
        PerfDataItem item = {0};
        double *ptr = (double *)(&item);

        for (uint32_t i = 0; i < nodes.size(); ++i)
        {
            if (sgn.second[i])
            {
                uint32_t pos = batch_index * output_size * 2;
                for (int j = 0; j < output_size; ++j)
                {
                    double num = std::exp(pred[pos + j * 2]) * std::pow(10, std::exp(pred[pos + j * 2 + 1]));
                    ptr[j] = std::max(num, 0.0);
                }
                batch_index += 1;
            }
            else if (zero_load_ops.count(nodes[i]->type()) ||
                     nodes[i]->attrs().count(Constants::remapping_node))
            {
                ptr[0] = 0;
            }
            else
            {
                ptr[0] = 1;
            }
            items.push_back(item);
        }
        if (items.size() > 1) // fused layers
        {
            item.reset();
            for (uint32_t i = 0; i < items.size(); ++i)
            {
                item.count_cycles += items[i].count_cycles;
                item.read_bytes += items[i].read_bytes;
                item.write_bytes += items[i].write_bytes;
                item.aiff_cycles += items[i].aiff_cycles;
                item.dma_cycles += items[i].dma_cycles;
            }
        }
        item.aiff_cycles = std::min(item.aiff_cycles, item.count_cycles);
        item.dma_cycles = std::min(item.dma_cycles, item.count_cycles);
        item.aiff_utilization = item.aiff_cycles / (item.count_cycles + 1e-7) * 100;
        item.dma_utilization = item.dma_cycles / (item.count_cycles + 1e-7) * 100;

        sgn.first->set_attr("count_cycles", uint64_t(item.count_cycles));
        sgn.first->set_attr("read_bytes", uint32_t(item.read_bytes));
        sgn.first->set_attr("write_bytes", uint32_t(item.write_bytes));
        sgn.first->set_attr("aiff_utilization", item.aiff_utilization);
        sgn.first->set_attr("dma_utilization", item.dma_utilization);
        sgn.first->set_attr(Constants::performance_info, item);

        DEBUG() << sgn.first->name() << ": ["
                << std::to_string(uint64_t(item.count_cycles)) << ", "
                << std::to_string(uint32_t(item.read_bytes)) << ", "
                << std::to_string(uint32_t(item.write_bytes)) << ", "
                << std::to_string(item.aiff_utilization) << ", "
                << std::to_string(item.dma_utilization) << "]";
        outputs[sgn.first] = item;
    }
}

static void run_graph(GraphPtr cost_model,
                      std::vector<std::pair<NodePtr, std::vector<bool>>> &perf_nodes,
                      PerfDataItems &outputs)
{
    if (!cost_model || perf_nodes.empty())
    {
        return;
    }
    std::vector<float> inputs;
    int batch_size = 0;
    preprocess(perf_nodes, inputs, batch_size);

    std::vector<float> pred;
    if (batch_size > 0)
    {
        cost_model->walk_on_node([batch_size](NodePtr node) {
            for (uint32_t i = 0; i < node->outputs().size(); ++i)
            {
                auto tensor = node->outputs()[i];
                std::vector<int> shape_list = tensor->shape();
                if (shape_list[0] != batch_size)
                {
                    shape_list[0] = batch_size;
                    tensor->shape() = TensorShape(shape_list, tensor->shape().layout());
                }
            }
        });

        TensorPtr inp = cost_model->get_input_nodes()[0]->outputs()[0];
        inp->share_data_from(ComputeTensor(inputs));

#if USE_OMP
        auto nt = omp_get_num_threads();
        omp_set_num_threads(1);
        {
#endif
            cost_model->walk_on_node(engine::run_node);
#if USE_OMP
        }
        omp_set_num_threads(nt);
#endif

        pred = ComputeTensor(cost_model->get_output_nodes()[0]->outputs()[0]);
    }
    postprocess(pred, perf_nodes, outputs);
}

static GraphPtr load_cost_model(const std::string &graph_path, const std::string &weight_path)
{
    static GraphPtr cost_model = nullptr;

    if (!cost_model)
    {
        if (!graph_path.empty() && !weight_path.empty())
        {
            if (!utils::file_exists(graph_path))
            {
                DEBUG() << "Cannot open file: " << graph_path;
                return nullptr;
            }
            if (!utils::file_exists(weight_path))
            {
                DEBUG() << "Cannot open file: " << weight_path;
                return nullptr;
            }
            cost_model = Graph::parse(graph_path);
            if (!cost_model)
            {
                DEBUG() << "Failed to load cost model";
                return nullptr;
            }
            cost_model->load_weight(weight_path);
        }
        else
        {
            auto txt_start = (const char *)&_binary_cost_model_x3p_txt_start;
            auto bin_start = (const char *)&_binary_cost_model_x3p_cbin_start;
            auto txt_data_len = (size_t)&_binary_cost_model_x3p_txt_end - (size_t)&_binary_cost_model_x3p_txt_start;
            auto bin_data_len = (size_t)&_binary_cost_model_x3p_cbin_end - (size_t)&_binary_cost_model_x3p_cbin_start;
            std::stringstream txt, weight;
            txt.write(txt_start, txt_data_len);
            cost_model = Graph::parse(txt);
            if (!cost_model)
            {
                DEBUG() << "Failed to load cost model";
                return nullptr;
            }
            weight.write(bin_start, bin_data_len);
            cost_model->load_weight(weight);
        }
        cost_model->topological_sort();
        cost_model->reset_layer_id();
    }

    return cost_model;
}

static void get_perf_nodes(SubGraphNodePtr sgn, std::vector<std::pair<NodePtr, std::vector<bool>>> &res)
{
    if (!sgn || !sgn->subgraph()->nodes().size())
    {
        return;
    }
    bool is_perf_subgraph = true;
    auto nodes = sgn->subgraph()->nodes();
    for (auto n : nodes)
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            is_perf_subgraph = false;
            break;
        }
    }
    if (is_perf_subgraph)
    {
        res.push_back(std::make_pair(sgn, std::vector<bool>(nodes.size(), true)));
    }
    else
    {
        for (auto n : nodes)
        {
            if (SubGraphNode::is_subgraph_node(n))
            {
                get_perf_nodes(SubGraphNode::from_node(n), res);
            }
            else
            {
                res.push_back(std::make_pair(n, std::vector<bool>(1, true)));
            }
        }
    }
}

static void get_perf_nodes(GraphPtr g, std::vector<std::pair<NodePtr, std::vector<bool>>> &res)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (SubGraphNode::is_subgraph_node(n))
        {
            get_perf_nodes(SubGraphNode::from_node(n), res);
        }
        else
        {
            res.push_back(std::make_pair(n, std::vector<bool>(1, true)));
        }
    }
}

static bool match_subgraph(const json &subgraphs, NodePtr sgn, PerfDataItem &res)
{
    std::vector<NodePtr> sg_nodes;
    if (SubGraphNode::is_subgraph_node(sgn))
    {
        sg_nodes = SubGraphNode::from_node(sgn)->subgraph()->nodes();
    }
    else
    {
        sg_nodes = {sgn};
    }
    for (auto &sg : subgraphs)
    {
        for (auto &n : sg["nodes"])
        {
            auto &inputs = n["input_shapes"];
            auto &outputs = n["output_shapes"];
            auto sg_inputs = sgn->inputs();
            auto sg_outputs = sgn->outputs();
            if (inputs.size() != sg_inputs.size() || outputs.size() != sg_outputs.size())
            {
                continue;
            }
            bool is_identical = true;
            for (uint32_t i = 0; i < inputs.size(); i++)
            {
                std::vector<int> in_shape = TensorShape(inputs[i].get<std::string>());
                std::vector<int> sg_in_shape = sg_inputs[i]->shape();
                if (in_shape != sg_in_shape)
                {
                    is_identical = false;
                    break;
                }
            }
            if (!is_identical)
            {
                continue;
            }
            for (uint32_t i = 0; i < outputs.size(); i++)
            {
                std::vector<int> out_shape = TensorShape(outputs[i].get<std::string>());
                std::vector<int> sg_out_shape = sg_outputs[i]->shape();
                if (out_shape != sg_out_shape)
                {
                    is_identical = false;
                    break;
                }
            }
            if (!is_identical)
            {
                continue;
            }
            auto &nodes = n["internal_nodes"];
            if (nodes.size() != sg_nodes.size())
            {
                continue;
            }
            for (uint32_t i = 0; i < nodes.size(); i++)
            {
                auto op_type = string_to_optype(nodes[i]["op_type"].get<std::string>());
                if (op_type != sg_nodes[i]->type())
                {
                    is_identical = false;
                    break;
                }
            }
            if (!is_identical)
            {
                continue;
            }
            for (uint32_t i = 0; i < nodes.size(); i++)
            {
                auto &params = nodes[i]["params"];
                auto sg_params = sg_nodes[i]->params();
                switch (sg_nodes[i]->type())
                {
                case OpType_Convolution:
                {
                    if (params["dilation_x"].get<std::string>() != std::to_string((int)sg_params["dilation_x"]) ||
                        params["dilation_y"].get<std::string>() != std::to_string((int)sg_params["dilation_y"]) ||
                        params["group"].get<std::string>() != std::to_string((int)sg_params["group"]) ||
                        params["kernel_x"].get<std::string>() != std::to_string((int)sg_params["kernel_x"]) ||
                        params["kernel_y"].get<std::string>() != std::to_string((int)sg_params["kernel_y"]) ||
                        params["stride_x"].get<std::string>() != std::to_string((int)sg_params["stride_x"]) ||
                        params["stride_y"].get<std::string>() != std::to_string((int)sg_params["stride_y"]) ||
                        params["with_activation"].get<std::string>() != sg_params["with_activation"])
                    {
                        is_identical = false;
                    }
                    break;
                }
                case OpType_Pooling:
                {
                    if (params["method"].get<std::string>() != sg_params["method"] ||
                        params["dilation_x"].get<std::string>() != std::to_string((int)sg_params["dilation_x"]) ||
                        params["dilation_y"].get<std::string>() != std::to_string((int)sg_params["dilation_y"]) ||
                        params["kernel_x"].get<std::string>() != std::to_string((int)sg_params["kernel_x"]) ||
                        params["kernel_y"].get<std::string>() != std::to_string((int)sg_params["kernel_y"]) ||
                        params["stride_x"].get<std::string>() != std::to_string((int)sg_params["stride_x"]) ||
                        params["stride_y"].get<std::string>() != std::to_string((int)sg_params["stride_y"]))
                    {
                        is_identical = false;
                    }
                    break;
                }
                default:
                {
                    for (auto &param : sg_params)
                    {
                        if (!params.contains(param.first))
                        {
                            is_identical = false;
                            break;
                        }
                        std::ostringstream oss;
                        oss << param.second;
                        if (params[param.first].get<std::string>() != oss.str())
                        {
                            is_identical = false;
                            break;
                        }
                    }
                    break;
                }
                }
                if (!is_identical)
                {
                    break;
                }
            }
            if (!is_identical)
            {
                continue;
            }
            res.count_cycles = n["cycles"].get<uint64_t>();
            res.read_bytes = n["aipu_read_bytes"].get<uint32_t>();
            res.write_bytes = n["aipu_write_bytes"].get<uint32_t>();
            res.aiff_utilization = n["aiff_utilization"].get<double>();
            res.dma_utilization = n["dma_utilization"].get<double>();
            res.aiff_cycles = res.aiff_utilization / 100.0 * res.count_cycles;
            res.dma_cycles = res.dma_utilization / 100.0 * res.count_cycles;
            return true;
        }
    }
    return false;
}

static void match_profile_data(std::vector<std::pair<NodePtr, std::vector<bool>>> &perf_nodes, PerfDataItems &outputs)
{
    static json perf_data;

    if (perf_data.is_null())
    {
        if (!BuilderConfig::exist(Constants::perf_json) || !BuilderConfig::exist(Constants::perf_bin))
        {
            return;
        }
        auto json_path = BuilderConfig::get<std::string>(Constants::perf_json);
        auto bin_path = BuilderConfig::get<std::string>(Constants::perf_bin);
        if (!utils::file_exists(json_path) || !utils::file_exists(bin_path))
        {
            return;
        }
        ProfileDataParser parser(json_path, bin_path);
        parser.parse(perf_data);
        if (perf_data.is_null())
        {
            return;
        }
        DEBUG() << "ProfileDataParser:\n"
                << "target: " << perf_data["target"].get<std::string>() << "\n"
                << "tec_num: " << perf_data["tec_num"].get<uint32_t>() << "\n"
                << "core_num: " << perf_data["core_num"].get<uint32_t>() << "\n"
                << "tcb_switch_cycle: " << perf_data["subgraph_switch_cycles"].get<uint64_t>() << "\n"
                << "total_cycles: " << perf_data["total_cycles"].get<uint64_t>() << "\n"
                << "total_time: " << perf_data["total_time"].get<double>() << "\n"
                << "fps: " << perf_data["fps"].get<double>() << "\n"
                << "total_dma_cycles: " << perf_data["total_dma_busy_cycles"].get<uint64_t>() << "\n"
                << "total_aiff_cycles: " << perf_data["total_aiff_busy_cycles"].get<uint64_t>() << "\n"
                << "total_read: " << perf_data["aipu_read_bytes"].get<uint32_t>() << "\n"
                << "total_write: " << perf_data["aipu_write_bytes"].get<uint32_t>() << "\n"
                << "total_dma_read: " << perf_data["dma_read_bytes"].get<uint32_t>() << "\n"
                << "total_dma_write: " << perf_data["dma_write_bytes"].get<uint32_t>() << "\n"
                << "subgraph_num: " << perf_data["subgraph_num"].get<uint32_t>() << "\n"
                << "node_num: " << perf_data["layer_num"].get<uint32_t>() << "\n"
                << "total_ops: " << perf_data["total_ops(million)"].get<double>();
        // for (auto &subgraph : perf_data["subgraphs"])
        // {
        //     for (auto &node : subgraph["nodes"])
        //     {
        //         DEBUG() << node["layer_name"].get<std::string>() << " {\n"
        //                 << "  op_type: " << node["op_type"].get<std::string>() << "\n"
        //                 << "  plugin: " << node["plugin"].get<std::string>() << "\n"
        //                 << "  internal_nodes: " << node["internal_nodes"].size() << "\n"
        //                 << "  core_id: " << node["core_id"].get<uint32_t>() << "\n"
        //                 << "  start: " << node["start"].get<uint64_t>() << "\n"
        //                 << "  cycles: " << node["cycles"].get<uint64_t>() << "\n"
        //                 << "  cycle_ratio: " << node["cycle_ratio"].get<double>() << "\n"
        //                 << "  aiff_busy_cycles: " << node["aiff_busy_cycles"].get<uint64_t>() << "\n"
        //                 << "  aiff_utilization: " << node["aiff_utilization"].get<double>() << "\n"
        //                 << "  mtp0_busy_cycles: " << node["mtp0_busy_cycles"].get<uint64_t>() << "\n"
        //                 << "  mtp0_utilization: " << node["mtp0_utilization"].get<double>() << "\n"
        //                 << "  mtp1_busy_cycles: " << node["mtp1_busy_cycles"].get<uint64_t>() << "\n"
        //                 << "  mtp1_utilization: " << node["mtp1_utilization"].get<double>() << "\n"
        //                 << "  itp_busy_cycles: " << node["itp_busy_cycles"].get<uint64_t>() << "\n"
        //                 << "  itp_utilization: " << node["itp_utilization"].get<double>() << "\n"
        //                 << "  ptp_busy_cycles: " << node["ptp_busy_cycles"].get<uint64_t>() << "\n"
        //                 << "  ptp_utilization: " << node["ptp_utilization"].get<double>() << "\n"
        //                 << "  dma_busy_cycles: " << node["dma_busy_cycles"].get<uint64_t>() << "\n"
        //                 << "  dma_utilization: " << node["dma_utilization"].get<double>() << "\n"
        //                 << "  dma_read_bytes: " << node["dma_read_bytes"].get<uint32_t>() << "\n"
        //                 << "  dma_write_bytes: " << node["dma_write_bytes"].get<uint32_t>() << "\n"
        //                 << "  aipu_read_bytes: " << node["aipu_read_bytes"].get<uint32_t>() << "\n"
        //                 << "  aipu_write_bytes: " << node["aipu_write_bytes"].get<uint32_t>() << "\n"
        //                 << "  algorithm_ops(million): " << node["algorithm_ops(million)"].get<double>() << "\n"
        //                 << "  BW(GB/s): " << node["BW(GB/s)"].get<double>() << "\n"
        //                 << "}";
        //     }
        // }
    }
    auto &subgraphs = perf_data["subgraphs"];
    for (auto &sgn : perf_nodes)
    {
        PerfDataItem item;
        if (match_subgraph(subgraphs, sgn.first, item))
        {
            sgn.first->set_attr("count_cycles", uint64_t(item.count_cycles));
            sgn.first->set_attr("read_bytes", uint32_t(item.read_bytes));
            sgn.first->set_attr("write_bytes", uint32_t(item.write_bytes));
            sgn.first->set_attr("aiff_utilization", item.aiff_utilization);
            sgn.first->set_attr("dma_utilization", item.dma_utilization);
            sgn.first->set_attr(Constants::performance_info, item);

            DEBUG() << sgn.first->name() << " matched: ["
                    << std::to_string(uint64_t(item.count_cycles)) << ", "
                    << std::to_string(uint32_t(item.read_bytes)) << ", "
                    << std::to_string(uint32_t(item.write_bytes)) << ", "
                    << std::to_string(item.aiff_utilization) << ", "
                    << std::to_string(item.dma_utilization) << "]";
            outputs[sgn.first] = item;
        }
    }
}

std::string PerfEstimator::summary()
{
    std::ostringstream oss;
    int64_t total_cycles = 0;
    for (auto &p : m_res)
    {
        total_cycles += p.second.count_cycles;
    }
    oss << "Total cycles estimated by cost model is: " << total_cycles;
    return oss.str();
}

bool PerfEstimator::operator()(GraphPtr g) const
{
    GraphPtr cost_model = load_cost_model(m_graph_path, m_weight_path);
    if (!g || !cost_model)
    {
        return true;
    }

    std::vector<std::pair<NodePtr, std::vector<bool>>> perf_nodes;
    get_perf_nodes(g, perf_nodes);

    run_graph(cost_model, perf_nodes, m_res);

    match_profile_data(perf_nodes, m_res);

    return false;
}

} // namespace aipubt

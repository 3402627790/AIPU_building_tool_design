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
 * =====================================================================================
 *
 *       Filename:  adjust_reduce_axis.cpp
 *
 *    Description:  Adjust reduce shape and axis if axis size > 1 and it's not continuous
 *    e.g. input_shape=[S0,S1,S2,S3,S4], axis=[0,2,4]
 *    1. Add Transpose:
 *      * AIFF surface reduce -> Reduce input_shape=[S0,S2,S4,S1,S3], axis=[0,1,2]
 *      * TPC/AIFF channel reduce -> Reduce input_shape=[S1,S3,S0,S2,S4], axis=[3,4,5]
 *    2. Update Reduce shape and axis:
 *    * If using TPC: input_shape=[S1S3,S0S2S4], axis=[1]
 *    * If using AIFF:
 *      - In X3:
 *          - If S0S2S4 <= AIFF channel spec (channel reduce) -> input_shape=[NHWC], N=1, HxW=S1S3, C=S0S2S4, axis=[3]
 *          - If S0S2S4 > AIFF channel spec (surface reduce) -> input_shape=[NHWC], N=1, HxW=S0S2S4, C=S1S3, axis=[1,2]
 *      - In X2 (surface reduce) -> input_shape=[NHWC], N=1, HxW=S0S2S4, C=S1S3, axis=[1,2]
 *
 *        Version:  1.0
 *        Created:  05/14/24 15:51:22
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Shuai Yuan
 *   Organization:  Arm China
 *
 * =====================================================================================
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "core/config.h"
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace aipubt
{
static int get_actual_axis(int dim, int axis)
{
    if (axis < 0)
    {
        return axis + dim;
    }
    else
    {
        return axis;
    }
};

static bool criterion(NodePtr node)
{
    if (node->type() != OpType_Reduce)
    {
        return false;
    }

    auto dim = node->inputs()[0]->shape().dim();
    std::vector<int> axis_vec = node->params()["axis"];
    if (axis_vec.size() <= 1)
    {
        return false;
    }
    int last_axis = get_actual_axis(dim, axis_vec[0]);
    for (size_t i = 1; i < axis_vec.size(); i++)
    {
        auto current_axis = get_actual_axis(dim, axis_vec[i]);
        if (current_axis != last_axis + 1)
        {
            return true;
        }
        else
        {
            last_axis = current_axis;
        }
    }

    return false;
}

static bool try_adjust_reduce_axis_for_aiff(GraphPtr graph, NodePtr node)
{
    if (node->is_dynamic())
    {
        DEBUG() << node->name() << " cannot use AIFF for dynamic shape temporarily";
        return false;
    }

    std::set<std::string> supported_methods = {
        "MEAN",
        "SUM",
        "MIN",
        "MAX",
    };
    std::string method = node->params()["method"];
    std::transform(method.begin(), method.end(), method.begin(), ::toupper);
    if (supported_methods.find(method) == supported_methods.end())
    {
        DEBUG() << node->name() << " cannot use AIFF due to unsupported method=" << method;
        return false;
    }

    std::set<Tensor::Dtype> supported_dtypes = {
        Tensor::INT8,
        Tensor::UINT8,
        Tensor::INT16,
        Tensor::UINT16,
    };
    auto arch = aipubt::get_target()->isainfo().type;
    if (arch >= aipubt::ISA_ZYv6)
    {
        supported_dtypes.insert(Tensor::FP16);
        supported_dtypes.insert(Tensor::BFP16);
    }
    for (auto &tensor_list : {node->inputs(), node->outputs()})
    {
        for (auto tensor : tensor_list)
        {
            auto dtype = tensor->type();
            if (supported_dtypes.find(dtype) == supported_dtypes.end())
            {
                DEBUG() << node->name() << " cannot use AIFF due to unsupported dtype=" << dtype;
                return false;
            }
        }
    }

    auto input_tensor = node->inputs()[0];
    auto input_shape = input_tensor->shape();
    std::vector<int> old_axis_vec = node->params()["axis"];
    auto old_dim = input_shape.dim();
    std::for_each(old_axis_vec.begin(), old_axis_vec.end(), [old_dim](int &axis) {
        if (axis < 0)
        {
            axis += old_dim;
        }
    });

    int reduce_size = 1;
    int non_reduce_size = 1;
    Expr s_reduce_size = 1;
    Expr s_non_reduce_size = 1;
    for (auto i = 0; i < input_shape.dim(); i++)
    {
        auto it = std::find(old_axis_vec.begin(), old_axis_vec.end(), i);
        if (it == old_axis_vec.end())
        {
            non_reduce_size *= input_shape[i];
            s_non_reduce_size *= input_shape.symbol()[i];
        }
        else
        {
            reduce_size *= input_shape[i];
            s_reduce_size *= input_shape.symbol()[i];
        }
    }

    auto max_surface_size = get_aiff_max_w() * get_aiff_max_h();
    auto max_channel_size = get_aiff_max_in_c();
    if (reduce_size > std::max(max_surface_size, max_channel_size))
    {
        DEBUG() << node->name() << " cannot use AIFF due to reduce_size=" << reduce_size << " exceeds AIFF spec=" << std::max(max_surface_size, max_channel_size);
        return false;
    }

    bool use_channel_reduce = false;
    if ((arch >= aipubt::ISA_ZYv6) && (reduce_size <= max_channel_size))
    {
        use_channel_reduce = true;
    }
    if (!use_channel_reduce)
    {
        int channel_size = non_reduce_size;
        int surface_size = reduce_size;
        if ((method == "MEAN") || (method == "SUM"))
        {
            int pe_w = 8;
            int pe_h = 4;
            get_pe_spec(pe_w, pe_h);
            int surface_size_thresold = pe_w * pe_h;
            const int channel_size_thresold = 10 * max_channel_size;
            if ((surface_size < surface_size_thresold) && (channel_size >= channel_size_thresold))
            {
                DEBUG() << node->name() << " had better not use AIFF due to too small surface_size=" << surface_size << " and poor utilization.";
                return false;
            }
        }
        if (((method == "MIN") || (method == "MAX")) && (arch < aipubt::ISA_ZYv6))
        {
            int max_kernel_size = 5;
            if (arch == aipubt::ISA_ZYv2)
            {
                max_kernel_size = 3;
            }
            else if (arch >= aipubt::ISA_ZYv3)
            {
                max_kernel_size = 8;
            }
            const int channel_size_thresold = 10 * max_channel_size;
            if ((surface_size > max_kernel_size) && (channel_size >= channel_size_thresold))
            {
                DEBUG() << node->name() << " had better not use AIFF due to surface_size=" << surface_size
                        << " is larger than max_kernel_size=" << max_kernel_size << " and will be splited to too many nodes.";
                return false;
            }
        }
    }

    if (use_channel_reduce)
    {
        int batch_size = 1;
        int channel_size = reduce_size;
        int surface_size = non_reduce_size;
        int h = 1;
        int w = surface_size;
        bool is_valid = try_optimize_aiff_reduce_shape(h, w, false, surface_size, method);
        if (!is_valid)
        {
            DEBUG() << node->name() << " cannot use AIFF due to surface_size=" << surface_size << " cannot have valid AIFF configuration.";
            return false;
        }
        Expr s_batch_size = 1;
        Expr s_channel_size = s_reduce_size;
        Expr s_h = 1;
        Expr s_w = s_non_reduce_size;

        std::vector<uint32_t> perm;
        for (auto i = 0; i < input_shape.dim(); i++)
        {
            auto it = std::find(old_axis_vec.begin(), old_axis_vec.end(), i);
            if (it == old_axis_vec.end())
            {
                perm.push_back(i);
            }
        }
        for (auto axis : old_axis_vec)
        {
            perm.push_back(axis);
        }
        auto transpose_tensor = ops::transpose(input_tensor, perm);
        auto transpose_node = transpose_tensor->node();
        ADD_LOCATION_BY_NODE(transpose_node, node);
        graph->add_node(transpose_node);
        transpose_node->infer_shape(true);

        std::vector<int> new_input_shape_vec{batch_size, h, w, channel_size};
        std::vector<int> new_output_shape_vec{batch_size, h, w, 1};

        TensorShape new_input_shape(new_input_shape_vec, DataLayout_NHWC);
        auto input_reshape_node = ops::reshape(transpose_tensor, new_input_shape)->node();
        graph->add_node(input_reshape_node);
        ADD_LOCATION_BY_NODE(input_reshape_node, node);
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            input_reshape_node->params()["ds_shape"] = std::vector<Expr>{s_batch_size, s_h, s_w, s_channel_size};
        }
        input_reshape_node->infer_shape(true);
        node->add_input(input_reshape_node->outputs()[0], node->remove_input(input_tensor));

        std::vector<int> new_axis_vec = {3};
        node->params()["axis"] = new_axis_vec;

        auto output_tensor = node->outputs()[0];
        auto output_shape = output_tensor->shape();
        TensorShape new_output_shape(new_output_shape_vec, DataLayout_NHWC);
        auto new_output_tensor = std::make_shared<Tensor>(output_tensor->name() + "/reshape", new_output_shape, output_tensor->type());
        new_output_tensor->quantization() = output_tensor->quantization();
        node->remove_output(0);
        node->add_output(new_output_tensor);
        auto output_reshape_node = ops::reshape(new_output_tensor, output_shape)->node();
        output_reshape_node->remove_output(0);
        output_reshape_node->add_output(output_tensor);
        graph->add_node(output_reshape_node);
        ADD_LOCATION_BY_NODE(output_reshape_node, node);
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            std::vector<Expr> ds_shape = output_shape.symbol();
            output_reshape_node->params()["ds_shape"] = ds_shape;
        }
        output_reshape_node->infer_shape(true);

        node->infer_shape(true);
    }
    else
    {
        int batch_size = 1;
        int channel_size = non_reduce_size;
        int surface_size = reduce_size;
        int h = 1;
        int w = surface_size;
        bool is_valid = try_optimize_aiff_reduce_shape(h, w, true, surface_size, method);
        if (!is_valid)
        {
            DEBUG() << node->name() << " cannot use AIFF due to surface_size=" << surface_size << " cannot have valid AIFF configuration.";
            return false;
        }
        Expr s_batch_size = 1;
        Expr s_channel_size = s_non_reduce_size;
        Expr s_h = 1;
        Expr s_w = s_reduce_size;

        std::vector<uint32_t> perm;
        for (auto axis : old_axis_vec)
        {
            perm.push_back(axis);
        }
        for (auto i = 0; i < input_shape.dim(); i++)
        {
            auto it = std::find(old_axis_vec.begin(), old_axis_vec.end(), i);
            if (it == old_axis_vec.end())
            {
                perm.push_back(i);
            }
        }
        auto transpose_tensor = ops::transpose(input_tensor, perm);
        auto transpose_node = transpose_tensor->node();
        ADD_LOCATION_BY_NODE(transpose_node, node);
        graph->add_node(transpose_node);
        transpose_node->infer_shape(true);

        std::vector<int> new_input_shape_vec{batch_size, h, w, channel_size};
        std::vector<int> new_output_shape_vec{batch_size, 1, 1, channel_size};

        TensorShape new_input_shape(new_input_shape_vec, DataLayout_NHWC);
        auto input_reshape_node = ops::reshape(transpose_tensor, new_input_shape)->node();
        graph->add_node(input_reshape_node);
        ADD_LOCATION_BY_NODE(input_reshape_node, node);
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            input_reshape_node->params()["ds_shape"] = std::vector<Expr>{s_batch_size, s_h, s_w, s_channel_size};
        }
        input_reshape_node->infer_shape(true);
        node->add_input(input_reshape_node->outputs()[0], node->remove_input(input_tensor));

        std::vector<int> new_axis_vec = {1, 2};
        node->params()["axis"] = new_axis_vec;
        node->infer_shape(true);

        auto output_tensor = node->outputs()[0];
        auto output_shape = output_tensor->shape();
        TensorShape new_output_shape(new_output_shape_vec, DataLayout_NHWC);
        auto new_output_tensor = std::make_shared<Tensor>(output_tensor->name() + "/reshape", new_output_shape, output_tensor->type());
        new_output_tensor->quantization() = output_tensor->quantization();
        node->remove_output(0);
        node->add_output(new_output_tensor);
        auto output_reshape_node = ops::reshape(new_output_tensor, output_shape)->node();
        output_reshape_node->remove_output(0);
        output_reshape_node->add_output(output_tensor);
        graph->add_node(output_reshape_node);
        ADD_LOCATION_BY_NODE(output_reshape_node, node);
        if (BuilderConfig::get<bool>(Constants::enable_ds))
        {
            std::vector<Expr> ds_shape = output_shape.symbol();
            output_reshape_node->params()["ds_shape"] = ds_shape;
        }
        output_reshape_node->infer_shape(true);
    }

    return true;
}

static void adjust_reduce_axis_for_tpc(GraphPtr graph, NodePtr node)
{
    auto input_tensor = node->inputs()[0];
    auto input_shape = input_tensor->shape();
    std::vector<int> old_axis_vec = node->params()["axis"];
    auto old_dim = input_shape.dim();
    std::for_each(old_axis_vec.begin(), old_axis_vec.end(), [old_dim](int &axis) {
        if (axis < 0)
        {
            axis += old_dim;
        }
    });

    std::vector<uint32_t> perm;
    std::vector<int> new_input_shape_vec(2, 1);
    std::vector<Expr> new_input_sym_shape_vec(2, 1);
    for (auto i = 0; i < input_shape.dim(); i++)
    {
        auto it = std::find(old_axis_vec.begin(), old_axis_vec.end(), i);
        if (it == old_axis_vec.end())
        {
            perm.push_back(i);
            new_input_shape_vec[0] *= input_shape[i];
            new_input_sym_shape_vec[0] *= input_shape.symbol()[i];
        }
    }
    for (auto axis : old_axis_vec)
    {
        perm.push_back(axis);
        new_input_shape_vec[1] *= input_shape[axis];
        new_input_sym_shape_vec[1] *= input_shape.symbol()[axis];
    }
    std::vector<int> new_output_shape_vec = new_input_shape_vec;
    new_output_shape_vec.back() = 1;

    auto transpose_tensor = ops::transpose(input_tensor, perm);
    auto transpose_node = transpose_tensor->node();
    ADD_LOCATION_BY_NODE(transpose_node, node);
    graph->add_node(transpose_node);
    transpose_node->infer_shape(true);

    TensorShape new_input_shape(new_input_shape_vec, DataLayout_Flat);
    auto input_reshape_node = ops::reshape(transpose_tensor, new_input_shape)->node();
    graph->add_node(input_reshape_node);
    ADD_LOCATION_BY_NODE(input_reshape_node, node);
    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        input_reshape_node->params()["ds_shape"] = new_input_sym_shape_vec;
    }
    input_reshape_node->infer_shape(true);
    node->add_input(input_reshape_node->outputs()[0], node->remove_input(input_tensor));

    std::vector<int> new_axis_vec = {1};
    node->params()["axis"] = new_axis_vec;

    auto output_tensor = node->outputs()[0];
    auto output_shape = output_tensor->shape();
    TensorShape new_output_shape(new_output_shape_vec, DataLayout_Flat);
    auto new_output_tensor = std::make_shared<Tensor>(output_tensor->name() + "/reshape", new_output_shape, output_tensor->type());
    new_output_tensor->quantization() = output_tensor->quantization();
    node->remove_output(0);
    node->add_output(new_output_tensor);
    auto output_reshape_node = ops::reshape(new_output_tensor, output_shape)->node();
    output_reshape_node->remove_output(0);
    output_reshape_node->add_output(output_tensor);
    graph->add_node(output_reshape_node);
    ADD_LOCATION_BY_NODE(output_reshape_node, node);
    if (BuilderConfig::get<bool>(Constants::enable_ds))
    {
        std::vector<Expr> ds_shape = output_shape.symbol();
        output_reshape_node->params()["ds_shape"] = ds_shape;
    }
    output_reshape_node->infer_shape(true);

    node->infer_shape(true);
}

void adjust_reduce_axis(GraphPtr graph)
{
    auto nodes = graph->nodes();
    for (auto node : nodes)
    {
        if (!criterion(node))
        {
            continue;
        }

        if (!try_adjust_reduce_axis_for_aiff(graph, node))
        {
            adjust_reduce_axis_for_tpc(graph, node);
        }
    }
}

}; // namespace aipubt

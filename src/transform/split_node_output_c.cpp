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
 * @Date: 2020-11-1 10:04:00
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 14:55:40
 */
#include "split_node_output_c.h"
#include "builder/lowering/lowering.h"
#include "builder/lowering/tiling.h"
#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/core.h"
#include "passes.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <sstream>

namespace aipubt
{
static int get_channel_dim(NodePtr n, TensorShape &shape)
{
    int dim = -1;
    if (n->type() == OpType_BatchNorm)
    {
        dim = n->params()["axis"];
        if (dim < 0)
        {
            dim += shape.dim();
        }
    }
    else
    {
        switch (shape.layout())
        {
        case DataLayout::DataLayout_NHWC:
        {
            if (shape.dim() == 4)
            {
                dim = 3;
            }
            break;
        }
        case DataLayout::DataLayout_NCHWC32:
        {
            if (shape.dim() == 5)
            {
                dim = 1;
            }
            break;
        }
        case DataLayout::DataLayout_Flat:
        {
            if (shape.dim() == 1 || shape.dim() == 2 || shape.dim() == 3 || shape.dim() == 4)
            {
                dim = shape.dim() - 1;
            }
            break;
        }
        default:
            break;
        }
    }
    return dim;
}

bool SplitNodeOutputC::split_output_c(GraphPtr g, NodePtr n) const
{
    auto is_board_cast_wo_c = [](TensorPtr t, NodePtr n) -> bool {
        if (n->outputs()[0]->shape() != t->shape() && t->shape().size() == 1)
        {
            return true;
        }
        return false;
    };
    auto is_board_cast_w_c = [](TensorPtr t, NodePtr n) -> bool {
        if (t->shape().dim() == 1 && t->shape().size() == n->outputs()[0]->shape().c_total())
        {
            return true;
        }
        return false;
    };

    static std::set<OpType> NON_INP_SPLIT_OPS =
        {
            OpType_Convolution,
            OpType_ConvTranspose,
            OpType_FullyConnected,
        };
    int num_split = m_output_c_vec.size();

    // create a concat node
    auto output = n->outputs()[0];
    int concat_dim = get_channel_dim(n, output->shape());
    if (concat_dim == -1)
    {
        WARN() << "unsupport layout layout: " << output->shape().layout() << " dim: " << output->shape().dim();
        return false;
    }

    NodePtr concat = Node::create(g->name() + "_" + n->name() + "_concat", OpType_Concat);

    concat->params()["axis"] = concat_dim;

    // no scale
    for (int i = 0; i < num_split; i++)
    {
        concat->params()["scale_value"].push_back(NodeParamValue(1));
        concat->params()["scale_type"].push_back(Tensor::Dtype::UINT8);
        concat->params()["shift_value"].push_back(NodeParamValue(0));
        concat->params()["shift_type"].push_back(Tensor::Dtype::UINT8);
    }

    TensorShape shape = n->outputs()[0]->shape();
    Tensor::Dtype output_type = n->outputs()[0]->type();

    concat->add_output(n->outputs());

    int out_c_start = 0;

    auto input = n->inputs()[0];

    // split node
    std::vector<NodePtr> split_nodes;
    int split_dim = -1;
    if (!contains(NON_INP_SPLIT_OPS, n->type()))
    {
        for (size_t i = 0; i < n->inputs().size(); ++i)
        {
            auto input = n->inputs()[i];
            if (is_board_cast_wo_c(input, n))
            {
                continue;
            }
            NodePtr split_node = Node::create(g->name() + "_" + n->name() + "/input" + std::to_string(i) + "_split", OpType_Split);
            split_dim = get_channel_dim(n, input->shape());
            if (split_dim == -1)
            {
                WARN() << "unsupport layout layout: " << input->shape().layout() << " dim: " << input->shape().dim();
                return false;
            }
            split_node->params()["axis"] = split_dim;
            split_node->add_input(input);
            split_nodes.push_back(split_node);
            g->add_node(split_node);
            ADD_LOCATION_BY_NODE(split_node, n);
        }
    }

    int original_out_c = std::accumulate(m_output_c_vec.begin(), m_output_c_vec.end(), 0);
    for (int i = 0; i < num_split; ++i)
    {
        // subnode
        std::string subnode_name = g->name() + "_" + n->name() + "/" + std::to_string(i);
        NodePtr subnode = Node::create(subnode_name, n->type());

        int channel = m_output_c_vec[i];

        for (size_t j = 0; j < n->inputs().size(); ++j)
        {
            if (!contains(NON_INP_SPLIT_OPS, n->type()))
            {
                if (is_board_cast_wo_c(n->inputs()[j], n))
                {
                    subnode->add_input(n->inputs()[j]);
                    continue;
                }
                auto split_node = split_nodes[j];
                split_node->params()["splits"].push_back(channel);
                split_dim = split_node->params()["axis"];
                auto input = split_node->inputs()[0];
                auto split_out_shape = input->shape();
                if (is_board_cast_w_c(n->inputs()[j], n))
                {
                    split_out_shape[0] = channel;
                }
                else if (n->type() == OpType_BatchNorm)
                {
                    if (n->inputs()[j]->shape().layout() == DataLayout_NCHWC32 && split_dim == 1)
                    {
                        split_out_shape[split_dim] = channel / 32;
                    }
                    else
                    {
                        split_out_shape[split_dim] = channel;
                    }
                }
                else if (n->inputs()[j]->shape().layout() == DataLayout_NCHWC32)
                {
                    split_out_shape[1] = channel / 32;
                }
                else if (n->inputs()[j]->shape().layout() == DataLayout_NHWC)
                {
                    split_out_shape[3] = channel;
                }
                else
                {
                    split_out_shape[split_dim] = channel;
                }
                split_out_shape.reset_strides();
                auto split_output = std::make_shared<Tensor>(split_node->name() + "/output/" + std::to_string(i),
                                                             split_out_shape, input->type());
                split_output->quantization() = input->quantization();
                split_node->add_output(split_output);
                split_node->infer_shape(true);
                subnode->add_input(split_output);
            }
            else
            {
                subnode->add_input(n->inputs()[j]);
            }
        }
        // copy gsize
        if (n->attrs().count(Constants::gsize))
            subnode->attrs()[Constants::gsize] = n->attrs()[Constants::gsize];
        g->add_node(subnode);
        ADD_LOCATION_BY_NODE(subnode, n);
        for (auto &&param : n->params())
        {
            subnode->params()[param.first] = param.second;
        }
        if (n->type() == OpType_Convolution || n->type() == OpType_FullyConnected)
        {
            subnode->params()["num_output"] = channel;
        }

        TensorShape out_shape_ = shape;
        if (n->type() == OpType_BatchNorm)
        {
            if (out_shape_.layout() == DataLayout_NCHWC32 && concat_dim == 1)
            {
                out_shape_[concat_dim] = channel / 32;
            }
            else
            {
                out_shape_[concat_dim] = channel;
            }
        }
        else
        {
            out_shape_[concat_dim] = channel;
        }

        // copy lut
        if (n->constants().count("lut"))
        {
            subnode->constants()["lut"] = std::make_shared<Tensor>(*n->constants()["lut"]);
        }
        if (n->constants().count("sqrt_lut"))
        {
            subnode->constants()["sqrt_lut"] = std::make_shared<Tensor>(*n->constants()["sqrt_lut"]);
        }

        tiling_set_constant(n, subnode, "scale0", out_c_start, channel);
        tiling_set_constant(n, subnode, "scale1", out_c_start, channel);
        tiling_set_constant(n, subnode, "zp0", out_c_start, channel);

        if (n->type() == OpType_LayerNorm)
        {
            // copy weight for LN
            if (n->constants().count("weights"))
            {
                subnode->constants()["weights"] = std::make_shared<Tensor>(*n->constants()["weights"]);
            }
            // copy bias for LN
            if (n->constants().count("biases"))
            {
                subnode->constants()["biases"] = std::make_shared<Tensor>(*n->constants()["biases"]);
            }
            // copy scale for LN
            if (n->constants().count("scale"))
            {
                subnode->constants()["scale"] = std::make_shared<Tensor>(*n->constants()["scale"]);
            }
            // copy shift for LN
            if (n->constants().count("shift"))
            {
                subnode->constants()["shift"] = std::make_shared<Tensor>(*n->constants()["shift"]);
            }
        }
        else
        {
            // copy weight for BN
            if (n->constants().count("weights"))
            {
                ComputeTensor weight = n->constants()["weights"];
                if (weight->shape().size() == 1)
                {
                    subnode->constants()["weights"] = std::make_shared<Tensor>(*n->constants()["weights"]);
                }
                else
                {
                    tiling_set_constant(n, subnode, "weights", out_c_start, channel);
                }

                // copy weight_scale
                auto wt_scales = n->constants()["weights"]->quantization().scales<ComputeTensor>();

                int wt_scales_size = wt_scales.size();
                if (wt_scales_size >= original_out_c)
                {
                    int group_num = wt_scales_size / original_out_c;
                    ComputeTensor new_wt_scales = wt_scales[{{out_c_start * group_num, (out_c_start + channel) * group_num}}];
                    new_wt_scales->shape() = change_shape_by_axis(new_wt_scales->shape(), 0 , channel*group_num);
                    new_wt_scales->monopolize_data();
                    subnode->constants()["weights"]->quantization().set_scales(new_wt_scales);
                }
                else
                {
                    subnode->constants()["weights"]->quantization().set_scales(wt_scales);
                }

                // copy weight_zp
                auto wt_zp = n->constants()["weights"]->quantization().offsets<ComputeTensor>();
                int wt_zp_size = wt_zp.size();
                if (wt_zp_size >= original_out_c)
                {
                    int group_num = wt_zp_size / original_out_c;
                    ComputeTensor new_wt_zp = wt_zp[{{out_c_start * group_num, (out_c_start + channel) * group_num}}];
                    new_wt_zp->shape() = change_shape_by_axis(new_wt_zp->shape(), 0 , channel*group_num);
                    new_wt_zp->monopolize_data();
                    subnode->constants()["weights"]->quantization().set_offsets(new_wt_zp);
                }
                else
                {
                    subnode->constants()["weights"]->quantization().set_offsets(wt_zp);
                }
            }
            // copy bias for BN
            if (n->constants().count("biases"))
            {
                ComputeTensor biases = n->constants()["biases"];
                if (biases->shape().size() == 1)
                {
                    subnode->constants()["biases"] = std::make_shared<Tensor>(*n->constants()["biases"]);
                }
                else
                {
                    ComputeTensor subnode_biases = biases[{{out_c_start, out_c_start + channel}}];
                    subnode_biases->shape() = change_shape_by_axis(subnode_biases->shape(), 0 , channel);
                    subnode_biases->monopolize_data();
                    subnode->constants()["biases"] = subnode_biases;
                    bool no_bias_compress = false;
                    auto op_plugin_options = BuilderConfig::get<std::map<std::string, std::string>>("op_plugin_options");
                    if (op_plugin_options.count("AIPU_32BIT_BIAS"))
                    {
                        auto value = std::atof(op_plugin_options["AIPU_32BIT_BIAS"].c_str());
                        if (value && subnode_biases->type() == aipubt::Tensor::INT32)
                        {
                            no_bias_compress = true;
                        }
                    }
                    if (!no_bias_compress)
                    {
                        WARN() << n->name() << " split conv output c to " << subnode->name()
                               << " for may cause accuracy loss due to bias compression in HW, try --op_plugin_options AIPU_32BIT_BIAS=1";
                    }
                }
            }

            // copy scale for BN
            if (n->constants().count("scale"))
            {
                ComputeTensor scale = n->constants()["scale"];
                if (scale->shape().size() == 1)
                {
                    subnode->constants()["scale"] = std::make_shared<Tensor>(*n->constants()["scale"]);
                }
                else
                {
                    tiling_set_constant(n, subnode, "scale", out_c_start, channel);
                }
            }
            // copy shift for BN
            if (n->constants().count("shift"))
            {
                ComputeTensor shift = n->constants()["shift"];
                if (shift->shape().size() == 1)
                {
                    subnode->constants()["shift"] = std::make_shared<Tensor>(*n->constants()["shift"]);
                }
                else
                {
                    tiling_set_constant(n, subnode, "shift", out_c_start, channel);
                }
            }

            // copy negative slope for prelu
            tiling_set_constant(n, subnode, "negative_slope", out_c_start, channel, shape.dim() - 1);
        }

        out_c_start = out_c_start + channel;

        // create subnode outputs
        out_shape_ = change_shape_by_axis(out_shape_, concat_dim, channel);
        out_shape_.reset_strides();
        TensorPtr subnode_out = std::make_shared<Tensor>(subnode_name, out_shape_, output_type);
        subnode_out->quantization() = n->outputs()[0]->quantization();
        subnode->add_output(subnode_out);
        subnode->infer_shape(true);
        concat->add_input(subnode_out);
    }
    concat->infer_shape(true);

    g->add_node(concat);
    ADD_LOCATION_BY_NODE(concat, n);
    g->remove_node(n);

    return true;
}

bool SplitNodeOutputC::operator()(GraphPtr g) const
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
            node->type() != OpType_Resize)
        {
            return false;
        }

        if (node->type() == OpType_Mul)
        {
            if (node->inputs()[1]->shape().size() > 1)
            {
                return false;
            }
        }

        auto out = node->outputs()[0];
        int axis = out->shape().dim() - 1;
        int32_t min_ch = 32 * split_num;
        if (out->shape()[axis] < min_ch)
        {
            return false;
        }

        if (split_num == 1)
        {
            return false;
        }

        // if (node->params().count("group") && node->params()["group"] > 1)
        // {
        //     return false;
        // }

        return true;
    };
    int split_num = m_output_c_vec.size();
    if (split_num == 0)
    {
        split_num = Target::get_the_target()->core_num();
    }
    auto m_nodes = g->nodes();
    for (auto n : m_nodes)
    {
        if (!hwa_criterion(n, split_num))
        {
            continue;
        }
        std::vector<int> output_c_vec = m_output_c_vec;
        if (m_output_c_vec.empty())
        {
            auto out = n->outputs()[0];
            int axis = out->shape().dim() - 1;
            int split_c = out->shape()[axis] / split_num;
            for (int i = 0; i < split_num; ++i)
            {
                if (i < split_num - 1)
                {
                    output_c_vec.push_back(split_c);
                }
                else
                {
                    output_c_vec.push_back(out->shape()[axis] - (split_c * i));
                }
            }
        }
        if (n->type() == OpType_Convolution)
        {
            split_conv_output_c_(g, n, output_c_vec);
        }
        else
        {
            set_output_c(output_c_vec);
            split_output_c(g, n);
        }
    }
    return true;
}

void SplitNodeOutputC::set_output_c(const std::vector<int> &output_c_vec) const
{
    m_output_c_vec = output_c_vec;
}
} // namespace aipubt

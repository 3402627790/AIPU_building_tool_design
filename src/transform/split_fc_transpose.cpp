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
 * @Date: 2025-4-01 13:38:32
 * @Last Modified by: Xialong Zhang
 * @Last Modified time: 2025-04-07 10:40:45
 *
 *=========Before===================================
 *        fc(0)
 *         |
 *      reshape(1)
 *         |
 *     transpose(2)
 *         |
 *      /  |  \
 * crop0 crop1, crop2
 *
 *=========After(split fc_cout)======================================
 *        fc(0_0)         fc(0_0)       fc(0_1)
 *         |                 |             |
 *      reshape(1_0)    reshape(1_1)   reshape(1_2)
 *         |                 |             |
 *     transpose(2_0)  transpose(2_1)  transpose(2_2)
 *         |                 |             |
 */
#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "quantization_transform/utils/quantize_utils.h"
#include <builder/target.h>

namespace aipubt
{

static inline bool criteria(std::vector<NodePtr> &m, const int split_num)
{
    auto fc = m[0];
    auto reshape = m[1];
    auto transpose = m[2];

    if (fc->inputs()[0]->is_io_tensor() || transpose->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    if (reshape->outputs()[0]->shape().dim() != 5)
    {
        return false;
    }

    auto arch = aipubt::get_target()->isainfo().type;
    if (arch < aipubt::ISA_ZYv6)
    {
        return false;
    }

    if (fc->outputs()[0]->mem_size() < 1 * 512 * 1024) // 1MB
    {
        return false;
    }

    int fc_co = fc->outputs()[0]->shape()[-1];
    if (fc_co % split_num != 0)
    {
        return false;
    }

    int reshape_check_num = reshape->outputs()[0]->shape()[2] * reshape->outputs()[0]->shape()[3] * reshape->outputs()[0]->shape()[4];
    if (fc_co != reshape_check_num || reshape->outputs()[0]->shape()[2] != split_num)
    {
        return false;
    }

    std::vector<int> trans_perm = transpose->params()["perm"];
    if (trans_perm.size() != 5 || trans_perm[0] != 2 || trans_perm[1] != 0 || trans_perm[2] != 3 || trans_perm[3] != 1 || trans_perm[4] != 4)
    {
        return false;
    }

    int count = 0;
    for (auto c : transpose->children()) // crops
    {
        if (c->type() != OpType_Crop || c->children().size() != 1 || c->outputs()[0]->is_io_tensor())
        {
            return false;
        }

        count++;
    }

    if (count != split_num)
    {
        return false;
    }

    return true;
}

bool split_fc_transpose(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_FullyConnected}, {"1", OpType_Reshape}, {"2", OpType_Transpose}}, {{"0", "1"}, {"1", "2"}, {"2", "3"}}}};

    auto tile_executor = [&](GraphPtr &g, const std::vector<NodePtr> &m, int split_num) -> bool {
        std::vector<int> split_step_vec;
        int out_c = m[0]->outputs()[0]->shape()[-1]; // fc cout
        int split_step = (out_c + split_num - 1) / split_num;
        for (int i = 0; i < split_num; ++i)
        {
            if (i < split_num - 1)
            {
                split_step_vec.push_back(split_step);
            }
            else
            {
                split_step_vec.push_back(out_c - (split_step * i));
            }
        }

        auto split_fc_outc = [&split_step_vec](NodePtr n, int split_num) -> TensorPtrList {
            auto g = n->graph();
            int out_c_start = 0;
            auto shape = n->outputs()[0]->shape();
            auto output_type = n->outputs()[0]->type();

            TensorPtrList split_out_tensors;
            ComputeTensor weight = n->constants()["weights"];
            ComputeTensor bias = n->constants()["biases"];
            auto weight_shape = weight->shape();

            for (int i = 0; i < split_num; i++)
            {
                std::string subnode_name = g->get_valid_node_name(n->name() + "/" + std::to_string(i));
                NodePtr subnode = Node::create(subnode_name, n->type());

                g->add_node(subnode);
                ADD_LOCATION_BY_NODE(subnode, n);

                for (auto &&param : n->params())
                {
                    subnode->params()[param.first] = param.second;
                }
                subnode->params()["num_output"] = split_step_vec[i];
                int channel = split_step_vec[i];
                TensorShape out_shape_ = shape;
                if (n->type() == OpType_FullyConnected)
                {
                    out_shape_[shape.dim() - 1] = channel;
                }
                else
                {
                    out_shape_.c() = channel;
                }
                ComputeTensor subnode_weight = weight[{{out_c_start, out_c_start + channel}}];
                subnode_weight->monopolize_data();
                subnode->constants()["weights"] = subnode_weight;

                int out_c = n->outputs()[0]->shape()[1];

                // copy weight_scale
                auto wt_scales = n->constants()["weights"]->quantization().scales<ComputeTensor>();
                int wt_scales_size = wt_scales.size();
                if (wt_scales_size >= out_c)
                {
                    int group_num = wt_scales_size / out_c;
                    ComputeTensor new_wt_scales = wt_scales[{{out_c_start * group_num, (out_c_start + channel) * group_num}}];
                    new_wt_scales->monopolize_data();
                    subnode->constants()["weights"]->quantization().set_scales(new_wt_scales);
                }

                // copy weight_zp
                auto wt_zp = n->constants()["weights"]->quantization().offsets<ComputeTensor>();
                int wt_zp_size = wt_zp.size();
                if (wt_zp_size >= out_c)
                {
                    int group_num = wt_zp_size / out_c;
                    ComputeTensor new_wt_zp = wt_zp[{{out_c_start * group_num, (out_c_start + channel) * group_num}}];;
                    new_wt_zp->monopolize_data();
                    subnode->constants()["weights"]->quantization().set_offsets(new_wt_zp);
                }

                // copy bias
                ComputeTensor w = bias[{{out_c_start, out_c_start + channel}}];
                w->monopolize_data();
                subnode->constants()["biases"] = w;
                bool no_bias_compress = false;
                auto op_plugin_options = BuilderConfig::get<std::map<std::string, std::string>>("op_plugin_options");
                if (op_plugin_options.count("AIPU_32BIT_BIAS"))
                {
                    auto value = std::atof(op_plugin_options["AIPU_32BIT_BIAS"].c_str());
                    if (value && w->type() == aipubt::Tensor::INT32)
                    {
                        no_bias_compress = true;
                    }
                }
                if (Tensor::is_float(w->type()))
                {
                    no_bias_compress = true;
                }
                if (!no_bias_compress)
                {
                    WARN() << n->name() << " split conv output c to " << subnode->name() << " for may cause accuracy loss due to bias compression in HW, try --op_plugin_options AIPU_32BIT_BIAS=1";
                }

                // copy negative slope for prelu
                if (n->constants().count("negative_slope"))
                {
                    ComputeTensor negative_slope = n->constants()["negative_slope"];
                    auto shape = negative_slope->shape();
                    std::vector<ComputeTensor::Slice> slices;
                    for (int i = 0; i < shape.dim(); ++i)
                    {
                        if (i == shape.dim() - 1)
                        {
                            ComputeTensor::Slice slice(out_c_start, out_c_start + channel);
                            slices.push_back(slice);
                        }
                        else
                        {
                            ComputeTensor::Slice slice(0, shape[i]);
                            slices.push_back(slice);
                        }
                    }
                    ComputeTensor w = negative_slope[slices];
                    w->monopolize_data();
                    subnode->constants()["negative_slope"] = w;
                }
                // copy scale
                if (n->constants().count("scale"))
                {
                    ComputeTensor scale = n->constants()["scale"];
                    ComputeTensor w = scale[{{out_c_start, out_c_start + channel}}];
                    w->monopolize_data();
                    subnode->constants()["scale"] = w;
                }

                // copy shift
                if (n->constants().count("shift"))
                {
                    WARN() << n->name() << " split conv output c to " << subnode->name() << " for channel-wise quantization may cause accuracy loss due to shift merge in HW";
                    ComputeTensor shift = n->constants()["shift"];
                    ComputeTensor w = shift[{{out_c_start, out_c_start + channel}}];
                    w->monopolize_data();
                    subnode->constants()["shift"] = w;
                }

                // copy lut
                if (n->constants().count("lut"))
                {
                    subnode->constants()["lut"] = std::make_shared<Tensor>(*n->constants()["lut"]);
                }
                out_c_start = out_c_start + channel;
                // redirect input/output and parents/children
                if (n->params().count("group") && n->params()["group"] == shape.c_total()) //this is depthwise
                {
                    subnode->params()["group"] = split_step_vec[i];
                    subnode->set_attr("is_dwconv", n->get_attr("is_dwconv"));
                    subnode->add_input(split_out_tensors[i]);
                }
                else
                {
                    subnode->add_input(n->inputs());
                }

                // create subnode outputs
                auto subnode_out = std::make_shared<Tensor>(g->get_valid_tensor_name(subnode_name), out_shape_, output_type);
                subnode_out->quantization() = n->outputs()[0]->quantization();
                subnode->add_output(subnode_out);
                subnode->infer_shape(true);

                split_out_tensors.push_back(subnode_out);
            }

            return split_out_tensors;
        };

        auto split_reshape_transpose = [&](NodePtr reshape, NodePtr transpose, TensorPtrList split_fc_out_tensors) -> TensorPtrList {
            TensorPtrList split_out_tensors;
            auto g = reshape->graph();
            for (uint32_t i = 0; i < split_fc_out_tensors.size(); ++i)
            {
                auto new_reshape_shape = reshape->outputs()[0]->shape();
                new_reshape_shape[2] = 1;
                auto reshape_out = ops::reshape(split_fc_out_tensors[i], new_reshape_shape);
                g->add_node(reshape_out->node());
                ADD_LOCATION_BY_NODE(reshape_out->node(), reshape);

                auto trans_out = ops::transpose(reshape_out, {2, 0, 3, 1, 4});
                g->add_node(trans_out->node());
                ADD_LOCATION_BY_NODE(trans_out->node(), transpose);
                split_out_tensors.push_back(trans_out);
            }
            return split_out_tensors;
        };

        auto fc = m[0];
        auto reshape = m[1];
        auto transpose = m[2];
        NodePtrList after_crop_nodes;
        TensorPtrList crops_out_tensor;
        for (uint32_t i = 0; i < transpose->children().size(); ++i)
        {
            after_crop_nodes.push_back(transpose->children()[i]->children()[0]); // only one child. this is crop
            crops_out_tensor.push_back(transpose->children()[i]->outputs()[0]);
        }

        auto fc_outs = split_fc_outc(fc, split_num);
        auto trans_out = split_reshape_transpose(reshape, transpose, fc_outs);

        for (uint32_t i = 0; i < trans_out.size(); ++i)
        {
            after_crop_nodes[i]->replace_input_by(crops_out_tensor[i], trans_out[i]);
        }

        fc->clear_inputs();
        fc->clear_outputs();
        g->remove_node(fc);
        reshape->clear_inputs();
        reshape->clear_outputs();
        g->remove_node(reshape);

        for (auto c : transpose->children()) // crops
        {
            c->clear_inputs();
            c->clear_outputs();
            g->remove_node(c);
        }

        transpose->clear_inputs();
        transpose->clear_outputs();
        g->remove_node(transpose);

        return true;
    };
    bool is_changed = false;
    const int split_num = 3;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);

        auto matched = matcher.match(g);
        for (auto &&m : matched)
        {
            if (!criteria(m, split_num))
            {
                continue;
            }

            is_changed |= tile_executor(g, m, split_num);
        }
    }
    return is_changed;
}
}; // namespace aipubt
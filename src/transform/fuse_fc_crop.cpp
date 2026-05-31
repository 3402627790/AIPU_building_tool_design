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
 * @Date: 2022-11-01 13:38:32
 * @Last Modified by: Xialong Zhang
 * @Last Modified time: 2022-07-05 10:40:45
 *
 *           |
 *           FC                     |
 *         /    \                  / \
 *    crop0     crop1   ==>     fc0   fc1
 *       |       |               |     |
 *
 *  fc + eltwise -> conv_32bit + eltwise
 *
 */

#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"
namespace aipubt
{
static inline bool criteria(std::vector<NodePtr> &m)
{
    auto is_one_dim_crop = [](NodePtr crop) -> bool
    {
        int32_t axis = 0;
        auto inp_shape = crop->inputs()[0]->shape();
        auto crop_params = crop->params()["crops"];
        int32_t count = 0;
        for (auto s : inp_shape)
        {
            int32_t start = crop_params[axis][0];
            int32_t end = crop_params[axis][1];
            if (s != end - start)
            {
                count++;
            }
            axis++;
        }

        if (count != 1)
        {
            return false;
        }

        return true;
    };

    auto get_key_axis = [](NodePtr crop) -> int32_t
    {
        int32_t axis = 0;
        auto inp_shape = crop->inputs()[0]->shape();
        auto crop_params = crop->params()["crops"];

        for (auto s : inp_shape)
        {
            // is_one_dim_crop() already make sure has one dim crop.
            int32_t start = crop_params[axis][0];
            int32_t end = crop_params[axis][1];
            if (s != end - start)
            {
                return axis;
            }
            axis++;
        }

        return axis;
    };

    auto crop_length_check = [get_key_axis](NodePtr crop0, NodePtr crop1) -> bool
    {
        auto key_axis = get_key_axis(crop0);
        auto crop0_params = crop0->params()["crops"];
        auto crop1_params = crop1->params()["crops"];
        int32_t ori_length = crop0->inputs()[0]->shape()[key_axis];

        auto crop0_start = static_cast<int32_t>(crop0_params[key_axis][0]);
        auto crop0_end = static_cast<int32_t>(crop0_params[key_axis][1]);
        auto crop1_start = static_cast<int32_t>(crop1_params[key_axis][0]);
        auto crop1_end = static_cast<int32_t>(crop1_params[key_axis][1]);

        if (crop1_end - crop1_start + crop0_end - crop0_start > ori_length)
        {
            return false;
        }

        // check overlap.
        if (crop1_end > crop0_end)
        {
            return crop1_start >= crop0_end;
        }
        else
        {
            // crop1_end <= crop0_end
            return crop0_start >= crop1_end;
        }
    };

    auto fc = m[0];
    if (fc->params()["with_activation"] != "NONE")
    {
        return false;
    }

    if (fc->children().size() > 2)
    {
        return false; // not support yet.
    }

    auto crop_0 = m[1];
    auto crop_1 = m[2];
    if (crop_0->outputs()[0]->is_io_tensor() || crop_1->outputs()[0]->is_io_tensor() || fc->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    if (crop_0->children().size() != 1 || crop_1->children().size() != 1)
    {
        return false;
    }

    auto crop0_node = m[1];
    auto crop1_node = m[2];
    if (!(is_one_dim_crop(crop0_node) && is_one_dim_crop(crop1_node)))
    {
        return false;
    }

    if (get_key_axis(crop0_node) != get_key_axis(crop1_node))
    {
        return false;
    }

    // only support split Cout
    int32_t dim = crop0_node->inputs()[0]->shape().dim() - 1;
    if (get_key_axis(crop0_node) != dim)
    {
        return false;
    }

    if (!crop_length_check(crop0_node, crop1_node))
    {
        return false;
    }

    return true;
}

bool fuse_fc_crop(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_FullyConnected}, {"1", OpType_Crop}, {"2", OpType_Crop}}, {{"0", "1"}, {"0", "2"}}},
    };

    using channel_pos = std::pair<int32_t, int32_t>;
    auto get_split_channel_pos = [](NodePtr crop) -> channel_pos
    {
        int32_t axis_ = 0;
        auto inp_shape = crop->inputs()[0]->shape();
        auto crop_params = crop->params()["crops"];
        channel_pos key_dim_pos = std::make_pair(0, 0);

        for (auto s : inp_shape)
        {
            // is_one_dim_crop() already make sure has one dim crop.
            int32_t delta = static_cast<int32_t>(crop_params[axis_][1]) - static_cast<int32_t>(crop_params[axis_][0]);
            if (s != delta)
            {
                key_dim_pos.first = crop_params[axis_][0];
                key_dim_pos.second = crop_params[axis_][1];
            }
            axis_++;
        }

        return key_dim_pos;
    };

    auto split_fc_outc = [g](NodePtr fc, channel_pos &pos0, std::string &name0) -> TensorPtr
    {
        ComputeTensor weights = fc->constants()["weights"];
        ComputeTensor biases = fc->constants()["biases"];
        auto weight_shape = weights->shape();

        // fc0 add node to graph
        auto channel0 = pos0.second - pos0.first;
        // std::string name0 = g->name() + "_" + fc->name() + "_0";
        NodePtr fc0 = Node::create(name0, fc->type());
        g->add_node(fc0);
        ADD_LOCATION_BY_NODE(fc0, fc);

        // copy constants
        ComputeTensor fc0_weight = weights[{{pos0.first, pos0.second}}];
        fc0_weight->monopolize_data();
        fc0->constants()["weights"] = fc0_weight;

        ComputeTensor fc0_bias = biases[{{pos0.first, pos0.second}}];
        fc0_bias->monopolize_data();
        fc0->constants()["biases"] = fc0_bias;

        int fc_oc = fc->outputs()[0]->shape()[1];

        auto w_scales = fc->constants()["weights"]->quantization().scales<ComputeTensor>();
        auto w_zps = fc->constants()["weights"]->quantization().offsets<ComputeTensor>();
        auto b_scales = fc->constants()["biases"]->quantization().scales<ComputeTensor>();
        auto b_zps = fc->constants()["biases"]->quantization().offsets<ComputeTensor>();
        int w_scales_size = w_scales.size();
        if (w_scales_size >= fc_oc)
        {
            int group_num = w_scales_size / fc_oc;
            ComputeTensor w_scale = w_scales[{{pos0.first * group_num, pos0.second * group_num}}];
            w_scale->monopolize_data();
            fc0->constants()["weights"]->quantization().set_scales(w_scale);
        }
        else
        {
            fc0->constants()["weights"]->quantization().set_scales(w_scales);
        }

        if (b_scales.size() > 1)
        {
            ComputeTensor b_scale = b_scales[{{pos0.first, pos0.second}}];
            b_scale->monopolize_data();
            fc0->constants()["biases"]->quantization().set_scales(b_scale);
        }
        else
        {
            fc0->constants()["biases"]->quantization().set_scales(b_scales);
        }

        int w_zps_size = w_zps.size();
        if (w_zps_size >= fc_oc)
        {
            int group_num = w_zps_size / fc_oc;
            ComputeTensor w_zp = w_zps[{{pos0.first * group_num, pos0.second * group_num}}];
            w_zp->monopolize_data();
            fc0->constants()["weights"]->quantization().set_offsets(w_zp);
        }
        else
        {
            fc0->constants()["weights"]->quantization().set_offsets(w_zps);
        }

        if (b_zps.size() > 1)
        {
            ComputeTensor b_zp = b_zps[{{pos0.first, pos0.second}}];
            b_zp->monopolize_data();
            fc0->constants()["biases"]->quantization().set_offsets(b_zp);
        }
        else
        {
            fc0->constants()["biases"]->quantization().set_offsets(b_zps);
        }

        // copy params
        for (auto &&param : fc->params())
        {
            fc0->params()[param.first] = param.second;
        }
        fc0->params()["num_output"] = channel0;

        // update out shape
        TensorShape out0_shape = fc->outputs()[0]->shape();
        auto out0_type = fc->outputs()[0]->type();
        out0_shape[out0_shape.dim() - 1] = channel0;

        // deal bias compress;
        bool no_bias_compress = false;
        auto op_plugin_options = BuilderConfig::get<std::map<std::string, std::string>>("op_plugin_options");
        if (op_plugin_options.count("AIPU_32BIT_BIAS"))
        {
            auto value = std::atof(op_plugin_options["AIPU_32BIT_BIAS"].c_str());
            if (value && fc0_bias->type() == aipubt::Tensor::INT32)
            {
                no_bias_compress = true;
            }
        }
        if (!no_bias_compress)
        {
            WARN() << fc->name() << " split FC output c to " << fc0->name() << " for may cause accuracy loss due to bias compression in HW, try --op_plugin_options AIPU_32BIT_BIAS=1";
        }

        // copy scale
        if (fc->constants().count("scale"))
        {
            ComputeTensor scale = fc->constants()["scale"];
            ComputeTensor w = scale[{{pos0.first, pos0.second}}];
            w->monopolize_data();
            fc0->constants()["scale"] = w;
        }

        // copy shift
        if (fc->constants().count("shift"))
        {
            ComputeTensor shift = fc->constants()["shift"];
            ComputeTensor w = shift[{{pos0.first, pos0.second}}];
            w->monopolize_data();
            fc0->constants()["shift"] = w;
        }

        // connect input
        fc0->add_input(fc->inputs());

        // connect output
        auto fc0_out = std::make_shared<Tensor>(name0, out0_shape, out0_type);
        fc0_out->quantization() = fc->outputs()[0]->quantization();
        fc0->add_output(fc0_out);
        fc0->infer_shape(true);

        return fc0_out;
    };

    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        for (auto &&m : matched)
        {
            if (!criteria(m))
            {
                continue;
            }

            // m[0]: FC
            // m[1]: crop
            // m[2]: crop
            auto fc_node = m[0];
            INFO() << "FC layer_id = " << fc_node->get_attr<int32_t>("layer_id");
            auto crop0_node = m[1];
            auto crop1_node = m[2];
            channel_pos pos_0 = get_split_channel_pos(crop0_node);
            channel_pos pos_1 = get_split_channel_pos(crop1_node);

            std::string name0 = g->name() + "_" + fc_node->name() + "_0";
            std::string name1 = g->name() + "_" + fc_node->name() + "_1";
            auto fc_split_out0 = split_fc_outc(fc_node, pos_0, name0);
            auto fc_split_out1 = split_fc_outc(fc_node, pos_1, name1);

            for (auto &c : crop0_node->children())
            {
                for (auto &inp : c->inputs())
                {
                    if (inp == crop0_node->outputs()[0])
                    {
                        c->replace_input_by(inp, fc_split_out0);
                    }
                }
            }

            for (auto &c : crop1_node->children())
            {
                for (auto &inp : c->inputs())
                {
                    if (inp == crop1_node->outputs()[0])
                    {
                        c->replace_input_by(inp, fc_split_out1);
                    }
                }
            }

            g->remove_node(fc_node);
            g->remove_node(crop0_node);
            g->remove_node(crop1_node);
        }
    }
    return is_changed;
}
}; // namespace aipubt
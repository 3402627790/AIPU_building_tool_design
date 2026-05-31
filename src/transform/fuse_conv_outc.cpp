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
 * @Author: Samuel Han
 * @Date: 2022-04-11 14:24:01
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-04-22 12:05:31
 */

#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include <algorithm>

namespace aipubt
{

namespace
{
    bool criteria(NodePtr n, NodePtrList &nodes)
    {
        if (n->type() != OpType_Concat || n->parents().empty() || n->parents()[0]->parents().empty())
        {
            return false;
        }
        int axis = n->params()["axis"];
        if (axis != 3) // concat output c
        {
            return false;
        }
        NodePtr pp = n->parents()[0]->parents()[0];
        for (auto p : n->parents())
        {
            if (p->type() != OpType_Convolution ||
                p->inputs()[0] != n->parents()[0]->inputs()[0] ||
                p->parents().empty() || p->parents()[0] != pp ||
                p->outputs()[0]->consumers().size() > 1)
            {
                return false;
            }
            if (!n->outputs()[0]->quantization().empty() && !p->outputs()[0]->quantization().empty())
            {
                if ((n->outputs()[0]->quantization().scale() != p->outputs()[0]->quantization().scale()) ||
                    (n->outputs()[0]->quantization().offset() != p->outputs()[0]->quantization().offset()))
                    return false;
            }
            nodes.push_back(p);
        }
        auto c0 = nodes[0]->params();
        auto wt = nodes[0]->constants()["weights"]->type();
        auto bt = nodes[0]->constants()["biases"]->type();
        if (!nodes[0]->constants()["weights"]->data<void>() || !nodes[0]->constants()["biases"]->data<void>())
        {
            return false;
        }
        for (uint32_t i = 1; i < nodes.size(); ++i)
        {
            if (wt != nodes[i]->constants()["weights"]->type() || bt != nodes[i]->constants()["biases"]->type() ||
                !nodes[i]->constants()["weights"]->data<void>() || !nodes[i]->constants()["biases"]->data<void>())
            {
                return false;
            }
            auto pp = nodes[i]->params();
            std::set<std::string> useless_params = {
                "num_output",
                "weights_range",
                "biases_range",
                "negative_slope_range",
                "weights_scale",
                "weights_zp",
                "biases_scale",
                "biases_zp",
            };
            for (auto kv : c0)
            {
                bool has_useless_params = false;
                if (!pp.count(kv.first) || !(kv.second == pp.at(kv.first)))
                {
                    for (auto up : useless_params)
                    {
                        if (kv.first == up)
                        {
                            // useless params, delete it can support different output c
                            has_useless_params = true;
                            break;
                        }
                    }
                    if (has_useless_params)
                    {
                        continue;
                    }
                    return false;
                }
            }
        }

        return true;
    }

    ComputeTensor scale_shift_tensor(NodePtr n, std::string key, int channel)
    {
        ComputeTensor cp_tensor = nullptr;
        if (n->constants().count(key))
        {
            cp_tensor = n->constants()[key];
        }
        else
        {
            int32_t scale_value = n->params()[key + "_value"];
            std::vector<int32_t> scale_values(channel, scale_value);
            cp_tensor = ComputeTensor(scale_values);
            Tensor::Dtype tensor_type = n->params()[key + "_type"];
            cp_tensor.astype(tensor_type);
        }
        return cp_tensor;
    }

    void fuse_quantization(NodePtr n, std::string key, int channel,
                           std::vector<float> &new_scales, std::vector<int32_t> &new_offsets)
    {
        std::vector<float> scales = n->constants()[key]->quantization().scales();
        std::vector<int32_t> offsets = n->constants()[key]->quantization().offsets();
        if (static_cast<int>(scales.size()) != channel)
        {
            new_scales.insert(new_scales.end(), channel, scales[0]);
            new_offsets.insert(new_offsets.end(), channel, offsets[0]);
        }
        else
        {
            new_scales.insert(new_scales.end(), scales.begin(), scales.end());
            new_offsets.insert(new_offsets.end(), offsets.begin(), offsets.end());
        }
    }

} // namespace

bool fuse_conv_outc(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        NodePtrList matched;
        if (!criteria(n, matched))
        {
            continue;
        }
        auto c0 = matched[0];
        int c0_channel = c0->constants()["weights"]->shape()[0];

        //fuse constant
        for (auto key : {"weights", "biases", "negative_slope"})
        {
            if (!c0->constants().count(key))
            {
                continue;
            }

            Quantization c0_tensor_qt = c0->constants()[key]->quantization();
            if (!c0->constants()[key]->quantization().empty())
            {
                std::vector<float> new_scales;
                std::vector<int32_t> new_offsets;
                fuse_quantization(c0, key, c0_channel, new_scales, new_offsets);
                for (uint32_t i = 1; i < matched.size(); ++i)
                {
                    int channel = matched[i]->constants()["weights"]->shape()[0];
                    fuse_quantization(matched[i], key, channel, new_scales, new_offsets);
                }
                c0_tensor_qt.set_scales(new_scales);
                c0_tensor_qt.set_offsets(new_offsets);
            }

            ComputeTensor c0_tensor = c0->constants()[key];
            for (uint32_t i = 1; i < matched.size(); ++i)
            {
                ComputeTensor cn_tensor = matched[i]->constants()[key];
                c0_tensor = ComputeTensor::concat({c0_tensor, cn_tensor});
            }
            c0_tensor->quantization() = c0_tensor_qt;
            c0->constants()[key] = c0_tensor;
        }

        //fuse weights/biases/negative_slope range
        for (auto key : {"weights_range", "biases_range", "negative_slope_range"})
        {
            if (!c0->params().count(key))
            {
                continue;
            }
            std::vector<std::vector<float>> xxx_range = c0->params()[key];
            for (uint32_t i = 1; i < matched.size(); ++i)
            {
                std::vector<std::vector<float>> _range = matched[i]->params()[key];
                for (auto _r : _range)
                {
                    xxx_range.push_back(_r);
                }
            }
            c0->params()[key] = xxx_range;
        }

        // fuse scale/shift to constant and remove from param
        if (c0->constants().count("scale") || c0->params().count("scale_value"))
        {
            for (auto key : {"scale", "shift"})
            {
                ComputeTensor c0_tensor = scale_shift_tensor(c0, key, c0_channel);
                for (uint32_t i = 1; i < matched.size(); ++i)
                {
                    int cn_channel = matched[i]->constants()["weights"]->shape()[0];
                    ComputeTensor cn_tensor = scale_shift_tensor(matched[i], key, cn_channel);
                    c0_tensor = ComputeTensor::concat({c0_tensor, cn_tensor});
                }
                c0->constants()[key] = c0_tensor;
            }
            for (auto key : {"scale_value", "shift_value", "scale_type", "shift_type"})
            {
                if (c0->params().count(key))
                    c0->params().erase(key);
            }
        }
        auto out = n->outputs()[0];
        n->clear_outputs();
        c0->clear_outputs();
        c0->add_output(out);
        MODIFY_NODE(c0);
        for (uint32_t i = 1; i < matched.size(); ++i)
        {
            g->remove_node(matched[i]);
        }
        g->remove_node(n);
        is_changed = true;
    }

    return is_changed;
}

} // namespace aipubt

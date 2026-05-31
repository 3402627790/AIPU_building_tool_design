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
 * @Author: Acheng Pan
 * @Date: 2022-10-19 11:25:43
 * @Last Modified by: Acheng Pan
 * @Last Modified time: 2022-10-19 13:37:25
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include <algorithm>
#include <set>

namespace aipubt
{

static bool is_symbol_ds_crops(NodePtr n)
{
    if (n->params().count("ds_crops") == 0)
        return false;
    std::vector<std::vector<tongue::Expr>> ds_crops = n->params()["ds_crops"];
    for (auto ds_crop : ds_crops)
    {
        for (auto expr : ds_crop)
        {
            if (!expr->is_constant())
            {
                return true;
            }
        }
    }
    return false;
}

static inline bool creteria(NodePtr n)
{
    if (n->children().empty())
    {
        return false;
    }
    if ((n->type() != OpType_Convolution) && (n->type() != OpType_ConvTranspose))
    {
        return false;
    }
    int group = n->params()["group"];
    if (group != 1)
    {
        return false;
    }

    for (auto &nc : n->children())
    {
        if ((nc->type() != OpType_Crop) && (nc->type() != OpType_Slice))
        {
            return false;
        }

        if (nc->type() == OpType_Slice)
        {
            std::vector<int> strides = nc->params()["strides"];
            if (std::any_of(strides.begin(), strides.end(), [](int i) { return i != 1; }))
            {
                return false;
            }
        }
        if (nc->outputs()[0]->shape().n() != n->outputs()[0]->shape().n())
        {
            return false;
        }
        if (nc->outputs()[0]->shape().h() != n->outputs()[0]->shape().h())
        {
            return false;
        }
        if (nc->outputs()[0]->shape().w() != n->outputs()[0]->shape().w())
        {
            return false;
        }
        if (is_symbol_ds_crops(nc))
        {
            return false;
        }
    }
    return true;
}

static inline void set_constant(NodePtr n, NodePtr subnode, std::string name,
                                TensorShape subnode_shape, Tensor::Dtype dtype, uint32_t size, uint32_t offset)
{
    TensorPtr subnode_const = std::make_shared<Tensor>(subnode->name() + "_" + name, subnode_shape, dtype);
    uint8_t *_data = new uint8_t[size];
    if (n->constants()[name]->data<uint8_t>())
    {
        uint8_t *_origin_ptr = n->constants()[name]->data<uint8_t>() + offset;
        memcpy(_data, _origin_ptr, size);
    }
    else
    {
        memset(_data, 0, size);
    }

    subnode_const->set_data(_data, size);

    subnode->constants()[name] = subnode_const;
}

static inline void split_constant(NodePtr n, NodePtr subnode, TensorShape shape, int start, int len)
{
    if (n->constants().count("weights"))
    {
        auto khwc = shape.c() * shape.w() * shape.h();

        Tensor::Dtype _dtype = n->constants()["weights"]->type();
        auto _data_size = Tensor::dtype_size(_dtype);
        auto curr_size = len * khwc * _data_size;
        auto curr_start_offset = start * khwc * _data_size;

        TensorShape subnode_shape = TensorShape(len, shape.h(), shape.w(), shape.c());
        set_constant(n, subnode, "weights", subnode_shape, _dtype, curr_size, curr_start_offset);
    }
}

static inline void split_constant(NodePtr n, NodePtr subnode, int start, int len)
{
    std::set<std::string> vector_const = {"biases", "negative_slope", "scale", "shift"};
    for (auto &vc : vector_const)
    {
        if (n->constants().count(vc))
        {
            Tensor::Dtype _dtype = n->constants()[vc]->type();
            auto _data_size = Tensor::dtype_size(_dtype);
            auto curr_size = len * _data_size;
            auto curr_start_offset = start * _data_size;

            TensorShape subnode_shape = TensorShape(len);
            set_constant(n, subnode, vc, subnode_shape, _dtype, curr_size, curr_start_offset);
        }
    }
}

static inline void split_quant_from_params(NodePtr n, NodePtr subnode, std::string name, int start, int len)
{
    if (n->params().count(name))
    {
        auto _origin_info = n->params()[name].arr;
        if (_origin_info.size() <= 1)
        {
            return;
        }

        subnode->params()[name].arr.assign(_origin_info.begin() + start, _origin_info.begin() + start + len);
    }
}

static inline void split_scale_from_quantization(NodePtr n, NodePtr subnode, std::string const_name, int start, int len)
{
    std::vector<float> _origin_info = n->constants()[const_name]->quantization().scales();
    std::vector<float> _info;
    if (_origin_info.size() <= 1)
    {
        return;
    }
    _info.assign(_origin_info.begin() + start, _origin_info.begin() + start + len);
    subnode->constants()[const_name]->quantization().set_scales(_info);
}

static inline void split_zp_from_quantization(NodePtr n, NodePtr subnode, std::string const_name, int start, int len)
{
    std::vector<int32_t> _origin_info = n->constants()[const_name]->quantization().offsets();
    std::vector<int32_t> _info;
    if (_origin_info.size() <= 1)
    {
        return;
    }
    _info.assign(_origin_info.begin() + start, _origin_info.begin() + start + len);
    subnode->constants()[const_name]->quantization().set_offsets(_info);
}

static inline void split_quant_from_quantization(NodePtr n, NodePtr subnode, std::string const_name, int start, int len)
{
    if (n->constants().count(const_name))
    {
        split_scale_from_quantization(n, subnode, const_name, start, len);
        split_zp_from_quantization(n, subnode, const_name, start, len);
    }
}

static inline void split_quantization(NodePtr n, NodePtr subnode, int start, int len)
{
    std::string delimiter = "_";
    std::set<std::string> vector_quantization =
        {
            "bias_top_scale",
            "bias_top_zp",
            "biases_scale",
            "biases_zp",
            "bias_scale",
            "bias_zp",
            "weight_top_scale",
            "weight_top_zp",
            "weights_scale",
            "weights_zp",
            "weight_scale",
            "weight_zp",
            "negative_slope_top_scale",
            "negative_slope_top_zp",
            "negative_slope_scale",
            "negative_slope_zp",
        };
    std::set<std::string> const_quantization_name = {};
    for (auto &vq : vector_quantization)
    {
        auto fst = vq.find(delimiter);
        auto const_name = vq.substr(0, fst);
        if (!const_name.compare("bias"))
            const_name = "biases";
        if (!const_name.compare("weight"))
            const_name = "weights";
        if (!const_name.compare("negative"))
            const_name = "negative_slope";
        if (!const_quantization_name.count(const_name))
        {
            const_quantization_name.insert(const_name);
        }

        split_quant_from_params(n, subnode, vq, start, len);
    }
    for (auto &cqn : const_quantization_name)
    {
        split_quant_from_quantization(n, subnode, cqn, start, len);
    }
}

bool merge_conv_output_crop(GraphPtr g)
{
    // From AI_benchamrk: xlsr, imdn
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }

        int out_c_start = 0;
        int channel = 0;
        int group = n->params()["group"];
        for (size_t i = 0; i < n->children().size(); i++)
        {
            if (n->children()[i]->type() == OpType_Slice)
            {
                auto curr_bengin = n->children()[i]->params()["begin"];
                auto curr_end = n->children()[i]->params()["end"];
                out_c_start = curr_bengin[3];
                channel = curr_end[3];
            }
            if (n->children()[i]->type() == OpType_Crop)
            {
                auto curr_crops = n->children()[i]->params()["crops"];
                out_c_start = curr_crops[3][0];
                channel = curr_crops[3][1];
            }
            channel -= out_c_start;

            std::string subnode_name = n->name() + "/" + std::to_string(i);
            NodePtr subnode = Node::create(subnode_name, n->type());

            g->add_node(subnode);
            ADD_LOCATION_BY_NODE(subnode, n);
            for (auto &&param : n->params())
            {
                subnode->params()[param.first] = param.second;
            }
            subnode->params()["group"] = group == 1 ? group : channel;
            subnode->params()["num_output"] = channel;
            // weights
            split_constant(n, subnode, n->constants()["weights"]->shape(), out_c_start, channel);
            // biases/scale/shift/negative_slope
            split_constant(n, subnode, out_c_start, channel);
            // weights/biases/negative_slope scale/zp
            split_quantization(n, subnode, out_c_start, channel);

            subnode->add_input(n->inputs());
            subnode->add_output(n->children()[i]->outputs()[0]);
            subnode->infer_shape(true);
        }

        for (auto &cn : n->children())
        {
            g->remove_node(cn);
        }
        g->remove_node(n);

        is_changed = true;
    }
    return is_changed;
}
}; // namespace aipubt

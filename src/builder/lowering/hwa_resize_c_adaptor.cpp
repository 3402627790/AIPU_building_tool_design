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
 * @Author: Tom Gao
 * @Date: 2021-12-29 10:07:56
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-09-26 15:03:25
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/config.h"

namespace aipubt
{
#define RETURN_FALSE_IF_FALSE(x) \
    if ((x) == false)            \
    {                            \
        return false;            \
    }

static bool check_required_params(NodePtr n, const std::string &pass_name, const std::vector<std::string> &required_params)
{
    for (auto &key : required_params)
    {
        if (!n->params().count(key))
        {
            WARN() << pass_name << ": " << n->name() << ", MissingParam: " << key;
            return false;
        }
    }
    return true;
}

static bool criteria_qian_r1p0(NodePtr n, int &align)
{
    RETURN_FALSE_IF_FALSE(n->type() == OpType_Resize);
    RETURN_FALSE_IF_FALSE(n->inputs()[0]->type() != Tensor::FP16 &&
                          n->inputs()[0]->type() != Tensor::BFP16 &&
                          n->outputs()[0]->type() != Tensor::FP16 &&
                          n->outputs()[0]->type() != Tensor::BFP16);
    RETURN_FALSE_IF_FALSE(n->inputs()[0]->shape().dim() == 4 ||
                          n->inputs()[0]->shape().dim() == 5);

    static const std::vector<std::string> required_params = {
        "method",
        "mode",
    };
    RETURN_FALSE_IF_FALSE(check_required_params(n, "hwa_resize_c_adaptor", required_params));

    std::string mode = n->params()["mode"];
    std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
    RETURN_FALSE_IF_FALSE(mode == "half_pixel" ||
                          mode == "asymmetric" ||
                          mode == "align_corners" ||
                          mode == "tf_half_pixel_for_nn" ||
                          mode == "pytorch_half_pixel");

    std::string method = n->params()["method"];
    std::transform(method.begin(), method.end(), method.begin(), ::tolower);
    RETURN_FALSE_IF_FALSE(method == "bilinear");
    RETURN_FALSE_IF_FALSE(n->inputs()[0]->type() != Tensor::UINT16);

    auto i_shape = n->inputs()[0]->shape();
    auto o_shape = n->outputs()[0]->shape();
    RETURN_FALSE_IF_FALSE((i_shape.h() >= 2) && (i_shape.h() <= 4096));
    RETURN_FALSE_IF_FALSE((i_shape.w() >= 2) && (i_shape.w() <= 4096));
    RETURN_FALSE_IF_FALSE((o_shape.h() >= 1) && (o_shape.h() <= 4096));
    RETURN_FALSE_IF_FALSE((o_shape.w() >= 1) && (o_shape.w() <= 4096));
    RETURN_FALSE_IF_FALSE((i_shape.c_total() <= 4096) && (o_shape.c_total() <= 4096));

    int32_t max_in_h = o_shape.h() * 16;
    int32_t max_in_w = o_shape.w() * 16;
    int32_t max_out_h = i_shape.h() * 64;
    int32_t max_out_w = i_shape.w() * 64;
    if (mode == "align_corners")
    {
        if (o_shape.h() > 1)
        {
            max_in_h = (o_shape.h() - 1) * 16 + 1;
            max_out_h = (i_shape.h() - 1) * 64 + 1;
        }
        if (o_shape.w() > 1)
        {
            max_in_w = (o_shape.w() - 1) * 16 + 1;
            max_out_w = (i_shape.w() - 1) * 64 + 1;
        }
    }
    RETURN_FALSE_IF_FALSE(i_shape.h() <= max_in_h);
    RETURN_FALSE_IF_FALSE(i_shape.w() <= max_in_w);
    RETURN_FALSE_IF_FALSE(o_shape.h() <= max_out_h);
    RETURN_FALSE_IF_FALSE(o_shape.w() <= max_out_w);

    if (!i_shape.symbol().c()->is_constant() ||
        (i_shape.symbol().c()->is_constant() && i_shape.c() % 8 != 0))
    {
        align = 8;
        return true;
    }
    return false;
}

static std::map<aipubt::ISAType, std::function<int(NodePtr, int &)>> criteria_map = {
    {aipubt::ISA_ZYv3, criteria_qian_r1p0},
    {aipubt::ISA_ZYv5, criteria_qian_r1p0},
};

void hwa_resize_c_adaptor(GraphPtr g)
{
    std::function<int(NodePtr, int &)> criteria;
    auto arch = get_target()->isainfo().type;
    if (!criteria_map.count(arch))
    {
        return;
    }
    criteria = criteria_map[arch];
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);

    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        int align = 1;
        if (!criteria(n, align))
        {
            continue;
        }

        TensorPtr inp = n->inputs()[0];
        int in_c = inp->shape().c();
        int new_in_c = ((in_c + align - 1) / align) * align;

        // create pad node
        NodePtr pad_node = Node::create(n->name() + "_pad", OpType_Pad);
        pad_node->params()["mode"] = "CONSTANT";
        pad_node->params()["constant_value"] = 0 - inp->quantization().offset();

        auto pad_shape = inp->shape();
        pad_shape.c() = new_in_c;
        TensorPtr pad_out = std::make_shared<Tensor>(inp->name() + "_pad", pad_shape, inp->type());

        pad_node->add_input(inp);
        pad_node->add_output(pad_out);
        g->add_node(pad_node);
        ADD_LOCATION_BY_NODE(pad_node, n);

        auto ndim = inp->shape().dim();
        std::vector<std::vector<int>> pads(ndim, {0, 0});
        pads.back() = {0, new_in_c - in_c};
        pad_node->params()["pads"] = pads;

        if (enable_ds)
        {
            std::vector<std::vector<Expr>> ds_pads(ndim, {0, 0});
            ds_pads.back()[1] = (align - inp->shape().symbol()[-1] % align) % align;
            pad_node->params()["ds_pads"] = ds_pads;
            pad_node->set_attr(Constants::c_align, align);
        }
        pad_node->infer_shape(true);

        // update node
        TensorPtr out = n->outputs()[0];
        auto resize_shape = out->shape();
        resize_shape.c() = new_in_c;
        TensorPtr resize_out = std::make_shared<Tensor>(out->name() + "_crop", resize_shape, out->type());
        resize_out->quantization() = out->quantization();

        n->clear_inputs();
        n->clear_outputs();
        n->add_input(pad_out);
        n->add_output(resize_out);
        n->infer_shape(true);

        // create crpp node
        NodePtr crop_node = Node::create(n->name() + "_crop", OpType_Crop);
        crop_node->add_input(resize_out);
        crop_node->add_output(out);

        std::vector<std::vector<int>> crops(ndim, {0, 0});
        for (int i = 0; i < out->shape().dim(); ++i)
        {
            crops[i] = {0, crop_node->outputs()[0]->shape()[i]};
        }
        crop_node->params()["crops"] = crops;

        if (enable_ds)
        {
            std::vector<std::vector<Expr>> ds_crops;
            for (int i = 0; i < out->shape().dim(); ++i)
            {
                ds_crops.push_back({0, out->shape().symbol()[i]});
            }
            crop_node->params()["ds_crops"] = ds_crops;
            crop_node->set_attr(Constants::c_align, align);
        }
        crop_node->infer_shape(true);

        g->add_node(crop_node);
        ADD_LOCATION_BY_NODE(crop_node, n);
    }
}
} // namespace aipubt

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
 * @Author: Tom.Gao
 * @Date: 2020-11-02 17:58:05
 * @Last Modified by: Tom.Gao
 * @Last Modified time: 2020-11-02 17:58:05
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/compute_tensor.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{
#define ALIGN_PAD(x, a) ((((x) + (a)-1) / (a)) * (a))

static void pad_weight_bias(NodePtr n, int old_c)
{
    int new_out_c = n->outputs()[0]->shape()[1];
    auto weight = n->constants()["weights"];
    auto w_shape = weight->shape();
    w_shape[0] = new_out_c;
    auto new_weight = ComputeTensor::zeros(w_shape, weight->type());
    new_weight[{{0, old_c}, {0, -1}}] = weight;
    new_weight->make_memory_continuous();
    n->constants()["weights"] = new_weight;

    auto bias = n->constants()["biases"];
    auto b_shape = bias->shape();
    b_shape[0] = new_out_c;
    auto new_bias = ComputeTensor::zeros(b_shape, bias->type());
    new_bias[{{0, old_c}}] = bias;
    new_bias->make_memory_continuous();
    n->constants()["biases"] = new_bias;
}

static int criteria_zyv1(NodePtr n)
{
    bool ret = (n->type() == OpType_FullyConnected);
    if (!ret)
        return 0;

    uint16_t mac_num = get_target()->ff_mac_num();
    int align_num;
    if (mac_num >= 1024)
    {
        align_num = 64;
    }
    else
    {
        align_num = 32;
    }

    int in_n = n->inputs()[0]->shape()[0];
    int out_n = n->outputs()[0]->shape()[0];
    int out_c = n->outputs()[0]->shape()[1];
    ret = ret && (in_n == out_n) && (out_n > 1);
    ret = ret && (out_c % align_num != 0);

    return ret ? align_num : 0;
}

static int criteria_qian(NodePtr n)
{
    bool ret = (n->type() == OpType_FullyConnected);
    if (!ret)
        return 0;

    int align_num = 16;
    if (n->outputs()[0]->type() == Tensor::INT8 || n->outputs()[0]->type() == Tensor::UINT8 ||
        n->inputs()[0]->type() == Tensor::INT8 || n->inputs()[0]->type() == Tensor::UINT8)
    {
        align_num = 32;
    }

    int in_n = n->inputs()[0]->shape()[0];
    int out_n = n->outputs()[0]->shape()[0];
    int out_c = n->outputs()[0]->shape()[1];
    ret = ret && (in_n == out_n) && (out_n > 1);
    ret = ret && (out_c % align_num != 0);

    return ret ? align_num : 0;
}

static int criteria_zyv3(NodePtr n)
{
    if (n->type() != OpType_FullyConnected)
    {
        return 0;
    }

    if (n->inputs()[0]->type() == Tensor::FP16 ||
        n->inputs()[0]->type() == Tensor::BFP16 ||
        n->outputs()[0]->type() == Tensor::FP16 ||
        n->outputs()[0]->type() == Tensor::BFP16)
    {
        return 0;
    }

    const int align_num = 8;
    int in_n = n->inputs()[0]->shape()[0];
    int out_n = n->outputs()[0]->shape()[0];
    int out_c = n->outputs()[0]->shape()[1];
    bool ret = (in_n == out_n) && (out_n > 1) && (out_c % align_num != 0);

    return ret ? align_num : 0;
}

static std::map<aipubt::ISAType, std::function<int(NodePtr)>> criteria_map = {
    {aipubt::ISA_ZY, criteria_zyv1},
    {aipubt::ISA_ZYv2, criteria_qian},
    {aipubt::ISA_ZYv3, criteria_zyv3},
    {aipubt::ISA_ZYv5, criteria_zyv3},
};

void hwa_fc_outc_adaptor(GraphPtr g)
{
    std::function<int(NodePtr)> criteria;
    auto arch = get_target()->isainfo().type;
    if (criteria_map.count(arch))
    {
        criteria = criteria_map[arch];
    }
    else
    {
        // ERROR() << "target arch : " << arch << " do NOT support!!";
        return;
    }

    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        int align_num = criteria(n);
        if (align_num == 0)
            continue;

        //update fc node
        TensorPtr n_tensor = n->outputs()[0];
        int out_c = n_tensor->shape()[-1];
        TensorShape conv_shape = n_tensor->shape();
        conv_shape[-1] = ALIGN_PAD(out_c, align_num);
        TensorPtr conv_out = std::make_shared<Tensor>(n->name() + "_fc_out_crop", conv_shape, n_tensor->type());
        conv_out->set_type(n->outputs()[0]->type());
        conv_out->quantization() = n_tensor->quantization();
        n->clear_outputs();
        n->add_output(conv_out);

        pad_weight_bias(n, out_c);

        if (n->constants().count("negative_slope"))
        {
            TensorPtr neg_slope = n->constants()["negative_slope"];
            auto ns_shape = neg_slope->shape();
            aipu_assert(ns_shape.dim() == 1);
            int new_out_c = conv_shape[1];
            ns_shape[0] = new_out_c;
            auto new_neg_slope = ComputeTensor::zeros(ns_shape, neg_slope->type());
            new_neg_slope[{{0, out_c}}] = neg_slope;
            new_neg_slope->make_memory_continuous();
            n->constants()["negative_slope"] = new_neg_slope;
        }
        if (n->constants().count("scale"))
        {
            int new_out_c = conv_shape[1];
            auto scale = n->constants()["scale"];
            auto s_shape = scale->shape();
            s_shape[0] = new_out_c;
            auto new_scale = ComputeTensor::zeros(s_shape, scale->type());
            new_scale[{{0, out_c}}] = scale;
            new_scale->make_memory_continuous();
            n->constants()["scale"] = new_scale;
        }
        if (n->constants().count("shift"))
        {
            int new_out_c = conv_shape[1];
            auto shift = n->constants()["shift"];
            auto s_shape = shift->shape();
            s_shape[0] = new_out_c;
            auto new_shift = ComputeTensor::zeros(s_shape, shift->type());
            new_shift[{{0, out_c}}] = shift;
            new_shift->make_memory_continuous();
            n->constants()["shift"] = new_shift;
        }
        n->params()["num_output"] = conv_shape[-1];
        n->infer_shape(true);

        // create crop node
        std::vector<std::pair<int32_t, int32_t>> crops;
        for (int i = 0; i < n_tensor->shape().dim(); ++i)
        {
            crops.push_back({0, n_tensor->shape()[i]});
        }
        auto crop_tensor = ops::crop(conv_out, crops);
        auto crop = crop_tensor->node();
        crop->clear_outputs();
        crop->add_output(n_tensor);
        if (enable_ds)
        {
            std::vector<std::vector<Expr>> ds_crops;
            for (int i = 0; i < n_tensor->shape().dim(); ++i)
            {
                ds_crops.push_back({0, n_tensor->shape().symbol()[i]});
            }
            crop->params()["ds_crops"] = ds_crops;
        }
        crop->infer_shape(true);

        g->add_node(crop);
        ADD_LOCATION_BY_NODE(crop, n);
    }
}
} // namespace aipubt

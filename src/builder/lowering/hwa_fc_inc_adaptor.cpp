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
 * @Date: 2020-11-02 18:35:05
 * @Last Modified by: Tom.Gao
 * @Last Modified time: 2020-11-02 18:35:05
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

static void pad_fc_weight_inc(NodePtr n, int new_in_c)
{
    TensorPtr weight = n->constants()["weights"];
    auto w_shape = weight->shape();
    int ori_in_c = w_shape[1];
    w_shape[1] = new_in_c;
    auto new_weight = ComputeTensor::zeros(w_shape, weight->type());
    new_weight[{{0, -1}, {0, ori_in_c}}] = weight;
    new_weight->make_memory_continuous();
    n->constants()["weights"] = new_weight;
}

static int criteria_zyv1(NodePtr n)
{
    return 0;
    // int ret = 0;

    // bool fc = ((n->type() == OpType_FullyConnected));
    // if (!(fc))
    //     return ret;

    // int align_num = 4;
    // int in_c = n->inputs()[0]->shape()[1];
    // ret = ALIGN_PAD(in_c, align_num);

    // return ret;
}

static int criteria_qian(NodePtr n)
{
    bool ret = (n->type() == OpType_FullyConnected);
    if (!ret)
        return 0;

    int align_num = 16;
    if (n->inputs()[0]->type() == Tensor::INT8 || n->inputs()[0]->type() == Tensor::UINT8)
    {
        align_num = 32;
    }

    int in_c = n->inputs()[0]->shape()[1];
    int in_n = n->inputs()[0]->shape()[0];
    int out_n = n->outputs()[0]->shape()[0];
    ret = ret && (in_n == out_n) && (out_n > 1);
    ret = ret && (in_c % align_num != 0);

    return ret ? ALIGN_PAD(in_c, align_num) : 0;
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
    int in_c = n->inputs()[0]->shape()[1];
    int in_n = n->inputs()[0]->shape()[0];
    int out_n = n->outputs()[0]->shape()[0];
    bool ret = (in_n == out_n) && (out_n > 1) && (in_c % align_num != 0);

    return ret ? ALIGN_PAD(in_c, align_num) : 0;
}

static std::map<aipubt::ISAType, std::function<int(NodePtr)>> criteria_map = {
    {aipubt::ISA_ZY, criteria_zyv1},
    {aipubt::ISA_ZYv2, criteria_qian},
    {aipubt::ISA_ZYv3, criteria_zyv3},
    {aipubt::ISA_ZYv5, criteria_zyv3},
};

void hwa_fc_inc_adaptor(GraphPtr g)
{
    std::function<int(NodePtr)> criteria;
    auto arch = get_target()->isainfo().type;
    if (criteria_map.count(arch))
    {
        criteria = criteria_map[arch];
    }
    else
    {
        // ERROR() << "target arch : " << arch << "do NOT support!!";
        return;
    }

    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        int new_in_c = criteria(n);
        if (new_in_c == 0)
            continue;

        // create pad node
        std::vector<std::pair<uint32_t, uint32_t>> pads;
        for (int i = 0; i < n->inputs()[0]->shape().dim() - 1; ++i)
        {
            pads.push_back({0, 0});
        }
        pads.push_back({0, new_in_c - n->inputs()[0]->shape()[-1]});
        auto pad_tensor = ops::pad(n->inputs()[0], pads, ops::PadMode::Constant, 0 - n->inputs()[0]->quantization().offset());
        auto pad = pad_tensor->node();
        if (enable_ds)
        {
            auto dims = n->inputs()[0]->shape().dim();
            std::vector<std::vector<Expr>> ds_pads(dims, {0, 0});
            ds_pads[dims - 1][1] = pads[dims - 1].second;
            pad->params()["ds_pads"] = ds_pads;
        }
        pad->infer_shape(true);

        g->add_node(pad);
        ADD_LOCATION_BY_NODE(pad, n);

        // update fc node
        n->clear_inputs();
        n->add_input(pad_tensor);
        if (!n->constants().count("weights"))
        {
            ERROR() << "hwa_fc_inc_adaptor : node " << n->name() << " do NOT have weights in constants map!";
            return;
        }
        pad_fc_weight_inc(n, new_in_c);
        n->infer_shape(true);
    }
}
} // namespace aipubt

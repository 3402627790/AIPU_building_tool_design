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
 * @Date: 2022-03-30 13:38:32
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 14:52:01
 */
#include "builder/target.h"
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"
#include <set>

namespace aipubt
{

static inline bool criteria(std::vector<NodePtr> m)
{
    auto target = get_target();
    if (target->isainfo().type >= ISA_ZYv6)
    {
        // DMA in X3 has boost much than X2
        return false;
    }

    auto crop = m[0];
    auto tile = m[1];
    auto params = crop->params();
    auto cropin = crop->inputs()[0];
    auto cropout = crop->outputs()[0];
    auto tilein = tile->inputs()[0];
    auto tileout = tile->outputs()[0];

    std::set<Tensor::Dtype> support_dtype = {Tensor::INT8, Tensor::UINT8, Tensor::INT16, Tensor::UINT16, Tensor::BFP16, Tensor::FP16};
    if (!support_dtype.count(cropin->type()) || !support_dtype.count(cropout->type()) ||
        !support_dtype.count(tilein->type()) || !support_dtype.count(tileout->type()))
    {
        return false;
    }
    if (Tensor::dtype_size(cropin->type()) != Tensor::dtype_size(cropout->type()) ||
        Tensor::dtype_size(cropout->type()) != Tensor::dtype_size(tileout->type()) ||
        Tensor::dtype_size(cropin->type()) != Tensor::dtype_size(tileout->type()))
    {
        return false;
    }
    if (cropin->shape().dim() != 4 ||
        cropout->shape().dim() != 4 ||
        tileout->shape().dim() != 4)
    {
        return false;
    }
    // channel after crop must be 1
    std::vector<std::vector<int>> crops = params["crops"];
    if (cropout->shape().n() - cropin->shape().n() != 0 ||
        cropout->shape().h() - cropin->shape().h() != 0 ||
        cropout->shape().w() - cropin->shape().w() != 0 ||
        crops[3][1] - crops[3][0] != 1)
    {
        return false;
    }
    // only repeat channel
    if (tileout->shape().n() - tilein->shape().n() != 0 ||
        tileout->shape().h() - tilein->shape().h() != 0 ||
        tileout->shape().w() - tilein->shape().w() != 0)
    {
        return false;
    }

    return true;
}

template <Tensor::Dtype dtype>
static TensorPtr create_1x1weights_detail(std::string &name, std::vector<std::vector<int>> &crops, int &ic, int &oc)
{
    using data_type = typename tensor_data_type<dtype>::type;

    TensorPtr weights = std::make_shared<Tensor>(name + "_weights", TensorShape(oc, 1, 1, ic), dtype);

    int size = oc * ic;
    data_type *_data = new data_type[size];
    memset((void *)_data, 0, size * Tensor::dtype_size(dtype));
    for (int o = 0; o < oc; ++o)
    {
        _data[(o * ic + crops[3][0])] = 1;
    }
    weights->set_data(_data, size * Tensor::dtype_size(dtype));

    return weights;
}

static inline TensorPtr create_1x1weights(NodePtr n, std::vector<std::vector<int>> &crops, int &ic, int &oc)
{
    auto name = n->name();
    auto precision = n->inputs()[0]->type();

    TensorPtr weights = nullptr;
    switch (precision)
    {
    case Tensor::INT8:
    case Tensor::UINT8:
        weights = create_1x1weights_detail<Tensor::INT8>(name, crops, ic, oc);
        break;
    case Tensor::INT16:
    case Tensor::UINT16:
        weights = create_1x1weights_detail<Tensor::INT16>(name, crops, ic, oc);
        break;
    case Tensor::FP16:
        weights = create_1x1weights_detail<Tensor::FP16>(name, crops, ic, oc);
        break;
    case Tensor::BFP16:
        weights = create_1x1weights_detail<Tensor::BFP16>(name, crops, ic, oc);
        break;
    default:
        ERROR() << "Connot support weight dtype: " << precision;
        break;
    }

    return weights;
}

static inline TensorPtr create_biases(NodePtr n, int &chan, int &val)
{
    auto name = n->name();
    auto precision = n->inputs()[0]->type();

    Tensor::Dtype btype = Tensor::INT32;
    if (precision == Tensor::FP16 || precision == Tensor::BFP16)
    {
        btype = Tensor::FP32;
        aipu_assert(val == 0);
    }

    TensorPtr biases = std::make_shared<Tensor>(name + "_biases", TensorShape(chan), btype);
    int size = chan * Tensor::dtype_size(btype);
    int *_data = new int[chan];
    memset(_data, 0, size);
    for (int i = 0; i < chan; ++i)
    {
        _data[i] = val;
    }
    biases->set_data(_data, size);

    return biases;
}

bool fuse_crop_tile_to_conv(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Crop}, {"1", OpType_Tile}},
        {{"0", "1"}}};
    static GraphMatcher matcher(pattern);
    matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    for (auto &&m : matched)
    {
        if (!criteria(m))
        {
            continue;
        }

        auto crop = m[0];
        auto tile = m[1];
        auto params = crop->params();
        auto cropin = crop->inputs()[0];
        auto tileout = tile->outputs()[0];
        auto ic = cropin->shape().c_total();
        auto oc = tileout->shape().c_total();
        int val = tileout->quantization().offset();
        std::vector<std::vector<int>> crops = params["crops"];

        auto weights = create_1x1weights(tile, crops, ic, oc);
        auto biases = create_biases(tile, oc, val);

        auto node = Node::create(tile->name() + "_conv", OpType_Convolution);
        node->constants()["weights"] = weights;
        node->constants()["biases"] = biases;
        node->params()["kernel_x"] = 1;
        node->params()["kernel_y"] = 1;
        node->params()["stride_x"] = 1;
        node->params()["stride_y"] = 1;
        node->params()["dilation_x"] = 1;
        node->params()["dilation_y"] = 1;
        node->params()["pad_bottom"] = 0;
        node->params()["pad_top"] = 0;
        node->params()["pad_right"] = 0;
        node->params()["pad_left"] = 0;
        if (cropin->type() != Tensor::FP16 && cropin->type() != Tensor::BFP16)
        {
            node->params()["scale_type"] = Tensor::INT8;
            node->params()["scale_value"] = 1;
            node->params()["shift_type"] = Tensor::INT8;
            node->params()["shift_value"] = 0;
        }
        node->params()["group"] = 1;
        node->params()["with_activation"] = "NONE";
        node->params()["num_output"] = oc;

        node->add_input(cropin);
        node->add_output(tileout);
        g->add_node(node);
        ADD_LOCATION_BY_NODE(node, crop, tile);
        for (auto _m : m)
        {
            g->remove_node(_m);
        }

        is_changed = true;
    }

    return is_changed;
}
}; // namespace aipubt
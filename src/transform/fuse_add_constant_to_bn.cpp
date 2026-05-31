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
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-07-05 10:40:45
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace aipubt
{

static inline bool criteria(std::vector<NodePtr> m)
{
    auto add_consts = m[0];
    auto add = m[1];
    auto params = add->params();
    auto const_output = add_consts->outputs()[0];
    auto const_shape = const_output->shape();
    auto const_weight = add_consts->constants()["weights"];
    auto in_shape = add->inputs()[0]->shape();
    if (const_output == add->inputs()[0])
    {
        in_shape = add->inputs()[1]->shape();
    }

    if (const_weight->type() != Tensor::INT8 &&
        const_weight->type() != Tensor::UINT8 &&
        const_weight->type() != Tensor::INT16 &&
        const_weight->type() != Tensor::UINT16)
    {
        return false;
    }
    if (params.count("with_activation") && params["with_activation"] != "NONE")
    {
        return false;
    }
    // add must be same shape and 4D(TODO)
    if (in_shape.dim() != 4)
    {
        return false;
    }
    // consts must be ONE constant value or per-channel
    if (const_shape.size() != 1)
    {
        bool ret = false;
        if (in_shape.dim() == const_shape.dim())
        {
            in_shape.set_layout(DataLayout_NHWC);
            const_shape.set_layout(DataLayout_NHWC);
            if (const_shape.c_total() == in_shape.c_total() &&
                const_shape.n() * const_shape.h() * const_shape.w() == 1)
            {
                ret = true;
            }
        }
        if (!ret)
            return ret;
    }

    return true;
}

static inline int get_constants_out(NodePtr consts, int idx = 0)
{
    int consts_output = 0;
    int zp = consts->outputs()[0]->quantization().offset();
    auto consts_weight = consts->constants()["weights"];
    // int8/uint8/int16/uint16
    if (!consts_weight->data<uint8_t>())
    {
        return consts_output;
    }
    switch (consts_weight->type())
    {
    case Tensor::INT8:
    {
        consts_output = *(consts_weight->data<int8_t>() + idx);
    }
    break;
    case Tensor::UINT8:
    {
        consts_output = *(consts_weight->data<uint8_t>() + idx);
    }
    break;
    case Tensor::INT16:
    {
        consts_output = *(consts_weight->data<int16_t>() + idx);
    }
    break;
    case Tensor::UINT16:
    {
        consts_output = *(consts_weight->data<uint16_t>() + idx);
    }
    break;
    default:
        break;
    }
    consts_output += zp;

    return consts_output;
}

static inline TensorPtr create_bn_weights(std::vector<NodePtr> m, int &w)
{
    auto add_consts = m[0];
    auto add = m[1];

    auto add_shape = add->outputs()[0]->shape();
    add_shape.set_layout(DataLayout_NHWC);
    int chan = add_shape.c_total();
    // add_scale1 must be connected with input
    int add_scale1 = add->params()["scale_value"].arr[1].v.i;
    if (add_consts->outputs()[0] == add->inputs()[0])
    {
        add_scale1 = add->params()["scale_value"].arr[2].v.i;
    }

    TensorPtr weights = std::make_shared<Tensor>(add_consts->name() + "_weights", TensorShape(chan), Tensor::INT16);
    int size = chan * Tensor::dtype_size(Tensor::INT16);
    int16_t *_data = new int16_t[chan];

    int64_t _w = add_scale1;
    if (_w > INT16_MAX || _w < INT16_MIN)
    {
        return nullptr;
    }
    w = _w;

    for (int i = 0; i < chan; ++i)
    {
        _data[i] = w;
    }
    weights->set_data(_data, size);

    return weights;
}

static inline TensorPtr create_bn_biases(std::vector<NodePtr> m, int &w)
{
    auto add_consts = m[0];
    auto add = m[1];

    auto add_shape = add->outputs()[0]->shape();
    add_shape.set_layout(DataLayout_NHWC);
    int chan = add_shape.c_total();
    // add_scale2 must be connected with constant output
    int add_scale2 = add->params()["scale_value"].arr[2].v.i;
    if (add_consts->outputs()[0] == add->inputs()[0])
    {
        add_scale2 = add->params()["scale_value"].arr[1].v.i;
    }

    TensorPtr biases = std::make_shared<Tensor>(add_consts->name() + "_biases", TensorShape(chan), Tensor::INT32);
    int size = chan * Tensor::dtype_size(Tensor::INT32);
    int *_data = new int[chan];
    if (add_consts->constants()["weights"]->data<uint8_t>())
    {
        // need add_input zp instead of add_const zp
        int zp = add->inputs()[0]->quantization().offset();
        if (add_consts->outputs()[0] == add->inputs()[0])
        {
            zp = add->inputs()[1]->quantization().offset();
        }

        int idx = 0;
        for (int i = 0; i < chan; ++i)
        {
            if (add_consts->outputs()[0]->shape().size() != 1)
            {
                idx = i;
            }
            int64_t b = zp * w + get_constants_out(add_consts, idx) * add_scale2;
            if (b > INT16_MAX || b < INT16_MIN)
            {
                // WARN() << "fuse_add_constant_to_bn may cause rounding error for " << add->name() << " and " << add_consts->name() << " on idx: " << idx;
                return nullptr;
            }
            // if (b > INT32_MAX || b < INT32_MIN)
            // {
            //     return nullptr;
            // }

            _data[i] = b;
        }
    }
    else
    {
        memset(_data, 0, size);
    }
    biases->set_data(_data, size);

    return biases;
}

bool fuse_add_constant_to_bn(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Constant}, {"1", OpType_Add}},
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

        auto add_consts = m[0];
        auto add = m[1];
        // y_add = ((x + add_in0_zp) * add_scale1 + (add_const + add_in1_zp) * add_scale2) * add_scale0 >> add_shift - add_out_zp
        // =====>
        // y = (ax + b) * add_scale0 >> add_shift - add_out_zp
        // a = add_scale1
        // b = add_in0_zp * add_scale1 + (add_const + add_in1_zp) * add_scale2

        auto inp = add->inputs()[0];
        if (inp == add_consts->outputs()[0])
        {
            inp = add->inputs()[1];
        }
        auto out = add->outputs()[0];

        int w = 0;
        auto weights = create_bn_weights(m, w);
        auto biases = create_bn_biases(m, w);
        if (weights == nullptr || biases == nullptr)
        {
            is_changed = false;
            break;
        }

        auto node = Node::create(add->name() + "_bn", OpType_BatchNorm);
        node->constants()["weights"] = weights;
        node->constants()["biases"] = biases;
        node->params()["scale_type"] = add->params()["scale_type"].arr[0].v.t;
        node->params()["scale_value"] = add->params()["scale_value"].arr[0].v.i;
        node->params()["shift_type"] = add->params()["shift_type"];
        node->params()["shift_value"] = add->params()["shift_value"];
        node->params()["axis"] = 3; //(TODO)

        node->add_input(inp);
        node->add_output(out);
        g->add_node(node);
        ADD_LOCATION_BY_NODE(node, add_consts, add);
        for (auto _m : m)
        {
            g->remove_node(_m);
        }

        is_changed = true;
    }

    return is_changed;
}
}; // namespace aipubt
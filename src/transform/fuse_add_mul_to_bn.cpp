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
    auto mul_consts = m[2];
    auto mul = m[3];

    auto criteria_detaial = [](NodePtr n0, NodePtr n1) -> bool
    {
        auto params = n1->params();
        auto const_output = n0->outputs()[0];
        auto const_weight = n0->constants()["weights"];
        auto in = n1->inputs()[0];
        if (const_output == n1->inputs()[0])
        {
            in = n1->inputs()[1];
        }

        if (params.count("with_activation") && params["with_activation"] != "NONE")
        {
            return false;
        }
        // consts must be ONE constant value(TODO)
        if (const_output->shape().size() != 1)
        {
            return false;
        }
        if (const_weight->type() != Tensor::INT8 &&
            const_weight->type() != Tensor::UINT8 &&
            const_weight->type() != Tensor::INT16 &&
            const_weight->type() != Tensor::UINT16)
        {
            return false;
        }
        // add/mul must be same shape and 4D(TODO)
        if (in->shape().dim() != 4)
        {
            return false;
        }
        return true;
    };

    if (!criteria_detaial(mul_consts, mul) || !criteria_detaial(add_consts, add))
    {
        return false;
    }
    return true;
}

static inline int get_constants_out(NodePtr consts)
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
        consts_output = *consts_weight->data<int8_t>();
    }
    break;
    case Tensor::UINT8:
    {
        consts_output = *consts_weight->data<uint8_t>();
    }
    break;
    case Tensor::INT16:
    {
        consts_output = *consts_weight->data<int16_t>();
    }
    break;
    case Tensor::UINT16:
    {
        consts_output = *consts_weight->data<uint16_t>();
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
    auto mul_consts = m[2];

    auto add_shape = add->outputs()[0]->shape();
    add_shape.set_layout(DataLayout_NHWC);
    int chan = add_shape.c_total();
    int add_shift = add->params()["shift_value"];
    int add_scale0 = add->params()["scale_value"].arr[0].v.i;
    // add_scale1 must be connected with input
    int add_scale1 = add->params()["scale_value"].arr[1].v.i;
    if (add_consts->outputs()[0] == add->inputs()[0])
    {
        add_scale1 = add->params()["scale_value"].arr[2].v.i;
    }

    TensorPtr weights = std::make_shared<Tensor>(add_consts->name() + "_weights", TensorShape(chan), Tensor::INT16);
    int size = chan * Tensor::dtype_size(Tensor::INT16);
    int16_t *_data = new int16_t[chan];

    // overflow
    int64_t _w = 1;
    _w *= get_constants_out(mul_consts);
    _w *= add_scale1;
    _w *= add_scale0;
    // _w >>= add_shift;
    _w = std::round((double)_w / ((int64_t)1 << add_shift));
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
    auto mul_consts = m[2];

    auto add_shape = add->outputs()[0]->shape();
    add_shape.set_layout(DataLayout_NHWC);
    int chan = add_shape.c_total();
    int add_shift = add->params()["shift_value"];
    int add_scale0 = add->params()["scale_value"].arr[0].v.i;
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

        // overflow
        int64_t b = 1;
        b *= get_constants_out(mul_consts);
        b *= get_constants_out(add_consts);
        b *= add_scale2;
        b *= add_scale0;
        // b >>= add_shift;
        b = std::round((double)b / ((int64_t)1 << add_shift));
        b += (w * zp);
        if (b > INT32_MAX || b < INT32_MIN)
        {
            return nullptr;
        }

        for (int i = 0; i < chan; ++i)
        {
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

bool fuse_add_mul_to_bn(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Constant}, {"1", OpType_Add}, {"2", OpType_Constant}, {"3", OpType_Mul}},
        {{"0", "1"}, {"1", "3"}, {"2", "3"}}};
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
        auto mul_consts = m[2];
        auto mul = m[3];
        // y_add = ((x + add_in0_zp) * add_scale1 + (add_const + add_in1_zp) * add_scale2) * add_scale0 >> add_shift - add_out_zp
        // y_mul = (y_add + add_out_zp) * (mul_const + mul_in1_zp) * mul_scale >> mul_shift - mul_out_zp
        // =====>
        // y = (ax + b) * mul_scale >> mul_shift - mul_out_zp
        // a =  add_scale1 * add_scale0 >> add_shift * (mul_const + mul_in1_zp)
        // b = add_in0_zp * a + (add_const + add_in1_zp) * add_scale2 * add_scale0 >> add_shift * (mul_const + mul_in1_zp)

        auto inp = add->inputs()[0];
        if (inp == add_consts->outputs()[0])
        {
            inp = add->inputs()[1];
        }
        auto out = mul->outputs()[0];

        int w = 0;
        auto weights = create_bn_weights(m, w);
        auto biases = create_bn_biases(m, w);
        if (weights == nullptr || biases == nullptr)
        {
            is_changed = false;
            break;
        }

        auto node = Node::create(mul->name() + "_bn", OpType_BatchNorm);
        node->constants()["weights"] = weights;
        node->constants()["biases"] = biases;
        node->params()["scale_type"] = mul->params()["scale_type"];
        node->params()["scale_value"] = mul->params()["scale_value"];
        node->params()["shift_type"] = mul->params()["shift_type"];
        node->params()["shift_value"] = mul->params()["shift_value"];
        node->params()["axis"] = 3; //(TODO)

        node->add_input(inp);
        node->add_output(out);
        g->add_node(node);
        ADD_LOCATION_BY_NODE(node, add_consts, add, mul_consts, mul);
        for (auto _m : m)
        {
            g->remove_node(_m);
        }

        is_changed = true;
    }

    return is_changed;
}
}; // namespace aipubt
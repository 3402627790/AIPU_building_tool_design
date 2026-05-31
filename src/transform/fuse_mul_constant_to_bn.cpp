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
    auto mul_consts = m[0];
    auto mul = m[1];
    auto params = mul->params();
    auto const_output = mul_consts->outputs()[0];
    auto const_weight = mul_consts->constants()["weights"];
    auto in = mul->inputs()[0];
    if (const_output->is_io_tensor() || mul->inputs().size() == 0)
    {
        return false;
    }
    if (const_output == mul->inputs()[0])
    {
        in = mul->inputs()[1];
    }
    auto in_shape = in->shape();
    auto const_shape = const_output->shape();

    if (const_weight->type() != Tensor::INT8 &&
        const_weight->type() != Tensor::UINT8 &&
        const_weight->type() != Tensor::INT16 &&
        const_weight->type() != Tensor::UINT16)
    {
        return false;
    }
    // mul must be without activation
    if (params.count("with_activation") && "NONE" != params["with_activation"])
    {
        return false;
    }
    // mul must be same shape and 4D(TODO)
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

static inline TensorPtr create_bn_weights(std::vector<NodePtr> m)
{
    auto mul_consts = m[0];

    auto mul_shape = m[1]->outputs()[0]->shape();
    mul_shape.set_layout(DataLayout_NHWC);
    int chan = mul_shape.c_total();

    TensorPtr weights = std::make_shared<Tensor>(mul_consts->name() + "_weights", TensorShape(chan), Tensor::INT16);
    int size = chan * Tensor::dtype_size(Tensor::INT16);
    int16_t *_data = new int16_t[chan];
    if (mul_consts->constants()["weights"]->data<uint8_t>())
    {
        int idx = 0;
        for (int i = 0; i < chan; ++i)
        {
            if (mul_consts->outputs()[0]->shape().size() != 1)
            {
                idx = i;
            }
            int64_t w = get_constants_out(mul_consts, idx);
            if (w > INT16_MAX || w < INT16_MIN)
            {
                return nullptr;
            }

            _data[i] = w;
        }
    }
    else
    {
        memset(_data, 0, size);
    }
    weights->set_data(_data, size);

    return weights;
}

static inline TensorPtr create_bn_biases(std::vector<NodePtr> m)
{
    auto mul_consts = m[0];
    auto mul = m[1];

    auto mul_shape = mul->outputs()[0]->shape();
    mul_shape.set_layout(DataLayout_NHWC);
    int chan = mul_shape.c_total();

    TensorPtr biases = std::make_shared<Tensor>(mul_consts->name() + "_biases", TensorShape(chan), Tensor::INT32);
    int size = chan * Tensor::dtype_size(Tensor::INT32);
    int *_data = new int[chan];
    if (mul_consts->constants()["weights"]->data<uint8_t>())
    {
        // need mul_input zp instead of mul_const zp
        int zp = mul->inputs()[0]->quantization().offset();
        if (mul_consts->outputs()[0] == mul->inputs()[0])
        {
            zp = mul->inputs()[1]->quantization().offset();
        }

        int idx = 0;
        for (int i = 0; i < chan; ++i)
        {
            if (mul_consts->outputs()[0]->shape().size() != 1)
            {
                idx = i;
            }
            int64_t b = get_constants_out(mul_consts, idx) * zp;
            if (b > INT16_MAX || b < INT16_MIN)
            {
                // WARN() << "fuse_mul_constant_to_bn may cause rounding error for " << mul->name() << " and " << mul_consts->name() << " on idx: " << idx;
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

bool fuse_mul_constant_to_bn(GraphPtr g)
{
    static GraphMatcherPattern pattern = {
        {{"0", OpType_Constant}, {"1", OpType_Mul}},
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

        auto mul_consts = m[0];
        auto mul = m[1];
        // y_mul = (x + mul_in0_zp) * (mul_const + mul_in1_zp) * mul_scale >> mul_shift - mul_out_zp
        // =====>
        // y = (ax + b) * mul_scale >> mul_shift - mul_out_zp
        // a = (mul_const + mul_in1_zp)
        // b = (mul_const + mul_in1_zp) * mul_in0_zp

        auto inp = mul->inputs()[0];
        if (inp == mul_consts->outputs()[0])
        {
            inp = mul->inputs()[1];
        }
        auto out = mul->outputs()[0];

        auto weights = create_bn_weights(m);
        auto biases = create_bn_biases(m);
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
        ADD_LOCATION_BY_NODE(node, mul_consts, mul);
        for (auto _m : m)
        {
            g->remove_node(_m);
        }

        is_changed = true;
    }

    return is_changed;
}
}; // namespace aipubt

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
 * @Author: Aaron Gao
 * @Date: 2022-03-22 10:25:43
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-05-16 10:28:17
 */
#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/utils/quantize_utils.h"
#include <algorithm>
#include <initializer_list>
#include <set>

// Attention: If different quantization params are between first layer and second activation layer,
//            when existing overboundray data in the first layer, it may not overboundray when fuse
//            the layers. Need more test...
// Relu6 layerlib suggests converting to Clip, so we will not fuse Relu6.
namespace aipubt
{

template <typename T>
int32_t get_data(TensorPtr t, int32_t index)
{
    if (t->data<int32_t>() == nullptr)
    {
        return -1;
    }
    int32_t val = t->data<T>()[index];
    return val;
}

template <typename T>
void set_data(TensorPtr t, int32_t index, int32_t val)
{
    if (t->data<int32_t>() == nullptr)
    {
        return;
    }
    T new_val = val;
    t->data<T>()[index] = new_val;
}

static std::map<Tensor::Dtype, std::function<int32_t(TensorPtr t, int32_t index)>> get_data_table = {
    {Tensor::UINT8, get_data<uint8_t>},
    {Tensor::INT8, get_data<int8_t>},
    {Tensor::UINT16, get_data<uint16_t>},
    {Tensor::INT16, get_data<int16_t>},
    {Tensor::INT16, get_data<int16_t>},
    {Tensor::INT32, get_data<int32_t>},
};

static std::map<Tensor::Dtype, std::function<void(TensorPtr t, int32_t index, int32_t val)>> set_data_table = {
    {Tensor::UINT8, set_data<uint8_t>},
    {Tensor::INT8, set_data<int8_t>},
    {Tensor::UINT16, set_data<uint16_t>},
    {Tensor::INT16, set_data<int16_t>},
    {Tensor::INT16, set_data<int16_t>},
    {Tensor::INT32, set_data<int32_t>},
};

// "scale_value" and "shift_value":
// - Convs per-tensor: int/int
// - Convs per-channel: constants/constants
// - Eltwise: mul: int/int, add/...: vector/int
static bool params_exist(NodePtr n,
                         const std::initializer_list<std::string> &l)
{
    for (auto &it : l)
    {
        if (n->params().count(it) == 0)
        {
            return false;
        }
    }
    return true;
}

static bool constants_exist(NodePtr n,
                            const std::initializer_list<std::string> &l)
{
    for (auto &it : l)
    {
        if (n->constants().count(it) == 0)
        {
            return false;
        }
    }
    return true;
}

inline static void get_multiplier_bits(NodePtr n, ComputeTensor ct, int32_t &mbits)
{

    if (n->attrs().count("multiplier_bits"))
    {
        mbits = n->get_attr("multiplier_bits");
    }
    else
    {
        mbits = ct.astype(Tensor::FP32).log2().floor().minimum(ComputeTensor((int32_t)15));
    }
}

static bool fuse_quant_params(NodePtr n1, NodePtr n2)
{
    if (!params_exist(n2, {"scale_value", "shift_value"}))
    {
        return false;
    }

    bool ret = true;
    float scale = 1.0;
    int8_t shift = 0;
    int32_t qbits = Tensor::dtype_bits(n2->outputs()[0]->type());
    // ATTENTION: QAT model, parser->qtlib->IR->BT will not inherit multiplier_bits
    if (n1->attrs().count("multiplier_bits"))
    {
        qbits = n1->get_attr("multiplier_bits");
    }
    float total_scale_2 = (float)n2->params()["scale_value"] / pow(2, (int)n2->params()["shift_value"]);
    if (params_exist(n1, {"scale_value", "shift_value"}))
    {
        uint32_t scale_size = n1->params()["scale_value"].arr.size();
        uint32_t shift_size = n1->params()["shift_value"].arr.size();
        if (scale_size == 0 && shift_size == 0)
        {
            float scale_value = static_cast<float>(n1->params()["scale_value"]);
            ComputeTensor ct = ComputeTensor(scale_value);
            get_multiplier_bits(n1, ct, qbits);
            float total_scale_1 = scale_value / pow(2, (int)n1->params()["shift_value"]);
            float total_scale = total_scale_1 * total_scale_2;
            quant_utils::get_scale_approximation_params(total_scale, scale, shift, qbits);
            n1->params()["scale_value"] = static_cast<int32_t>(scale);
            n1->params()["shift_value"] = static_cast<int8_t>(shift);
            n1->outputs()[0]->quantization() = n2->outputs()[0]->quantization();
        }
        else if (scale_size != 0 && scale_size == shift_size)
        {
            for (uint32_t i = 0; i < scale_size; ++i)
            {
                float total_scale_1 = (float)n1->params()["scale_value"].arr[i] / pow(2, (int)n1->params()["shift_value"].arr[i]);
                float total_scale = total_scale_1 * total_scale_2;
                quant_utils::get_scale_approximation_params(total_scale, scale, shift, qbits);
                n1->params()["scale_value"].arr[i] = static_cast<int32_t>(scale);
                n1->params()["shift_value"].arr[i] = static_cast<int8_t>(shift);
            }
            n1->outputs()[0]->quantization() = n2->outputs()[0]->quantization();
        }
        else if (scale_size != 0 && shift_size == 0)
        {
            int32_t real_shift = (int)n1->params()["shift_value"] - qbits;
            float total_scale_1 = (float)n1->params()["scale_value"].arr[0] / pow(2, real_shift);
            float total_scale = total_scale_1 * total_scale_2;
            quant_utils::get_scale_approximation_params(total_scale, scale, shift, qbits);
            n1->params()["scale_value"].arr[0] = static_cast<int32_t>(scale);
            n1->params()["shift_value"] = static_cast<int8_t>(shift) + qbits;
            n1->outputs()[0]->quantization() = n2->outputs()[0]->quantization();
        }
        else
        {
            ret = false;
        }
    }
    else if (constants_exist(n1, {"scale", "shift"}))
    {
        auto scale_t = n1->constants()["scale"];
        auto shift_t = n1->constants()["shift"];
        uint32_t eles = scale_t->mem_size() / Tensor::dtype_size(scale_t->type());
        Tensor::Dtype scale_d = scale_t->type();
        Tensor::Dtype shift_d = shift_t->type();
        ComputeTensor ct = ComputeTensor(scale_t);
        get_multiplier_bits(n1, ct, qbits);
        for (uint32_t i = 0; i < eles; ++i)
        {
            int32_t scale_value = get_data_table[scale_d](scale_t, i);
            int32_t shift_value = get_data_table[shift_d](shift_t, i);
            float total_scale_1 = (float)scale_value / pow(2, shift_value);
            float total_scale = total_scale_1 * total_scale_2;
            quant_utils::get_scale_approximation_params(total_scale, scale, shift, qbits);
            set_data_table[scale_d](scale_t, i, static_cast<int32_t>(scale));
            set_data_table[shift_d](shift_t, i, static_cast<int32_t>(shift));
        }
        n1->outputs()[0]->quantization() = n2->outputs()[0]->quantization();
    }
    else
    {
        ret = false;
    }
    return ret;
}

static bool relu(NodePtr n1, NodePtr n2)
{
    n1->params()["with_activation"] = Constants::act_relu;
    if (fuse_quant_params(n1, n2))
    {
        auto output_dtype = Tensor::dtype_to_unsigned(n2->outputs()[0]->type());
        n1->outputs()[0]->set_type(output_dtype);
        n1->outputs()[0]->quantization().set_aipu_dtype(output_dtype);
    }
    return true;
}

static bool clip(NodePtr n1, NodePtr n2)
{
    n1->params()["with_activation"] = Constants::act_clip;
    n1->params()["clip_max"] = n2->params()["clip_max"];
    n1->params()["clip_min"] = n2->params()["clip_min"];
    fuse_quant_params(n1, n2);
    return true;
}

static bool leakyrelu(NodePtr n1, NodePtr n2)
{
    n1->params()["with_activation"] = Constants::act_leakyrelu;
    n1->params()["negative_slope_value"] = n2->params()["negative_slope_value"];
    if (fuse_quant_params(n1, n2))
    {
        n1->params()["negative_slope_type"] = n2->params()["negative_slope_type"];
    }
    return true;
}

static std::vector<int> eliminate_prelu_shape(const TensorShape &shape)
{
    // prelu has some shape format:
    // 1. [c]
    // 2. [1,1,c] or [1,1,1,c]
    // 3. [h,w,c]
    // 4. [n,h,w,c]
    std::vector<int> new_shape = shape;
    for (auto intr = shape.begin(); intr != shape.end(); ++intr)
    {
        if (*intr != 1)
        {
            new_shape.assign(intr, shape.end());
            break;
        }
    }
    return new_shape;
}

static bool prelu(NodePtr n1, NodePtr n2)
{
    auto negative_slop = n2->constants()["negative_slope"];
    if (negative_slop->shape().dim() > 1)
    {
        // for 2 or 3 or 4
        auto new_shape = eliminate_prelu_shape(negative_slop->shape());
        auto shrink_axis = negative_slop->shape().dim() - new_shape.size();
        auto t = negative_slop; // for 2
        if (new_shape.size() > 1)
        {
            // for 3 or 4
            do
            {
                t = shrink_tensor_data(t, shrink_axis);
            } while (t && t->shape().dim() != 1 && t->shape()[-1] != t->shape().size()); // for [1, xxx]; shrink_axis isn't 0
        }

        if (t)
        {
            negative_slop = t;
        }
        else
        {
            return false;
        }
    }
    n1->params()["with_activation"] = Constants::act_prelu;
    negative_slop->shape().reshape(eliminate_prelu_shape(negative_slop->shape()));
    n1->constants()["negative_slope"] = negative_slop;
    if (fuse_quant_params(n1, n2))
    {
        n1->params()["negative_slope_shift"] = n2->params()["negative_slope_shift"];
        if (n1->params().count("negative_slope_type"))
        {
            n1->params()["negative_slope_type"] = n2->params()["negative_slope_type"];
        }
    }
    return true;
}

static std::map<std::string, std::function<bool(NodePtr, NodePtr)>> activation_table = {
    {"RELU", relu},
    {"CLIP", clip},
    {"LEAKYRELU", leakyrelu},
    {"PRELU", prelu},
};

static inline bool creteria(NodePtr n1, NodePtr n2)
{
    std::string act = n1->params()["with_activation"];
    transform(act.begin(), act.end(), act.begin(), ::toupper);
    if (act != "NONE" || n1->outputs()[0]->is_io_tensor())
    {
        return false;
    }

    std::string method = n2->params()["method"];
    transform(method.begin(), method.end(), method.begin(), ::toupper);
    if (activation_table.count(method) == 0)
    {
        return false;
    }
    if (method == "PRELU" && n1->type() == OpType_Eltwise)
    {
        return false;
    }
    bool n1_quant = params_exist(n1, {"scale_value", "shift_value"}) || constants_exist(n1, {"scale", "shift"});
    bool n2_quant = params_exist(n2, {"scale_value", "shift_value"});
    if (n1_quant ^ n2_quant)
    {
        return false;
        // Note: activation will not change the quantization params.
        // constexpr float EPSILON = 0.004f; // 0.004 > 255/256
        // int32_t scale_value = n2->params()["scale_value"];
        // int32_t shift_value = n2->params()["shift_value"];
        // float total_scale = (float)scale_value / (float)pow(2, shift_value);
        // if (fabs(total_scale - 1.f) > EPSILON)
        // {
        //     return false;
        // }
    }
    if (method == "PRELU" && n1->outputs()[0]->type() != n2->outputs()[0]->type())
    {
        return false;
    }
    return true;
}

// static bool is_float_node(NodePtr node)
// {
//     for (const auto &tensors : {node->inputs(), node->outputs()})
//     {
//         for (auto tensor : tensors)
//         {
//             if (Tensor::is_float(tensor->type()))
//             {
//                 return true;
//             }
//         }
//     }
//     return false;
// }

bool fuse_activation(GraphPtr g)
{
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Convolution}, {"1", OpType_Activation}},
         {{"0", "1"}}},
        {{{"0", OpType_Convolution3D}, {"1", OpType_Activation}},
         {{"0", "1"}}},
        {{{"0", OpType_DepthwiseConv}, {"1", OpType_Activation}},
         {{"0", "1"}}},
        {{{"0", OpType_ConvTranspose}, {"1", OpType_Activation}},
         {{"0", "1"}}},
        {{{"0", OpType_ConvTranspose3D}, {"1", OpType_Activation}},
         {{"0", "1"}}},
        {{{"0", OpType_Eltwise}, {"1", OpType_Activation}},
         {{"0", "1"}}},
        {{{"0", OpType_FullyConnected}, {"1", OpType_Activation}},
         {{"0", "1"}}},
    };
    bool is_changed = false;
    for (auto &pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                // bool is_float_ir = false;
                // for (auto node : m)
                // {
                //     is_float_ir |= is_float_node(node);
                // }
                // if (is_float_ir)
                // {
                //     continue;
                // }

                if (!creteria(m[0], m[1]))
                {
                    continue;
                }

                DEBUG() << "Fuse " << m[0]->name() << " and " << m[1]->name();

                std::string method = m[1]->params()["method"];
                transform(method.begin(), method.end(), method.begin(), ::toupper);
                if (activation_table.count(method) == 0)
                {
                    continue;
                }
                if (!activation_table[method](m[0], m[1]))
                {
                    continue;
                }

                auto out = m[1]->outputs()[0];
                m[0]->remove_output(0);
                m[0]->add_output(out);
                g->remove_node(m[1]);

                is_changed = true;
                is_merged = true;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}

}; // namespace aipubt

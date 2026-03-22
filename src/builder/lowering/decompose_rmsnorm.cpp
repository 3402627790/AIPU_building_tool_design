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
 * =====================================================================================
 *
 *       Filename:  decompose_rmsnorm.cpp
 *
 *    Description:  decompose rmsnorm to multiple OPs
 *
 *        Version:  1.0
 *        Created:  06/14/23 11:19:00
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Yi Gao
 *   Organization:  Arm China
 *
 * =====================================================================================
 */

#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/core.h"
#include "core/compute_tensor.h"
#include "core/ops/ops.h"
#include "core/internal_optype.h"
#include <vector>
#include <cmath>

namespace aipubt
{

static void opt_hw(int *ret, int hw)
{
    auto target = get_target();
    int max_div = 2048;
    int pe_w = 8;
    int pe_h = 4;
    // select best size according to different ISA config
    // base PE size w * h:
    // _1104 (mac >= 2048): 8 * 2
    // _10xx (max <  2048): 8 * 1
    if (target->isainfo().type == ISA_ZYv2)
    {
        max_div = 1920;
        uint16_t mac_num = target->ff_mac_num();
        if (mac_num >= 2048)
        {
            pe_w = 8;
            pe_h = 2;
        }
        else
        {
            pe_w = 8;
            pe_h = 1;
        }
    }
    else if (target->isainfo().type >= ISA_ZYv3)
    {
        max_div = 4096;
        uint16_t mac_num = target->ff_mac_num();
        if (mac_num >= 4096)
        {
            pe_w = 8;
            pe_h = 2;
        }
        else
        {
            pe_w = 8;
            pe_h = 1;
        }
    }

    int s_h = 0, s_w = 0;
    ret[0] = 0;
    ret[1] = hw;
    ret[2] = 1;
    int sqrt_n = int(std::ceil(std::sqrt(hw)));
    auto get_ret = [&]() -> void {
        if (hw % s_h == 0)
        {
            s_w = hw / s_h;
            if (s_w <= max_div && s_h <= max_div)
            {
                ret[0] = 1;
                ret[1] = s_h;
                ret[2] = s_w;
                return;
            }
        }
        return;
    };

    if (hw % (pe_w * pe_h) == 0 && sqrt_n >= pe_h && sqrt_n >= pe_w)
    {
        // from big to small
        auto new_sqrt_n = sqrt_n / pe_h * pe_h;
        for (s_h = new_sqrt_n; s_h < max_div + 1 && s_h > 0; s_h = s_h - pe_h)
        {
            get_ret();
            if (ret[0])
            {
                return;
            }
        }
    }
    else
    {
        for (s_h = sqrt_n; s_h > 0; s_h--)
        {
            get_ret();
            if (ret[0])
            {
                return;
            }
        }
    }
}

void decompose_rmsnorm(GraphPtr graph)
{
    constexpr auto c_alignment = 8;
    auto arch = aipubt::get_target()->isainfo().type;
    if (arch < aipubt::ISA_ZYv3)
    {
        return;
    }

    auto copy_param = [](BaseNodePtr node, BaseNodePtr origin_node, const std::string& old_param_name, const std::string& new_param_name)
    {
        auto it = origin_node->params().find(old_param_name);
        if (it != origin_node->params().end())
        {
            node->params()[new_param_name] = it->second;
        }
    };

    auto criterion = [](NodePtr node) -> bool {
        if (node->type() != OpType_RMSNorm)
        {
            return false;
        }
        auto arch = aipubt::get_target()->isainfo().type;
        if (Tensor::is_float(node->inputs()[0]->type()) && (arch < aipubt::ISA_ZYv6))
        {
            return false;
        }

        auto shape = node->inputs()[0]->shape();
        std::vector<int> axises = node->params()["axis"];
        if (axises.size() != 1)
        {
            return false;
        }
        auto axis = axises[0];
        if ((axis != shape.dim() - 1) && (axis != -1))
        {
            return false;
        }

        if ((shape.dim() != 4) && (shape.dim() != 3) && (shape.dim() != 2))
        {
            return false;
        }

        return true;
    };

    auto nodes = graph->nodes();
    for (auto node : nodes)
    {
        if (!criterion(node))
        {
            continue;
        }

        auto input = node->inputs()[0];
        auto input_shape = input->shape();
        auto output = node->outputs()[0];
        auto origin_output_shape = output->shape();
        int hw = 1;
        for (auto i = 0; i < input_shape.dim() - 1; i++)
        {
            hw *= input_shape[i];
        }
        int opt_hw_ret[3] = {0, 0, 0};
        opt_hw(opt_hw_ret, hw);
        int rms_h = opt_hw_ret[1];
        int rms_w = opt_hw_ret[2];
        TensorShape new_input_shape{1, rms_h, rms_w, input_shape[input_shape.dim() - 1]};
        new_input_shape.set_layout(DataLayout_NHWC);
        auto input_reshape_output = ops::reshape(input, new_input_shape);
        auto input_reshape_node = input_reshape_output->node();
        graph->add_node(input_reshape_node);
        ADD_LOCATION_BY_NODE(input_reshape_node, node);

        if (arch < aipubt::ISA_ZYv6)
        {
            auto new_input = input_reshape_output;
            bool need_pad_c = (new_input->shape().c() % c_alignment != 0);
            if (need_pad_c)
            {
                auto unpad_shape = new_input->shape();
                TensorShape pad_shape(unpad_shape);
                pad_shape.c() = static_cast<int>(std::ceil(unpad_shape.c() * 1.0 / c_alignment)) * c_alignment;
                std::vector<std::pair<uint32_t, uint32_t>> pads;
                for (int i = 0; i < new_input->shape().dim() - 1; ++i)
                {
                    pads.push_back({0, 0});
                }
                pads.push_back({0, pad_shape.c() - unpad_shape.c()});
                auto pad_tensor = ops::pad(new_input, pads, ops::PadMode::Constant, 0 - new_input->quantization().offset());
                new_input = pad_tensor;
                auto pad_node = pad_tensor->node();
                graph->add_node(pad_node);
                ADD_LOCATION_BY_NODE(pad_node, node);
            }

            auto rms_rsqrt_node_name = graph->get_valid_node_name(node->name() + "/rms_rsqrt");
            auto rms_rsqrt_node = Node::create(rms_rsqrt_node_name, static_cast<aipubt::OpType>(internal_ops::OpType_RMSRsqrt));
            rms_rsqrt_node->constants()["lut"] = node->constants()["lut"];
            bool has_ngamma_quant_params = (node->params().find("ngamma_scale_type") != node->params().end());
            if (has_ngamma_quant_params)
            {
                copy_param(rms_rsqrt_node, node, "ngamma_scale_type", "ngamma_scale_type");
                copy_param(rms_rsqrt_node, node, "ngamma_scale_value", "ngamma_scale_value");
                copy_param(rms_rsqrt_node, node, "ngamma_shift_type", "ngamma_shift_type");
                copy_param(rms_rsqrt_node, node, "ngamma_shift_value", "ngamma_shift_value");
            }
            if (!has_ngamma_quant_params)
            {
                copy_param(rms_rsqrt_node, node, "norm_scale_type", "norm_scale_type");
                copy_param(rms_rsqrt_node, node, "norm_scale_value", "norm_scale_value");
                copy_param(rms_rsqrt_node, node, "norm_shift_type", "norm_shift_type");
                copy_param(rms_rsqrt_node, node, "norm_shift_value", "norm_shift_value");
            }
            std::vector<int> squaresum_axises{3};
            rms_rsqrt_node->params()["axis"] = squaresum_axises;
            rms_rsqrt_node->add_input(new_input);
            TensorPtr rms_rsqrt_output = std::make_shared<Tensor>(rms_rsqrt_node->name(), new_input->shape(), input->type());
            auto squaresum_out_mem_size = rms_rsqrt_output->mem_size();
            uint8_t* squaresum_data = new uint8_t[squaresum_out_mem_size];
            rms_rsqrt_output->set_data(squaresum_data, squaresum_out_mem_size);
            rms_rsqrt_node->add_output(rms_rsqrt_output);
            auto it = node->params().find("ngamma_zp_value");
            if (it != node->params().end())
            {
                rms_rsqrt_output->quantization().set_offset(it->second);
            }
            graph->add_node(rms_rsqrt_node);
            ADD_LOCATION_BY_NODE(rms_rsqrt_node, node);

            if (need_pad_c)
            {
                std::vector<std::pair<int32_t, int32_t>> crops;
                auto crop_shape = new_input_shape;
                for (int i = 0; i < crop_shape.dim(); ++i)
                {
                    crops.push_back({0, crop_shape[i]});
                }
                auto crop_tensor = ops::crop(rms_rsqrt_output, crops);
                rms_rsqrt_output = crop_tensor;
                auto crop_node = crop_tensor->node();
                graph->add_node(crop_node);
                ADD_LOCATION_BY_NODE(crop_node, node);
            }

            auto bn_node_name = graph->get_valid_node_name(node->name() + "/bn");
            auto bn_node = Node::create(bn_node_name, OpType_BatchNorm);
            copy_param(bn_node, node, "scale_type", "scale_type");
            copy_param(bn_node, node, "scale_value", "scale_value");
            copy_param(bn_node, node, "shift_type", "shift_type");
            copy_param(bn_node, node, "shift_value", "shift_value");
            bn_node->params()["axis"] = 3;
            auto origin_node_consts = node->constants();
            bn_node->constants()["weights"] = origin_node_consts["weights"];

            auto weight = origin_node_consts["weights"];
            TensorShape bias_shape(weight->shape()[0]);
            // ComputeTensor::zeros constructs tensor with default layout and strides
            auto bn_bias = ComputeTensor::zeros(bias_shape, Tensor::INT32);
            bn_bias->make_memory_continuous();
            bn_node->constants()["biases"] = bn_bias;

            if (origin_node_consts["weights"]->type() == Tensor::INT8)
            {
                bn_node->params()["use_8bit_weight"] = 1;
            }
            if (origin_node_consts.find("scale") != origin_node_consts.end())
            {
                bn_node->constants()["scale"] = origin_node_consts["scale"];
            }
            if (origin_node_consts.find("shift") != origin_node_consts.end())
            {
                bn_node->constants()["shift"] = origin_node_consts["shift"];
            }
            bn_node->add_input(rms_rsqrt_output);
            TensorPtr bn_output = std::make_shared<Tensor>(bn_node->name(), rms_rsqrt_output->shape(), input->type());
            auto bn_out_mem_size = bn_output->mem_size();
            uint8_t* bn_data = new uint8_t[bn_out_mem_size];
            bn_output->set_data(bn_data, bn_out_mem_size);
            bn_node->add_output(bn_output);
            graph->add_node(bn_node);
            ADD_LOCATION_BY_NODE(bn_node, node);

            auto output_reshape_node = ops::reshape(bn_output, node->outputs()[0]->shape())->node();
            output_reshape_node->remove_output(0);
            output_reshape_node->add_output(node->outputs()[0], 0);
            graph->add_node(output_reshape_node);
            ADD_LOCATION_BY_NODE(output_reshape_node, node);

            graph->remove_node(node);
        }
        else
        {
            TensorPtr out_tensor = node->outputs()[0];
            auto tensor_name = graph->get_valid_tensor_name(node->name() + "out_reshape");
            TensorPtr output_tmp = std::make_shared<Tensor>(tensor_name, new_input_shape, out_tensor->type());
            node->add_input(input_reshape_output, node->remove_input(input));
            node->add_output(output_tmp, node->remove_output(out_tensor));

            node->params()["axis"] = std::vector<int>{-1};

            auto output_reshape_out = ops::reshape(output_tmp, origin_output_shape);
            auto output_reshape_node = output_reshape_out->node();
            output_reshape_node->remove_output(0);
            output_reshape_node->add_output(out_tensor, 0);
            graph->add_node(output_reshape_node);
            ADD_LOCATION_BY_NODE(output_reshape_node, node);
        }
    }
}

} // namespace aipubt

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
 * @Author: Lang Hu
 * @Date: 2024-05-09 22:11:39
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-05-09 22:11:39
 */
#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ops/ops.h"

namespace aipubt
{

bool check_plugin_option(const std::string &opt)
{
    auto options = BuilderConfig::get<std::map<std::string, std::string>>("op_plugin_options");
    return options.count(opt) > 0 && std::stoi(options[opt]) != 0;
}

static int get_batch_alignment(NodePtr n)
{
    return 16;
}

bool is_btconv_supported(NodePtr node)
{
    if (node->type() != OpType_FullyConnected &&
        node->type() != OpType_Convolution)
    {
        return false;
    }

    if (get_target()->isainfo().type < ISA_ZYv6)
    {
        return false;
    }

    if (!check_plugin_option("ENABLE_BTCONV_PLUGIN"))
    {
        auto is_fp8_type = [](TensorPtr t) {
            return t->type() == Tensor::FP8E4M3FN || t->type() == Tensor::FP8E5M2;
        };
        bool is_fp8_conv = false;
        if (is_fp8_type(node->inputs()[0]) || is_fp8_type(node->outputs()[0]))
        {
            is_fp8_conv = true;
        }
        if (!is_fp8_conv && !node->is_dynamic())
        {
            return false;
        }
    }

    auto input = node->inputs()[0];
    auto output = node->outputs()[0];
    auto weight = node->constants()["weights"];
    auto bias = node->constants()["biases"];

    if ((input->type() != Tensor::FP16 && input->type() != Tensor::FP8E4M3FN &&
         input->type() != Tensor::FP8E5M2) ||
        (output->type() != Tensor::FP16 &&
         output->type() != Tensor::FP8E4M3FN && output->type() != Tensor::FP8E5M2) ||
        (weight->type() != Tensor::ALIGNED_INT4 && weight->type() != Tensor::FP16 &&
         weight->type() != Tensor::FP8E4M3FN && weight->type() != Tensor::FP8E5M2 &&
         weight->type() != Tensor::FP4E2M1FN) ||
        bias->type() != Tensor::FP32)
    {
        return false;
    }

    int aiff_num = get_target()->aiff_num(), bs, oc;
    if (node->type() == OpType_FullyConnected)
    {
        bs = 1;
        oc = output->shape().h();
    }
    else
    {
        bs = output->shape().n();
        oc = output->shape().c();
    }
    if (bs != 1 || oc < aiff_num * 32)
    {
        return false;
    }

    if (node->type() == OpType_Convolution)
    {
        if (!node->params().count("kernel_x") || !node->params().count("kernel_y") ||
            !node->params().count("stride_x") || !node->params().count("stride_y"))
        {
            return false;
        }

        if ((int)node->params()["kernel_x"] != 1 || (int)node->params()["kernel_y"] != 1 ||
            (int)node->params()["stride_x"] != 1 || (int)node->params()["stride_y"] != 1 ||
            (int)node->params()["pad_top"] != 0 || (int)node->params()["pad_bottom"] != 0 ||
            (int)node->params()["pad_left"] != 0 || (int)node->params()["pad_right"] != 0 ||
            (int)node->params()["dilation_x"] != 1 || (int)node->params()["dilation_y"] != 1 ||
            (int)node->params()["group"] != 1)
        {
            return false;
        }
    }

    if ((std::string)node->params()["with_activation"] != "NONE")
    {
        return false;
    }

    if (aiff_num > 4 || aiff_num > (int)get_target()->tec_num())
    {
        return false;
    }

    return true;
}

static bool criteria(NodePtr n, bool &pad_in, bool &crop_out)
{
    if (n->type() != OpType_FullyConnected)
    {
        return false;
    }

    if (get_target()->isainfo().type == ISA_ZYv5 &&
        (n->inputs()[0]->type() == Tensor::FP16 ||
         n->inputs()[0]->type() == Tensor::BFP16 ||
         n->outputs()[0]->type() == Tensor::FP16 ||
         n->outputs()[0]->type() == Tensor::BFP16))
    {
        return false;
    }

    if (is_btconv_supported(n))
    {
        return false;
    }

    bool force_batchfc_to_conv = false;
    if (check_plugin_option("FORCE_BATCHFC_TO_CONV"))
    {
        force_batchfc_to_conv = true;
    }
    if (check_plugin_option("AIPU_LLM_SAME_WEIGHT") &&
        get_target()->isainfo().type >= ISA_ZYv6)
    {
        force_batchfc_to_conv = true;
    }
    auto wt_scales = n->constants()["weights"]->quantization().scales();
    if ((int)wt_scales.size() > n->outputs()[0]->shape()[1])
    {
        force_batchfc_to_conv = true;
    }
    if (Tensor::is_float(n->inputs()[0]->type()) && Tensor::is_float(n->outputs()[0]->type()))
    {
        force_batchfc_to_conv = true;
    }

    int max_in_c = force_batchfc_to_conv ? 1048576 : 8192;
    if (n->inputs()[0]->shape().n() == 1 || n->inputs()[0]->shape().h() > max_in_c)
    {
        return false;
    }

    if (!n->inputs()[0]->shape().symbol().n()->is_constant())
    {
        pad_in = true;
        auto p = n->parents()[0];
        if (p->type() == OpType_Pad && p->attrs().count(Constants::n_align))
        {
            auto n_align = p->get_attr<int>(Constants::n_align);
            if (get_batch_alignment(n) == n_align)
            {
                pad_in = false;
            }
        }
    }

    if (!n->outputs()[0]->shape().symbol().n()->is_constant())
    {
        crop_out = true;
    }

    return pad_in || crop_out;
}

static NodePtr pad_input(GraphPtr g, NodePtr n)
{
    int alignment = get_batch_alignment(n);
    auto inp = n->inputs()[0];
    auto in_shape = inp->shape();
    std::vector<std::pair<uint32_t, uint32_t>> pads(in_shape.dim(), {0, 0});
    pads[0] = {0, (alignment - in_shape[0] % alignment) % alignment};
    int32_t constant_value = 0 - inp->quantization().offset();

    auto consumers = inp->consumers();
    auto pad_out = ops::pad(inp, pads, ops::PadMode::Constant, constant_value);
    auto pad_node = pad_out->node();
    for (auto c : consumers)
    {
        if (c->type() == OpType_FullyConnected)
        {
            c->add_input(pad_out, c->remove_input(inp));
        }
    }

    std::vector<std::vector<Expr>> ds_pads(in_shape.dim(), {0, 0});
    ds_pads[0][1] = (alignment - in_shape.symbol().n() % alignment) % alignment;
    pad_node->params()["ds_pads"] = ds_pads;
    pad_node->set_attr(Constants::n_align, alignment);
    pad_node->infer_shape(true);
    g->add_node(pad_node);

    return pad_node;
}

static NodePtr crop_output(GraphPtr g, NodePtr n)
{
    int alignment = get_batch_alignment(n);
    auto out = n->outputs()[0];
    auto out_shape = out->shape();
    auto consumers = out->consumers();
    std::vector<int> indexs;
    for (auto c : consumers)
    {
        indexs.push_back(c->remove_input(out));
    }
    auto crop_in = std::make_shared<Tensor>(*out);
    crop_in->set_io_tensor(false);
    crop_in->set_name(g->get_valid_tensor_name(out->name()));
    crop_in->shape()[0] = (out_shape[0] + alignment - 1) / alignment * alignment;
    n->add_output(crop_in, n->remove_output(out));
    n->infer_shape(true);

    std::vector<std::pair<int32_t, int32_t>> crops;
    for (int i = 0; i < out_shape.dim(); ++i)
    {
        crops.push_back({0, out_shape[i]});
    }

    auto crop_out = ops::crop(crop_in, crops);
    auto crop_node = crop_out->node();
    crop_node->add_output(out, crop_node->remove_output(crop_out));
    for (uint32_t i = 0; i < consumers.size(); ++i)
    {
        consumers[i]->add_input(out, indexs[i]);
    }

    std::vector<std::vector<Expr>> ds_crops;
    for (int i = 0; i < out_shape.dim(); ++i)
    {
        ds_crops.push_back({0, out_shape.symbol()[i]});
    }
    crop_node->params()["ds_crops"] = ds_crops;
    crop_node->set_attr(Constants::n_align, alignment);
    crop_node->infer_shape(true);
    g->add_node(crop_node);

    return crop_node;
}

void fc_batch_adaptor(GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        bool pad_in = false, crop_out = false;

        if (!criteria(n, pad_in, crop_out))
        {
            continue;
        }

        if (pad_in)
        {
            auto pad_node = pad_input(g, n);
            ADD_LOCATION_BY_NODE(pad_node, n);
        }

        if (crop_out)
        {
            auto crop_node = crop_output(g, n);
            ADD_LOCATION_BY_NODE(crop_node, n);
        }
    }
}

} // namespace aipubt

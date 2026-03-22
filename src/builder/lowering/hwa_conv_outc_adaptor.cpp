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
 * @Author: Samule Han
 * @Date: 2020-03-03 13:01:13
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-06-13 14:56:10
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

static int get_ds_alignment(NodePtr node)
{
    auto out = node->outputs()[0];
    if (out->type() == Tensor::INT16 || out->type() == Tensor::UINT16 ||
        out->type() == Tensor::BFP16 || out->type() == Tensor::FP16)
    {
        return 16;
    }
    if (out->type() == Tensor::INT8 || out->type() == Tensor::UINT8)
    {
        return 32;
    }
    return 8;
}

static ComputeTensor pad_weight(ComputeTensor w, int new_oc)
{
    std::vector<int> s = w->shape();
    s[0] = new_oc;
    auto new_w = ComputeTensor::zeros(TensorShape(s, w->shape().layout()), w->type());
    new_w[{{0, w->shape()[0]}}] = w;
    new_w->make_memory_continuous();
    return new_w;
}

static void pad_weight_bias(NodePtr n, int ori_oc)
{
    int new_oc = n->outputs()[0]->shape().c();
    // pad weight/bias
    ComputeTensor weight = n->constants()["weights"];
    ComputeTensor bias = n->constants()["biases"];
    n->constants()["weights"] = pad_weight(weight, new_oc);
    n->constants()["biases"] = pad_weight(bias, new_oc);

    // pad weight scale/zp
    auto wt_scale = weight->quantization().scales<ComputeTensor>();
    if ((int)wt_scale.size() >= ori_oc)
    {
        int group_num = wt_scale.size() / ori_oc;
        auto new_wt_scale = pad_weight(wt_scale, new_oc * group_num);
        n->constants()["weights"]->quantization().set_scales(new_wt_scale);
    }
    auto wt_zp = weight->quantization().offsets<ComputeTensor>();
    if ((int)wt_zp.size() >= ori_oc)
    {
        int group_num = wt_zp.size() / ori_oc;
        auto new_wt_zp = pad_weight(wt_zp, new_oc * group_num);
        n->constants()["weights"]->quantization().set_offsets(new_wt_zp);
    }

    // pad zp0/scale0/scale1
    for (auto cst_name : {"zp0", "scale0", "scale1"})
    {
        if (!n->constants().count(cst_name))
        {
            continue;
        }
        ComputeTensor ct = n->constants()[cst_name];
        n->constants()[cst_name] = pad_weight(ct, new_oc);
    }
}

static void checking_mixed_output_align_within_eltwise(NodePtr conv, int32_t &alignment)
{
    // see CP-4880/CP-9589
    // conv + eltwise
    // checking conv_input precision and eltwise the other input precision
    if (conv->children().size() == 1 && conv->children()[0]->type() == OpType_Eltwise)
    {
        auto i0 = conv->inputs()[0];
        auto i1 = conv->children()[0]->inputs()[1];
        if (conv->outputs()[0] == i1)
        {
            i1 = conv->children()[0]->inputs()[0];
        }
        if (Tensor::dtype_size(i0->type()) != Tensor::dtype_size(i1->type()))
        {
            alignment = 32;
        }
    }
}

static int criteria_zyv1(NodePtr n)
{
    auto params = n->params();
    if (params.find("kernel_x") == params.end() ||
        params.find("kernel_y") == params.end() ||
        params.find("stride_x") == params.end() ||
        params.find("stride_y") == params.end())
    {
        ERROR() << "hwa_conv_inc_adaptor: node " << n->name() << "does NOT have parameters of kx,ky,sx,sy!";
        return 0;
    }
    //skip depthwise
    if (n->attrs().count("is_dwconv") && n->get_attr("is_dwconv"))
    {
        return 0;
    }

    int align_num = 16;
    bool conv = ((n->type() == OpType_Convolution));
    bool align = n->outputs()[0]->shape().c() % align_num != 0;
    conv = conv && align;

    bool deconv = ((n->type() == OpType_ConvTranspose));
    deconv = deconv && align;
    return conv || deconv ? align_num : 0;
}

static int criteria_qian(NodePtr n)
{
    auto &params = n->params();
    if (params.find("kernel_x") == params.end() ||
        params.find("kernel_y") == params.end() ||
        params.find("stride_x") == params.end() ||
        params.find("stride_y") == params.end())
    {
        ERROR() << "hwa_conv_inc_adaptor: node " << n->name() << "does NOT have parameters of kx,ky,sx,sy!";
        return 0;
    }
    //skip depthwise
    if (n->attrs().count("is_dwconv") && n->get_attr("is_dwconv"))
    {
        return 0;
    }
    auto layout = n->outputs()[0]->shape().layout();
    if (layout != DataLayout_NHWC && layout != DataLayout_NDHWC)
    {
        return 0;
    }
    int align_num = 16;
    if (n->outputs()[0]->type() == Tensor::INT8 || n->outputs()[0]->type() == Tensor::UINT8 ||
        n->inputs()[0]->type() == Tensor::INT8 || n->inputs()[0]->type() == Tensor::UINT8)
    {
        align_num = 32;
    }

    int out_c = n->outputs()[0]->shape().c();
    return (out_c % align_num != 0) ? align_num : 0;
}

static int criteria_qian_r1p0(NodePtr n)
{
    auto &params = n->params();
    if (params.find("kernel_x") == params.end() ||
        params.find("kernel_y") == params.end() ||
        params.find("stride_x") == params.end() ||
        params.find("stride_y") == params.end())
    {
        ERROR() << "hwa_conv_inc_adaptor: node " << n->name() << "does NOT have parameters of kx,ky,sx,sy!";
        return 0;
    }
    auto arch = get_target()->isainfo().type;
    if (arch < aipubt::ISA_ZYv6 &&
        (n->inputs()[0]->type() == Tensor::FP16 ||
         n->inputs()[0]->type() == Tensor::BFP16 ||
         n->outputs()[0]->type() == Tensor::FP16 ||
         n->outputs()[0]->type() == Tensor::BFP16))
    {
        return 0;
    }
    //skip depthwise
    if (n->attrs().count("is_dwconv") && n->get_attr("is_dwconv"))
    {
        return 0;
    }
    auto layout = n->outputs()[0]->shape().layout();
    if (layout != DataLayout_NHWC && layout != DataLayout_NDHWC)
    {
        return 0;
    }
    int align_num = arch < aipubt::ISA_ZYv6 ? 8 : 1;
    if (n->is_dynamic())
    {
        align_num = get_ds_alignment(n);
    }
    else
    {
        checking_mixed_output_align_within_eltwise(n, align_num);
        // // Conv3D/Deconv3D for Z3 do NOT support T8 now
        // if (n->type() == OpType_Convolution3D || n->type() == OpType_ConvTranspose3D)
        // {
        //     align_num = 16;
        //     if (n->inputs()[0]->type() == Tensor::INT8 || n->inputs()[0]->type() == Tensor::UINT8)
        //     {
        //         align_num = 32;
        //     }
        // }
    }
    int out_c = n->outputs()[0]->shape().c();
    return (out_c % align_num != 0) ? align_num : 0;
}

static std::map<aipubt::ISAType, std::function<int(NodePtr)>> criteria_map = {
    {aipubt::ISA_ZY, criteria_zyv1},
    {aipubt::ISA_ZYv2, criteria_qian},
    {aipubt::ISA_ZYv3, criteria_qian_r1p0},
    {aipubt::ISA_ZYv5, criteria_qian_r1p0},
    {aipubt::ISA_ZYv6, criteria_qian_r1p0},
};

void hwa_conv_outc_adaptor(GraphPtr g)
{
    auto arch = get_target()->isainfo().type;
    if (!criteria_map.count(arch))
    {
        // ERROR() << "target arch : " << arch << " do NOT support!!";
        return;
    }
    auto criteria = criteria_map[arch];
    auto enable_ds = BuilderConfig::get<bool>(Constants::enable_ds);
    auto nodes = g->nodes();
    for (auto &&n : nodes)
    {
        if (n->type() != OpType_Convolution &&
            n->type() != OpType_ConvTranspose &&
            n->type() != OpType_Convolution3D &&
            n->type() != OpType_ConvTranspose3D)
        {
            continue;
        }
        int align_num = criteria(n);
        if (align_num == 0)
        {
            continue;
        }

        // update conv node
        TensorPtr conv_out = n->outputs()[0];
        int out_c = conv_out->shape().c();
        int new_out_c = ALIGN_PAD(out_c, align_num);
        TensorShape conv_out_shape = conv_out->shape();
        conv_out_shape.c() = new_out_c;
        auto conv_out_name = g->get_valid_tensor_name(conv_out->name());
        TensorPtr new_conv_out = std::make_shared<Tensor>(conv_out_name, conv_out_shape, conv_out->type());
        new_conv_out->quantization() = conv_out->quantization();
        n->clear_outputs();
        n->add_output(new_conv_out);

        pad_weight_bias(n, out_c);

        if (n->constants().count("negative_slope"))
        {
            TensorPtr neg_slope = n->constants()["negative_slope"];
            auto ns_shape = neg_slope->shape();
            ns_shape[ns_shape.dim() - 1] = new_out_c;
            auto new_neg_slope = ComputeTensor::zeros(ns_shape, neg_slope->type());
            new_neg_slope[{ComputeTensor::Ellipse, {0, out_c}}] = neg_slope;
            new_neg_slope->make_memory_continuous();
            n->constants()["negative_slope"] = new_neg_slope;
        }
        if (n->constants().count("scale"))
        {
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
            auto shift = n->constants()["shift"];
            auto s_shape = shift->shape();
            s_shape[0] = new_out_c;
            auto new_shift = ComputeTensor::zeros(s_shape, shift->type());
            new_shift[{{0, out_c}}] = shift;
            new_shift->make_memory_continuous();
            n->constants()["shift"] = new_shift;
        }
        n->params()["num_output"] = new_out_c;
        n->infer_shape(true);
        MODIFY_NODE(n);

        // create crop node
        std::vector<std::pair<int32_t, int32_t>> crops;
        for (int i = 0; i < conv_out->shape().dim(); ++i)
        {
            crops.push_back({0, conv_out->shape()[i]});
        }
        auto crop_out = ops::crop(new_conv_out, crops);
        auto crop = crop_out->node();
        crop->set_name(n->name() + "/conv_out_crop");
        crop->clear_outputs();
        crop->add_output(conv_out);
        if (enable_ds)
        {
            std::vector<std::vector<Expr>> ds_crops;
            for (int i = 0; i < conv_out->shape().dim(); ++i)
            {
                ds_crops.push_back({0, conv_out->shape().symbol()[i]});
            }
            crop->params()["ds_crops"] = ds_crops;
        }
        crop->infer_shape(true);

        g->add_node(crop);
        ADD_LOCATION_BY_NODE(crop, n);
    }
}
} // namespace aipubt

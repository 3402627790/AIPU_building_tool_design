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
 * @Date: 2020-03-02 10:07:56
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-04-01 17:42:24
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
    auto inp = node->inputs()[0];
    if (inp->type() == Tensor::INT16 || inp->type() == Tensor::UINT16 || inp->type() == Tensor::BFP16 || inp->type() == Tensor::FP16)
    {
        return 16;
    }
    if (inp->type() == Tensor::INT8 || inp->type() == Tensor::UINT8)
    {
        return 32;
    }
    return 8;
}

static ComputeTensor pad_conv_weight_inc(ComputeTensor weight, int new_in_c)
{
    // weight->make_memory_continuous();
    TensorShape w_shape = weight->shape();
    int i_c = w_shape[-1];
    w_shape[-1] = new_in_c;
    auto w = ComputeTensor::zeros(w_shape, weight->type());
    if (weight->shape().dim() == 5)
    {
        w[{{0, -1}, {0, -1}, {0, -1}, {0, -1}, {0, i_c}}] = weight;
    }
    else
    {
        w[{{0, -1}, {0, -1}, {0, -1}, {0, i_c}}] = weight;
    }
    w->make_memory_continuous();

    return w;
}

static bool checking_node_before_conv(NodePtr n, int32_t align_num)
{
    // return val is whether needing pad by front node type
    // true: need pad, only not align, only front node type is crop, e.g aiff+crop+rgb, add pad to remove crop+pad pair
    // false: no need pad, confirm RGB and not align both, e.g. input/transpose+rgb, concat+rgb
    bool ret = false;
    if ((n->parents()[0]->type() == OpType_Crop) &&
        (n->parents()[0]->children().size() == 1) &&
        (n->parents()[0]->inputs()[0]->shape().n() == n->parents()[0]->outputs()[0]->shape().n()) &&
        (n->parents()[0]->inputs()[0]->shape().h() == n->parents()[0]->outputs()[0]->shape().h()) &&
        (n->parents()[0]->inputs()[0]->shape().w() == n->parents()[0]->outputs()[0]->shape().w()) &&
        (n->parents()[0]->inputs()[0]->shape().c_total() == ALIGN_PAD(n->inputs()[0]->shape().c_total(), align_num)) &&
        (n->parents()[0]->params().count("crops")))
    {
        ret = true;
        std::vector<std::vector<int>> crops = n->params()["crops"];
        for (auto &c : crops)
        {
            ret = ret && (c[0] == 0);
        }
    }
    return ret;
}

static int criteria_zyv1(NodePtr n)
{
    int ret = 0;
    if (!(n->params().count("kernel_x") && n->params().count("kernel_y") &&
          n->params().count("stride_x") && n->params().count("stride_y")))
    {
        ERROR() << "hwa_conv_inc_adaptor: node " << n->name() << "does NOT have parameters of kx,ky,sx,sy!";
        return ret;
    }

    bool conv = ((n->type() == OpType_Convolution));
    bool deconv = ((n->type() == OpType_ConvTranspose));
    if (!(conv || deconv))
        return ret;
    //skip depthwise
    if (n->attrs().count("is_dwconv") && n->get_attr("is_dwconv"))
    {
        return 0;
    }
    int align_num = 16;
    int in_c = n->inputs()[0]->shape().c();
    int in_c_max = (n->inputs()[0]->type() == Tensor::INT8 || n->inputs()[0]->type() == Tensor::UINT8) ? 6 : 3;
    if ((in_c > in_c_max) && (in_c < 16))
    {
        ret = 16;
    }
    else if ((in_c > 16) && (in_c % align_num != 0))
    {
        ret = ALIGN_PAD(in_c, align_num);
    }

    return ret;
}

static int criteria_qian(NodePtr n)
{
    //skip depthwise
    if (n->attrs().count("is_dwconv") && n->get_attr("is_dwconv"))
    {
        return 0;
    }
    if (!(n->params().count("kernel_x") && n->params().count("kernel_y") &&
          n->params().count("stride_x") && n->params().count("stride_y")))
    {
        ERROR() << "hwa_conv_inc_adaptor: node " << n->name() << "does NOT have parameters of kx,ky,sx,sy!";
        return 0;
    }
    auto layout = n->inputs()[0]->shape().layout();
    if (layout == DataLayout_NCHWC16 || layout == DataLayout_NCHWC32 ||
        layout == DataLayout_NDCHWC16 || layout == DataLayout_NDCHWC32)
    {
        return 0;
    }

    int align_num = 16;
    if (n->inputs()[0]->type() == Tensor::INT8 || n->inputs()[0]->type() == Tensor::UINT8)
    {
        align_num = 32;
    }

    auto in_shape = n->inputs()[0]->shape();
    int in_c = in_shape.c();
    int merge_ic = (layout == DataLayout_NDHWC) ? (in_c * in_shape[in_shape.indexing().d]) : in_c;
    if ((checking_node_before_conv(n, align_num) ||
         ((merge_ic != 1) && (merge_ic != 3) && (merge_ic != 4)) ||
         n->type() == OpType_Convolution3D ||
         n->type() == OpType_ConvTranspose3D) &&
        (in_c % align_num != 0))
    {
        if (merge_ic == 2)
        {
            return 3;
        }
        return ALIGN_PAD(in_c, align_num);
    }

    return 0;
}

static int criteria_qian_r1p0(NodePtr n)
{
    auto arch = aipubt::get_target()->isainfo().type;
    auto inp = n->inputs()[0];
    int in_c = inp->shape().c();

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
    if (!(n->params().count("kernel_x") && n->params().count("kernel_y") &&
          n->params().count("stride_x") && n->params().count("stride_y")))
    {
        ERROR() << "hwa_conv_inc_adaptor: node " << n->name() << "does NOT have parameters of kx,ky,sx,sy!";
        return 0;
    }
    auto layout = inp->shape().layout();
    if (layout == DataLayout_NCHWC16 ||
        layout == DataLayout_NCHWC32 ||
        layout == DataLayout_NCHWC32T8 ||
        layout == DataLayout_NCHWC16T8 ||
        layout == DataLayout_NCHWC32T1 ||
        layout == DataLayout_NCHWC16T1 ||
        layout == DataLayout_NDCHWC16 ||
        layout == DataLayout_NDCHWC32 ||
        layout == DataLayout_NDCHWC16T8 ||
        layout == DataLayout_NDCHWC32T8 ||
        layout == DataLayout_NDCHWC16T1 ||
        layout == DataLayout_NDCHWC32T1)
    {
        return 0;
    }

    if (n->is_dynamic())
    {
        int align_num = get_ds_alignment(n);
        std::set<int> special_c_set = {1, 2, 3, 4};
        if (!special_c_set.count(in_c) && in_c % align_num != 0)
        {
            return ALIGN_PAD(in_c, align_num);
        }
        return 0;
    }
    if (arch == aipubt::ISA_ZYv6)
    {
        return 0;
    }
    int align_num = 8;
    // Conv3D/Deconv3D for Z3 do NOT support T8 now.
    if (n->type() == OpType_Convolution3D || n->type() == OpType_ConvTranspose3D)
    {
        align_num = 16;
        if (inp->type() == Tensor::INT8 || inp->type() == Tensor::UINT8)
        {
            align_num = 32;
        }
    }
    auto kernel_shape = n->constants()["weights"]->shape();
    int merge_ic = (layout == DataLayout_NDHWC) ? (in_c * kernel_shape[3]) : in_c;
    bool is_conv3d_before_x2 = (n->type() == OpType_Convolution3D || n->type() == OpType_ConvTranspose3D);
    if (arch == aipubt::ISA_ZYv5)
    {
        align_num = 8;
        merge_ic = in_c;
        is_conv3d_before_x2 = false;
    }
    if ((checking_node_before_conv(n, align_num) ||
         ((merge_ic != 1) && (merge_ic != 3) && (merge_ic != 4)) ||
         is_conv3d_before_x2) &&
        (in_c % align_num != 0))
    {
        if ((merge_ic == 2) && !is_conv3d_before_x2)
        {
            if (arch == aipubt::ISA_ZYv5)
            {
                // AIFF support channel 2 from Z5
                return 0;
            }
            return 3;
        }
        return ALIGN_PAD(in_c, align_num);
    }

    return 0;
}

static std::map<aipubt::ISAType, std::function<int(NodePtr)>> criteria_map = {
    {aipubt::ISA_ZY, criteria_zyv1},
    {aipubt::ISA_ZYv2, criteria_qian},
    {aipubt::ISA_ZYv3, criteria_qian_r1p0},
    {aipubt::ISA_ZYv5, criteria_qian_r1p0},
    {aipubt::ISA_ZYv6, criteria_qian_r1p0},
};

void hwa_conv_inc_adaptor(GraphPtr g)
{
    std::function<int(NodePtr)> criteria;
    auto arch = aipubt::get_target()->isainfo().type;
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
        if (n->type() != OpType_Convolution &&
            n->type() != OpType_ConvTranspose &&
            n->type() != OpType_Convolution3D &&
            n->type() != OpType_ConvTranspose3D)
        {
            continue;
        }
        int new_in_c = criteria(n);
        if (new_in_c == 0)
            continue;

        // create pad node
        auto inp = n->inputs()[0];
        std::vector<std::pair<uint32_t, uint32_t>> pads;
        for (int i = 0; i < inp->shape().dim() - 1; ++i)
        {
            pads.push_back({0, 0});
        }
        pads.push_back({0, new_in_c - inp->shape().c() % new_in_c});
        auto pad_out = ops::pad(inp, pads, ops::PadMode::Constant, 0 - inp->quantization().offset());
        auto pad = pad_out->node();
        if (enable_ds)
        {
            std::vector<std::vector<Expr>> ds_pads(inp->shape().dim(), {0, 0});
            ds_pads.back()[1] = new_in_c - inp->shape().c() % new_in_c;
            pad->params()["ds_pads"] = ds_pads;
        }
        pad->infer_shape(true);

        g->add_node(pad);
        ADD_LOCATION_BY_NODE(pad, n);

        // inputs/outputs
        n->clear_inputs();
        n->add_input(pad_out);
        if (!n->constants().count("weights"))
        {
            ERROR() << "hwa_conv_inc_adaptor : node " << n->name() << " do NOT have weights in constants map!";
            return;
        }
        auto weight = n->constants()["weights"];
        n->constants()["weights"] = pad_conv_weight_inc(weight, new_in_c);
        n->infer_shape(true);
    }
}
} // namespace aipubt

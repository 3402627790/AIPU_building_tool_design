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
 * @Author: Matt Chen
 * @Date: 2022-04-06 19:22:58
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-11-08 16:30:45
 */

#include "quantization_transform/ops/concat.h"

namespace qtlibn
{
void ConcatOp::quantize(aipubt::NodePtr node)
{
    auto oup = node->outputs()[0];

    // collect input quantinfo
    float min_scale = std::numeric_limits<float>::max();
    float max_scale = std::numeric_limits<float>::min();
    int32_t sign_branches = 0;
    int input_num = node->inputs().size();
    std::vector<int> zps;
    std::vector<float> scs;
    std::vector<bool> qinv_s;
    oup->quantization().set_qinvariant(false);
    int qin_bits_max = 8;
    for (int i = 0; i < input_num; i++)
    {
        auto inpq = node->inputs()[i]->quantization();
        auto local_scale = inpq.scale();
        min_scale = min_scale > local_scale ? local_scale : min_scale;
        max_scale = max_scale < local_scale ? local_scale : max_scale;
        if (aipubt::Tensor::is_signed(node->inputs()[i]->type()))
        {
            sign_branches++;
        }
        zps.push_back(inpq.offset());
        scs.push_back(inpq.scale());
        qinv_s.push_back(inpq.qinvariant());
        qin_bits_max = inpq.bits() > qin_bits_max ? inpq.bits() : qin_bits_max;
        if (inpq.qinvariant())
        {
            oup->quantization().set_qinvariant(true);
        }
    }

    // set output quant info
    bool osign = sign_branches > 0;
    oup->quantization().set_bits(qin_bits_max);
    dtype_utils::set_aipu_dtype(oup, osign);

    if (oup->quantization().qinvariant()) // bypass if this layer dont need quant
    {
        oup->quantization().set_scale(1.0);
        oup->quantization().set_offset(0);
        // dtype_utils::set_aipu_dtype(oup, osign);
    }
    else
    {
        Op::quantize(node);
    }

    // if (1)
    // {
    //     auto oot = node->outputs()[0];
    //     auto oo_s = oot->quantization().scale();
    //     auto oo_z = oot->quantization().offset();
    //     std::vector<float> in_scales;
    //     std::vector<int32_t> in_offsets;
    //     for (auto ip : node->inputs())
    //     {
    //         auto ii_s = ip->quantization().scale();
    //         auto ii_z = ip->quantization().offset();
    //         in_scales.push_back(ii_s);
    //         in_offsets.push_back(ii_z);
    //     }
    //     std::all_of(in_scales.begin(), in_scales.end(), [&oo_s](float inss){return oo_s == inss;})
    // }

    int32_t multiplier_bits = node->outputs()[0]->quantization().bits();
    if (node->attrs().count("multiplier_bits"))
    {
        multiplier_bits = node->get_attr("multiplier_bits");
    }
    // do_scale,do_shift
    zps.push_back(oup->quantization().offset());
    scs.push_back(oup->quantization().scale());
    std::vector<int> branch_scales, branch_shifts;
    std::vector<aipubt::Tensor::Dtype> branch_types1, branch_types2;
    for (int i = 0; i < input_num; i++)
    {
        float iscale = node->inputs()[i]->quantization().scale();
        float local_rescale = oup->quantization().scale() / iscale;
        float do_scales;
        int8_t do_shifts;
        quant_utils::get_scale_approximation_params(local_rescale, do_scales, do_shifts, multiplier_bits);
        branch_scales.push_back(static_cast<uint32_t>(do_scales));
        branch_types1.push_back(dtype_utils::bits_to_dtype(multiplier_bits, false));
        branch_shifts.push_back(do_shifts);
        branch_types2.push_back(dtype_utils::bits_to_dtype(8, true));
    }

    float bs_threshold = 1.05;
    if (node->attrs().count("unify_scales_for_concat_threshold"))
    {
        bs_threshold = node->get_attr<aipubt::NodeParamValue>("unify_scales_for_concat_threshold"); // default0 opt change 1.05
    }
    std::sort(scs.begin(), scs.end());
    std::sort(zps.begin(), zps.end());
    bool cond1 = (float(scs.back()) / (float(scs.front() + 0.000001))) <= bs_threshold;
    bool cond2 = (zps.back() - zps.front()) < 0.0001;
    if ((cond1 && cond2) || oup->quantization().qinvariant())
    {
        node->set_attr("need_align_scales", false);
        DEBUG() << node << ": the the largest scale / the smallest scale < 1.05 in all input scales and output scale.";
    }
    else
    {
        node->set_attr("need_align_scales", true);
    }
    if (node->get_attr<bool>("need_align_scales"))
    {
        node->params()["scale_value"] = branch_scales;
        node->params()["shift_value"] = branch_shifts;
        node->params()["scale_type"] = branch_types1;
        node->params()["shift_type"] = branch_types2;
    }
}

REGISTER_QUANTIZATION_OP(Concat);

}; // namespace qtlibn

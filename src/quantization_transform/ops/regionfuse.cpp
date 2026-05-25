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
 * @Author: Samuel Han
 * @Date: 2023-10-27 14:02:05
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-10-27 16:17:45
 */

#include "quantization_transform/ops/regionfuse.h"

namespace qtlibn
{
void RegionFuseOp::quantize(aipubt::NodePtr node)
{
    auto inp0_t = node->inputs()[0];
    auto inp1_t = node->inputs()[1];
    auto inp2_t = node->inputs()[2];
    auto inp3_t = node->inputs()[3];
    auto inp4_t = node->inputs()[4];
    auto inp5_t = node->inputs()[5];
    auto inp6_t = node->inputs()[6];
    auto inp7_t = node->inputs()[7];
    auto inp8_t = node->inputs()[8];
    auto inp9_t = node->inputs()[9];
    int32_t bits = inp0_t->quantization().bits();
    int32_t bits2 = inp2_t->quantization().bits();
    auto dtype2 = inp2_t->quantization().aipu_dtype();
    auto dtype2_signed = aipubt::Tensor::is_signed(dtype2);
    int32_t scale = dtype_utils::bits_to_range_max(bits, false);
    node->params()["score_scale_value"] = std::vector<int32_t>{scale, scale};
    node->params()["score_shift_value"] = std::vector<int32_t>{bits, bits};

    node->params()["score_scale_type"] = std::vector<aipubt::Tensor::Dtype>(2, dtype_utils::range_to_dtype(0, scale, false));
    node->params()["score_shift_type"] = std::vector<aipubt::Tensor::Dtype>(2, aipubt::Tensor::INT8);

    auto in_scale2 = inp2_t->quantization().scale();
    auto in_scale3 = inp3_t->quantization().scale();
    std::vector<float> box_scales = {in_scale2, in_scale3};
    auto min_box_scale = in_scale2 < in_scale3 ? in_scale3 : in_scale3;
    box_scales[0] = round(min_box_scale / box_scales[0] * 4096);
    box_scales[1] = round(min_box_scale / box_scales[1] * 4096);
    int32_t reduce_shift = bits2 - int(dtype2_signed) - floor(std::log2(min_box_scale));
    node->params()["box_scale_value"] = {int(box_scales[0]), int(box_scales[1])};
    node->params()["box_shift_value"] = {12 - reduce_shift, 12 - reduce_shift};

    node->params()["box_scale_type"] = std::vector<aipubt::Tensor::Dtype>{
        dtype_utils::range_to_dtype(0, box_scales[0], false),
        dtype_utils::range_to_dtype(0, box_scales[1], false),
    };
    node->params()["box_shift_type"] = std::vector<aipubt::Tensor::Dtype>(2, aipubt::Tensor::INT8);

    std::map<int32_t, std::vector<aipubt::TensorPtr>> _id_to_inps = {
        {0, {inp0_t, inp1_t}},
        {1, {inp2_t, inp3_t}},
        {2, {inp4_t, inp5_t}},
        {3, {inp6_t, inp7_t}},
        {4, {inp8_t, inp9_t}},
    };

    for (int32_t i = 0; i < (int32_t)node->outputs().size(); i++)
    {
        auto inps = _id_to_inps[i];
        auto scale0 = inps[0]->quantization().scale();
        auto scale1 = inps[1]->quantization().scale();
        auto min_in_scale = scale0 <= scale1 ? scale0 : scale1;
        auto min_in_scale_index = min_in_scale == scale0 ? 0 : 1;

        node->outputs()[i]->quantization() = inps[min_in_scale_index]->quantization();
        node->outputs()[i]->quantization().set_scales({min_in_scale});
        node->outputs()[i]->quantization().set_offsets({0});
        dtype_utils::set_aipu_dtype(node->outputs()[i], aipubt::Tensor::is_signed(inps[min_in_scale_index]->quantization().aipu_dtype()));
    }

    node->outputs()[0]->quantization().set_scales({float(scale)});
    node->outputs()[1]->quantization().set_scales({float(dtype_utils::dtype_to_range_max(node->outputs()[1]->quantization().aipu_dtype()))});
}

REGISTER_QUANTIZATION_OP(RegionFuse);

}; // namespace qtlibn

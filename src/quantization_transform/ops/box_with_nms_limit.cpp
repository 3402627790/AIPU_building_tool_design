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
 * @Last Modified time: 2022-04-20 15:51:09
 */

#include "quantization_transform/ops/box_with_nms_limit.h"
#include "quantization_transform/ops/no_quantization.h"

namespace qtlibn
{
void BoxWithNmsLimitOp::quantize(aipubt::NodePtr node)
{
    // infer output dtype
    bool out_signed = aipubt::Tensor::is_signed(node->inputs()[0]->quantization().aipu_dtype());
    if (node->outputs()[0]->quantization().mins().size() > 0)
    {
        auto mins = node->outputs()[0]->quantization().mins()[0];
        if (mins >= 0)
        {
            out_signed = false;
        }
    }
    dtype_utils::set_aipu_dtype(node->outputs()[0], out_signed);

    Op::quantize(node);

    float score_threshold = node->params()["scoreThreshold"];
    float nms_score_threshold = node->params()["nmsScoreThreshold"];
    float aipu_input_scale = node->inputs()[0]->quantization().scale();
    node->params()["scoreThreshold"] = static_cast<int32_t>(floor(score_threshold * aipu_input_scale));
    node->params()["nmsScoreThreshold"] = static_cast<int32_t>(floor(nms_score_threshold * aipu_input_scale));

    std::map<int32_t, std::string> _nms_kernel_method = {
        {0, "hard"},
        {1, "linear"},
        {2, "gaussian"},
    };

    int32_t nms_kernel_method = node->params()["softNmsKernel"];
    std::string method = _nms_kernel_method[nms_kernel_method];
    if (method == "hard" || method == "linear")
    {
        float iou_threshold_val = 1.0;
        if (node->params().count("iouThreshold"))
        {
            iou_threshold_val = node->params()["iouThreshold"];
        }
        else
        {
            ERROR() << "miss iou_threshold param in BoxWithNmsLimit op, please check, and qtlib will use iou_threshold=1";
        }

        int32_t factor = pow(2, 16) - 1; // passed down by iou_scale
        node->params()["iouThreshold"] = int32_t(floor(iou_threshold_val * factor));
        node->params()["iou_scale"] = factor;
    }
    else if (method == "gaussian")
    {
        uint32_t lut_len = 10001;
        if (node->attrs().count(Constants::lut_items_in_bits))
        {
            uint32_t bits = node->get_attr<aipubt::NodeParamValue>(Constants::lut_items_in_bits);
            bits = bits > 16 ? 16 : bits;
            uint32_t lut_bit_len = pow(2, bits);
            lut_len = lut_len < lut_bit_len ? lut_bit_len : lut_len;
        }

        uint16_t *avg_lut = new uint16_t[lut_len];
        float sigma = node->params()["sigma"];

        if (!node->params().count("guass_lut_shift"))
        {
            node->params()["guass_lut_shift"] = aipubt::NodeParamValue(15);
        }
        int32_t lut_shift = node->params()["guass_lut_shift"];
        int32_t i = 0;
        for (float step = 0; step <= 1.0; step = step + 1.0 / (lut_len - 1))
        {
            float val = exp(-pow(step, 2) / sigma);
            uint16_t iou = quant_utils::clip<float, uint16_t>(round(val * (1 << lut_shift)), 0, (1 << 16) - 1);
            avg_lut[i++] = iou;
        }
        std::string node_name = node->name();
        aipubt::TensorPtr lut_constant = std::make_shared<aipubt::Tensor>(node_name + "box_with_nms_limit_lut", (void *)avg_lut,
                                                                          lut_len * sizeof(uint16_t));
        aipubt::TensorShape lut_shape(lut_len);
        lut_constant->shape() = lut_shape;
        lut_constant->set_type(aipubt::Tensor::UINT16);
        node->constants()["lut"] = lut_constant;
    }
    else
    {
        ERROR() << "unsupported nms kenerl method, which only support [hard, linear, gaussian]";
    }
}

REGISTER_QUANTIZATION_OP(BoxWithNmsLimit);

}; // namespace qtlibn

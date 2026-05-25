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
 * @Date: 2020-08-24 17:13:09
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-03 11:09:50
 */

#include "quantization_transform/ops/op.h"

namespace qtlibn
{

void Op::quantize(aipubt::NodePtr node)
{
    auto _check_input_quantization = [](std::vector<aipubt::TensorPtr> ts, const std::string node_name) -> bool {
        for (auto t : ts)
        {
            if (t->quantization().scales().size() == 0 || t->quantization().offsets().size() == 0)
            {
                ERROR() << "Please quantize the input tensors = " << t->name() << " firstly when calculating the "
                        << node_name << " quantization infos.";
                return false;
            }
            if (!t->quantization().quantized())
            {
                DEBUG() << "please notice, the input tensor = " << t->name() << " of node(name = "
                        << node_name << ") has not quantized, please quantize the input tensor first.";
            }
        }
        return true;
    };

    if (node->type() != aipubt::OpType::OpType_Quantize &&
        !_check_input_quantization(node->inputs(), node->name()))
    {
        return;
    }

    auto adaptor = get_adaptor();
    if (!m_adaptor)
    {
        DEBUG() << "now node(" << node->name()
                << ") doesnot set Adaptor, qtlib will use defaultly calculate scale/zp.";
        quant_utils::calculate_scale_zp(node);
    }
    else
    {
        adaptor->calculate_scale_zp(node);
    }
}

void Op::approximate(aipubt::NodePtr node)
{
    auto _set_float_dtype = [](aipubt::TensorPtr t, aipubt::Tensor::Dtype _dtype) -> bool {
        auto is_fp = aipubt::Tensor::is_float(t->type());
        auto is_iot = t->is_io_tensor();
        if (is_fp && !is_iot)
        {
            t->set_type(_dtype);
            return true;
        }
        else if (is_fp && is_iot) // input/output is io_tensor and float, will keep the origin float,becasue we insert the cast op
        {
            return true;
        }
        return false;
    };
    std::vector<aipubt::Tensor::Dtype> setted_dtypes = quant_utils::preferred_fp_dtype(node);
    // INFO() << "weight dtype = " << setted_dtypes[0] << ", cativation dtype = " << setted_dtypes[1];
    if (setted_dtypes[0] == aipubt::Tensor::VOID && setted_dtypes[1] == aipubt::Tensor::VOID)
    {
        return;
    }

    auto act_dtype = setted_dtypes[0];
    // auto wht_dtype = setted_dtypes[1];
    bool float_dtype = false;
    for (auto &tensors : {node->inputs(), node->outputs()})
    {
        for (auto t : tensors)
        {
            float_dtype |= _set_float_dtype(t, act_dtype);
        }
    }
    std::vector<aipubt::Tensor::Dtype> fp_quantization_dtype = {aipubt::Tensor::FP4E2M1FN, aipubt::Tensor::FP8E5M2, aipubt::Tensor::FP8E4M3FN};
    for (auto &ct : node->constants())
    {
        if (aipubt::Tensor::is_float(ct.second->type()) && ct.first != "biases")
        {
            float_dtype |= true;
            if (ct.second->type() == act_dtype)
            {
                continue;
            }
            if (!aipubt::Tensor::is_float(ct.second->type()))
            {
                continue;
            }
            //fp4e2m1fn, fp8 will not change its dtype and data, which already fpx quantize.
            if (aipubt::contains(fp_quantization_dtype, ct.second->type()))
            {
                continue;
            }
            aipubt::ComputeTensor data_ct = ct.second;
            data_ct = data_ct.astype(act_dtype);
            node->constants()[ct.first]->set_type(act_dtype);
            node->constants()[ct.first] = data_ct;
        }
        if (aipubt::Tensor::is_float(ct.second->type()) &&
            ct.first == "biases" &&
            ct.second->type() != aipubt::Tensor::FP32)
        {
            aipubt::ComputeTensor data_ct = ct.second;
            data_ct = data_ct.astype(aipubt::Tensor::FP32);
            node->constants()[ct.first]->set_type(aipubt::Tensor::FP32);
            node->constants()[ct.first] = data_ct;
        }
    }
    if (float_dtype)
    {
        node->params()["unquantifiable"] = true;
    }

    if (node->type() == aipubt::OpType_Cast && float_dtype)
    {
        node->params()["to_dtype"] = node->outputs()[0]->type();
    }
}

AdaptorPtr Op::get_adaptor(void) const
{
    return m_adaptor;
}

void Op::set_adaptor(AdaptorPtr adaptor)
{
    m_adaptor = adaptor;
}

} // namespace qtlibn

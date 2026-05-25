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
 * @Date: 2022-04-06 21:49:37
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-09-02 10:54:27
 */

#include "quantization_transform/ops/depthwiseconv.h"

namespace qtlibn
{

void DepthwiseConvOp::quantize(aipubt::NodePtr node)
{
    if (node->attrs().count("asym_sym_trans") && (int32_t)node->get_attr<aipubt::NodeParamValue>("asym_sym_trans"))
    {
        DEBUG() << "this is fake depthwiseconv for input/output data transfer.";
        std::string mode = node->get_attr<aipubt::NodeParamValue>("mode");
        if (mode != "input" && mode != "output")
        {
            ERROR() << "please check the fake depthwiseconv 'mode', only support[input, output] mode list.";
            return ;
        }
        bool is_unsigend = node->params().count("with_activation") && ((std::string)node->params()["with_activation"] == "RELU");
        dtype_utils::set_aipu_dtype(node->outputs()[0], !is_unsigend);
        //get aipu_scale for outputs
        Op::quantize(node);
        // save real dwconv quantization params.
        auto quant = node->outputs()[0]->quantization();
        //set weights and biases
        datatransfer_utils::data_transfer(mode, node);
        //quantize constant tensor and cal scale_value and shift_value
        BaseLinearOp::quantize(node);
        //resume real dwconv quantization params
        node->outputs()[0]->quantization() = quant;
    }
    else
    {
        BaseLinearOp::quantize(node);
    }
}

REGISTER_QUANTIZATION_OP(DepthwiseConv);

} // namespace qtlibn
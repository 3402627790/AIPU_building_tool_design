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
 * @Date: 2022-10-20 17:11:37
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-03-27 17:36:47
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

void convert_to_16bit(aipubt::GraphPtr g)
{
    auto _zps_nonzero = [](const std::vector<int32_t> &zerops) -> bool {
        for (auto zp : zerops)
        {
            if (zp != 0)
            {
                return true;
            }
        }
        return false;
    };
    const std::vector<aipubt::OpType> _to_convert_optype = {

        aipubt::OpType_Convolution,
        aipubt::OpType_Convolution3D,
        aipubt::OpType_ConvTranspose,
        aipubt::OpType_ConvTranspose3D,
        aipubt::OpType_FullyConnected,
    };

    auto _convert_dtype = [&_zps_nonzero](aipubt::TensorPtr &t, const aipubt::Tensor::Dtype target_dtype) {
        aipubt::ComputeTensor t_ct = aipubt::ComputeTensor(t).astype(target_dtype);

        auto t_qinfo = t->quantization();
        auto zps = t_qinfo.offsets();
        bool has_nonzero_zp = _zps_nonzero(zps);
        if (has_nonzero_zp)
        {
            auto zps_ct = aipubt::ComputeTensor(zps);
            t_ct = t_ct - zps_ct.astype(target_dtype);
        }
        t = t_ct;
        t->set_type(target_dtype);
        t->quantization() = t_qinfo;
        // t->quantization().set_primitive_dtype(target_dtype);
        if (has_nonzero_zp)
        {
            std::vector<int32_t> new_zps(zps.size(), 0);
            t->quantization().set_offsets(new_zps);
        }
    };

    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        if (node->type() == aipubt::OpType_BatchNorm)
        {
            auto weights = node->constants()["weights"];
            if (weights->type() == aipubt::Tensor::INT8 || weights->type() == aipubt::Tensor::UINT8)
            {
                _convert_dtype(weights, aipubt::Tensor::INT16);
            }
            auto biases = node->constants()["biases"];
            if (biases->type() == aipubt::Tensor::INT32 || biases->type() == aipubt::Tensor::INT8)
            {
                _convert_dtype(biases, aipubt::Tensor::INT32);
            }
        }

        /*
        before X3, the input dtype should be same with the weight dtype in conv serial and fc serial.
        */
        std::string target = "X2";
        if (node->attrs().count("min_compatible_zhouyi_target"))
        {
            target = (std::string)node->get_attr<aipubt::NodeParamValue>("min_compatible_zhouyi_target");
        }
        target = utils::split_string(target, '_')[0];
        if (target[0] == 'Z' || (target.size() > 1 && target[0] == 'X' && std::stoi(target.substr(1, target.size() - 1)) < 3))
        {

            if (aipubt::contains(_to_convert_optype, node->type()))
            {
                auto inp_dtype = node->inputs()[0]->type();
                auto weight_dtype = node->constants()["weights"]->type();
                auto bias_dtype = node->constants()["biases"]->type();
                auto inp_bits = dtype_utils::dtype_to_bits(inp_dtype);
                auto weight_bits = dtype_utils::dtype_to_bits(weight_dtype);
                auto bias_bits = dtype_utils::dtype_to_bits(bias_dtype);
                if (weight_bits < inp_bits && !aipubt::Tensor::is_float(weight_dtype) && !aipubt::Tensor::is_float(inp_dtype))
                {
                    _convert_dtype(node->constants()["weights"], aipubt::Tensor::dtype_to_signed(inp_dtype));
                }

                if ((bias_bits < 32) && !aipubt::Tensor::is_float(bias_dtype))
                {
                    _convert_dtype(node->constants()["biases"], aipubt::Tensor::INT32);
                }
            }
        }
    }
}
} // namespace qtlibn

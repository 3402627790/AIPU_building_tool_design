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
 * @Author: Alex Zheng
 * @Date: 2023-10-20 22:04:33
 * @Last Modified by: Alex Zheng
 * @Last Modified time: 2022-08-13 22:33:03
 */
#include <core/core.h>
#include <core/node.h>
#include <core/ops/ops.h>
#include <core/ops/ops_utils.h>
namespace aipubt
{
namespace ops
{
    std::vector<TensorPtr> divmod(TensorPtr input1,
                                  TensorPtr input2,
                                  DivmodMethod mode,
                                  const std::vector<Quantization> &quants)
    {
        if (Tensor::dtype_bits(input1->type()) != Tensor::dtype_bits(input2->type()))
        {
            ERROR() << "inputs should have same bitwidth!";
            return {};
        }
        std::map<DivmodMethod, std::string> mod_str{
            {DivmodMethod::TRUNC, "TRUNC"},
            {DivmodMethod::FLOOR, "FLOOR"},
        };

        auto g = input1->node()->graph();
        TensorShape div_shape(input1->shape());
        TensorPtr div = std::make_shared<Tensor>(g->get_valid_tensor_name("divmod/div"), div_shape);
        if ((quants.size() > 0) && (!quants[0].empty()))
        {
            add_tensor_info(input1, div, quants[0]);
        }
        else
        {
            add_tensor_info(input1, div);
        }

        TensorShape mod_shape(input1->shape());
        TensorPtr mod = std::make_shared<Tensor>(g->get_valid_tensor_name("divmod/mod"), mod_shape);
        if ((quants.size() > 1) && (!quants[1].empty()))
        {
            add_tensor_info(input1, mod, quants[1]);
        }
        else if ((quants.size() > 0) && (!quants[0].empty()))
        {
            add_tensor_info(input1, mod, quants[0]);
        }
        else
        {
            add_tensor_info(input1, mod);
        }

        auto node = Node::create(g->get_valid_node_name("divmod"), OpType_DivMod);
        node->add_input(input1);
        node->add_input(input2);
        node->add_output(div);
        node->add_output(mod);
        g->add_node(node);
        auto &params = node->params();
        params["mod"] = mod_str[mode];
        node->infer_shape(true);
        quantize(node);
        return {div, mod};
    }

}; // namespace ops
}; // namespace aipubt
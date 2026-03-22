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
 * @Date: 2020-03-12 19:36:04
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-05-22 11:18:38
 */

#include "builder/ro_generator.h"
#include "builder/text_generator.h"
#include "core/core.h"
namespace aipubt
{
RO_GENERATOR(OpType_Customize)
{
    // param param_value[0,1,...,param_num - 1]          4 * param_num
    // param input_addr[0,1,...,input_num - 1]           4 * input_num
    // param output_addr[0,1,...,output_num - 1]         4 * output_num
    // param const_addr[0,1,...,const_num - 1]           4 * const_num
    if (node->params().find("param_num") != node->params().end())
    {
        for (int i = 0; i < (int)node->params()["param_num"]; i++)
        {
            ro.push_back(node->params()["param_value"].arr[i]);
        }
    }
    else
    {
        INFO() << "user defined param value is empty.";
    }

    for (auto it : node->inputs())
    {
        ro.push_back(it);
    }
    for (auto it : node->outputs())
    {
        ro.push_back(it);
    }

    for (auto it : node->constants())
    {
        if (it.first != "weights" && it.first != "biases" && it.first != "lut" && it.first != "scales" &&
            it.first != "shifts" && it.first != "negative_slope" && it.first != "out_constant")
        {
            ro.push_back(it.second);
        }
    }
}
RO_GENERATOR_END(OpType_Customize);

TEXT_GENERATOR(OpType_Customize)
{
    std::string name = "customize.bin";
    return (complete_text_code_name(name));
}
TEXT_GENERATOR_END(OpType_Customize);
}; // namespace aipubt
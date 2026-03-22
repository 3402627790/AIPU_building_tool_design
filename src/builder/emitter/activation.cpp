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
 * @Author: Kyle Huang
 * @Date: 2026-01-08 16:50:53
 */
#include "code_emitter.h"
#include <string>

namespace aipubt
{

std::stringstream CodeEmitter::emitter_activation(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars)
{
    std::string method = n->params()["method"];
    std::stringstream code;
    if (method == "LEAKYRELU")
    {
        int alpha = n->params()["negative_slope_value"];
        if (alpha == 1)
        {
            auto arg0 = vars.at(n->outputs()[0]);
            auto arg1 = vars.at(n->inputs()[0]);

            auto type = get_type(n->outputs()[0]);
            code << "        " << arg0 << " = (" << type << ")" << arg1 << ";";

            code << "\n";
        }
        else
        {
            ERROR();
        }
    }
    else
    {
        ERROR();
    }
    return code;
}
} // namespace aipubt
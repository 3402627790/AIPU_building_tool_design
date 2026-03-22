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
#include <core/compute_tensor.h>
#include <string>

namespace aipubt
{

std::stringstream CodeEmitter::emitter_constant(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars)
{
    // std::string method = n->params()["method"];
    // std::string relu = n->params()["with_activation"];
    // int scale_type = n->params()["scale_type"];
    // float scale_value = n->params()["scale_value"];
    // int shift_type = n->params()["shift_type"];
    // float shift_value = n->params()["shift_value"];

    ComputeTensor wt = n->constants()["weights"];

    auto arg0 = vars.at(n->outputs()[0]);
    std::stringstream code;

    code << "        " << arg0 << " = " << wt[0] << ";";

    code << "\n";

    if (n->outputs()[0]->is_io_tensor())
    {
        INFO() << n << " is io tensor!";
    }

    // int idx = get_arg_idx(n->outputs()[0]);
    // std::string type = get_type(n->outputs()[0]);
    // code << "        *(" << type << "*)pp[" << idx << "] = " << arg0 << ";";
    // code << "\n";
    return code;
}
} // namespace aipubt
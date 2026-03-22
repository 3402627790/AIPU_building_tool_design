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

std::stringstream CodeEmitter::emitter_gather(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars)
{

    auto arg0 = vars.at(n->outputs()[0]);
    auto arg1 = vars.at(n->inputs()[0]);
    auto arg2 = vars.at(n->inputs()[1]);

    if (!contains(m_ptrs, n->inputs()[0]))
    {
        ERROR();
    }
    std::stringstream code;

    code << "        " << arg0 << " = " << arg1 << "[" << arg2 << "]"
         << ";";
    // code << "        " << arg0 << " = " << arg1 << ";";

    code << "\n";
    // int idx = get_arg_idx(n->outputs()[0]);
    // std::string type = get_type(n->outputs()[0]);
    // code << "        *(" << type << "*)pp[" << idx << "] = " << arg0 << ";";
    // code << "\n";
    return code;
}
} // namespace aipubt
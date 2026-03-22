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

std::stringstream CodeEmitter::emitter_logical(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars)
{
    std::string method = n->params()["method"];
    std::string op;
    if (method == "EQUAL")
    {
        op = "==";
    }
    else if (method == "NOT_EQUAL")
    {
        op = "!=";
    }
    else if (method == "GREATER")
    {
        op = ">";
    }
    else if (method == "GREATER_EQUAL")
    {
        op = ">=";
    }
    else if (method == "LESS")
    {
        op = "<";
    }
    else if (method == "LESS_EQUAL")
    {
        op = "<=";
    }
    else if (method == "OR")
    {
        op = "OR";
    }
    else if (method == "AND")
    {
        op = "AND";
    }
    else if (method == "XOR")
    {
        op = "XOR";
    }
    else if (method == "NOT")
    {
        op = "NOT";
    }
    else
    {
        ERROR() << "unsupported logical method";
    }

    auto arg0 = vars.at(n->outputs()[0]);
    auto arg1 = vars.at(n->inputs()[0]);
    auto arg2 = vars.at(n->inputs()[1]);
    std::stringstream code;
    if (op == "XOR")
    {
        code << "        " << arg0 << " = (" << arg1 << " OR " << arg2 << ") AND (!(" << arg1 << " AND " << arg2 << "));";
    }
    else if (op == "NOT")
    {
        code << "        " << arg0 << " = NOT " << arg1;
    }
    else
    {
        code << "        " << arg0 << " = " << arg1 << op << arg2 << ";";
    }
    code << "\n";
    // int idx = get_arg_idx(n->outputs()[0]);
    // std::string type = get_type(n->outputs()[0]);
    // code << "        *(" << type << "*)pp[" << idx << "] = " << arg0 << ";";
    // code << "\n";
    return code;
}
} // namespace aipubt
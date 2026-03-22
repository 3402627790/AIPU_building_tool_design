
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
#include <core/node.h>
#include <string>
#include <vector>

namespace aipubt
{

class CodeEmitter
{
    struct EmitInfo
    {
        std::string kernel_name;
        std::string code_name;
        std::string file_name;
        std::vector<std::pair<TensorPtr, std::string>> args;
    };

private:
    // std::vector<std::pair<TensorPtr, std::string>> m_args;
    // std::map<TensorPtr, std::string> m_vars;
    std::set<TensorPtr> m_ptrs;
    void init_args(const std::vector<BaseNodePtr> &nodes);
    std::stringstream emitter_op(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars);
    std::string get_type(TensorPtr t);
    std::stringstream emitter_logical(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars);
    std::stringstream emitter_eltwise(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars);
    std::stringstream emitter_activation(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars);
    std::stringstream emitter_cast(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars);
    std::stringstream emitter_constant(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars);
    std::stringstream emitter_gather(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars);
    std::stringstream emitter_reshape(BaseNodePtr n, const std::map<TensorPtr, std::string> &vars);
    int get_ro_idx(TensorPtr t, const std::vector<std::pair<TensorPtr, std::string>> &args);
    std::map<std::string, EmitInfo> m_info;

public:
    void emit_func(const std::vector<BaseNodePtr> &nodes, BaseNodePtr node, const std::string &name);
    void compiler_Driver(const std::string &name);
    std::string code_name(const std::string &name);
    std::string kernel_name(const std::string &name);
    std::vector<TensorPtr> args(const std::string &name);
};
} // namespace aipubt
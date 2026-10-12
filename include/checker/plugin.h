//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Tom Gao
 * @Date: 2021-01-20 17:26:41
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-21 10:51:09
 */
#ifndef CHECKER_PLUGIN_H
#define CHECKER_PLUGIN_H

#include "core/logger.h"
#include "core/optype.h"
#include "core/plugin.h"
#include "core/tensor_shape.h"

namespace aipubt
{
enum IRErrorType
{
    IRErr_NoError = 0u,
    IRErr_MAX_COUNT = (1 << 16) - 1,
    IRErr_MissingParam = 1,
    IRErr_Syntax = 1000,
    IRErr_InvalidParam = 2000,
    // IRErr_InvalidTensorShape = 3000,
    // IRErr_OutOfRange = 4000,
};

class IRError
{
public:
    template <class T>
    IRError &operator<<(const T &x)
    {
        (*oss) << x;
        return *this;
    }
    IRError(IRErrorType type, std::string key) : m_type(type), m_key(key) {}
    template <class T>
    IRError(IRErrorType type, std::string key, const T &t) : m_type(type), m_key(key) { (*oss) << t; }
    IRErrorType type() { return m_type; };
    const std::string &key() const { return m_key; }
    const std::string msg() const { return (*oss).str(); }

private:
    IRErrorType m_type;
    std::string m_key;
    std::shared_ptr<std::ostringstream> oss = std::make_shared<std::ostringstream>();
};

class CheckerOpPlugin
{
public:
    virtual OpType get_op_type() = 0;
    virtual ~CheckerOpPlugin() = default;
    virtual bool check_IR(BaseNodePtr node, std::vector<IRError> &errors) { return true; };
    virtual std::vector<TensorShape> infer_shape(BaseNodePtr node) { return {}; };
    virtual std::vector<IRError> check(BaseNodePtr node)
    {
        std::vector<IRError> errors;
        check_IR(node, errors);
        return errors;
    };
};
} // namespace aipubt

#endif //!CHECKER_PLUGIN_H
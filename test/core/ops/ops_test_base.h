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
#ifndef TEST_OPS_TEST_BASE_H
#define TEST_OPS_TEST_BASE_H

#include <builder/builder.h>
#include <gcompiler_api.h>

#include <core/graph.h>
#include <core/ops/ops.h>
#include <gcompiler_api.h>

namespace test_ops
{
class OpTestBase
{
public:
    virtual ~OpTestBase() {}
    virtual bool run_test() = 0;
};

using OpTestTable = std::vector<std::tuple<OpTestBase *, std::string, bool>>;

class OpTestObj
{
public:
    static OpTestObj &get_instance()
    {
        static OpTestObj ops_test_obj;
        return ops_test_obj;
    }

    void insert_test_op(OpTestBase *base_op, const std::string &str, bool res)
    {
        std::tuple<OpTestBase *, std::string, bool> ele = std::make_tuple(base_op, str, res);
        m_ops_table.push_back(ele);
    }

    OpTestTable get_ops_table()
    {
        return m_ops_table;
    }

    OpTestTable get_ops_by_name(const std::string &str)
    {
        OpTestTable ops;
        for (auto &it : m_ops_table)
        {
            if (std::get<1>(it) == str)
                ops.push_back(it);
        }
        if (ops.size() == 0)
            ERROR() << "cannot find op: " << str;
        return ops;
    }

private:
    OpTestObj() {}
    ~OpTestObj() {}

private:
    OpTestTable m_ops_table;
};

class OpTestRegister
{
public:
    ~OpTestRegister() {}
    OpTestRegister(OpTestBase *base_op, const std::string &str, bool res)
    {
        OpTestObj::get_instance().insert_test_op(base_op, str, res);
    }
};

#define REGISTER_TEST_OP(NAME, str, res)    \
    static OpTest##NAME g_op_##NAME##_test; \
    static OpTestRegister g_test_register_##NAME(&g_op_##NAME##_test, str, res);

} // namespace test_ops

#endif
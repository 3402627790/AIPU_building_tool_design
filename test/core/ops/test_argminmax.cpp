//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file
// may only be used by a person authorised under and to the extent permitted
// by a subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2018 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//
//       Release Information:
//
//--------------------------------------------------------------------------------

#include <builder/builder.h>
#include <gcompiler_api.h>

#include <core/graph.h>
#include <core/ops/ops.h>
#include <gcompiler_api.h>

#include "ops_test_base.h"

using namespace aipubt;
namespace test_ops
{

class OpTestArgmin : public OpTestBase
{
public:
    OpTestArgmin() {};
    virtual ~OpTestArgmin() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("argmin");
        auto inp = ops::input({2, 2, 2, 8}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::argmin(inp, 2, true, false);
        if (!a)
        {
            return false;
        }

        g->to_png("argmin.png");
        g->serialize("graph_argmin.def");
        return true;
    }
};

class OpTestArgmax : public OpTestBase
{
public:
    OpTestArgmax() {};
    virtual ~OpTestArgmax() {};
    bool run_test() override
    {
        GraphPtr g = Graph::create("argmax");
        auto inp = ops::input({2, 2, 2, 8}, Tensor::Dtype::UINT8, {1.0, 128}, g);
        auto a = ops::argmin(inp, 2, true, false);
        if (!a)
        {
            return false;
        }

        g->to_png("argmax.png");
        g->serialize("graph_argmax.def");
        return true;
    }
};

REGISTER_TEST_OP(Argmin, "argminmax", true);
REGISTER_TEST_OP(Argmax, "argminmax", true);

} // namespace test_ops
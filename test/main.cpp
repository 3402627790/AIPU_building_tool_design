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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "core/ops/test_ops.h"
#include <builder/builder.h>
#include <builder/mem_allocator.h>
#include <builder/scheduler.h>
#include <gcompiler_api.h>
#include <utils/string_utils.h>

#include "builder/test_builder.h"
#include "core/test_core.h"
#include "dtypes/test_op_dtypes.h"
#include "qtlib/test_qtlib.h"
#include "simplifier/test_simplifier.h"
#include "utils/test_utils.h"
using namespace std;
using namespace aipubt;

int main0()
{
    GraphPtr g = Graph::parse("test.def");
    g->topological_sort();
    g->load_weight("alexnet.bin2");
    std::set<NodePtr> ran_nodes;
    MemoryAllocator allocator;
    for (auto n : *g)
    {
        for (auto t : n->outputs())
        {
            allocator.allocate(t->size(), t->memory_offset());
        }
        ran_nodes.insert(n);
        for (auto p : n->parents())
        {
            // try to free parents
            bool is_free = true;
            for (auto child : p->children())
            {
                if (!ran_nodes.count(child))
                {
                    is_free = false;
                    break;
                }
            }
            if (is_free)
            {
                for (auto t : p->outputs())
                {
                    allocator.free(t->memory_offset());
                }
            }
        }
    }
    g->to_dot();
    GraphBuilder builder(g);
    builder.build();
    std::string s = "[1,2,3455,78],[1,23,4,5,6]";

    GraphMatcher gm(
        {
            {"n0", OpType_Convolution},
            {"n1", OpType_Convolution},
            {"n2", OpType_DepthwiseConv},
            {"n3", OpType_Convolution},
            {"n4", OpType_Eltwise},
        },
        {
            {"n0", "n1"},
            {"n1", "n2"},
            {"n2", "n3"},
            {"n3", "n4"},
            {"n0", "n4"},

        });
    auto m = gm.match(g);
    std::cout << m.size();
    auto ret = utils::split_string(s, "],", 1);
    ;
    int j = 0;
    return j;
}

typedef std::function<void(void)> Test_Func;

std::vector<Test_Func> test_all_func = {
    // test_mmu_allocator,
    // test_ar_loader,
    // test_gmater0,
    // test_json,
    // test_elfreader,
    test_qtlib,
    // test_cpp_ops,
    // test_op_dtypes,
    // test_eliminate_cast,
    // test_ct_slicing,
};

int main()
{
    int idx = 1;
    for (auto test_func : test_all_func)
    {
        DEBUG() << "\t****Begin to test [" << idx++ << "/" << test_all_func.size()
                << "]****";
        test_func();
    }

    return 0;
}

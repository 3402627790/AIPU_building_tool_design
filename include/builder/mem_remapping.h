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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-01-29 14:23:23
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-05-28 08:50:18
 */
#ifndef BUILDER_MEM_REMAPPING_H
#define BUILDER_MEM_REMAPPING_H
#include "builder/bldtypes.h"
#include "core/node.h"
#include "core/optype.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace aipubt
{
// text and its size
typedef std::function<void(NodePtr)> MemoryRemappingFunc;
class MemoryRemappingOpRegister
{
public:
    MemoryRemappingOpRegister(OpType t, MemoryRemappingFunc func);
    static std::map<OpType, MemoryRemappingFunc> &get_memory_remapping_map()
    {
        static std::map<OpType, MemoryRemappingFunc> MEM_REMAPPING_MAP;
        return MEM_REMAPPING_MAP;
    }
};
void memory_remapping_op(NodePtr n);

#define MEMORY_REMAPPING_OP(optype) \
    void memory_remmaping_of_op_##optype(NodePtr node)

#define MEMORY_REMAPPING_OP_REGISTER(optype)                             \
    static MemoryRemappingOpRegister _create_mempory_remapping_##optype( \
        optype, memory_remmaping_of_op_##optype);

#define MEMORY_REMAPPING_OP_END(optype) MEMORY_REMAPPING_OP_REGISTER(optype);
};     // namespace aipubt
#endif //! BUILDER_MEM_REMAPPING_H
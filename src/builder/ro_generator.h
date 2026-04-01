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
 * @Author: Neo Wang 
 * @Date: 2020-01-29 14:27:22 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-05-21 17:27:04
 */
#ifndef BUILDER_RO_GENERATOR_H
#define BUILDER_RO_GENERATOR_H
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "builder/readonly_param.h"
#include "builder/bldtypes.h"
#include "core/optype.h"
#include "core/node.h"

#define DEFAULT_RO_NUMBER_PER_LAYER 64
namespace aipubt
{
typedef std::function<void(NodePtr  , std::vector<ReadOnlyParam> &)> ro_generator_func;

ro_generator_func get_ro_generator(const OpType t);
ro_generator_func get_descriptor_generator(const OpType t);
class ROGeneratorRegister
{
public:
    ROGeneratorRegister(OpType t, ro_generator_func func);
    static std::map<OpType, ro_generator_func> &get_ro_generator_map()
    {
        static std::map<OpType, ro_generator_func> RO_GENERATORS_MAP;
        return RO_GENERATORS_MAP;
    }
};

class DescriptorGeneratorRegister
{
public:
    DescriptorGeneratorRegister(OpType t, ro_generator_func func);
    static std::map<OpType, ro_generator_func> &get_descriptor_generator_map()
    {
        static std::map<OpType, ro_generator_func> GENERATORS_MAP;
        return GENERATORS_MAP;
    }
};

#define RO_GENERATOR_REGISTER(optype, func) static ROGeneratorRegister _create_ro_generator_##optype(optype, func);
#define RO_GENERATOR(optype) \
    void ro_generator_func_##optype(NodePtr  node, std::vector<ReadOnlyParam> &ro)

#define RO_GENERATOR_END(optype) RO_GENERATOR_REGISTER(optype, ro_generator_func_##optype);
#define RO_GENERATOR_REGISTER_SHARED_WITH(optype, shared_optype)                        \
    void ro_generator_func_##shared_optype(NodePtr  node, std::vector<ReadOnlyParam> &ro); \
    static ROGeneratorRegister _create_ro_generator_##optype(optype,                    \
                                                             (ro_generator_func_##shared_optype));

#define DESC_GENERATOR_REGISTER(optype, func) static DescriptorGeneratorRegister _create_descriptor_generator_##optype(optype, func);
#define DESC_GENERATOR(optype) \
    void descriptor_generator_func_##optype(NodePtr  node, std::vector<ReadOnlyParam> &descriptor)

#define DESC_GENERATOR_END(optype) DESC_GENERATOR_REGISTER(optype, descriptor_generator_func_##optype);
#define DESC_GENERATOR_REGISTER_SHARED_WITH(optype, shared_optype)                        \
    void descriptor_generator_func_##shared_optype(NodePtr  node, std::vector<ReadOnlyParam> &descriptor); \
    static DescriptorGeneratorRegister _create_descriptor_generator_##optype(optype,                    \
                                                             (descriptor_generator_func_##shared_optype));

}; // namespace aipubt

#endif //!BUILDER_RO_GENERATOR_H
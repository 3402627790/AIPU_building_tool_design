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
 * @Date: 2020-01-29 14:27:25 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-04-01 17:18:34
 */
#include "ro_generator.h"
#include "core/core.h"
namespace aipubt
{

ro_generator_func get_ro_generator(const OpType t)
{
    if (!ROGeneratorRegister::get_ro_generator_map().count(t))
    {
        return nullptr;
    }
    return ROGeneratorRegister::get_ro_generator_map()[t];
}

ro_generator_func get_descriptor_generator(const OpType t)
{
    if (!DescriptorGeneratorRegister::get_descriptor_generator_map().count(t))
    {
        return nullptr;
    }
    return DescriptorGeneratorRegister::get_descriptor_generator_map()[t];
}

ROGeneratorRegister::ROGeneratorRegister(OpType t, ro_generator_func func)
{
    ROGeneratorRegister::get_ro_generator_map()[t] = func;
}
DescriptorGeneratorRegister::DescriptorGeneratorRegister(OpType t, ro_generator_func func)
{
    DescriptorGeneratorRegister::get_descriptor_generator_map()[t] = func;
}
}; // namespace aipubt

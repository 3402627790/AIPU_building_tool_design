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
 * @Author: Lang Hu
 * @Date: 2024-05-29 17:57:05
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-05-29 17:57:05
 */
#pragma once

#include "builder/builder.h"
#include "core/core.h"

namespace aipubt
{

bool is_a_remapping_convertor(NodePtr node, bool remap = true);

NodeParamValue get_control_param(NodePtr n, const std::string &param);

uint32_t get_memory_base(TensorPtr t);

} // namespace aipubt

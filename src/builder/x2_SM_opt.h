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
 * @Date: 2020-07-28 11:11:21 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-07-28 14:41:00
 */
#ifndef BUILDER_X2_SM_OPT_H
#define BUILDER_X2_SM_OPT_H

#include "core/core.h"
#include "core/node.h"
#include "mem_allocator.h"
#include "optimizer.h"
#include <set>
#include <vector>
namespace aipubt
{
// pre load weight to sm for next layer
// only apply to a single core subgraph
void x2_sm_opt(SubGraphNodePtr sgn);
};     // namespace aipubt
#endif //!BUILDER_X2_SM_OPT_H

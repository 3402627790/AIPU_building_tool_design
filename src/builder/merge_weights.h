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
#ifndef BUILDER_MERGE_WEIGHTS_H
#define BUILDER_MERGE_WEIGHTS_H

#include "builder/mem_allocator.h"
#include "builder/optimizer.h"
#include "core/core.h"
#include "core/node.h"
#include <set>
#include <vector>
namespace aipubt
{
// check which layer need to merge all weights in single tensors
void merge_weights(GraphPtr g);
};     // namespace aipubt
#endif //!BUILDER_MERGE_WEIGHTS_H

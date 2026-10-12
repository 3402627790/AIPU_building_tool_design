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
 * @Author: Neo Wang 
 * @Date: 2020-01-29 14:26:24 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-04-01 17:10:29
 */
#ifndef BUILDER_ALLOCATOR_H
#define BUILDER_ALLOCATOR_H

#include "core/core.h"
namespace aipubt
{
class Allocator
{
public:
    virtual ~Allocator() {}
    virtual MemoryObjectPtr allocate(unsigned int size, MemoryObjectPtr ptr) = 0;
    virtual void free(MemoryObjectPtr ptr) = 0;
};
};     // namespace aipubt
#endif //!BUILDER_ALLOCATOR_H
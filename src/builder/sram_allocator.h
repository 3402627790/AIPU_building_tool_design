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
 * @Date: 2023-04-12 16:26:38 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2023-04-12 16:59:42
 */
#include "amat_allocator.h"

namespace aipubt
{
class SRAMAllocator : public AMATAllocator
{
public:
    SRAMAllocator(uint32_t sram_base, size_t sram_size)
    {
        add_soc_sram(sram_base, sram_size);
        enable_soc_sram(true);
    }
    std::vector<MemoryObjectPtr> slots() override
    {
        std::vector<MemoryObjectPtr> ret;
        apply_dtcm_opt();
        return ret;
    }
};
} // namespace aipubt
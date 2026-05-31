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
 * @Author: Aaron Gao
 * @Date: 2022-08-16 14:02:50
 * @Last Modified by: Aaron Gao
 * @Last Modified time: 2022-08-16 14:09:58
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{
// Once unquantifiable is set true, OpsAPI will not call Quantizer(), Qtransform will not use this.
void set_unquantifiable_default(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        for (auto &t : node->outputs())
        {
            t->quantization().set_unquantifiable(false);
        }
        for (auto &tmap : node->constants())
        {
            tmap.second->quantization().set_unquantifiable(false);
        }
    }
}
} // namespace qtlibn

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
 * @Author: Samuel Han
 * @Date: 2024-04-25 11:31:53
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-04-25 13:06:53
 */

#include "passes.h"

namespace qtlibn
{
/*
    node.serialize would export the layer_top_range using the tensor.min and tensor.max.
    and sometimes lib or opt forward would use the layer_top_range, so now should change the
    tensor.min and tensor.max to qmin and qmax
*/
void flash_min_max_to_quantized(aipubt::GraphPtr g)
{
    DEBUG() << "after qt, the min/max is updated by dtype_min and dtype_max";
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        for (auto &ot : node->outputs())
        {
            if (ot->quantization().quantized())
            {
                auto mins = ot->quantization().mins();
                auto maxs = ot->quantization().maxs();
                if (mins.size() || maxs.size())
                {
                    auto dtype = ot->type();
                    float dtype_min = dtype_utils::dtype_to_range_min(dtype);
                    float dtype_max = dtype_utils::dtype_to_range_max(dtype);
                    ot->quantization().set_mins({dtype_min});
                    ot->quantization().set_maxs({dtype_max});
                }
            }
        }
    }
}
} // namespace qtlibn

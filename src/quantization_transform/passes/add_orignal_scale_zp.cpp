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
 * @Date: 2022-03-07 18:57:23
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-09 13:49:28
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

// add original scales/zps to node.attrs()
void backup_primitive_scale_zp(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        if (!node->attrs().count(Constants::original_scale))
        {
            std::vector<aipubt::NodeParamValue> scales;
            std::vector<aipubt::NodeParamValue> zps;
            for (auto ot : node->outputs())
            {
                aipubt::NodeParamValue scale = ot->quantization().scales();
                aipubt::NodeParamValue zp = ot->quantization().offsets();
                scales.push_back(scale);
                zps.push_back(zp);
            }
            node->set_attr(Constants::original_scale, scales);
            node->set_attr(Constants::original_zp, zps);
        }
    }
}
} // namespace qtlibn

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
 * @Author: Yuchou.Gan
 * @Date: 2020-02-25 17:58:05
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2020-12-14 15:01:10
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"

namespace aipubt
{

/*
TODO! since some  op does not support NCHWC32 in param_check, this pass will change the datalayou to NHWC for these ops.
*/
void datalayout_adaptor(GraphPtr g)
{
    static std::set<OpType> support_NHWC = {
        OpType_Pooling,
        OpType_Convolution,
        OpType_ConvTranspose,
        OpType_DepthwiseConv,
    };
    static std::set<OpType> support_NDHWC = {
        OpType_Pooling3D,
        OpType_ConvTranspose3D,
        OpType_Convolution3D,
    };

    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (support_NHWC.count(n->type()))
        {
            for (auto inp : n->inputs())
            {
                inp->shape().set_layout(DataLayout_NHWC);
            }
            for (auto out : n->outputs())
            {
                out->shape().set_layout(DataLayout_NHWC);
            }
            // update params of subsequent layoutconvertor
            for (auto c : n->children())
            {
                if (c->type() == OpType_LayoutConvertor)
                {
                    auto nhwc = datalayout_to_string(DataLayout_NHWC);
                    if (c->params().count("from") && c->params()["from"] != nhwc)
                    {
                        c->params()["from"] = nhwc;
                    }
                }
            }
        }
        else if (support_NDHWC.count(n->type()))
        {
            for (auto inp : n->inputs())
            {
                inp->shape().set_layout(DataLayout_NDHWC);
            }
            for (auto out : n->outputs())
            {
                out->shape().set_layout(DataLayout_NDHWC);
            }
            // update params of subsequent layoutconvertor
            for (auto c : n->children())
            {
                if (c->type() == OpType_LayoutConvertor)
                {
                    auto ndhwc = datalayout_to_string(DataLayout_NDHWC);
                    if (c->params().count("from") && c->params()["from"] != ndhwc)
                    {
                        c->params()["from"] = ndhwc;
                    }
                }
            }
        }
    }
}
}; // namespace aipubt

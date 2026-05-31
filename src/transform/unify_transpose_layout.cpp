
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2023 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Yongjie Li
 * @Date: 2025-3-17  13:15:32
 * @Last Modified by: Yongjie Li
 * @Last Modified time: 2025-3-17 15:34:00
 */

#include "builder/lowering/lowering.h"
#include "builder/target.h"
#include "core/compute_tensor.h"
#include "core/core.h"
#include "core/node.h"
#include "core/ops/ops.h"
#include "core/config.h"

/**
 * This pass will insert a layout convertor to make transpose's input and output have the same data layout, which can better performance.
 */

namespace aipubt
{
static inline int criteria(NodePtr node, const std::vector<DataLayout> forbidden_layout_comb)
{
    auto node_type = node->type();
    if (node_type == OpType_SubGraph)
    {
        node_type = get_real_node(node)->type();
    }
    if (node_type == OpType_Transpose && node->inputs()[0]->shape().layout() != node->outputs()[0]->shape().layout())
    {
        auto output_layout = node->outputs()[0]->shape().layout();
        auto input_layout = node->inputs()[0]->shape().layout();
        // insert layout convertor before transpose
        auto forbidden_output_layout = std::find(forbidden_layout_comb.begin(), forbidden_layout_comb.end(), output_layout);
        if (forbidden_output_layout == forbidden_layout_comb.end())
        {
            return 0;
        }    
        // insert layout convertor after transpose
        auto forbidden_input_layout = std::find(forbidden_layout_comb.begin(), forbidden_layout_comb.end(), input_layout);
        if (forbidden_input_layout == forbidden_layout_comb.end())
        {
            return 1;
        }
    }    
    return -1;
}


void unify_transpose_layout(GraphPtr g)
{
    std::vector<DataLayout> forbidden_layout_comb = {
        DataLayout_NCHWC32,DataLayout_NCHWC16,DataLayout_NCHWC8,DataLayout_NCHWC4,DataLayout_NDCHWC32
    };
    
    if (aipubt::get_target()->isainfo().type != aipubt::ISA_ZYv6)
    {
        return ; 
    }
    auto nodes = g->nodes();
    
    for (auto node : nodes)
    {
        int criteria_idx = criteria(node, forbidden_layout_comb);
        if (criteria_idx > -1)
        {
            auto input = node->inputs()[0];
            auto output = node->outputs()[0];
            auto input_shape = input->shape();
            auto output_shape = output->shape();
            const DataLayout output_layout = output->shape().layout();
            const DataLayout input_layout = input->shape().layout();
            // create layout convertor
            auto base_name = input->name() + "/layout/adaptor";
            auto layout_convertor = Node::create(g->get_valid_node_name(base_name), OpType_LayoutConvertor);
            layout_convertor->params()["from"] = datalayout_to_string(input_layout);
            layout_convertor->params()["to"] = datalayout_to_string(output_layout);
            // insert layout convertor before transpose
            if (criteria_idx == 0)
            {
                if (!input_shape.change_layout(output_layout))
                {
                    WARN() << "the input layout is:" << input_layout << " which can not change to layout:" << output_layout;
                    return;
                }
                
                layout_convertor->add_input(input);
                auto cvt_output = std::make_shared<Tensor>(g->get_valid_tensor_name(base_name), input_shape, input->type());
                cvt_output->quantization() = input->quantization();
                layout_convertor->add_output(cvt_output);
                g->add_node(layout_convertor);
                node->replace_input_by(input, cvt_output);  
            }
            // insert layout convertor after transpose
            else
            {
                if (!output_shape.change_layout(input_layout))
                {
                    WARN() << "the output layout is:" << output_layout << " which can not change to layout:" << input_layout;
                    return;
                }
                layout_convertor->add_output(output);
                auto cvt_input = std::make_shared<Tensor>(g->get_valid_tensor_name(base_name), output_shape, output->type());
                cvt_input->quantization() = output->quantization();
                layout_convertor->add_input(cvt_input);
                g->add_node(layout_convertor);
                node->remove_output(output);
                node->add_output(cvt_input);
            }
            ADD_LOCATION_BY_NODE(layout_convertor, node);
        }
    }
}
} // namespace aipubt
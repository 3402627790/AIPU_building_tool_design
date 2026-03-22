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
 * @Date: 2020-05-26 17:25:56
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-21 16:16:37
 */
#include "builder/lowering/lowering.h"
#include "core/core.h"
#include <algorithm>
#include <set>

namespace aipubt
{
static std::set<OpType> forbidden_input_strides = {OpType_Slice, OpType_InternalSlice};
static std::set<OpType> forbidden_output_strides = {OpType_Concat, OpType_InternalConcatenated};
static inline bool support_strides(TensorPtr t)
{
    auto &attrs = t->attrs();
    if (attrs.count("support_strides") && cast_any<bool>(t->get_attr("support_strides")) == true)
    {
        return true;
    }
    else
    {
        return false;
    }
}

void adapt_tensor_strides(GraphPtr g)
{
    auto m_nodes = g->nodes();
    auto output_nodes = g->get_output_nodes();
    for (auto &&n : m_nodes)
    {
        auto &inputs = n->inputs();
        auto &outputs = n->outputs();
        auto t = get_real_op_type(n);
        if (forbidden_input_strides.count(t))
        {
            for (auto &inp : inputs)
            {
                inp->set_attr("support_strides", false);
            }
        }
        if (forbidden_output_strides.count(t))
        {
            for (auto &out : outputs)
            {
                out->set_attr("support_strides", false);
            }
        }
    }
    for (auto &&n : m_nodes)
    {
        auto &inputs = n->inputs();
        auto &outputs = n->outputs();

        if (n->attrs().count("support_output_strides"))
        {
            std::vector<uint32_t> index = n->get_attr<std::vector<uint32_t>>("support_output_strides");
            for (uint32_t idx = 0; idx < outputs.size(); idx++)
            {
                auto &attrs = outputs[idx]->attrs();
                if (std::find(index.begin(), index.end(), idx) != index.end())
                {
                    if (attrs.count("support_strides") == 0)
                    {
                        outputs[idx]->set_attr("support_strides", true);
                    }
                }
                else
                {
                    outputs[idx]->set_attr("support_strides", false);
                }
            }
        }
        else
        {
            for (uint32_t idx = 0; idx < outputs.size(); idx++)
            {
                outputs[idx]->set_attr("support_strides", false);
            }
        }

        if (n->attrs().count("support_input_strides"))
        {
            std::vector<uint32_t> index = n->get_attr<std::vector<uint32_t>>("support_input_strides");
            for (uint32_t idx = 0; idx < inputs.size(); idx++)
            {
                auto &attrs = inputs[idx]->attrs();
                if (std::find(index.begin(), index.end(), idx) != index.end())
                {
                    if (attrs.count("support_strides") == 0)
                    {
                        inputs[idx]->set_attr("support_strides", true);
                    }
                }
                else
                {
                    inputs[idx]->set_attr("support_strides", false);
                }
            }
        }
        else
        {
            for (uint32_t idx = 0; idx < inputs.size(); idx++)
            {
                inputs[idx]->set_attr("support_strides", false);
            }
        }

        if (std::find(output_nodes.begin(), output_nodes.end(), n) != output_nodes.end())
        {
            // for graph output, not support strides
            for (uint32_t idx = 0; idx < outputs.size(); idx++)
            {
                outputs[idx]->set_attr("support_strides", false);
            }
        }
    }
}
void adapt_slice_strides(GraphPtr g)
{
    auto criterion = [](NodePtr n) -> bool {
        OpType t = get_real_op_type(n);

        if (t != OpType_Slice)
        {
            return false;
        }

        auto input = n->inputs()[0];
        auto output = n->outputs()[0];

        // input tensor stride should be compact
        if (!input->shape().is_continuous())
        {
            return false;
        }

        // output would be strided tensor
        if (!support_strides(output))
        {
            return false;
        }
        return true;
    };

    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        if (!criterion(n))
        {
            continue;
        }
        std::vector<int> begin, end, strides;
        for (auto it : n->params()["begin"].arr)
        {
            begin.push_back(it);
        }
        for (auto it : n->params()["end"].arr)
        {
            end.push_back(it);
        }
        for (auto it : n->params()["strides"].arr)
        {
            strides.push_back(it);
        }

        TensorPtr input = n->inputs()[0];
        std::vector<int> origin_shape;
        for (auto it = input->shape().begin(); it != input->shape().end(); it++)
        {
            origin_shape.push_back(*it);
        }

        bool valid_slice = true;
        uint32_t begin_offset = 0;
        uint32_t lane = 1;
        std::vector<uint32_t> new_strides(begin.size(), 1);
        ssize_t dsize = Tensor::dtype_size(input->type());
        for (uint32_t i = begin.size(); i != 0; i--)
        {
            auto index = i - 1;
            if (strides[index] != 1)
            {
                valid_slice = false;
                break;
            }
            begin_offset = begin_offset + lane * begin[index];
            new_strides[index] = lane;
            lane = lane * origin_shape[index];
        }

        begin_offset = begin_offset * dsize;
        uint32_t align = n->outputs()[0]->memory_offset()->align;
        if (begin_offset % align != 0)
        {
            valid_slice = false;
        }
        if (!valid_slice)
        {
            continue;
        }
        NodePtr internal_slice_node = Node::create(n->name(), OpType_InternalSlice);
        ADD_LOCATION_BY_NODE(internal_slice_node, n);
        g->replace_node_by(n, internal_slice_node);
        internal_slice_node->params() = n->params();
        internal_slice_node->constants() = n->constants();

        internal_slice_node->params()["start_offset"] = begin_offset;
        TensorPtr output = internal_slice_node->outputs()[0];

        output->shape().set_strides(new_strides);
        auto children = internal_slice_node->children();
        for (auto c : children)
        {
            auto inputs = c->inputs();
            if (std::find(inputs.begin(), inputs.end(), output) != inputs.end())
            {
                c->set_attr("specified_input_strides", true);
            }
        }
        input->set_attr("support_strides", false);
    }
}

void remove_dummy_slice(GraphPtr g)
{
    auto m_nodes = g->nodes();
    for (auto &&n : m_nodes)
    {
        OpType t = n->type();
        auto params = n->params();

        if (SubGraphNode::is_subgraph_node(n))
        {
            SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(n);
            auto nodes = sgn->subgraph()->nodes();
            if (nodes.size() == 1)
            {
                t = nodes[0]->type();
                params = nodes[0]->params();
            }
        }
        if (t == OpType_Slice)
        {
            uint32_t offset = 0;
            std::vector<int> begin, end, strides;
            std::vector<int> input_shape;
            std::vector<int> output_shape;
            if (!n->inputs()[0]->shape().is_continuous())
            {
                continue;
            }
            if (!n->outputs()[0]->shape().is_continuous())
            {
                continue;
            }
            for (auto it : params["begin"].arr)
            {
                begin.push_back(it);
            }
            for (auto it : params["end"].arr)
            {
                end.push_back(it);
            }
            for (auto it : params["strides"].arr)
            {
                strides.push_back(it);
            }
            for (auto it : n->inputs()[0]->shape())
            {
                input_shape.push_back(it);
            }
            for (auto it : n->outputs()[0]->shape())
            {
                output_shape.push_back(it);
            }

            int ignore_axis = -1;
            bool slice_stride_check = true;
            for (uint32_t i = 0; i != begin.size(); i++)
            {
                if (strides[i] != 1)
                {
                    slice_stride_check = false;
                    break;
                }
                if (begin[ignore_axis + 1] == 0 && end[ignore_axis + 1] == 1 && input_shape[ignore_axis + 1] == 1)
                {
                    ignore_axis = ignore_axis + 1;
                }
            }
            if (!slice_stride_check)
            {
                continue;
            }
            if (ignore_axis == (int)begin.size() - 1)
            {
                // This is a dummy slice and could be removed directly.
                continue;
            }
            uint32_t start_check_axis = ignore_axis + 1;

            uint32_t back_axis = begin.size();
            for (uint32_t i = begin.size(); i != start_check_axis; i--)
            {
                uint32_t index = i - 1;
                back_axis = index;
                if (!(begin[index] == 0 && end[index] == input_shape[index]))
                {
                    break;
                }
            }

            if (back_axis == start_check_axis)
            {
                auto input = n->inputs()[0];
                auto output = n->outputs()[0];
                auto strides = input->shape().strides();
                offset = strides[back_axis] * begin[back_axis];
                ssize_t dsize = Tensor::dtype_size(input->type());
                offset = offset * dsize;
                uint32_t align = output->memory_offset()->align;
                if (offset % align != 0)
                {
                    continue;
                }

                NodePtr reshape_node = Node::create(n->name() + "_dummy", OpType_InternalSlice);
                ADD_LOCATION_BY_NODE(reshape_node, n);
                g->replace_node_by(n, reshape_node);
                reshape_node->params() = n->params();
                reshape_node->constants() = n->constants();
                reshape_node->params()["start_offset"] = offset;
            }
        }
    }
}

} // namespace aipubt

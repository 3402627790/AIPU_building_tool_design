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
 * @Date: 2025-02-14 16:58:12
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-02-14 18:22:28
 */
#include "checker/checker.h"
#include "core/core.h"
#include "core/node.h"
#include "core/ops/ops.h"
#include "core/ops/ops_utils.h"

namespace aipubt
{
namespace ops
{
    TensorPtrList call(TensorPtrList inputs,
                       GraphPtr call_subgraph,
                       const std::string &txt_path)
    {
        if (inputs.size() != call_subgraph->get_input_tensors().size())
        {
            ERROR() << "number of input tensors not matched";
            return TensorPtrList();
        }
        for (size_t i = 0; i < inputs.size(); ++i)
        {
            auto t0 = inputs[i];
            auto t1 = call_subgraph->get_input_tensors()[i];
            if (t0->type() != t1->type() || t0->shape() != t1->shape())
            {
                ERROR() << "input tensor type or shape not matched";
                return TensorPtrList();
            }
        }
        auto g = inputs[0]->node()->graph();
        auto sgn_name = g->get_valid_node_name("call");
        std::vector<NodePtr> m;
        auto sg = call_subgraph->clone();
        sg->replace_symbols(inputs);
        auto sg_inputs = sg->get_input_tensors();
        auto sg_nodes = sg->nodes();
        for (auto n : sg_nodes)
        {
            if (n->type() == OpType_Input)
            {
                sg->remove_node(n);
                continue;
            }
            for (auto inp : n->inputs())
            {
                for (size_t i = 0; i < sg_inputs.size(); ++i)
                {
                    if (inp == sg_inputs[i])
                    {
                        n->replace_input_by(inp, inputs[i]);
                        break;
                    }
                }
            }
            for (auto out : n->outputs())
            {
                out->set_name(g->get_valid_tensor_name(sgn_name + "/" + out->name()));
            }
            n->set_name(g->get_valid_node_name(sgn_name + "/" + n->name()));
            sg->remove_node(n, false);
            g->add_node(n);
            m.push_back(n);
        }
        auto nsg = SubGraph::create(g, m);
        for (auto &kv : call_subgraph->attrs())
        {
            nsg->set_attr(kv.first, kv.second);
        }
        auto sgn = nsg->as_node(true, true);
        // sgn->set_attr("own_weight", true);
        sgn->set_attr("is_call", true);
        sgn->params()["call_subgraph"] = call_subgraph->name();
        sgn->set_name(sgn_name);
        if (txt_path.size())
        {
            sgn->set_attr("txt_path", txt_path);
        }
        return sgn->outputs();
    }

    TensorPtrList call(TensorPtrList inputs,
                       const std::string &call_subgraph,
                       const std::string &txt_path)
    {
        std::vector<NodePtr> nodes;
        for (auto &inp : inputs)
        {
            auto n = inp->node();
            if (!contains(nodes, n))
            {
                nodes.push_back(n);
            }
        }
        auto g = nodes[0]->graph();
        auto nsg = SubGraph::create(g, nodes);
        auto sgn = nsg->as_node(true, true);
        if (!is_DAG(nsg))
        {
            ERROR() << "please check the subgraph in call op, which is not DAG graph.";
            return TensorPtrList();
        }
        sgn->set_attr("is_call", true);
        std::string call_subg = call_subgraph;
        if (call_subg.size() == 0)
        {
            call_subg = sgn->name();
        }
        sgn->params()["call_subgraph"] = call_subg;
        if (txt_path.size())
        {
            sgn->set_attr("txt_path", txt_path);
        }
        return sgn->outputs();
    }
} //namespace ops
} // namespace aipubt
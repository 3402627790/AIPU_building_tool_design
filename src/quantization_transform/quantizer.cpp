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
 * @Date: 2022-03-14 19:06:04
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-03-26 10:38:23
 */

#include "quantizer.h"
#include "inference.h"
#include "observer.h"
#include "ops/op.h"
#include "ops/op_register.h"
#include "passes/passes.h"
#include "qinfo.h"

using namespace aipubt;
namespace qtlibn
{

Quantizer::Quantizer(aipubt::GraphPtr graph, TensorQInfo_t *tensor_qinfo)
    : m_graph(graph)
{
    update_qinfo_to_tensor_in_graph(graph, tensor_qinfo);
}

Quantizer::Quantizer(aipubt::GraphPtr graph, std::string scheme, const int32_t bits)
    : m_graph(graph), m_scheme(scheme)
{
    // configured symmetric and configured bits quantization
    update_graph_qinfo(graph, scheme, bits, false);
}

Quantizer::Quantizer(aipubt::GraphPtr graph, const int32_t bits) : m_graph(graph)
{
    // symmetric and configured bits quantization
    update_graph_qinfo(graph, m_scheme, bits, false);
}

Quantizer::Quantizer(aipubt::GraphPtr graph) : m_graph(graph)
{
    // symmetric and 8bits quantization
    int32_t bits = 8;
    update_graph_qinfo(graph, m_scheme, bits, false);
}

Quantizer::Quantizer(aipubt::NodePtr node, std::string scheme, const int32_t bits)
{
    m_scheme = scheme;
    update_node_qinfo(node, m_scheme, bits, false);
    update_aipu_dtype(node);
    // backup_primitive_scale_zp(node->graph());

    forward(node);
    _node_quantize(node);
}

Quantizer::Quantizer(aipubt::NodePtr node, const int32_t bits) : Quantizer(node, "Symmetric", bits) {}

Quantizer::Quantizer(aipubt::NodePtr node) : Quantizer(node, "Symmetric", 8) {}

Quantizer::~Quantizer()
{
    if (m_graph)
    {
        m_graph = nullptr;
    }
}

void Quantizer::quantize_subgraph(aipubt::NodePtr n)
{
    auto sgn = SubGraphNode::from_node(n);
    auto sg = sgn->subgraph();
    for (auto &subg_node : sg->nodes())
    {
        quantize(subg_node);
    }
}

void Quantizer::quantize(void)
{
    for (auto &node : *m_graph)
    {
        quantize(node);
    }
}

void Quantizer::quantize(aipubt::NodePtr node)
{
    if (aipubt::SubGraphNode::is_subgraph_node(node))
    {
        quantize_subgraph(node);
    }
    else
    {
        _node_quantize(node);
    }

    node->update_constants();
}

void Quantizer::_node_quantize(aipubt::NodePtr node)
{

    if (node)
    {
        auto quant_op = OpRegistry::get_quant_op(node->type());
        if (quant_op)
        {
            if (node->attrs().count("unquantifiable") && (bool)node->get_attr<bool>("unquantifiable"))
            {
                DEBUG() << "node name = " << node->name() << " will do approximate quantization, because node unquantifiable=true";
                // TODO: it is not reasonable to directly return, we need to add new method to deal with the float32 to fp16 or bf16 in transform,
                // like in asym mode, when unquantifiable=true, we only to set the dtype = fp16/bf16
                quant_op->approximate(node);
                return;
            }
            else
            {
                if (!check_node_neccessary_qinfo(node))
                {
                    WARN() << node->name() << ": please check the tensor quantization which should include either scale/zp or max/min.";
                    return;
                }
                quant_op->quantize(node);
            }
        }
        else
        {
            ERROR() << "unsupport node type: " << optype_to_string(node->type())
                    << " node name: " << node->name();
        }
    }
    else
    {
        ERROR() << "please check the quantize node, which now is nullptr";
    }
}

void Quantizer::forward(void)
{
    for (auto &node : *m_graph)
    {
        forward(node);
    }
}

void Quantizer::forward(aipubt::NodePtr node)
{
    std::vector<std::string> forward_ops = {
        // "InstanceNorm",
        // "LayerNorm",
        "BasicLSTM",
        // "MVN",
        "Pow",
        "LogSoftmax",
        "Reduce",
        "SquaredDifference",
        "NormalizedMoments",
        // "GroupNorm",
        "Pooling"};
    if (aipubt::contains(forward_ops, aipubt::optype_to_string(node->type())))
    {
        auto unquantifiable = false;
        if (node->attrs().count("unquantifiable"))
        {
            unquantifiable = node->get_attr("unquantifiable");
        }
        if (unquantifiable)
        {
            DEBUG() << node << "is float op which should not inference to get placeholder for quantization.";
            return;
        }
        if (node->type() == aipubt::OpType_BasicLSTM && node->placeholders().size() > 6)
        {
            bool skip = true;
            for (size_t i = 0; i < node->placeholders().size(); i++)
            {
                if (node->placeholders()[i]->quantization().maxs().size() == 0 || node->placeholders()[i]->quantization().scales().size() == 0)
                {
                    skip = false;
                    break;
                }
            }
            if (skip)
            {
                DEBUG() << "BasicLSTM alreadey has placeholders with statistics set, no need to forward";
                return;
            }
        }
        if (node->type() == aipubt::OpType_Pooling && node->params()["method"] != "L2")
        {
            DEBUG() << "Pooling doesnot need to do forward to save placeholder when method != L2";
            return;
        }
        if (node->type() == aipubt::OpType_Reduce && node->params()["method"] != "L2")
        {
            DEBUG() << "Reduce op doesnot need to do forward to save placeholder when method != L2";
            return;
        }
        Inferencer::inference(node);
    }
}

}; // namespace qtlibn

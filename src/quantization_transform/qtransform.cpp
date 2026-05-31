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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-26 14:51:25
 */

#include "quantization/qtransform.h"
#include "quantization/dequantizer.h"
#include "quantization_transform/adaptor.h"
#include "quantization_transform/ops/op.h"
#include "quantization_transform/ops/op_register.h"
#include "quantization_transform/passes/passes.h"
#include "quantization_transform/qinfo.h"
#include "quantization_transform/quantizer.h"
#include "quantization_transform/run_mode.h"

namespace qtlibn
{

QuantizationTransform::QuantizationTransform(aipubt::GraphPtr graph) : m_graph(graph) {}

QuantizationTransform::QuantizationTransform(aipubt::GraphPtr graph,
                                             std::string run_mode,
                                             TensorQInfo_t *tensor_qinfo)
    : m_graph(graph), m_run_mode(run_mode)
{
    update_qinfo_to_tensor_in_graph(graph, tensor_qinfo);
}

QuantizationTransform::QuantizationTransform(std::string run_mode, int32_t bits) : m_run_mode(run_mode)
{
    get_tensor_qinfo(run_mode, bits, true);
}

QuantizationTransform::QuantizationTransform(aipubt::GraphPtr graph,
                                             std::string run_mode,
                                             int32_t bits)
    : m_graph(graph), m_run_mode(run_mode)
{
    update_graph_qinfo(graph, run_mode, bits, true);
}

QuantizationTransform::~QuantizationTransform()
{
    // FIXME: clear shared_ptr of adaptor, otherwise, it will impact following graph.
    for (auto &node : *m_graph)
    {
        auto quant_op = OpRegistry::get_quant_op(node->type());
        if (quant_op)
        {
            quant_op->set_adaptor(nullptr);
        }
    }
    if (m_graph)
    {
        m_graph = nullptr;
    }
}

void QuantizationTransform::transform(void)
{
    auto dequantizer = Dequantizer(m_graph, m_run_mode);
    dequantizer.dequantize();
}

void QuantizationTransform::quantize(void)
{
    pre_quantize_passes(m_graph);
    for (auto &node : *m_graph)
    {
        RunMode run_mode = string_to_runmode(m_run_mode);
        auto adaptor = std::make_shared<Adaptor>(run_mode);

        auto quant_op = OpRegistry::get_quant_op(node->type());
        if (quant_op)
        {
            quant_op->set_adaptor(adaptor);
        }
        else
        {
            ERROR() << "unsupport node type: " << optype_to_string(node->type())
                    << " node name: " << node->name();
        }
    }

    for (auto &node : *m_graph)
    {
        quantize(node);
    }
    post_quantize_passes(m_graph);
}

void QuantizationTransform::quantize(aipubt::NodePtr node)
{
    auto quantizer = Quantizer();
    quantizer.quantize(node);
}

void QuantizationTransform::forward(void)
{
    pre_forward_passes(m_graph);
    for (auto &node : *m_graph)
    {
        forward(node);
    }
}

void QuantizationTransform::forward(aipubt::NodePtr node)
{
    if (node)
    {
        auto quantizer = Quantizer();
        quantizer.forward(node);
    }
    else
    {
        ERROR() << "please check the forward node, which now is nullptr";
    }
}

void QuantizationTransform::clear_placeholders(void)
{
    for (auto &node : *m_graph)
    {
        node->placeholders().clear();
    }
    return;
}

void QuantizationTransform::serialize(const std::string ir_txt, const std::string ir_bin)
{
    auto nodes = m_graph->nodes();
    for (auto &node : nodes)
    {
        std::vector<std::vector<float>> layer_top_range;
        for (auto &ot : node->outputs())
        {
            if (ot->quantization().mins().size() && ot->quantization().maxs().size())
            {
                auto mins = ot->quantization().mins();
                auto maxs = ot->quantization().maxs();

                layer_top_range.push_back((std::vector<float>){mins[0], maxs[0]});
            }
            ot->set_attr("dtype", ot->type());
            ot->set_type(aipubt::Tensor::FP32);
        }
        node->params()["layer_top_range"] = layer_top_range;

        for (auto &ct : node->constants())
        {
            auto t = ct.second;
            std::string key = ct.first + "_range";
            if (t->quantization().mins().size() && t->quantization().maxs().size())
            {
                auto mins = t->quantization().mins();
                auto maxs = t->quantization().maxs();
                if (mins.size() == 1)
                {
                    node->params()[key] = (std::vector<float>){mins[0], maxs[0]};
                }
                else
                {
                    node->params()[key] = (std::vector<std::vector<float>>){mins, maxs};
                }
            }
        }
    }
    m_graph->serialize(ir_txt, ir_bin);
    for (auto &node : nodes)
    {
        for (auto &ot : node->outputs())
        {
            ot->set_type(ot->get_attr("dtype"));
        }
    }
}

void QuantizationTransform::run(void)
{
    pre_passes(m_graph);
    transform();
    forward();
    quantize();
    clear_placeholders();
    post_passes(m_graph);
}

}; // namespace qtlibn

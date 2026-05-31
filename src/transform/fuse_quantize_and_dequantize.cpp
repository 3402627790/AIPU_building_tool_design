//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Samuel Han
 * @Date: 2022-07-22 16:19:07
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-12-04 15:12:20
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"

namespace aipubt
{

static inline bool criteria(NodePtr n0, NodePtr n1)
{
    return !n0->outputs()[0]->is_io_tensor();
}

static void _substitude_tensor(aipubt::NodePtr target_node, aipubt::TensorPtr src_t, aipubt::TensorPtr dst_t, std::string direct)
{
    if (direct == "in")
    {
        auto idx = target_node->remove_input(src_t);
        target_node->add_input(dst_t, idx);
    }
    else
    {
        auto idx = target_node->remove_output(src_t);
        target_node->add_output(dst_t, idx);
    }
}

static bool _eliminate_quantize_dequantize(std::vector<aipubt::NodePtr> nodes, aipubt::GraphPtr g)
{
    auto in = nodes[0]->inputs()[0];
    auto out = nodes[1]->outputs()[0];
    auto m0_parents = nodes[0]->parents();
    bool to_eliminate = true;
    if (in->is_io_tensor() && !out->is_io_tensor())
    {
        // in substitudes the out tensor
        for (auto &out_c_n : out->consumers())
        {
            _substitude_tensor(out_c_n, out, in, "in");
            in->quantization() = out->quantization();
        }
    }
    else if (out->is_io_tensor() && !in->is_io_tensor())
    {
        // out substitudes the in tensor
        for (auto &in_c_n : nodes[0]->parents())
        {
            _substitude_tensor(in_c_n, in, out, "out");
        }
    }
    else if (!in->is_io_tensor() && !out->is_io_tensor())
    {
        // in substitudes the out tensor
        for (auto &out_c_n : out->consumers())
        {
            _substitude_tensor(out_c_n, out, in, "in");
        }
    }
    else
    {
        // not fuse
        DEBUG() << "because quantize input tensor = " << in->name() << ", dequantize output tensor = "
                << out->name() << " are both the graph input/output tensor, so we will not fuse these two op.";
        to_eliminate = false;
    }

    if (to_eliminate)
    {
        g->remove_node(nodes[0]);
        g->remove_node(nodes[1]);
    }
    return to_eliminate;
}

static bool _merge_quantize_dequantize_to_cast(std::vector<aipubt::NodePtr> nodes, aipubt::GraphPtr g)
{
    auto q_node = nodes[0];
    auto deq_node = nodes[1];
    auto q_inp = q_node->inputs()[0];
    auto deq_out = deq_node->outputs()[0];
    auto ignore_scale_zp = true;
    auto mode = ops::ClipMode::Saturation;
    auto cast_o = ops::cast(q_inp, deq_out->type(), ignore_scale_zp, mode);
    auto cast_node = cast_o->node();
    cast_node->remove_output(cast_o);
    cast_node->add_output(deq_out);

    ADD_LOCATION_BY_NODE(cast_node, deq_node);
    ADD_LOCATION_BY_NODE(cast_node, q_node);
    g->remove_node(deq_node);
    g->remove_node(q_node);
    return true;
}

static bool _merge_dequantize_quantize_to_cast(std::vector<aipubt::NodePtr> nodes, aipubt::GraphPtr g)
{
    auto deq_node = nodes[0];
    auto q_node = nodes[1];
    auto deq_inp = deq_node->inputs()[0];
    auto q_out = q_node->outputs()[0];
    auto inp_quantization = deq_node->inputs()[0]->quantization();
    auto out_quantization = q_node->outputs()[0]->quantization();

    bool ignore_scale_zp = inp_quantization == out_quantization;
    bool is_merged = false;
    if (!ignore_scale_zp)
    {
        // fp8 does not has inst in tpc cast
        if (Tensor::is_float(q_out->type()) && Tensor::is_int(deq_inp->type()) && (Tensor::dtype_size(q_out->type()) < 2))
        {
            // int deq than qt to fp8 = int deq fp8
            //     = ((int + zp_deq) / scale_deq) * scale_q - zp_q
            //     = (int + zp_deq) / (scale_deq / scale_q) - zp_q
            //     = (int + (zp_deq - zp_q * (scale_deq / scale_q))) / (scale_deq / scale_q)
            // new_deq_scale = (scale_deq / scale_q)
            // new_deq_zp = (zp_deq - zp_q * new_deq_scale)
            auto new_deq_scale = inp_quantization.scale() / out_quantization.scale();
            auto new_deq_zp = (int)(inp_quantization.offset() - out_quantization.offset() * new_deq_scale);
            deq_node->clear_outputs();
            deq_node->add_output(q_out);
            deq_inp->quantization().set_scale(new_deq_scale);
            deq_inp->quantization().set_offset(new_deq_zp);

            if (deq_node->params().count("quantize_scale"))
            {
                deq_node->params()["quantize_scale"] = new_deq_scale;
            }
            if (deq_node->params().count("quantize_zp"))
            {
                deq_node->params()["quantize_zp"] = new_deq_zp;
            }

            g->remove_node(q_node);
        }
        else if (Tensor::is_float(deq_inp->type()) && Tensor::is_int(q_out->type()) && (Tensor::dtype_size(deq_inp->type()) < 2))
        {
            // fp8 deq than q to int = fp8 q int
            //     = ((int + zp_deq) / scale_deq) * scale_q - zp_q
            //     = (int + zp_deq) * (scale_q / scale_deq) - zp_q
            //     = int * (scale_q / scale_deq)  - (zp_q + zp_deq * (scale_q / scale_deq))
            // new_q_scale = (scale_q / scale_deq)
            // new_q_zp = (zp_q + zp_deq * new_q_scale)
            auto new_q_scale = out_quantization.scale() / inp_quantization.scale();
            auto new_q_zp = (int)(out_quantization.offset() + inp_quantization.offset() * new_q_scale);
            q_node->clear_inputs();
            q_node->add_input(deq_inp);
            q_out->quantization().set_scale(new_q_scale);
            q_out->quantization().set_offset(new_q_zp);

            if (q_node->params().count("quantize_scale"))
            {
                q_node->params()["quantize_scale"] = new_q_scale;
            }
            if (q_node->params().count("quantize_zp"))
            {
                q_node->params()["quantize_zp"] = new_q_zp;
            }

            g->remove_node(deq_node);
        }
        else
        {
            auto mode = ops::ClipMode::Saturation;
            auto cast_o = ops::cast(deq_inp, q_out->type(), ignore_scale_zp, mode, out_quantization);
            auto cast_node = cast_o->node();
            cast_node->remove_output(cast_o);
            cast_node->add_output(q_out);

            ADD_LOCATION_BY_NODE(cast_node, deq_node);
            ADD_LOCATION_BY_NODE(cast_node, q_node);

            g->remove_node(deq_node);
            g->remove_node(q_node);
        }
        is_merged = true;
    }
    else
    {
        is_merged = _eliminate_quantize_dequantize(nodes, g) || is_merged;
    }

    return is_merged;
}

bool fuse_quantize_and_dequantize(GraphPtr g)
{
    // TODO: dequantize + quantize
    std::vector<GraphMatcherPattern> patterns = {
        {{{"0", OpType_Quantize}, {"1", OpType_DeQuantize}}, {{"0", "1"}}},
        {{{"0", OpType_DeQuantize}, {"1", OpType_Quantize}}, {{"0", "1"}}},
    };
    bool is_changed = false;
    for (auto pattern : patterns)
    {
        GraphMatcher matcher(pattern);
        matcher.set_mode(GraphMatcher::MatchMode::StrictMode);
        auto matched = matcher.match(g);
        while (matched.size())
        {
            bool is_merged = false;
            for (auto &&m : matched)
            {
                if (!criteria(m[0], m[1]))
                {
                    continue;
                }

                if (m[0]->type() == OpType_Quantize)
                {
                    auto in_dtype = m[0]->inputs()[0]->type();
                    auto out_dtype = m[1]->outputs()[0]->type();
                    if (in_dtype == out_dtype)
                    {
                        is_merged = _eliminate_quantize_dequantize(m, g) || is_merged;
                    }
                    else
                    {
                        is_merged = _merge_quantize_dequantize_to_cast(m, g) || is_merged;
                    }
                }
                else // dequantize + quantize
                {
                    is_merged = _merge_dequantize_quantize_to_cast(m, g) || is_merged;
                }
                is_changed = is_merged || is_changed;
            }
            if (!is_merged)
            {
                break;
            }
            matched = matcher.match(g);
        }
    }
    return is_changed;
}
}; // namespace aipubt

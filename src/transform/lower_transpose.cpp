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
 * @Author: Neo Wang
 * @Date: 2020-05-26 17:25:56
 * @Last Modified by: Ryan Lai
 * @Last Modified time: 2023-03-26 14:17:20
 */
#include "core/core.h"
#include <algorithm>
#include <map>
#include <set>

namespace aipubt
{

static std::vector<std::vector<int>> squeeze_perm(const std::vector<int> &perm, std::vector<int> &sqz_flat_perm)
{
    std::vector<std::vector<int>> sqz_perm;
    std::vector<int> sqz = {perm[0]};
    for (int i = 0; i < (int)perm.size() - 1; ++i)
    {
        if (perm[i] + 1 == perm[i + 1])
        {
            sqz.push_back(perm[i + 1]);
        }
        else
        {
            if (sqz.size())
            {
                sqz_perm.push_back(sqz);
                sqz.clear();
            }
            sqz.push_back(perm[i + 1]);
        }
    }
    if (sqz.size())
    {
        sqz_perm.push_back(sqz);
    }

    sqz_flat_perm.clear();
    sqz_flat_perm.resize(sqz_perm.size());
    int d = 0;
    for (uint32_t i = 0; i < sqz_perm.size(); ++i)
    {
        for (uint32_t p = 0; p < sqz_perm.size(); ++p)
        {
            if (sqz_perm[p][0] == d)
            {
                sqz_flat_perm[p] = i;
                d += sqz_perm[p].size();
                break;
            }
        }
    }

    return sqz_perm;
}
static bool lower_reshape(NodePtr node, const std::vector<int> &perm, NodePtr reshape)
{
    auto inp = node->inputs()[0]->shape();
    auto out = reshape->outputs()[0]->shape();
    std::vector<int> sqz_flat_perm;
    auto sqz_perm = squeeze_perm(perm, sqz_flat_perm);
    if (out.dim() < (int)sqz_perm.size())
    {
        return false;
    }
    std::vector<int> sqz;
    for (uint32_t i = 0; i < sqz_perm.size(); ++i)
    {
        int d = 1;
        for (uint32_t j = 0; j < sqz_perm[i].size(); ++j)
        {
            d *= inp[sqz_perm[i][j]];
        }
        sqz.push_back(d);
    }
    // std::vector<std::vector<int>> embed_shape(sqz.size());
    std::vector<std::vector<int>> before_shape(sqz.size());
    uint32_t d = 0;
    for (int i = 0; i < out.dim(); ++i)
    {
        if (d >= sqz.size())
        {
            before_shape[sqz_flat_perm[sqz.size() - 1]].push_back(out[i]);
            continue;
        }
        if (sqz[d] == 1)
        {
            d++;
            if (out[i] == 1)
            {
                before_shape[sqz_flat_perm[d - 1]].push_back(1);
                continue;
            }
        }
        if (sqz[d] % out[i] != 0)
        {
            return false;
        }
        sqz[d] = sqz[d] / out[i];
        // embed_shape[d].push_back(out[i]);
        before_shape[sqz_flat_perm[d]].push_back(out[i]);
    }

    std::vector<int> perm_diff(sqz_flat_perm.size());
    std::vector<std::vector<int>> nest_perm(sqz_flat_perm.size());
    std::vector<int> flat_shape, flat_perm;
    d = 0;
    for (uint32_t i = 0; i < before_shape.size(); ++i)
    {
        perm_diff[i] = before_shape[i].size() - 1;
        for (auto s : before_shape[i])
        {
            flat_shape.push_back(s);
            nest_perm[i].push_back(d);
            d++;
        }
    }
    for (uint32_t i = 0; i < nest_perm.size(); ++i)
    {
        for (auto s : nest_perm[sqz_flat_perm[i]])
        {
            flat_perm.push_back(s);
        }
    }
    node->params()["perm"] = flat_perm;
    reshape->params()["shape"] = flat_shape;
    return true;
}

static bool creteria(NodePtr node)
{
    if (node->type() != OpType_Transpose || node->children().size() == 0 || node->outputs()[0]->is_io_tensor())
    {
        return false;
    }
    return true;
}

static bool ds_creteria(NodePtr node)
{
    if (node->children()[0]->type() == OpType_Softmax)
    {
        auto c = node->children()[0];
        if (c->is_dynamic())
        {
            return false;
        }
    }
    return true;
}

static bool update_empty_param(const std::vector<int> &perm, NodePtr child)
{
    // empty, no params within shape dim
    return true;
}

static bool update_singleaxis_param(const std::vector<int> &perm, NodePtr child)
{
    int axis = child->params()["axis"];
    for (uint32_t i = 0; i < perm.size(); ++i)
    {
        if (i == (uint32_t)axis)
        {
            axis = perm[i];
            break;
        }
    }
    if (axis < 1)
    {
        return false;
    }
    child->params()["axis"] = axis;
    return true;
}

static bool update_multiaxis_param(const std::vector<int> &perm, NodePtr child)
{
    std::vector<int> axis = child->params()["axis"];
    std::vector<int> axis2 = axis;
    for (uint32_t i = 0; i < axis.size(); ++i)
    {
        axis2[i] = perm[axis[i]];
    }
    child->params()["axis"] = axis2;
    return true;
}

static bool update_pad_param(const std::vector<int> &perm, NodePtr child)
{
    std::vector<std::vector<int>> pads = child->params()["pads"];
    auto new_pads = pads;
    for (uint32_t i = 0; i < perm.size(); ++i)
    {
        new_pads[perm[i]] = pads[i];
    }
    child->params()["pads"] = new_pads;

    if (child->params().count("ds_pads"))
    {
        std::vector<std::vector<Expr>> ds_pads = child->params()["ds_pads"];
        auto new_ds_pads = ds_pads;
        for (uint32_t i = 0; i < perm.size(); ++i)
        {
            new_ds_pads[perm[i]] = ds_pads[i];
        }
        child->params()["ds_pads"] = new_ds_pads;
    }
    return true;
}
static bool update_crop_param(const std::vector<int> &perm, NodePtr child)
{
    std::vector<std::vector<int>> crops = child->params()["crops"];
    auto new_crops = crops;
    for (uint32_t i = 0; i < perm.size(); ++i)
    {
        new_crops[perm[i]] = crops[i];
    }
    child->params()["crops"] = new_crops;

    if (child->params().count("ds_crops"))
    {
        std::vector<std::vector<Expr>> ds_crops = child->params()["ds_crops"];
        auto new_ds_crops = ds_crops;
        for (uint32_t i = 0; i < perm.size(); ++i)
        {
            new_ds_crops[perm[i]] = ds_crops[i];
        }
        child->params()["ds_crops"] = new_ds_crops;
    }
    return true;
}

static bool update_quant_param(const std::vector<int> &perm, NodePtr child)
{
    auto quant_param = child->outputs()[0]->quantization();
    if (quant_param.scales().size() > 1 || quant_param.offsets().size() > 1)
    {
        return false;
    }
    return true;
}

static bool update_dequant_param(const std::vector<int> &perm, NodePtr child)
{
    auto quant_param = child->inputs()[0]->quantization();
    if (quant_param.scales().size() > 1 || quant_param.offsets().size() > 1)
    {
        return false;
    }
    return true;
}

static void update_gather_perm(NodePtr node, NodePtr child)
{
    std::vector<int> perm = node->params()["perm"];
    int axis = child->params()["axis"];
    uint32_t index_dims = 1;

    // if index form input
    if (2 == child->inputs().size())
    {
        index_dims = child->inputs()[1]->shape().dim();
    }
    // if index form weight
    if (child->constants().count("lut"))
        index_dims = child->constants()["lut"]->shape().dim();

    // if index has batch
    if (child->params().count("batch_dims"))
    {
        index_dims = index_dims - (uint32_t)child->params()["batch_dims"];
    }

    if (index_dims != 1)
    // 1: no need change perm
    // 0: need remove
    // >1: need enlarge
    {

        std::vector<int> new_perm;
        for (int &p : perm)
        {
            if (p < axis)
                new_perm.push_back(p);
            else if (p == axis)
            {
                int p_need_enlarge = p;
                // or remove
                for (size_t i = 0; i < index_dims; i++)
                    new_perm.push_back(p_need_enlarge++);
            }
            else // p > asix
            {
                new_perm.push_back(p + index_dims - 1);
            }
        }
        perm = new_perm;
        node->params()["perm"] = new_perm;
    }
}

static void update_shape(NodePtr node, NodePtr child)
{
    auto inp = node->inputs()[0];
    auto inter = node->outputs()[0];
    auto out = child->outputs()[0];

    // gather may changes perm
    if (OpType_Gather == child->type())
        update_gather_perm(node, child);

    std::vector<int> perm = node->params()["perm"];
    std::vector<int> inter_shape(out->shape().dim());
    for (uint32_t i = 0; i < perm.size(); ++i)
    {
        inter_shape[perm[i]] = out->shape()[i];
    }
    inter->shape() = TensorShape(inter_shape, out->shape().layout());
    if (child->is_dynamic() && child->type() == OpType_Reshape)
    {
        std::vector<tongue::Expr> exprs(out->shape().dim());
        for (uint32_t i = 0; i < perm.size(); ++i)
        {
            exprs[perm[i]] = out->shape().symbol()[i];
        }
        child->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{exprs};
    }

    node->clear_inputs();
    node->clear_outputs();
    child->remove_input(0);
    child->clear_outputs();

    child->add_input(inp, 0);
    child->add_output(inter);
    if (child->is_dynamic())
    {
        child->infer_shape(true);
    }
    // when child is cast, its dtype may different between in and out
    // the transpose out(inter) dtype may diff with cast out(out)
    child->outputs()[0]->set_type(out->type());
    child->outputs()[0]->quantization() = out->quantization();

    node->add_input(inter);
    node->add_output(out);
    node->infer_shape(true);
    return;
}

static void update_shape(NodePtr node, NodePtrList children)
{
    auto num_children = node->children().size();
    auto inp = node->inputs()[0];
    auto trans_out = node->outputs()[0];
    std::vector<int> trans_perm = node->params()["perm"];

    for (auto child : children)
    {
        auto out = child->outputs()[0];

        auto inter = std::make_shared<Tensor>(*trans_out);
        inter->set_io_tensor(false);
        inter->set_name(node->graph()->get_valid_tensor_name(inter->name()));
        auto inter_shape = inter->shape();

        // auto new_trans = node->clone();
        // new_trans->set_name(node->graph()->get_valid_node_name(new_trans->name()));
        // CP-24562: replace "node->clone()"
        auto new_trans = Node::create(node->graph()->get_valid_node_name("transpose"), OpType_Transpose);
        new_trans->params()["perm"] = trans_perm;

        // gather may changes perm
        if (OpType_Gather == child->type())
            update_gather_perm(new_trans, child);

        for (uint32_t i = 0; i < trans_perm.size(); ++i)
        {
            inter_shape[trans_perm[i]] = out->shape()[i];
        }

        inter->shape() = TensorShape(inter_shape, inp->shape().layout());
        child->remove_input(0);
        child->clear_outputs();
        child->add_input(inp, 0);
        child->add_output(inter);
        child->outputs()[0]->set_type(out->type());
        child->outputs()[0]->quantization() = out->quantization();
        child->infer_shape(true);

        node->graph()->add_node(new_trans);

        new_trans->add_input(inter);
        new_trans->add_output(out);
        new_trans->infer_shape(true);
    }
    if (num_children == children.size()) // all lower
    {
        node->clear_inputs();
        node->clear_outputs();
        node->graph()->remove_node(node);
    }
}

static bool lower_reshape(NodePtr node)
{
    if (node->children().size() == 1 && node->children()[0]->type() == OpType_Reshape)
    {
        std::vector<int> perm = node->params()["perm"];
        auto reshape = node->children()[0];
        if (lower_reshape(node, perm, reshape))
        {
            update_shape(node, reshape);
            // reshape->outputs()[0]->shape()=reshape->params()["shape"];
            return true;
        }
    }
    return false;
}

static bool lower_one_level(NodePtr node)
{
    if (lower_reshape(node))
    {
        return true;
    }
    static std::map<OpType, std::function<bool(const std::vector<int> &, NodePtr)>> lower_map =
        {
            {OpType_Softmax, update_singleaxis_param},
            {OpType_BatchNorm, update_singleaxis_param},
            {OpType_ArgMinMax, update_singleaxis_param},
            {OpType_GroupNorm, update_singleaxis_param},
            {OpType_Gather, update_singleaxis_param},
            {OpType_Reduce, update_multiaxis_param},
            {OpType_Pad, update_pad_param},
            {OpType_Crop, update_crop_param},
            {OpType_Cast, update_empty_param},
            {OpType_Activation, update_empty_param},
            {OpType_Quantize, update_quant_param},
            {OpType_DeQuantize, update_dequant_param},
            {OpType_Square, update_empty_param},
            {OpType_Negative, update_empty_param},
        };
    if (node->children().size() > 1)
    {
        auto n = node->children().size();
        uint32_t n_t = 0;
        for (auto c : node->children())
        {
            if (c->type() == OpType_Reshape) // lower reshape only support one child case now.
            {
                return false;
            }
            if (c->type() == OpType_Gather && c->inputs()[1]->shape().dim() == 1)
            {
                // TODO: refine me CA-3376
                // for swim_transformer:    need lower
                // for resa:                don't need lower
                // it conflict with pass: swap_gather_trans
                return false;
            }
            if (c->type() == OpType_Activation && (std::string)c->params()["method"] == "PRELU")
            {
                return false;
            }
            if (lower_map.count(c->type())) // the child support lower trans
            {
                for (auto gc : c->children())
                {
                    if (gc->type() == OpType_Transpose)
                    {
                        n_t++;
                    }
                }
            }
        }
        if (n_t + 1 < n)
        {
            return false;
        }
    }
    if (node->children().size() == 1)
    {
        auto child = node->children()[0];
        if (!lower_map.count(child->type()))
        {
            return false;
        }
        if (child->type() == OpType_Activation && (std::string)child->params()["method"] == "PRELU")
        {
            return false;
        }
        if (child->type() == OpType_Gather && child->inputs()[1]->shape().dim() == 1)
        {
            // TODO: refine me CA-3376
            // for swim_transformer:    need lower
            // for resa:                don't need lower
            // it conflict with pass: swap_gather_trans
            return false;
        }
        if (child->type() == OpType_Cast && child->outputs()[0]->type() == Tensor::FP32)
        {
            // Cast: XX type -> float32
            // Transpose not support float32 input
            return false;
        }
        std::vector<int> perm = node->params()["perm"];

        if (!lower_map[child->type()](perm, child))
        {
            return false;
        }
        update_shape(node, child);
    }
    else
    {
        NodePtrList to_be_lower;
        for (auto child : node->children())
        {
            if (lower_map.count(child->type()))
            {
                std::vector<int> perm = node->params()["perm"];
                if (!lower_map[child->type()](perm, child))
                {
                    return false;
                }
                to_be_lower.push_back(child);
                // update_shape(node, child);
            }
        }
        update_shape(node, to_be_lower);
    }

    return true;
}
/**
 * @brief move the transpose to lower position
 *
 * @param g
 * @return true if the graph changed
 * @return false
 */

bool lower_transpose(GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!creteria(n))
        {
            continue;
        }
        if (!ds_creteria(n))
        {
            continue;
        }
        is_changed = lower_one_level(n) || is_changed;
    }
    return is_changed;
}
}; // namespace aipubt

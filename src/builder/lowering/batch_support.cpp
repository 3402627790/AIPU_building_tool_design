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
 * @Date: 2020-01-29 14:25:20
 * @Last Modified by: Ryan Lai
 * @Last Modified time: 2024-09-09 14:38:38
 */
#include "builder/lowering/lowering.h"
#include "builder/lowering/tiling.h"
#include "builder/target.h"
#include "transform/passes.h"
#include <fstream>
#include <set>
#include <numeric>

namespace aipubt
{
static std::set<OpType> SUPPORT_CUT_BATCH_OPS =
{
    OpType_Constant,
    OpType_Convolution,
    OpType_Activation,
    OpType_DepthwiseConv,
    OpType_Eltwise,
    OpType_Pooling,
    OpType_Cast,
    OpType_Mul,
    OpType_Tile,
    OpType_Softmax,
    OpType_Reshape,
    OpType_SquaredDifference,
    OpType_Add,
    OpType_Sub,
    OpType_LayerNorm,
    OpType_Quantize,
    OpType_DeQuantize,
    OpType_Pad,
    OpType_ConvTranspose,
    OpType_ConvTranspose3D,
    OpType_BatchNorm,
    OpType_Sign,
    OpType_Abs,
    OpType_Transpose,
    OpType_Resize,
    OpType_Concat,
    OpType_MatMul,
    OpType_FullyConnected,
    OpType_GridSample,
    OpType_Split,
    OpType_Crop,
    OpType_Reduce,
    OpType_TopK,
    OpType_Gather,
    OpType_ArgMinMax,
    OpType_RMSNorm
};

static void add_spliter(NodePtr node, TensorPtr inp, NodePtrList &split_nodes)
{
    auto g = node->graph();
    NodePtr spliter = Node::create(node->name() + "_" + inp->name() + "_spliter", OpType_InternalSpliter);
    spliter->params()["axis"] = 0;
    spliter->params()["splits"] = {};
    spliter->add_input(inp);

    g->add_node(spliter);
    split_nodes.push_back(spliter);
    ADD_LOCATION_BY_NODE(spliter, node);
}

static NodePtrList apply_split(NodePtr node, std::vector<int> batches)
{
    auto g = node->graph();
    NodePtrList subnodes;
    NodePtrList split_nodes;
    int batch = std::accumulate(batches.begin(), batches.end(), 0);
    int batch_axis = 0;

    if (is_special_eltwise(node) &&
        node->inputs()[0]->shape().dim() != node->inputs()[1]->shape().dim())
    {
        if (node->inputs()[0]->shape().dim() > node->inputs()[1]->shape().dim())
        {
            add_spliter(node, node->inputs()[0], split_nodes);
            split_nodes.push_back(nullptr);
        }
        else
        {
            split_nodes.push_back(nullptr);
            add_spliter(node, node->inputs()[1], split_nodes);
        }
    }
    else if (node->type() == OpType_Gather)
    {
        // gather's input1 are indices, they are independent in batch dim, so add spliter for input1
        split_nodes.push_back(nullptr);
        add_spliter(node, node->inputs()[1], split_nodes);
        batch_axis = node->params()["axis"];
    }
    else
    {
        for (auto inp : node->inputs())
        {
            if (inp->shape().n() == batch)
            {
                add_spliter(node, inp, split_nodes);
            }
            else
            {
                split_nodes.push_back(nullptr);
            }
        }
    }

    NodePtrList concat_nodes;
    for (auto outp : node->outputs())
    {
        auto insert_optype = OpType_Concat;
        auto concat = Node::create(node->name() + "_" + outp->name() + "_concat", insert_optype);
        concat->params()["axis"] = batch_axis;
        concat->add_output(outp);

        g->add_node(concat);
        concat_nodes.push_back(concat);
        ADD_LOCATION_BY_NODE(concat, node);
    }

    int b = 0;
    for (int step : batches)
    {
        auto subnode = Node::create(node->name() + "_subnode_" + std::to_string(b), node->type());
        subnode->set_attr("original_id", node->get_layer_id());
        g->add_node(subnode);
        subnodes.push_back(subnode);
        ADD_LOCATION_BY_NODE(subnode, node);

        // copy gsize
        if (node->attrs().count(Constants::gsize))
            subnode->attrs()[Constants::gsize] = node->attrs()[Constants::gsize];

        // spliter
        for (int i = 0; i < int(node->inputs().size()); i++)
        {
            auto spliter = split_nodes[i];
            if (spliter)
            {
                auto t = std::make_shared<Tensor>(*spliter->inputs()[0]);
                t->shape()[0] = step;
                t->memory_offset()->size = t->memory_offset()->size / batch * step;
                t->set_name(g->get_valid_tensor_name(spliter->name() + "_" + std::to_string(b / step)));
                spliter->params()["splits"].push_back(t->shape()[0]);
                spliter->add_output(t);
                subnode->add_input(t);
            }
            else
            {
                subnode->add_input(node->inputs()[i]);
            }
        }

        // concat
        for (auto concat : concat_nodes)
        {
            auto t = std::make_shared<Tensor>(*concat->outputs()[0]);
            t->set_io_tensor(false);
            t->shape()[batch_axis] = step;
            t->memory_offset()->size = t->memory_offset()->size / batch * step;
            t->set_name(g->get_valid_tensor_name(concat->name() + "_" + std::to_string(b / step)));
            subnode->add_output(t);
            concat->add_input(t);
        }

        // copy flags
        if (node->attrs().count(Constants::keep_isolate_when_scheduling))
            subnode->attrs()[Constants::keep_isolate_when_scheduling] = node->attrs()[Constants::keep_isolate_when_scheduling];

        // copy & fix params
        subnode->params() = node->params();
        tiling_update_params(subnode, "shape", 0, step);
        tiling_update_params(subnode, "crops", 0, 0, step);
        tiling_update_params(subnode, "repeats", 0);

        // copy weights
        subnode->constants() = node->constants();

        if ((subnode->type() == OpType_BatchNorm && subnode->params().count("axis") && subnode->params()["axis"] == 0) ||
            subnode->type() == OpType_Constant)
        {
            tiling_set_constant(node, subnode, "weights", b, step);
            tiling_set_constant(node, subnode, "biases", b, step);
        }

        b += step;
    }

    g->remove_node(node);

    return subnodes;
}

static std::vector<int> step_to_batches(int batch, int step)
{
    std::vector<int> batches((batch + step - 1) / step, step);
    batches.back() -= batches.size() * step - batch;
    return batches;
}

bool check_manual_tile_n(NodePtr n)
{
    if (!n)
    {
        return false;
    }

    auto type = n->type();
    if(!SUPPORT_CUT_BATCH_OPS.count(type))
    {
        return false;
    }

    auto o_shape = n->outputs()[0]->shape();
    if (o_shape.dim() < 1 || o_shape.n() <= 1)
    {
        return false;
    }

    if (type == OpType_Softmax)
    {
        auto axis = n->params()["axis"];
        if (axis == 0)
        {
            // some ops deal with batch axis should not be cut
            return false;
        }
    }

    if (type == OpType_Activation)
    {
        // constant should be split according batch
        if (n->params()["method"] == "PRELU")
        {
            return false;
        }
    }

    if (type == OpType_Concat)
    {
        if (n->params().count("axis") && (int32_t)n->params()["axis"] == 0)
        {
            return false;
        }
    }

    if (type == OpType_Transpose)
    {
        if (n->params().count("perm"))
        {
            std::vector<int> perm = n->params()["perm"];
            if (perm[0] != 0)
            {
                return false;
            }
        }
    }

    if (type == OpType_Gather)
    {
        if (n->params().count("batch_dims") && n->params()["batch_dims"])
        {
            return false;
        }
    }

    return true;
}

bool check_auto_tile_n(NodePtr n)
{
    auto type = n->type();
    if (type == OpType_Sub || type == OpType_Add || type == OpType_Mul)
    {
        bool is_in0_full = n->inputs()[0]->shape().size() > n->inputs()[1]->shape().size();
        auto full_idx = is_in0_full ? 0 : 1;
        int batch = n->inputs()[full_idx]->shape().n();
        auto full_batch_size = n->inputs()[full_idx]->mem_size() / batch;
        if (full_batch_size % 8 != 0)
        {
            return false;
        }
    }

    if (type == OpType_Constant)
    {
        auto child = n->children()[0];
        if (is_special_eltwise(child) &&
            child->outputs()[0]->shape().dim() > n->outputs()[0]->shape().dim())
        {
            return false;
        }
        else if (child->type() != OpType_Quantize) /* for SD only; supports more ops if necessary */
        {
            return false;
        }
    }

    if (type == OpType_Gather)
    {
        if (n->inputs()[1]->shape().n() == 1)
        {
            return false;
        }
    }

    return true;
}

bool check_batched_graph_tile_n(GraphPtr g)
{
    auto arch = aipubt::get_target()->isainfo().type;
    if (arch < aipubt::ISA_ZYv5)
    {
        return false;
    }

    auto graph_batch = g->get_input_tensors()[0]->shape()[0];
    if (graph_batch == 1 || graph_batch > 32)
    {
        return false;
    }

    for (auto n : g->nodes())
    {
        if (!check_manual_tile_n(n) || !check_auto_tile_n(n))
        {
            if (n->type() == OpType_Input ||
                (n->type() == OpType_Constant && is_special_eltwise(n->children()[0])))
            {
                continue;
            }
            else
            {
                return false;
            }
        }

        for (auto t : n->outputs())
        {
            auto local_batch = t->shape().n();
            if (local_batch != graph_batch && !is_special_eltwise(n))
            {
                return false;
            }

            /* AIFF concat only support dims 2 ~ 5 */
            if (t->shape().dim() > 5)
            {
                return false;
            }
        }
    }

    return true;
}

NodePtrList apply_tile_n(NodePtr n, std::vector<int> &batches)
{
    return apply_split(n, batches);
}

void apply_tile_batched_graph(GraphPtr g, std::string tile_cfg, bool dump_tile_cfg)
{
    auto inputs = g->get_input_tensors();
    auto outputs = g->get_output_tensors();
    int batch = inputs[0]->shape()[0];
    auto aiff_num = aipubt::get_target()->aiff_num();
    int step = 1;
    TileSubGraphMap mp_bp_map;
    TileNumMap max_num_map;

    // TODO: always make batch = 2 for X3
    if (2 == aiff_num && batch % 2 == 0 && tile_cfg == "fps_dev")
    {
        // X3 have 2 aiffs, set 2 batch per slice to enable broadcast inside core
        step = 2;
    }

    auto nodes = g->nodes();
    auto batches = step_to_batches(batch, step);
    DEBUG() << "Split batched graph with step: " << step;

    /* dump before split because the original nodes will be removed */
    if (dump_tile_cfg)
    {
        if (outputs.size() == 1)
        {
            auto mp = outputs[0]->node();
            NodePtrList bps;

            for (auto t : inputs)
            {
                for (auto c : t->consumers())
                {
                    bps.push_back(c);
                }
            }

            mp_bp_map[mp] = bps;
            max_num_map[mp] = batches.size();
        }
        else
        {
            for (auto n : nodes)
            {
                if (n->type() == OpType_Input ||
                    (n->type() == OpType_Constant && n->outputs()[0]->shape().n() == 1))
                {
                    continue;
                }

                NodePtrList bps = {n};
                mp_bp_map[n] = bps;
                max_num_map[n] = batches.size();
            }
        }

        utils::JsonObject json;
        dump_cfg_to_json(json, "tile_n", mp_bp_map, max_num_map);
        std::ofstream os("tile_cfg_dump.json");
        if (os.is_open())
        {
            os << json;
            os << std::endl;
            os.close();
        }
    }

    for (auto n : nodes)
    {
        if (n->type() == OpType_Input ||
            (n->type() == OpType_Constant && n->outputs()[0]->shape().n() == 1))
        {
            continue;
        }

        apply_split(n, batches);
    }

    eliminate_crop(g);
    fuse_concat_splits(g);
}

void tile_graph_n_criteria(GraphPtr g, std::function<bool(NodePtr)> criteria, int step)
{
    auto nodes = g->nodes();
    for (auto node : nodes)
    {
        auto n = get_real_node(node);

        /* check the internal condition of tiling module */
        if (!check_manual_tile_n(n))
        {
            continue;
        }

        /* check the user providing condition */
        if (criteria && !criteria(n))
        {
            continue;
        }

        apply_split(n, step_to_batches(n->outputs()[0]->shape().n(), step));
    }
}
}; // namespace aipubt

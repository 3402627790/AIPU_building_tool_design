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
 * @Date: 2023-11-15 18:24:50
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2026-04-08 17:07:56
 */

#include "core/core.h"
#include "core/ops/ops.h"
#include "passes.h"
#include "quantization_transform/common.h"
#include "quantization_transform/ops/no_quantization.h"
#include "quantization_transform/ops/op.h"
#include "transform/passes.h"

namespace qtlibn
{

// step1: lower dequantize to transparent the NoQuantization op
// step2: lift the quantize to transparent the NoQuantization op
// step3: if cur node all input.pnode is dequantize, all output.consumers is quantize, then absorb dequantize and quantize op to cur node
// step4: eliminate the dequantize->quantize pattern

static aipubt::TensorPtr tensor_clone(aipubt::TensorPtr t)
{
    aipubt::TensorPtr new_t = std::make_shared<aipubt::Tensor>(t->node()->graph()->get_valid_tensor_name(t->name()), t->shape(), t->type());
    return new_t;
}

static bool is_no_quantization(aipubt::NodePtr node)
{
    auto quant_op = OpRegistry::get_quant_op(node->type());
    auto op = dynamic_cast<NoQuantization *>(quant_op);
    // if (op == nullptr && node->type() == aipubt::OpType_Concat)
    // {
    //     auto inputs = node->inputs();
    //     auto ip = std::unique(inputs.begin(), inputs.end());
    //     auto uniqued_size = std::distance(inputs.begin(), ip);
    //     return uniqued_size == 1;
    // }
    return op != nullptr;
}

static bool is_quantized_node(aipubt::NodePtr node)
{
    for (auto in_t : node->inputs())
    {
        if (aipubt::Tensor::is_float(in_t->type()))
        {
            return false;
        }
    }
    for (auto out_t : node->outputs())
    {
        if (aipubt::Tensor::is_float(out_t->type()))
        {
            return false;
        }
    }
    for (auto ct : node->constants())
    {
        if (aipubt::Tensor::is_float(ct.second->type()))
        {
            return false;
        }
    }
    return true;
}

static bool absorb_dequantize_quantize_to_op(aipubt::NodePtr node, aipubt::GraphPtr g)
{
    bool quantized = is_quantized_node(node);
    if (quantized)
    // if (quantized || node->type() == aipubt::OpType_LRN)
    {
        return false;
    }
    if (node->type() == aipubt::OpType_Input || node->type() == aipubt::OpType_DeQuantize || node->type() == aipubt::OpType_Quantize)
    {
        return false;
    }

    auto _criteria = [&node](void) -> bool {
        for (auto &in_t : node->inputs())
        {
            // if ((in_t->node()->type() != aipubt::OpType_DeQuantize && aipubt::Tensor::is_float(in_t->type())) || in_t->is_io_tensor())
            if (in_t->node()->type() != aipubt::OpType_DeQuantize || in_t->is_io_tensor())
            {
                return false;
            }
        }
        for (auto &out_t : node->outputs())
        {
            if (out_t->consumers().size() < 1)
            {
                return false;
            }
            for (auto consumer : out_t->consumers())
            {
                if (consumer->type() != aipubt::OpType_Quantize || out_t->is_io_tensor())
                // if ((consumer->type() != aipubt::OpType_Quantize && aipubt::Tensor::is_float(out_t->type())) || out_t->is_io_tensor())
                {
                    return false;
                }
                // consumer outputs's consumers is_io_tensor ....
            }
        }

        /*
            now we want to fuse the fp_constant + quantize to qnn_constant
        */
        if (node->type() != aipubt::OpType_Constant)
        {
            for (auto &ckv : node->constants())
            {
                auto ct = ckv.second;
                if (aipubt::Tensor::is_float(ct->type()))
                {
                    return false;
                }
                // if (ct->quantization().scales().size() == 0 || ct->quantization().offsets().size() ==0)
                // {
                //     return false;
                // }
            }
        }

        return true;
    };

    if (!_criteria())
    {
        return false;
    }
    std::vector<aipubt::NodePtr> del_nodes;
    for (int32_t i = 0; i < (int32_t)node->inputs().size(); i++)
    {
        auto dq_node = node->inputs()[i]->node();
        auto idx = node->remove_input(node->inputs()[i]);
        node->add_input(dq_node->inputs()[0], idx);
        del_nodes.push_back(dq_node);
    }

    for (int32_t o = 0; o < (int32_t)node->outputs().size(); o++)
    {
        if (node->outputs()[o]->consumers().size() == 0)
        {
            continue;
        }
        auto cur_output = node->outputs()[o];
        auto all_cur_output_consumers = cur_output->consumers();
        for (int32_t c = 0; c < (int32_t)cur_output->consumers().size(); c++)
        {
            auto cur_consumer = all_cur_output_consumers[c];
            del_nodes.push_back(cur_consumer);
            if (c == 0)
            {
                node->remove_output(o);
                node->add_output(cur_consumer->outputs()[0], o);
                continue;
            }
            for (auto &c_out_t : cur_consumer->outputs())
            {
                for (auto &c_out_t_c : c_out_t->consumers())
                {
                    auto idx = c_out_t_c->remove_input(c_out_t);
                    c_out_t_c->add_input(node->outputs()[o], idx);
                }
            }
        }
    }

    if (node->type() == aipubt::OpType_Constant)
    {
        auto const_qinfo = node->outputs()[0]->quantization();
        aipubt::ComputeTensor w_ct = node->constants()["weights"];
        node->set_attr("back_weights", node->constants()["weights"]);
        auto unique_w = w_ct.unique();
        auto uq_size = unique_w->shape().size();
        if (uq_size == 1)
        {
            node->set_attr("fused_multiplier", float(unique_w[0]));
        }
        auto out_dtype = node->outputs()[0]->type();
        // tflite quantization function: f = (q - zp) * scale
        aipubt::ComputeTensor scale_ct = aipubt::ComputeTensor(const_qinfo.scales());
        aipubt::ComputeTensor zp_ct = aipubt::ComputeTensor(const_qinfo.offsets());
        auto clip_min = dtype_utils::dtype_to_range_min(out_dtype);
        auto clip_max = dtype_utils::dtype_to_range_max(out_dtype);
        w_ct = w_ct.astype(aipubt::Tensor::FP32) / scale_ct + zp_ct;
        w_ct = w_ct.clamp(clip_min, clip_max).astype(out_dtype);
        node->constants()["weights"] = w_ct;
        node->constants()["weights"]->set_type(out_dtype);
        node->constants()["weights"]->quantization() = const_qinfo;
    }

    for (auto &dn : del_nodes)
    {
        dn->clear_outputs();
        dn->clear_outputs();
        g->remove_node(dn);
    }

    return true;
}

static bool absorb_dequantize_quantize_to_op_for_graph(aipubt::GraphPtr g)
{

    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        auto ret = absorb_dequantize_quantize_to_op(n, g);
        is_changed = is_changed || ret;
    }
    return is_changed;
}
static bool lower_dequantize_with_special_case(aipubt::NodePtr node, aipubt::GraphPtr g)
{
    /* jiar: CSM-360
     *  pattern: - > dequantize ----\
     *                         \     | --> concat ->
     *                          \---/
     *  so concat has same input, both from the dequantize node's output
     */

    auto _crietia = [node](void) -> bool {
        if (node->type() != aipubt::OpType_DeQuantize)
        {
            return false;
        }
        auto consumers = node->outputs()[0]->consumers();

        if (consumers.size() == 0)
        {
            return false;
        }
        auto concat_node = consumers[0];
        for (auto consumer : consumers)
        {
            if (consumer != concat_node || consumer->type() != aipubt::OpType_Concat)
            {
                return false;
            }
        }

        for (auto &inp : concat_node->inputs())
        {
            if (inp->node() != node)
            {
                return false;
            }
        }
        return true;
    };

    if (!_crietia())
    {
        return false;
    }
    INFO() << "qtlib now lower dequantize + concat, which concat has multi same inputs.";
    // INFO() << concat_node << " has multi same inputs.";
    auto dq_input = node->inputs()[0];
    auto concat_node = node->outputs()[0]->consumers()[0];

    for (auto &inp : concat_node->inputs())
    {
        auto idx = concat_node->remove_input(inp);
        concat_node->add_input(dq_input, idx);
    }

    auto concat_out = concat_node->outputs()[0];
    auto new_concat_out = tensor_clone(concat_out);
    new_concat_out->set_type(dq_input->type());
    new_concat_out->quantization() = dq_input->quantization();

    concat_node->clear_outputs();
    concat_node->add_output(new_concat_out);

    node->clear_inputs();
    node->add_input(new_concat_out);
    node->clear_outputs();
    node->add_output(concat_out);

    return true;
}

static bool lower_dequantize(aipubt::NodePtr node, aipubt::GraphPtr g)
{
    if (node->type() != aipubt::OpType_DeQuantize)
    {
        return false;
    }

    bool is_lower_dq = false;
    auto dq_consumers = node->outputs()[0]->consumers();
    std::vector<aipubt::NodePtr> _to_walk_dequantize_node = {node};
    auto _clone_dequantize_to_other_branches = [&_to_walk_dequantize_node, dq_consumers, g](aipubt::NodePtr dq) -> void {
        auto base_input = dq->inputs()[0];
        for (int32_t d = 1; d < (int32_t)dq_consumers.size(); d++)
        {
            auto cur_consumer = dq_consumers[d];
            aipubt::NodePtr new_dq_node = aipubt::Node::create(g->get_valid_node_name("dequantize"), aipubt::OpType_DeQuantize);
            auto new_dq_out = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(base_input->name()), dq->outputs()[0]->shape(), aipubt::Tensor::FP32);
            new_dq_out->quantization() = dq->outputs()[0]->quantization();
            new_dq_node->add_input(base_input);
            new_dq_node->add_output(new_dq_out);
            auto idx = cur_consumer->remove_input(dq->outputs()[0]);
            cur_consumer->add_input(new_dq_out, idx);
            g->add_node(new_dq_node);
            _to_walk_dequantize_node.push_back(new_dq_node);
        }
    };

    if (dq_consumers.size() > 1)
    {
        _clone_dequantize_to_other_branches(node);
    }

    auto _swap_node = [g](aipubt::NodePtr n0, aipubt::NodePtr n1) -> bool {
        // TODO: if follower node has multiple outputs, also do lower dequantize
        if (n0->inputs().size() != n1->inputs().size() ||
            n0->outputs().size() != n1->outputs().size())
        {
            return false;
        }
        // n0: dequantize node, n1: noquantization node
        // -t0-> n0 -t1-> n1 -t2->
        // -t0-> n1 -t3-> n0 -t2->
        auto t0 = n0->inputs()[0];
        auto t2 = n1->outputs()[0];
        auto t3 = tensor_clone(t2);
        t3->set_type(t0->type());
        t3->quantization() = t0->quantization();

        n1->clear_inputs();
        n1->clear_outputs();
        n0->clear_inputs();
        n0->clear_outputs();

        n1->add_input(t0);
        n1->add_output(t3);
        n0->add_input(t3);
        n0->add_output(t2);
        n1->set_attr("is_lower_by_dequantize", true);
        return true;
    };
    for (auto walk_dq_node : _to_walk_dequantize_node)
    {
        if (walk_dq_node->outputs()[0]->is_io_tensor() || walk_dq_node->outputs()[0]->consumers().size() == 0)
        {
            continue;
        }
        // because _clone_dequantize to other branches, so now this walk_dq_node only has one consumer
        auto child_node = walk_dq_node->outputs()[0]->consumers()[0];
        if (is_no_quantization(child_node))
        {
            if (_swap_node(walk_dq_node, child_node))
            {
                is_lower_dq = is_lower_dq || true;
                is_lower_dq = lower_dequantize(walk_dq_node, g) || is_lower_dq;
            }
        }
    }

    return is_lower_dq;
}

static bool lower_dequantize_for_graph(aipubt::GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        is_changed = lower_dequantize_with_special_case(n, g) || is_changed;
        is_changed = lower_dequantize(n, g) || is_changed;
    }
    return is_changed;
}

static bool lift_quantize(aipubt::NodePtr node, aipubt::GraphPtr g)
{
    if (node->type() != aipubt::OpType_Quantize)
    {
        return false;
    }

    if (node->inputs().size() == 0)
    {
        return false;
    }
    // quantize op lifts to transparent the noquantization node
    auto parent_node = node->inputs()[0]->node();
    if (!parent_node || !is_no_quantization(parent_node))
    {
        return false;
    }

    bool is_all_consumers_of_parent_node_have_quantize_op = true;
    auto gt_quantization = node->outputs()[0]->quantization();
    std::vector<aipubt::NodePtr> _all_branch_quantize_op;
    /*
     * if float pooling, we assume that quantized pooling has the same input scale and out scale, not like global pooling, which
     * may have different input and output scale
     *          | Ta
     *       pooling
     *        |   |  Tb    (pooling out tensor has two consumers)
     *       /     \
     * quantize0  quantize1
     *    |  Tc       | Td
     *    n0          n1
     *
     *  Ta, Tb is float dtype, Tc, Td is int dtype
     * criteria: tb is not is_io_tensor, the all consumers of Tb have Quantize child node, Tc.quantization == Td.quantization
     *
     *          |Ta
     *        quantize0
     *          |  Te   (new tensor Te)
     *        pooling
     *         |   |
     *      Tc /    |  Td
     *        n0    n1
     * Ta is float dtype, Te Tc Td is int dtype
     * keep quantize0 op, and other quantize op (like quantize1) would be deleted.
     * create new tensor Te from Tc
     *
     * the above case is assuming that the parent node of quantize op is single input, but it would be multi inputs, so the quantize0
     * op should transparent the parent node and insert one quantize op to each input branch of parent node
     */
    for (auto &p_out : parent_node->outputs())
    {
        if (p_out->is_io_tensor() || p_out->consumers().size() < 1)
        {
            return false;
        }
        auto p_out_consumers = p_out->consumers();
        for (auto &pc : p_out_consumers)
        {
            if (pc->type() != aipubt::OpType_Quantize || gt_quantization != pc->outputs()[0]->quantization())
            {
                is_all_consumers_of_parent_node_have_quantize_op = false;
                break;
            }
            _all_branch_quantize_op.push_back(pc);
        }
    }

    if (!is_all_consumers_of_parent_node_have_quantize_op)
    {
        return false;
    }

    std::vector<aipubt::TensorPtr> _all_quantize_outs;
    for (auto bq : _all_branch_quantize_op)
    {
        _all_quantize_outs.push_back(bq->outputs()[0]);
    }

    // update parent node output
    for (int32_t oidx = 0; oidx < (int32_t)parent_node->outputs().size(); oidx++)
    {
        auto cur_t_consumers = parent_node->outputs()[oidx]->consumers();
        parent_node->remove_output(oidx);
        parent_node->add_output(cur_t_consumers[0]->outputs()[0], oidx);
        cur_t_consumers[0]->clear_outputs();
        for (int32_t cidx = 1; cidx < (int32_t)cur_t_consumers.size(); cidx++)
        {
            auto c_n_cs = cur_t_consumers[cidx]->outputs()[0]->consumers();
            for (auto cncs : c_n_cs)
            {
                auto cncs_inp_idx = cncs->remove_input(cur_t_consumers[cidx]->outputs()[0]);
                cncs->add_input(cur_t_consumers[0]->outputs()[0], cncs_inp_idx);
            }
        }
    }

    bool status = false;
    // lift the quantize node and update the parent node input
    int32_t branch_quantize_op_size = _all_branch_quantize_op.size();
    int32_t input_size_of_parent_node = parent_node->inputs().size();
    // auto ref_quantize_op_out_tensor = _all_branch_quantize_op[0]->outputs()[0];
    auto ref_quantize_op_out_tensor = parent_node->outputs()[0];
    std::vector<aipubt::NodePtr> lifted_quantize_ops;
    for (int32_t ip = 0; ip < input_size_of_parent_node; ip++)
    {
        auto inp_of_parent_node = parent_node->inputs()[ip];

        /* if the input[ip] of parent node is also int dtype,
         * we will not insert quantize to keep the intdtype. like in gather(jira CSM-180),
         *    inp0(data)   inp1(index)
         *      \             /
         *          gather
         *           |
         *         quantize
         * becasue gather is noquantization, quantize will lift the gather, and insert quantiza op in
         * inp0 edge and inp1 edge, but inp1 edge is int32 dtype, if insert quantize op, this edge will be
         * int32--> quantize --> int8, lib not support this quantize op.
         *
         * so if one input edge already is int dtype, we will not insert quantize op.
         */
        if (!aipubt::Tensor::is_float(inp_of_parent_node->type()))
        {
            continue;
        }

        auto lift_quantize_op = aipubt::Node::create(g->get_valid_node_name(parent_node->name()), aipubt::OpType_Quantize);
        auto q_out_tensor = tensor_clone(inp_of_parent_node);
        q_out_tensor->set_type(ref_quantize_op_out_tensor->type());
        q_out_tensor->quantization() = ref_quantize_op_out_tensor->quantization();

        lift_quantize_op->clear_outputs();
        lift_quantize_op->clear_inputs();
        lift_quantize_op->add_input(inp_of_parent_node);
        lift_quantize_op->add_output(q_out_tensor);
        auto xx = parent_node->remove_input(inp_of_parent_node);
        parent_node->add_input(q_out_tensor, xx);
        g->add_node(lift_quantize_op);
        status = lift_quantize(lift_quantize_op, g) || status;
        lifted_quantize_ops.push_back(lift_quantize_op);
    }

    int32_t ssss = 0;
    // if (branch_quantize_op_size > input_size_of_parent_node)
    if (branch_quantize_op_size > ssss)
    {
        // del the redundant quantize op
        //  for (int32_t start_idx = input_size_of_parent_node; start_idx < branch_quantize_op_size; start_idx++)
        for (int32_t start_idx = ssss; start_idx < branch_quantize_op_size; start_idx++)
        {
            auto del_node = _all_branch_quantize_op[start_idx];
            del_node->clear_outputs();
            del_node->clear_inputs();
            g->remove_node(del_node);
        }
    }

    // for (auto &lifted_q : lifted_quantize_ops)
    // {
    //     INFO() << "lift q type = " << lifted_q;
    //     status = lift_quantize(lifted_q, g) || status;
    // }

    return status;
}

static bool lift_quantize_for_graph(aipubt::GraphPtr g)
{
    bool is_changed = false;
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        is_changed = lift_quantize(n, g) || is_changed;
    }
    return is_changed;
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

static bool eliminate_dequantize_quantize(aipubt::GraphPtr g)
{
    // now only fuse dequantize + quantize pattern
    auto _criteria = [](aipubt::NodePtr n0, aipubt::NodePtr n1) -> bool {
        if (n0->outputs()[0]->is_io_tensor())
        {
            return false;
        }
        auto deq_inp_qinfo = n0->inputs()[0]->quantization();
        auto q_out_qinfo = n1->outputs()[0]->quantization();
        if (deq_inp_qinfo != q_out_qinfo)
        {
            return false;
        }
        return true;
    };

    aipubt::GraphMatcherPattern pattern = {{{"0", aipubt::OpType_DeQuantize}, {"1", aipubt::OpType_Quantize}}, {{"0", "1"}}};
    aipubt::GraphMatcher matcher(pattern);
    matcher.set_mode(aipubt::GraphMatcher::MatchMode::StrictMode);
    auto matched = matcher.match(g);
    bool is_changed = false;
    while (matched.size())
    {
        bool is_merged = false;
        for (auto &&m : matched)
        {
            if (!_criteria(m[0], m[1]))
            {
                continue;
            }
            auto in = m[0]->inputs()[0];
            auto out = m[1]->outputs()[0];
            auto m0_parents = m[0]->parents();
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
                for (auto &in_c_n : m[0]->parents())
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
                INFO() << "because dequantize input tensor = " << in->name() << ", quantize output tensor = "
                       << out->name() << " are both the graph input/output tensor, so we will not fuse these two op.";
            }
            g->remove_node(m[0]);
            g->remove_node(m[1]);
            is_changed = true;
            is_merged = true;
        }
        if (!is_merged)
        {
            break;
        }
        matched = matcher.match(g);
    }
    return is_changed;
}

static std::vector<aipubt::NodePtr> _existed_quantization_consumer(std::vector<aipubt::NodePtr> consumers, aipubt::OpType target_optype)
{
    std::vector<aipubt::NodePtr> _quantize_consumers;
    for (auto consumer : consumers)
    {
        if (consumer->type() == target_optype)
        {
            _quantize_consumers.push_back(consumer);
        }
    }
    return _quantize_consumers;
}

static bool insert_quantize_and_dequantize_in_float_edge(aipubt::GraphPtr g)
{
    bool is_changed = false;
    auto _criteria = [](aipubt::TensorPtr t) -> std::pair<bool, aipubt::Quantization> {
        if (!aipubt::Tensor::is_float(t->type()))
        {
            return std::make_pair<bool, aipubt::Quantization>(false, aipubt::Quantization());
        }

        /* check one branch of this tensor whether has quantize, other doesnot have.
        * if true, will insert quantize + dequantize in the other branches
        */
        auto q = t->quantization();
        bool ret = true;
        auto _quantized_consumers = _existed_quantization_consumer(t->consumers(), aipubt::OpType_Quantize);
        if (_quantized_consumers.size() == 0)
        {
            ret = false;
        }
        else
        {
            auto base_qc = _quantized_consumers[0];
            for (int32_t j = 1; j < (int32_t)_quantized_consumers.size(); j++)
            {
                if (base_qc->outputs()[0]->quantization() != _quantized_consumers[j]->outputs()[0]->quantization())
                {
                    ret = false;
                    break;
                }
            }
            if (ret)
            {
                q = base_qc->outputs()[0]->quantization();
            }
        }

        // check special op, like cast ( int--> cast --> fp), or constant (constant -->fp)
        /* jira: CSM-360
         *  cast op: if cast is from int to fp, the output edge can insert quantize + dequantize, and quantizaiton(1.0, 0)
         * constant op: if constant is float op, the output edge can insert quantize + dequantize, and quantization is calculated by constant data
         */
        if (!ret)
        {
            auto op = t->node();
            auto is_out_dtype_fp = aipubt::Tensor::is_float(op->outputs()[0]->type());
            if (op->type() == aipubt::OpType_Cast)
            {
                auto is_in_dtype_fp = aipubt::Tensor::is_float(op->inputs()[0]->type());
                if (!is_in_dtype_fp && is_out_dtype_fp)
                {
                    ret = true;
                    q.set_aipu_dtype(op->inputs()[0]->type());
                    q.set_scales({1.0});
                    q.set_offsets({0});
                }
            }
            else if (op->type() == aipubt::OpType_Constant)
            {
                if (is_out_dtype_fp && op->outputs()[0]->consumers().size() == 1)
                {
                    ret = true;
                    std::vector<float> mins;
                    std::vector<float> maxs;
                    calibrate_utils::statistic_max_min_from_float_tensor(op->constants()["weights"], maxs, mins);
                    q.set_mins(mins);
                    q.set_maxs(maxs);
                    std::vector<float> scales;
                    std::vector<int32_t> zps;
                    bool is_signed = true;
                    quant_utils::get_quant_params_from_statistic_value(maxs, mins, q.bits(), is_signed, true, false, scales, zps);
                    q.set_aipu_dtype(dtype_utils::bits_to_dtype(q.bits(), is_signed));
                    for (uint32_t i = 0; i < scales.size(); i++)
                    {
                        scales[i] = 1.0 / scales[i];
                        zps[i] = 0 - zps[i];
                    }
                    q.set_scales(scales);
                    q.set_offsets(zps);
                }
            }
        }
        return std::make_pair(ret, q);
    };

    auto _insert_func = [g, _criteria](aipubt::NodePtr n) -> bool {
        bool inserted = false;
        for (auto in_t : n->inputs())
        {
            if (!aipubt::Tensor::is_float(in_t->type()) && n->type() != aipubt::OpType_Cast)
            {
                return inserted;
            }
        }
        for (auto out_t : n->outputs())
        {
            auto ret = _criteria(out_t);
            if (!ret.first)
            {
                continue;
            }
            auto qinfo = ret.second;
            for (auto c : out_t->consumers())
            {
                if (c->type() == aipubt::OpType_Quantize)
                {
                    continue;
                }

                auto new_quantize_node = aipubt::Node::create(g->get_valid_node_name(n->name()), aipubt::OpType_Quantize);
                auto quantize_out = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(out_t->name()), out_t->shape(), qinfo.aipu_dtype());

                new_quantize_node->add_input(out_t);
                new_quantize_node->add_output(quantize_out);
                new_quantize_node->outputs()[0]->quantization() = qinfo;
                g->add_node(new_quantize_node);

                auto dequantize_node = aipubt::Node::create(g->get_valid_node_name(n->name()), aipubt::OpType_DeQuantize);
                auto dequantize_out = std::make_shared<aipubt::Tensor>(g->get_valid_tensor_name(out_t->name()), out_t->shape(), out_t->type());
                dequantize_node->add_input(quantize_out);
                dequantize_node->add_output(dequantize_out);
                dequantize_out->quantization() = out_t->quantization();

                auto idx = c->remove_input(out_t);
                c->add_input(dequantize_out, idx);

                g->add_node(dequantize_node);
                inserted = true;
            }
        }
        return inserted;
    };

    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        is_changed = _insert_func(n) || is_changed;
    }
    return is_changed;
}

static bool _is_lower_by_dequantize(aipubt::NodePtr cur_node)
{
    if (cur_node->attrs().count("is_lower_by_dequantize") && cur_node->get_attr<bool>("is_lower_by_dequantize"))
    {
        return true;
    }
    return false;
}
static bool _revert_walk_through_lower_dequantize(aipubt::NodePtr node, std::vector<aipubt::NodePtr> &walk_nodes)
{
    auto consumers = node->outputs()[0]->consumers();
    bool ret = true;
    if (consumers.size() > 1)
    {
        WARN() << " when _revert lower dequantize, the node " << node << " has multi consumers";
    }
    for (auto cs : consumers)
    {
        if (cs->type() != aipubt::OpType_DeQuantize)
        {
            if (is_no_quantization(cs) && _is_lower_by_dequantize(cs))
            {
                // walk_nodes.push_back(cs);
                ret = _revert_walk_through_lower_dequantize(cs, walk_nodes) && ret;
            }
            else
            {
                return false;
            }
        }
        else
        {
            walk_nodes.push_back(cs);
            return true;
        }
    }
    return ret;
}
static bool revert_matched_subgraph(aipubt::GraphPtr g)
{
    bool is_changed = false;
    auto _merge_to_one = [g](aipubt::NodePtr one_n, aipubt::NodePtr keeped_n, aipubt::OpType target_optype) -> bool {
        auto consumers = one_n->outputs()[0]->consumers();
        if (consumers.size() <= 0)
        {
            return true;
        }

        if (!aipubt::contains(consumers, keeped_n))
        {
            WARN() << "keeped_node = " << keeped_n << " not in parent node = " << one_n << " consumers.";
            return false;
        }

        for (auto consumer : consumers)
        {
            if (consumer->type() != target_optype)
            {
                return false;
            }
            if (keeped_n->outputs()[0]->quantization() != consumer->outputs()[0]->quantization())
            {
                return false;
            }
        }

        auto keeped_out = keeped_n->outputs()[0];
        auto cons_size = consumers.size();
        std::vector<aipubt::NodePtr> _needed_delete_nodes = {};
        for (uint32_t j = 0; j < cons_size; j++)
        {
            auto cur_deq_consumer = consumers[j];
            if (cur_deq_consumer == keeped_n)
            {
                continue;
            }
            auto cur_deq_out = cur_deq_consumer->outputs()[0];
            auto cur_deq_out_consumers = cur_deq_out->consumers();
            for (auto &one_cur_deq_out_consumer : cur_deq_out_consumers)
            {
                auto idx = one_cur_deq_out_consumer->remove_input(cur_deq_out);
                one_cur_deq_out_consumer->add_input(keeped_out, idx);
            }
            _needed_delete_nodes.push_back(cur_deq_consumer);
        }

        for (auto &del_node : _needed_delete_nodes)
        {
            del_node->clear_inputs();
            del_node->clear_outputs();
            g->remove_node(del_node);
        }
        return true;
    };
    auto revert_matched_subgraph_criteria = [](aipubt::NodePtr n, std::vector<aipubt::NodePtr> &walk_nodes) -> bool {
        if (n->type() != aipubt::OpType_Constant)
        {
            return false;
        }
        if (!n->attrs().count("back_weights"))
        {
            return false;
        }
        return _revert_walk_through_lower_dequantize(n, walk_nodes);
    };
    auto _swap_node = [g, &_merge_to_one](aipubt::NodePtr deq_p_n, aipubt::NodePtr deq_n) -> bool {
        // deq_p_n: dequanize parent node, deq_n: cur dequantize node

        if (deq_p_n->type() == aipubt::OpType_Constant)
        {
            return false;
        }

        // TODO: if follower node has multiple outputs, also do lower dequantize
        if (deq_p_n->inputs().size() != deq_n->inputs().size() ||
            deq_p_n->outputs().size() != deq_n->outputs().size())
        {
            return false;
        }

        auto p_out = deq_p_n->outputs()[0];
        auto consumers = p_out->consumers();
        for (auto &consumer : consumers)
        {
            if (consumer->type() != aipubt::OpType_DeQuantize)
            {
                return false;
            }
        }

        if (consumers.size() > 1)
        {
            /* like:      /-> dequantize
             *    slice --|-> dequantize
             *            \-> dequantize
             * so we should merge the multi dequantize to one dequantize, and use one dequantize to swap
             */
            _merge_to_one(deq_p_n, deq_n, aipubt::OpType_DeQuantize);
        }

        auto n0 = deq_p_n;
        auto n1 = deq_n;
        // n0: noquantization node, n1: dequantize node
        // -t0-> n0 -t1-> n1 -t2->
        // -t0-> n1 -t3-> n0 -t2->
        auto t0 = n0->inputs()[0];
        auto t2 = n1->outputs()[0];
        auto t3 = tensor_clone(t0);
        t3->set_type(t2->type());
        t3->quantization() = t2->quantization();

        n1->clear_inputs();
        n1->clear_outputs();
        n0->clear_inputs();
        n0->clear_outputs();

        n1->add_input(t0);
        n1->add_output(t3);
        n0->add_input(t3);
        n0->add_output(t2);
        return true;
    };
    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        std::vector<aipubt::NodePtr> walk_nodes;
        if (revert_matched_subgraph_criteria(n, walk_nodes))
        {
            auto walkn_size = walk_nodes.size();
            if (walkn_size)
            {
                // all saved walk node are dequantize, we recored the end node, and then back-tranverse
                for (size_t i = 0; i < walkn_size; i++)
                {
                    auto dq = walk_nodes[i];
                    bool ret = true;
                    auto pn = dq->inputs()[0]->node();
                    do
                    {
                        // forward-tranverse: constant -> reshape -> slice -> dequantize
                        // back-tranverse: dequantize -> slice -> reshape -> constant
                        ret = _swap_node(pn, dq);
                        pn = dq->inputs()[0]->node();
                        ret = pn->type() != aipubt::OpType_Constant && ret;
                    } while (ret);
                }
                auto const_consumers = n->outputs()[0]->consumers();
                auto keeped_n = const_consumers[0];
                if (const_consumers.size() > 1)
                {
                    _merge_to_one(n, keeped_n, aipubt::OpType_DeQuantize);
                }
                auto fp_weight = n->get_attr("back_weights");
                auto fp_out = keeped_n->outputs()[0];

                n->remove_output(0);
                n->add_output(fp_out);
                n->constants()["weights"] = fp_weight;
                n->constants()["weights"]->set_type(fp_out->type());
                keeped_n->clear_inputs();
                keeped_n->clear_outputs();
                g->remove_node(keeped_n);
                is_changed = true;
            }
        }
    }

    // remove the backup weights in constant node
    nodes = g->nodes();
    for (auto n : nodes)
    {
        if (n->type() == aipubt::OpType_Constant && n->attrs().count("back_weights"))
        {
            n->erase_attr("back_weights");
        }
    }
    return is_changed;
}

// If a tensor is followed by two identical dequantizes, they are merged into one.
//  tensor -> dequantize1
//     |                        >>> dequantize2
//     -----> dequantize2
//
static void eliminate_extra_dequantize(aipubt::GraphPtr g)
{
    auto is_same_dequantize = [](aipubt::NodePtr n1, aipubt::NodePtr n2) {
        if (n2->type() != aipubt::OpType_DeQuantize)
        {
            return false;
        }
        if (n1->inputs()[0] != n2->inputs()[0])
        {
            return false;
        }
        auto n1_qs = n1->params()["quantize_scale"];
        auto n1_qz = n1->params()["quantize_zp"];
        auto n2_qs = n2->params()["quantize_scale"];
        auto n2_qz = n2->params()["quantize_zp"];
        if (n1_qs != n2_qs || n1_qz != n2_qz)
        {
            return false;
        }
        return true;
    };

    auto criteria = [is_same_dequantize](aipubt::NodePtr node) {
        if (node->type() != aipubt::OpType_DeQuantize)
        {
            return false;
        }
        auto input = node->inputs()[0];
        if (input->consumers().size() < 2)
        {
            return false;
        }
        bool ret = true;
        for (auto c : input->consumers())
        {
            if (c == node)
            {
                continue;
            }
            ret = is_same_dequantize(node, c) && ret;
            if (!ret)
            {
                break;
            }
        }
        return ret;
    };

    auto nodes = g->nodes();
    for (auto n : nodes)
    {
        if (!criteria(n))
        {
            continue;
        }
        auto input = n->inputs()[0];
        if (n->outputs()[0]->is_io_tensor())
        {
            continue;
        }

        aipubt::NodePtr dq = nullptr;
        for (auto c : input->consumers())
        {
            if (c == n)
                continue;
            if (is_same_dequantize(n, c))
            {
                dq = c;
                break;
            }
        }
        if (dq == nullptr)
        {
            continue;
        }
        for (auto c : n->outputs()[0]->consumers())
        {
            c->replace_input_by(n->outputs()[0], dq->outputs()[0]);
        }
        g->remove_node(n, true);
    }
    return;
}

static void revert_unsupported_woq(aipubt::GraphPtr g)
{
    auto _criteria = [](aipubt::NodePtr n) -> bool {
        if (n->type() != aipubt::OpType_Convolution)
        {
            return false;
        }
        auto inp = n->inputs()[0];
        auto out = n->outputs()[0];
        auto wt = n->constants()["weights"];
        auto is_inp_float = aipubt::Tensor::is_float(inp->type());
        auto is_out_float = aipubt::Tensor::is_float(out->type());
        auto is_wt_float = aipubt::Tensor::is_float(wt->type());
        if (is_inp_float && is_out_float && !is_wt_float)
        {
            auto wscale = wt->quantization().scales();
            auto wzp = wt->quantization().offsets();
            if (wscale.size() == 1 && wzp.size() == 1)
            {
                return true;
            }
        }
        return false;
    };
    auto nodes = g->nodes();
    for (auto &node : nodes)
    {
        if (!_criteria(node))
        {
            continue;
        }

        auto wt = node->constants()["weights"];
        aipubt::ComputeTensor wct = wt;
        wct = wct.astype(aipubt::Tensor::FP32);
        auto wtscale = wt->quantization().scales();
        auto wtzp = wt->quantization().offsets();
        auto deq_wct = (wct - wtzp) * wtscale;
        node->constants()["weights"] = deq_wct;
        node->set_attr("weight_quantization_scale", std::vector<float>(wtscale));
        node->set_attr("weight_quantization_zp", std::vector<float>(wtzp));

        auto bt = node->constants()["biases"];
        if (!aipubt::Tensor::is_float(bt->type()))
        {
            aipubt::ComputeTensor bct = bt;
            bct = bct.astype(aipubt::Tensor::FP32);
            auto btscale = bt->quantization().scales();
            auto btzp = bt->quantization().offsets();
            auto deq_bct = (bct - btzp) * btscale;
            node->constants()["biases"] = deq_bct;
        }
    }
}

void simplify_dequantize_quantize(aipubt::GraphPtr g)
{
    bool is_simplify = true;
    if (g->nodes()[0]->attrs().count("compat_quantized_model_simplify_dequantize_quantize") && !(bool)g->nodes()[0]->get_attr("compat_quantized_model_simplify_dequantize_quantize"))
    {
        INFO() << "Disable simplify dequantize quantize op for qat model";
        is_simplify = false;
    }
    if (is_simplify)
    {
        insert_quantize_and_dequantize_in_float_edge(g);
        lower_dequantize_for_graph(g);
        lift_quantize_for_graph(g);
        absorb_dequantize_quantize_to_op_for_graph(g);
        revert_matched_subgraph(g);
        eliminate_dequantize_quantize(g);
    }
    eliminate_extra_dequantize(g);
    revert_unsupported_woq(g);

    //qtlib will do fuse_quantize_and_dequantize, gsim does it
    // fuse_quantize_and_dequantize(g);
}

} // namespace qtlibn

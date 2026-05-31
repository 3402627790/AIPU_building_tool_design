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
 * @Date: 2024-10-28 22:29:05
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-11-05 10:08:30
 */

#include "quantization_transform/insertop.h"
#include "utils/utils.h"

namespace qtlibn
{

static aipubt::TensorPtr tensor_clone(aipubt::TensorPtr t)
{
    aipubt::TensorPtr new_t = std::make_shared<aipubt::Tensor>(t->node()->graph()->get_valid_tensor_name(t->name() + std::to_string(t->node()->graph()->nodes().size())), t->shape(), t->type());
    return new_t;
}

void BaseInsertOp::insert(aipubt::GraphPtr g)
{
}

bool InsertCastOp::is_matched(dtypes_list &in_dtypes, dtypes_list &out_dtypes, std::vector<aipubt::Op_Dtype> &lib_spec_dtypes)
{
    for (auto &lib_dtype : lib_spec_dtypes)
    {
        auto lib_in_dtypes = lib_dtype.in_dtypes;
        auto lib_out_dtypes = lib_dtype.out_dtypes;
        if (lib_in_dtypes == in_dtypes && lib_out_dtypes == out_dtypes)
        {
            return true;
        }
    }
    return false;
}

InsertCastOp::Status InsertCastOp::criteria(aipubt::NodePtr n)
{
    std::vector<aipubt::OpType> _skip_ops = {aipubt::OpType_Concat, aipubt::OpType_Reshape, aipubt::OpType_Input, aipubt::OpType_Constant};
    if (aipubt::contains(_skip_ops, n->type()))
    {
        return Status::SKIP;
    }
    std::vector<aipubt::Op_Dtype> lib_dtype_specs;
    if (m_cache_dtypes.count(n->type()))
    {
        lib_dtype_specs = m_cache_dtypes[n->type()];
    }
    else
    {
        lib_dtype_specs = aipubt::Op_Dtypes::get_dtypes(n->type());
        if (lib_dtype_specs.size() == 0)
        {
            WARN() << n << " get_dtypes from lib spec failed.";
            return Status::SKIP;
        }
        m_cache_dtypes.insert({n->type(), lib_dtype_specs});
    }

    std::vector<aipubt::Tensor::Dtype> ir_in_dtypes;
    std::vector<aipubt::Tensor::Dtype> ir_out_dtypes;
    for (auto inpt : n->inputs())
    {
        ir_in_dtypes.push_back(inpt->type());
    }
    for (auto out : n->outputs())
    {
        ir_out_dtypes.push_back(out->type());
    }

    bool has_output_dtype = false;
    for (auto &lib_dtype : lib_dtype_specs)
    {
        auto lib_in_dtypes = lib_dtype.in_dtypes;
        auto lib_out_dtypes = lib_dtype.out_dtypes;
        if (lib_in_dtypes.size() != ir_in_dtypes.size() || lib_out_dtypes.size() != ir_out_dtypes.size())
        {
            continue;
        }

        if (lib_in_dtypes == ir_in_dtypes && lib_out_dtypes == ir_out_dtypes)
        {
            return Status::MATCHED;
        }

        if (lib_out_dtypes == ir_out_dtypes)
        {
            has_output_dtype = true;
        }
    }

    if (!has_output_dtype)
    {
        std::string trigger_float_op = n->get_attr<aipubt::NodeParamValue>(Constants::trigger_float_op);
        bool unquantifiable = n->params()["unquantifiable"];
        if (trigger_float_op != std::string("disable") && unquantifiable)
        {
            // DEBUG() << "=== trigger_float_op = " << trigger_float_op << ". its unquantifiable = " <<unquantifiable;
            bool output_has_int = false;
            for (uint32_t i = 0; i < n->outputs().size(); i++)
            {
                if (!aipubt::Tensor::is_float(n->outputs()[i]->type()))
                {
                    output_has_int = true;
                    break;
                }
            }
            // bool input_has_int = false;
            // for (uint32_t i = 0; i < n->inputs().size(); i++)
            // {
            //     if (!aipubt::Tensor::is_float(n->inputs()[i]->type()))
            //     {
            //         output_has_int = true;
            //         break;
            //     }
            // }
            // if (output_has_int && input_has_int)
            if (output_has_int)
            {
                return Status::OUTPUT_UNMATCHED;
            }
        }
        return Status::SKIP;
    }

    return Status::UNMATCHED;
}

std::vector<dtypes_list> InsertCastOp::_match_algo_for_output_dtype(dtypes_list &cur_in_dtypes,
                                                                    dtypes_list &cur_out_dtypes,
                                                                    std::vector<aipubt::Op_Dtype> &lib_spec_dtypes,
                                                                    const aipubt::Tensor::Dtype target_dtype)
{
    auto _get_lower_dtypes = [](aipubt::Tensor::Dtype dtype) -> dtypes_list {
        dtypes_list lower_dtypes;
        std::vector<int32_t> candidate_bits;
        auto bits = aipubt::Tensor::dtype_bits(dtype);
        bool signed_f = aipubt::Tensor::is_signed(dtype);
        if (!signed_f)
        {
            candidate_bits.push_back(bits);
        }

        while (bits > 8)
        {
            bits = bits / 2;
            candidate_bits.push_back(bits);
        }
        for (auto &can_bits : candidate_bits)
        {
            lower_dtypes.push_back(dtype_utils::bits_to_dtype(can_bits, true));
            lower_dtypes.push_back(dtype_utils::bits_to_dtype(can_bits, false));
        }
        return lower_dtypes;
    };

    std::vector<uint32_t> o_int_index;
    std::vector<uint32_t> i_int_index;
    for (uint32_t i = 0; i < cur_in_dtypes.size(); i++)
    {
        if (!aipubt::Tensor::is_float(cur_in_dtypes[i]))
        {
            i_int_index.push_back(i);
        }
    }
    for (uint32_t i = 0; i < cur_out_dtypes.size(); i++)
    {
        if (!aipubt::Tensor::is_float(cur_out_dtypes[i]))
        {
            o_int_index.push_back(i);
        }
    }
    std::vector<dtypes_list> candidate_output_dtypes;
    for (auto &int_idx : o_int_index)
    {
        auto lower_dtypes = _get_lower_dtypes(cur_out_dtypes[int_idx]);
        candidate_output_dtypes.push_back(lower_dtypes);
    }
    auto cartesian_product_result = math_utils::cartesian_product(candidate_output_dtypes);
    bool matched = false;

    // transverse the possible output integer dtype, if matched, indicate the ir_in_dtypes is ok, and
    // only insert output cast. if not matched, would be get the best in and out dtypes for casting.
    for (auto &out_dtypes : cartesian_product_result)
    {
        dtypes_list tmp_out_dtypes(cur_out_dtypes.begin(), cur_out_dtypes.end());
        for (uint32_t t = 0; t < out_dtypes.size(); t++)
        {
            tmp_out_dtypes[o_int_index[t]] = out_dtypes[t];
        }
        matched = is_matched(cur_in_dtypes, tmp_out_dtypes, lib_spec_dtypes);
        if (matched)
        {
            // matched
            // INFO() << "matchend";
            return std::vector<dtypes_list>{cur_in_dtypes, tmp_out_dtypes};
        }
    }

    // not matched
    auto _has_float_and_not_target_type = [](aipubt::Op_Dtype odtype, aipubt::Tensor::Dtype td) -> bool {
        bool tmp_flag = false;
        for (auto &spec : {odtype.in_dtypes, odtype.out_dtypes})
        {
            for (auto &sdt : spec)
            {
                tmp_flag = aipubt::Tensor::is_float(sdt) && sdt != td;
                if (tmp_flag)
                {
                    break;
                }
            }
            if (tmp_flag)
            {
                break;
            }
        }
        return tmp_flag;
    };

    // INFO() << "unmatched";

    std::vector<dtypes_list> best_candidate;
    uint32_t best_cbits = 0;
    for (auto &lib_spec : lib_spec_dtypes)
    {
        // input and output has float dtype, but not match current lib spec, like
        // input and output has bfloat16 dtype, but current lib spec is float16 dtype
        if (lib_spec.in_dtypes.size() != cur_in_dtypes.size() || lib_spec.out_dtypes.size() != cur_out_dtypes.size())
        {
            continue;
        }
        if (_has_float_and_not_target_type(lib_spec, target_dtype))
        {
            continue;
        }
        uint32_t cbits = 0;
        for (auto &int_index : o_int_index)
        {
            if (int_index >= lib_spec.out_dtypes.size())
            {
                continue;
            }
            auto cur_int_odtype = lib_spec.out_dtypes[int_index];
            if ((!aipubt::Tensor::is_float(cur_int_odtype)) &&
                !(aipubt::Tensor::is_signed(cur_out_dtypes[int_index]) && !aipubt::Tensor::is_signed(cur_int_odtype)))
            {
                cbits += aipubt::Tensor::dtype_bits(cur_int_odtype);
            }
        }

        for (auto &int_index : i_int_index)
        {
            if (int_index >= lib_spec.in_dtypes.size())
            {
                continue;
            }
            auto cur_int_idtype = lib_spec.in_dtypes[int_index];
            if ((!aipubt::Tensor::is_float(cur_int_idtype)) &&
                !(aipubt::Tensor::is_signed(cur_in_dtypes[int_index]) && !aipubt::Tensor::is_signed(cur_int_idtype)))
            {
                cbits += aipubt::Tensor::dtype_bits(cur_int_idtype);
            }
        }

        if (cbits > best_cbits)
        {
            best_cbits = cbits;
            best_candidate.clear();
            best_candidate.push_back(lib_spec.in_dtypes);
            best_candidate.push_back(lib_spec.out_dtypes);
        }
    }

    return best_candidate;
}

dtypes_list InsertCastOp::_match_algo(dtypes_list &cur_dtypes, std::vector<dtypes_list> &dtypes_pool)
{
    auto _get_bits = [](dtypes_list &dtypes, std::vector<int32_t> &bits) {
        for (auto &d : dtypes)
        {
            bits.push_back(aipubt::Tensor::dtype_bits(d));
        }
        // std::transform(dtypes.begin(), dtypes.end(), bits.begin(), [](aipubt::Tensor::Dtype dt)->int32_t{return aipubt::Tensor::dtype_bits(dt);});
    };
    auto _diff = [](aipubt::Tensor::Dtype dt0, aipubt::Tensor::Dtype dt1) {
        return int32_t(aipubt::Tensor::is_signed(dt0) xor aipubt::Tensor::is_signed(dt1));
    };
    auto _get_sign_diff = [&_diff](dtypes_list &dl0, dtypes_list &dl1, std::vector<int32_t> &diff) {
        for (uint32_t m = 0; m < dl0.size(); m++)
        {
            diff.push_back(_diff(dl0[m], dl1[m]));
        }
        // std::transform(dl0.cbegin(),dl0.cend(), dl1.cbegin(), diff.begin(), _diff);
    };

    auto _add = [](std::vector<int32_t> &b, std::vector<int32_t> &d) {
        std::transform(b.cbegin(), b.cend(), d.begin(), d.begin(), [](int32_t b0, int32_t b1) { return b0 + b1; });
    };
    auto _sub = [](std::vector<int32_t> &b, std::vector<int32_t> &d) {
        std::transform(b.cbegin(), b.cend(), d.begin(), d.begin(), [](int32_t b0, int32_t b1) { return b0 - b1; });
    };

    std::vector<int32_t> cur_bits;
    _get_bits(cur_dtypes, cur_bits);
    dtypes_list best_dtype;
    int32_t best_score = -32768;
    for (auto &dtype_spec : dtypes_pool)
    {
        std::vector<int32_t> spec_bits;
        _get_bits(dtype_spec, spec_bits);

        std::vector<int32_t> sign_diff;
        _get_sign_diff(cur_dtypes, dtype_spec, sign_diff);

        _add(cur_bits, sign_diff);
        _sub(spec_bits, sign_diff);

        auto cur_score = std::accumulate(sign_diff.begin(), sign_diff.end(), 0);
        if (std::abs(cur_score) < std::abs(best_score))
        {
            best_score = cur_score;
            best_dtype = dtype_spec;
        }
    }
    if (best_dtype.size() == 0)
    {
        WARN() << "not found the best spec dtype when insert cast op";
        return dtypes_pool[0];
    }
    return best_dtype;
}

void InsertCastOp::_insert_for_one_node(aipubt::NodePtr n)
{
    auto ret = criteria(n);
    if (ret == Status::SKIP || ret == Status::MATCHED)
    {
        return;
    }

    INFO() << "[insertcatop] begin to insert cast op for node: " << n;

    std::vector<aipubt::Op_Dtype> lib_dtype_specs = m_cache_dtypes[n->type()];
    std::vector<aipubt::Tensor::Dtype> ir_in_dtypes;
    std::vector<aipubt::Tensor::Dtype> ir_out_dtypes;
    bool is_fp_dtype_in_ir_dtype = false;
    for (auto inpt : n->inputs())
    {
        ir_in_dtypes.push_back(inpt->type());
        is_fp_dtype_in_ir_dtype = aipubt::Tensor::is_float(inpt->type()) || is_fp_dtype_in_ir_dtype;
    }
    for (auto out : n->outputs())
    {
        ir_out_dtypes.push_back(out->type());
    }

    if (ret == Status::OUTPUT_UNMATCHED)
    {
        std::string trigger_float_op = n->get_attr<aipubt::NodeParamValue>(Constants::trigger_float_op);
        auto best_in_out_dtypes = _match_algo_for_output_dtype(ir_in_dtypes, ir_out_dtypes, lib_dtype_specs, aipubt::Tensor::string_to_dtype(trigger_float_op));
        auto best_in_dtypes = best_in_out_dtypes[0];
        auto best_out_dtypes = best_in_out_dtypes[1];

        if (ir_in_dtypes.size() != best_in_dtypes.size())
        {
            WARN() << "[OUTPUT_UNMATCHED]the best_in_dtypes.size = " << best_in_dtypes.size()
                   << " is not matched with ir_in_dtypes.size=" << ir_in_dtypes.size() << ", so return.";
            return;
        }
        INFO() << n << "[OUTPUT_UNMATCHED] output_dtypes un-matched lib'spec, qtlib will insert cast op in in tensor edge and output tensor edge.";
        if (ir_in_dtypes != best_in_dtypes)
        {
            for (uint32_t o = 0; o < ir_in_dtypes.size(); o++)
            {
                if (ir_in_dtypes[o] != best_in_dtypes[o])
                {
                    auto inpt = n->inputs()[o];
                    auto cast_out = std::make_shared<aipubt::Tensor>(n->graph()->get_valid_tensor_name(inpt->name() + std::to_string(n->graph()->nodes().size())), inpt->shape(), best_in_dtypes[o]);
                    auto cast_node = aipubt::Node::create(n->graph()->get_valid_node_name(n->name() + std::to_string(n->graph()->nodes().size())), aipubt::OpType_Cast);
                    cast_node->params()["ignore_scale_zp"] = true;
                    cast_node->params()["clip_mode"] = "TRUNCATION";
                    cast_node->params()["to_dtype"] = best_in_dtypes[o];
                    cast_node->params()["unquantifiable"] = true;
                    cast_out->quantization() = inpt->quantization();

                    cast_node->add_input(inpt);
                    cast_node->add_output(cast_out);
                    int32_t idx = n->remove_input(inpt);
                    n->add_input(cast_out, idx);
                    n->graph()->add_node(cast_node);
                    DEBUG() << "[OUTPUT_UNMATCHED] now insert cast op at node: " << n << "'s input idx = " << o << " and insert cast node :" << cast_node;
                }
            }
        }
        if (ir_out_dtypes != best_out_dtypes)
        {
            for (uint32_t o = 0; o < ir_out_dtypes.size(); o++)
            {
                if (ir_out_dtypes[o] != best_out_dtypes[o])
                {
                    auto outt = n->outputs()[o];
                    if (outt->consumers().size() == 0 && !outt->is_io_tensor())
                    {
                        // if no consumer and not io tensor, directly change the dtype
                        n->outputs()[o]->set_type(best_out_dtypes[o]);
                    }
                    else
                    {
                        auto consumers = outt->consumers();
                        auto cast_node = aipubt::Node::create(n->graph()->get_valid_node_name(n->graph()->name() + n->name() + std::to_string(n->graph()->nodes().size())), aipubt::OpType_Cast);
                        // cast_node->set_name(cast_node->name() + "_node_output_dtype_samuel" + std::to_string(o));
                        cast_node->params()["ignore_scale_zp"] = true;
                        cast_node->params()["clip_mode"] = "TRUNCATION";
                        cast_node->params()["to_dtype"] = best_out_dtypes[o];
                        cast_node->params()["unquantifiable"] = true;

                        auto cast_out = qtlibn::tensor_clone(outt);
                        // cast_out->set_name(cast_out->name() + "_output_dtype_samuel" + std::to_string(o));
                        cast_out->quantization() = outt->quantization();
                        outt->set_type(best_out_dtypes[o]);
                        cast_node->add_input(outt);
                        cast_node->add_output(cast_out);

                        if (outt->is_io_tensor())
                        {
                            cast_out->set_io_tensor(false);
                        }
                        for (auto &consumer : consumers)
                        {
                            int32_t idx = consumer->remove_input(outt);
                            consumer->add_input(cast_out, idx);
                        }
                        n->graph()->add_node(cast_node);
                        DEBUG() << "[OUTPUT_UNMATCHED] now insert cast op at node: " << n << "'s output idx = " << o << " and insert cast node :" << cast_node;
                    }
                }
            }
        }

        return;
    }

    INFO() << "[UNMATCHED]in_dtypes of node(" << n << ") un-matched lib'spec, begin to insert cast op process.";

    std::vector<std::vector<aipubt::Tensor::Dtype>> candidate_lib_in_dtypes;
    std::vector<std::vector<aipubt::Tensor::Dtype>> matched_candidate_lib_in_dtypes;
    for (auto &lib_dtype : lib_dtype_specs)
    {
        auto lib_in_dtypes = lib_dtype.in_dtypes;
        auto lib_out_dtypes = lib_dtype.out_dtypes;
        if (lib_out_dtypes == ir_out_dtypes)
        {
            bool is_fp_dtype_in_lib_spec_dtypes = false;
            for (auto &lib_dt : lib_in_dtypes)
            {
                is_fp_dtype_in_lib_spec_dtypes = aipubt::Tensor::is_float(lib_dt) || is_fp_dtype_in_lib_spec_dtypes;
            }
            if (!(is_fp_dtype_in_lib_spec_dtypes xor is_fp_dtype_in_ir_dtype))
            {
                candidate_lib_in_dtypes.push_back(lib_in_dtypes);
            }
        }
    }

    if (candidate_lib_in_dtypes.size() == 0)
    {
        WARN() << "[UNMATCHED] node(=" << n << ") does not find candidate lib in dtypes, will not insert cast op";
        return;
    }

    //TODO: eliminate float dtype in lib specs if all in dtyps is integer
    for (uint32_t i = 0; i < ir_in_dtypes.size(); i++)
    {
        auto ir_dtype = ir_in_dtypes[i];
        for (auto &lib_dtype : candidate_lib_in_dtypes)
        {
            if (lib_dtype[i] == ir_dtype)
            {
                matched_candidate_lib_in_dtypes.push_back(lib_dtype);
            }
        }
    }

    if (matched_candidate_lib_in_dtypes.size() == 0)
    {
        DEBUG() << n << " all input edge should insert cast in [UNMATCHED] process.";
        matched_candidate_lib_in_dtypes = candidate_lib_in_dtypes;
    }

    auto best_in_dtypes = _match_algo(ir_in_dtypes, matched_candidate_lib_in_dtypes);

    for (uint32_t i = 0; i < ir_in_dtypes.size(); i++)
    {
        auto ir_dtype = ir_in_dtypes[i];
        auto lib_dtype = best_in_dtypes[i];
        if (ir_dtype != lib_dtype)
        {
            auto uuid = utils::get_uuid();
            auto inpt = n->inputs()[i];
            auto bkunq = inpt->quantization().unquantifiable();
            inpt->quantization().set_unquantifiable(true);
            auto cast_out = aipubt::ops::cast(inpt, lib_dtype, true, aipubt::ops::ClipMode::Truncation);
            cast_out->node()->set_name(cast_out->node()->name() + inpt->name() + uuid);
            cast_out->set_name(cast_out->name() + inpt->name() + std::to_string(i) + uuid);
            cast_out->quantization() = inpt->quantization();
            inpt->quantization().set_unquantifiable(bkunq);
            auto idx = n->remove_input(inpt);
            n->add_input(cast_out, (int32_t)idx);
            INFO() << "[UNMATCHED] now insert cast at node: " << n << "'s input idx = " << i << " and insert cast node :" << cast_out->node();
        }
    }
}
void InsertCastOp::_insert_subgraph_node(aipubt::NodePtr n)
{
    auto sgn = aipubt::SubGraphNode::from_node(n);
    auto sg = sgn->subgraph();
    auto sgnodes = sg->nodes();
    for (auto &subg_node : sgnodes)
    {
        _insert(subg_node);
    }
}

void InsertCastOp::_insert(aipubt::NodePtr n)
{
    if (aipubt::SubGraphNode::is_subgraph_node(n))
    {
        _insert_subgraph_node(n);
    }
    else
    {
        _insert_for_one_node(n);
    }
}

void InsertCastOp::insert(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &n : nodes)
    {
        _insert(n);
    }
}
} //namespace qtlibn

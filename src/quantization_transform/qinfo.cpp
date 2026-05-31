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
 * @Date: 2022-04-25 13:38:44
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-07-14 11:44:26
 */

#include "quantization_transform/qinfo.h"
#include "passes/passes.h"

namespace qtlibn
{
void update_qinfo_to_tensor(aipubt::TensorPtr tensor, QInfo_t &qinfo)
{
    bool qinvariant = qinfo.qinvariant;
    bool data_unchanged = qinfo.data_unchanged;
    int32_t bits = qinfo.bits;
    auto scheme = qinfo.scheme;
    auto granu = qinfo.granularity;
    auto qrange = qinfo.quantized_range;
    if (tensor->quantization().scheme() == aipubt::Quantization::Scheme_Void)
    {
        tensor->quantization().set_scheme(scheme);
    }

    if (tensor->quantization().granularity() == aipubt::Quantization::Granularity_Void)
    {
        tensor->quantization().set_granularity(granu);
    }
    if (tensor->quantization().quantized_range() == aipubt::Quantization::Range_Void)
    {
        tensor->quantization().set_quantized_range(qrange);
    }

    if (tensor->quantization().bits() == -1)
    {
        tensor->quantization().set_bits(bits);
    }
    // Only qinvariant is true, we set it, otherwise, it will override user setting
    if (qinvariant)
    {
        tensor->quantization().set_qinvariant(qinvariant);
    }
    if (data_unchanged)
    {
        tensor->quantization().set_data_unchanged(data_unchanged);
    }
}

static inline bool _data_changed_rule(aipubt::TensorPtr t, bool force_weight_asym, aipubt::OpType optype)
{
    bool ret = false;
    auto primitive_dtype = t->type();
    if (force_weight_asym)
    {
        // if setted force_weight_asym=false, will not change the weight data
        return false;
    }

    auto dtype_bits = aipubt::Tensor::dtype_bits(primitive_dtype);
    if (aipubt::Tensor::is_float(primitive_dtype) && dtype_bits <= 8)
    {
        return false;
    }

    std::vector<float> zps = t->quantization().offsets();
    auto zps_size = zps.size();
    if (zps_size > 0)
    {
        float one_zp = zps[0];
        bool dtype_is_signed = aipubt::Tensor::is_signed(primitive_dtype);
        if (zps_size == 1 && (one_zp != 0 || !dtype_is_signed) && optype != aipubt::OpType_Constant)
        {
            return true;
        }
    }

    return ret;
}

void update_qinfo_to_tensor_in_node(aipubt::NodePtr node, TensorQInfo_t *tensor_qinfo)
{
    bool force_weight_asym = false;
    if (node->attrs().count("force_weight_asym"))
    {
        force_weight_asym = node->get_attr("force_weight_asym");
    }

    for (auto ot : node->outputs())
    {
        auto activation_tensor_qinfo = tensor_qinfo->activation_qinfo;
        update_qinfo_to_tensor(ot, activation_tensor_qinfo);
    }

    for (auto plh : node->placeholders())
    {
        auto activation_tensor_qinfo = tensor_qinfo->activation_qinfo;
        update_qinfo_to_tensor(plh, activation_tensor_qinfo);
    }

    for (auto ct : node->constants())
    {
        if (ct.first == "weights")
        {
            auto weight_tensor_qinfo = tensor_qinfo->weight_qinfo;
            if (_data_changed_rule(ct.second, force_weight_asym, node->type()))
            {
                weight_tensor_qinfo.data_unchanged = false;
            }
            update_qinfo_to_tensor(ct.second, weight_tensor_qinfo);
        }
        else if (ct.first == "biases")
        {
            auto bias_tensor_qinfo = tensor_qinfo->bias_qinfo;
            update_qinfo_to_tensor(ct.second, bias_tensor_qinfo);
        }
        else if (ct.first == "negative_slope")
        {
            auto weight_tensor_qinfo = tensor_qinfo->weight_qinfo;
            weight_tensor_qinfo.data_unchanged = false;
            update_qinfo_to_tensor(ct.second, weight_tensor_qinfo);
        }
        else
        {
            DEBUG() << ct.first << " in Constants data will use the weight_qinfo  to update tensor qinfo.";
            auto weight_tensor_qinfo = tensor_qinfo->weight_qinfo;
            update_qinfo_to_tensor(ct.second, weight_tensor_qinfo);
        }
    }
    if (node->constants().count("biases") && node->constants().count("weights"))
    {
        //bias quantization should be aligned with weight quantization. if weight needs dequantize to float data, biases data
        // also need to dequantize to float data
        auto weight_qinvariant = node->constants()["weights"]->quantization().qinvariant();
        auto weight_unchanged = node->constants()["weights"]->quantization().data_unchanged();
        node->constants()["biases"]->quantization().set_qinvariant(weight_qinvariant);
        node->constants()["biases"]->quantization().set_data_unchanged(weight_unchanged);
    }
    update_tensor_group_and_granularity(node);
}

void _update(aipubt::NodePtr n, TensorQInfo_t *tqinfo)
{
    if (aipubt::SubGraphNode::is_subgraph_node(n))
    {
        _update_subgraph(n, tqinfo);
    }
    else
    {
        update_qinfo_to_tensor_in_node(n, tqinfo);
    }
}
void _update_subgraph(aipubt::NodePtr n, TensorQInfo_t *tqinfo)
{
    auto sgn = aipubt::SubGraphNode::from_node(n);
    auto sg = sgn->subgraph();
    for (auto &subg_node : sg->nodes())
    {
        _update(subg_node, tqinfo);
    }
}

void update_qinfo_to_tensor_in_graph(aipubt::GraphPtr graph, TensorQInfo_t *tensor_qinfo)
{
    if (!graph || !tensor_qinfo)
    {
        ERROR() << "please set Graph and TensorQInfo firstly.";
    }
    for (auto &node : *graph)
    {
        _update(node, tensor_qinfo);
    }
}

TensorQInfo_t get_tensor_qinfo(const std::string run_mode_or_scheme, const int32_t bits, bool is_run_mode)
{
    // default value bits in [8, 9, 16]
    auto scm = aipubt::Quantization::Symmetric;
    bool constant_data_unchanged = false;
    if (is_run_mode)
    {
        RunMode enum_run_mode = string_to_runmode(run_mode_or_scheme);
        scm = runmode_to_scheme(enum_run_mode);

        if (enum_run_mode == RunMode::FAsym)
        {
            constant_data_unchanged = true;
        }
    }
    else
    {
        scm = string_to_scheme(run_mode_or_scheme);
    }

    TensorQInfo_t bits_qinfo = {
        .activation_qinfo = {
            .qinvariant = false,
            .data_unchanged = false,
            .bits = bits,
            .scheme = scm,
            .granularity = aipubt::Quantization::Per_Tensor,
            .quantized_range = aipubt::Quantization::Restricted_Range},
        .weight_qinfo = {.qinvariant = false, .data_unchanged = constant_data_unchanged, .bits = bits, .scheme = aipubt::Quantization::Symmetric, .granularity = aipubt::Quantization::Per_Tensor, .quantized_range = aipubt::Quantization::Restricted_Range},
        .bias_qinfo = {.qinvariant = false, .data_unchanged = constant_data_unchanged, .bits = bits < 10 ? 32 : 32 + (bits - 8) * 2, .scheme = aipubt::Quantization::Symmetric, .granularity = aipubt::Quantization::Per_Tensor, .quantized_range = aipubt::Quantization::Restricted_Range},
    };
    return bits_qinfo;
}

static void update_qinfo_to_tensor_in_graph(aipubt::GraphPtr graph, const std::string run_mode_or_scheme, bool is_run_mode, const int32_t setted_bits)
{
    auto scm = aipubt::Quantization::Symmetric;
    bool constant_data_unchanged = false;
    if (is_run_mode)
    {
        RunMode enum_run_mode = string_to_runmode(run_mode_or_scheme);
        scm = runmode_to_scheme(enum_run_mode);

        if (enum_run_mode == RunMode::FAsym)
        {
            constant_data_unchanged = true;
        }
    }
    else
    {
        scm = string_to_scheme(run_mode_or_scheme);
    }

    auto _get_activation_qinfo = [&scm](int32_t bits) {
        QInfo_t q = {
            .qinvariant = false,
            .data_unchanged = false,
            .bits = bits,
            .scheme = scm,
            .granularity = aipubt::Quantization::Per_Tensor,
            .quantized_range = aipubt::Quantization::Restricted_Range,
        };
        return q;
    };
    auto _get_weight_qinfo = [&constant_data_unchanged](int32_t bits) {
        QInfo_t q = {
            .qinvariant = false,
            .data_unchanged = constant_data_unchanged,
            .bits = bits,
            .scheme = aipubt::Quantization::Symmetric,
            .granularity = aipubt::Quantization::Per_Tensor,
            .quantized_range = aipubt::Quantization::Restricted_Range,
        };
        return q;
    };
    auto _get_bias_qinfo = [&constant_data_unchanged](int32_t bits) {
        QInfo_t q = {
            .qinvariant = false,
            .data_unchanged = constant_data_unchanged,
            .bits = bits < 10 ? 32 : 32 + (bits - 8) * 2,
            .scheme = aipubt::Quantization::Symmetric,
            .granularity = aipubt::Quantization::Per_Tensor,
            .quantized_range = aipubt::Quantization::Restricted_Range,
        };
        return q;
    };

    auto _get_t_bits = [&setted_bits](aipubt::TensorPtr out) {
        // if setted_bits has setted by user, like midware, qtlib direcltly use it
        // if setted_bits is defaultly -1, we assume qtlib deduces the bits, prority:
        // tensor->quantization().bits() > tensor->type() bits or default value(=8)
        if (setted_bits != -1)
        {
            return setted_bits;
        }
        else
        {
            auto bits = out->quantization().bits();
            if (bits == -1)
            {
                if (aipubt::Tensor::is_float(out->type()))
                {
                    bits = 8; // defalut 8bits quantization for activation, if not meets your quantization, please set the bits first.
                }
                else
                {
                    bits = aipubt::Tensor::dtype_bits(out->type());
                }
            }
            return bits;
        }
    };

    auto _update_one = [&_get_t_bits, &_get_activation_qinfo, &_get_bias_qinfo, &_get_weight_qinfo](aipubt::NodePtr node) {
        std::vector<aipubt::Tensor::Dtype> setted_dtypes = quant_utils::preferred_fp_dtype(node);
        auto act_dtype = setted_dtypes[0];
        auto wht_dtype = setted_dtypes[1];

        bool force_weight_asym = false;
        if (node->attrs().count("force_weight_asym"))
        {
            force_weight_asym = node->get_attr("force_weight_asym");
        }

        for (auto &out : node->outputs())
        {
            if (!aipubt::Tensor::is_float(out->type()) && out->quantization().empty() && !aipubt::Tensor::is_float(act_dtype))
            {
                ERROR() << node << ", please set the output = " << out << " quantization info, which now is empty.";
            }
            auto bits = _get_t_bits(out);
            auto q = _get_activation_qinfo(bits);
            update_qinfo_to_tensor(out, q);
        }

        for (auto plh : node->placeholders())
        {
            auto bits = node->outputs()[0]->quantization().bits();
            auto q = _get_activation_qinfo(bits);
            update_qinfo_to_tensor(plh, q);
        }

        for (auto &ct : node->constants())
        {
            if (!aipubt::Tensor::is_float(ct.second->type()) && ct.second->quantization().empty() && !aipubt::Tensor::is_float(wht_dtype))
            {
                ERROR() << node << ", please set the output = " << ct.second << " quantization info, which now is empty.";
            }

            if (ct.first == "biases")
            {
                auto bits = node->outputs()[0]->quantization().bits();
                auto q = _get_bias_qinfo(bits);
                update_qinfo_to_tensor(ct.second, q);
            }
            else if (ct.first == "weights")
            {
                auto bits = _get_t_bits(ct.second);
                auto q = _get_weight_qinfo(bits);
                // if (!force_weight_asym && ((aipubt::Tensor::is_float(primitive_dtype) && primitive_dtype != aipubt::Tensor::FP4E2M1FN) ||
                //                            (ct.second->quantization().offsets().size() == 1 && (ct.second->quantization().offset() != 0 || !aipubt::Tensor::is_signed(primitive_dtype)) &&
                //                             node->type() != aipubt::OpType_Constant)))
                if (_data_changed_rule(ct.second, force_weight_asym, node->type()))
                {
                    q.data_unchanged = false;
                }
                update_qinfo_to_tensor(ct.second, q);
            }
            else
            {
                auto bits = _get_t_bits(ct.second);
                auto q = _get_weight_qinfo(bits);
                q.data_unchanged = false;
                update_qinfo_to_tensor(ct.second, q);
            }
        }
        if (node->constants().count("biases") && node->constants().count("weights"))
        {
            //bias quantization should be aligned with weight quantization. if weight needs dequantize to float data, biases data
            // also need to dequantize to float data
            auto weight_qinvariant = node->constants()["weights"]->quantization().qinvariant();
            auto weight_unchanged = node->constants()["weights"]->quantization().data_unchanged();
            node->constants()["biases"]->quantization().set_qinvariant(weight_qinvariant);
            node->constants()["biases"]->quantization().set_data_unchanged(weight_unchanged);
        }
        update_tensor_group_and_granularity(node);
    };

    std::function<void(aipubt::NodePtr)> _update;
    _update = [&_update_one, &_update](aipubt::NodePtr node) {
        if (aipubt::SubGraphNode::is_subgraph_node(node))
        {
            auto sgn = aipubt::SubGraphNode::from_node(node);
            auto sg = sgn->subgraph();
            for (auto &subg_node : sg->nodes())
            {
                _update(subg_node);
            }
        }
        else
        {
            _update_one(node);
        }
    };
    for (auto &node : graph->nodes())
    {
        _update(node);
    }
}

void update_graph_qinfo(aipubt::GraphPtr graph, const std::string scheme, const int32_t bits, bool is_run_mode)
{
    if (bits == -1)
    {
        INFO() << "QTlib will use tensor's dtype bits to set quantization bits, "
               << "like if tensor's dtype bits is 16, tensor.quantization.bits=16";
    }
    update_qinfo_to_tensor_in_graph(graph, scheme, is_run_mode, bits);
}

void update_node_qinfo(aipubt::NodePtr node, const std::string scheme, const int32_t bits, bool is_run_mode)
{
    auto lbits = bits;
    if (bits == -1)
    {
        ERROR() << node << "'s update node qinfo uses the bits == -1, which does not support. defaultly using bits==8";
        lbits = 8;
    }
    auto bits_qinfo = get_tensor_qinfo(scheme, lbits, is_run_mode);
    update_qinfo_to_tensor_in_node(node, &bits_qinfo);
}

bool check_node_neccessary_qinfo(aipubt::NodePtr node)
{
    if (node->type() == aipubt::OpType::OpType_Quantize)
    {
        if (!node->outputs()[0]->quantization().empty())
        {
            return true;
        }
    }

    if (node->type() == aipubt::OpType::OpType_DeQuantize)
    {
        if (!node->inputs()[0]->quantization().empty())
        {
            return true;
        }
    }
    if (node->type() == aipubt::OpType::OpType_IfHeader or node->type() == aipubt::OpType_IfEnder)
    {
        return true;
    }

    // mainly scales/offsets/statistic_mins/statistic_maxs
    auto get_all_tensors = [](aipubt::NodePtr node) -> std::vector<aipubt::TensorPtr> {
        std::vector<aipubt::TensorPtr> all_ts;
        for (auto &in_t : node->inputs())
        {
            all_ts.push_back(in_t);
        }
        for (auto &out_t : node->outputs())
        {
            all_ts.push_back(out_t);
        }
        // for (auto &ct : node->constants())
        // {
        //     all_ts.push_back(ct.second);
        // }
        for (auto &pt : node->placeholders())
        {
            all_ts.push_back(pt);
        }
        return all_ts;
    };

    auto has_tensor_qinfo = [](aipubt::TensorPtr t) -> bool {
        if (t->quantization().empty())
        {
            DEBUG() << t->name() << " has no min/max, also has no scale/zp.";
            return false;
        }
        return true;
    };

    auto all_ts = get_all_tensors(node);
    auto ret = true;
    for (auto ts : all_ts)
    {
        ret = ret && has_tensor_qinfo(ts);
    }
    return ret;
}

void set_default_qinfo_to_tensor(aipubt::TensorPtr t, const int32_t bits)
{
    auto qinfos = get_tensor_qinfo("symmetric", bits, false);
    auto activation_tensor_qinfo = qinfos.activation_qinfo;
    update_qinfo_to_tensor(t, activation_tensor_qinfo);
}

void update_aipu_dtype(aipubt::NodePtr node)
{
    /*
     * this situation is for when opsAPI has called in lowering, which input tensor is not created by opsAPI,
     * but using the input tensor to create a opsAPI tensor, so when quantizing this created node, the input
     * tensor doesnot include valid aipu_dtype, so we now use the tensor dtype as aipu_dtype in input tensor.
     */
    for (auto &it : node->inputs())
    {
        if (it->quantization().aipu_dtype() == aipubt::Tensor::VOID)
        {
            auto tdtype = it->type();
            it->quantization().set_aipu_dtype(tdtype);
        }
    }
}

} // namespace qtlibn

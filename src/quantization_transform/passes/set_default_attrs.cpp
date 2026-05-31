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
 * @Date: 2024-03-25 10:57:28
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2024-03-28 21:20:22
 */

#include "quantization_transform/passes/passes.h"

namespace qtlibn
{

static inline void _set_trigger_float_op(aipubt::NodePtr n)
{
    /*
    now qtlib sets the trigger_float_op attrs for qat model, which can align the android and bt application.
    and this pass must not apply to floatIR which wants to quantize.

    if qat model is fp16*aligned_int4, we want to defaultly set trigger_float_op;
                                               act
                |   float32  |  float16  |  float8  |  float4  |  int8   |  int16  |  int32  |
        --------------------------------------------------------------------------------------
        not wht |    fp16    |   fp16    |    fp8   |    fp4   |    /    |     /   |    /    |
        float32 |    fp16    |   fp16    |afp8_wfp16|afp4_wfp16|ai8_wfp32|
        float16 |    fp16    |   fp16    |afp8_wfp16|afp4_wfp16|ai8_wfp16|
    wht float8  | afp16_wfp8 | afp16_wfp8|afp8_wfp8 |afp4_wfp8 |ai8_wfp8|
        float4  |
        int8    |
        int16   |
        int32   |

        afp8_wfp16, afp4wfp816 afp8_wfp16 temporaly donot support
        fp16/fp8/fp4: trigger_float_op=true
        ai8_wfp32: weight should be to quantize: trigger_float_op = "disable"
    */
    const std::vector<aipubt::Tensor::Dtype> orig_fp_dtype = {aipubt::Tensor::FP32, aipubt::Tensor::FP64};
    if (!n->attrs().count(Constants::trigger_float_op))
    {
        std::string act_dtype;
        std::string wht_dtype;
        std::string trigger_float_op;
        std::string def_dtype = "float16";
        for (auto &ot : n->outputs())
        {
            if (aipubt::Tensor::is_float(ot->type()))
            {
                if (aipubt::contains(orig_fp_dtype, ot->type()))
                {
                    act_dtype = def_dtype;
                }
                else
                {
                    act_dtype = aipubt::Tensor::dtype_to_string(ot->type());
                }
                break;
            }
        }

        if (n->constants().count("weights"))
        {
            auto w = n->constants()["weights"];
            if (aipubt::contains(orig_fp_dtype, w->type()))
            {
                wht_dtype = def_dtype;
            }
            else
            {
                wht_dtype = aipubt::Tensor::dtype_to_string(w->type());
            }
        }

        if (act_dtype.size() > 0 && wht_dtype.size() > 0)
        {
            trigger_float_op = act_dtype;
            if (act_dtype != wht_dtype)
            {
                trigger_float_op += "_ACT_" + wht_dtype + "_WHT";
            }
        }
        else if (act_dtype.size() > 0)
        {
            trigger_float_op = act_dtype;
        }
        // else if (act_dtype.size() == 0 && wht_dtype.size() > 0)
        // {
        //     DEBUG() << "When set_default_attrs, we deduce the node(" << n << ") needs to quantize constant data.";
        // }

        if (trigger_float_op.size() > 0)
        {
            n->set_attr(Constants::trigger_float_op, aipubt::NodeParamValue(trigger_float_op));
            // INFO() << "node(" << n << ")'s trigger_float_op = " << (trigger_float_op.size() > 0 ? trigger_float_op : "disable");
        }
    }
}

static inline void set_multiplier_bits(aipubt::NodePtr n)
{
    if (!n->attrs().count("multiplier_bits"))
    {
        n->set_attr("multiplier_bits", aipubt::NodeParamValue(13));
    }
}

static inline void set_lut_items_in_bits(aipubt::NodePtr n)
{
    // we appoint that first output tensor should include 'lut_items_bits'.
    int32_t lut_items_bits = n->outputs()[0]->quantization().lut_items_bits();
    if (lut_items_bits == -1)
    {
        std::vector<int32_t> out_bits;
        for (auto ot : n->outputs())
        {
            out_bits.push_back(ot->quantization().bits());
        }
        auto max_bits = *std::max_element(out_bits.begin(), out_bits.end());

        if (max_bits <= 8)
        {
            lut_items_bits = 8;
        }
        else
        {
            lut_items_bits = 12;
        }
    }
    n->set_attr(Constants::lut_items_in_bits, aipubt::NodeParamValue(lut_items_bits));
}

void set_node_default_attr(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &n : nodes)
    {
        _set_trigger_float_op(n);
        set_multiplier_bits(n);
        set_lut_items_in_bits(n);
    }
}
void set_trigger_float_op(aipubt::GraphPtr g)
{
    auto nodes = g->nodes();
    for (auto &n : nodes)
    {
        _set_trigger_float_op(n);
    }
}

} // namespace qtlibn
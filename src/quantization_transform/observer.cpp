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
 * @Date: 2023-03-24 10:22:04
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2025-11-05 09:46:00
 */

#include "quantization_transform/observer.h"
#include "quantization_transform/utils/calibrate_utils.h"
#include "quantization_transform/utils/data_type.h"

namespace qtlibn
{

void statistic(aipubt::NodePtr node)
{
    auto tensor_statistic = [](aipubt::TensorPtr t) -> void {
        auto flag = t->quantization().maxs().size() && t->quantization().mins().size();
        flag = flag && t->quantization().scales().size() && t->quantization().offsets().size();
        if (!flag)
        {
            auto ttype = t->type();
            if (aipubt::Tensor::is_float(ttype) && t->data<float>())
            {
                // statistic constant data float max/min. and qtlib only receive float data defaultly
                std::vector<float> statistic_max;
                std::vector<float> statistic_min;
                calibrate_utils::statistic_max_min_from_float_tensor(t, statistic_max, statistic_min);
                t->quantization().set_mins(statistic_min);
                t->quantization().set_maxs(statistic_max);
            }
        }
    };

    for (auto &it : node->inputs())
    {
        tensor_statistic(it);
    }
    for (auto &ot : node->outputs())
    {
        tensor_statistic(ot);
    }
    for (auto &ct : node->constants())
    {
        tensor_statistic(ct.second);
    }
    for (auto &pt : node->placeholders())
    {
        pt->quantization().set_granularity(node->outputs()[0]->quantization().granularity());
        pt->quantization().set_scheme(node->outputs()[0]->quantization().scheme());
        pt->quantization().set_quantized_range(node->outputs()[0]->quantization().quantized_range());
        pt->quantization().set_bits(node->outputs()[0]->quantization().bits());
        tensor_statistic(pt);
    }
}

} // namespace qtlibn

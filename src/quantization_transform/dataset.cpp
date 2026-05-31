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
 * @Date: 2022-08-18 10:45:48
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-18 18:02:48
 */

#include "quantization_transform/dataset.h"
#include "quantization_transform/utils/quantize_utils.h"

namespace qtlibn
{
Dataset::Dataset(std::vector<float> &ranges, std::vector<aipubt::TensorShape> &shapes) : m_ranges(ranges), m_shapes(shapes)
{
}

Dataset::~Dataset()
{
    m_ranges.clear();
    m_shapes.clear();
    for (auto &d : m_datas)
    {
        if (d)
        {
            d = nullptr;
        }
    }
    m_datas.clear();
    m_tensor = nullptr;
}

bool Dataset::prepare(void)
{
    bool ret = false;
    if (m_ranges.size() == 0 && m_shapes.size() == 0 && m_tensor)
    {
        if (m_tensor->quantization().mins().size() == 0 || m_tensor->quantization().maxs().size() == 0)
        {
            std::vector<float> scales = m_tensor->quantization().scales();
            auto offsets = m_tensor->quantization().offsets();
            auto dtype = m_tensor->type();
            if (aipubt::Tensor::is_float(dtype))
            {
                WARN() << "tensor.name = " << m_tensor->name() << " has the float dtype, so the deduce the min/max value failed. and we now use the int8 to deduce.";
                // dtype = aipubt::Tensor::INT8;
                return ret;
            }

            std::vector<float> statistic_max;
            std::vector<float> statistic_min;

            std::function<float(float, float, int32_t)> formula = quant_utils::dequantize_formula_aipu;
            quant_utils::dequantize_from_primitive_quant_info(dtype,
                                                              scales,
                                                              offsets,
                                                              statistic_max,
                                                              statistic_min,
                                                              formula);
            m_tensor->quantization().set_mins(statistic_min);
            m_tensor->quantization().set_maxs(statistic_max);
        }
        m_ranges.push_back(m_tensor->quantization().maxs()[0]);
        m_ranges.push_back(m_tensor->quantization().mins()[0]);
        m_shapes.push_back(m_tensor->shape());
        ret = true;
    }
    return ret;
}

void Dataset::generate_data(void)
{
    if (!prepare())
    {
        DEBUG() << "we cannot generate the input data";
        return;
    }

    if (m_ranges.size() == 0 || m_shapes.size() == 0)
    {
        ERROR() << "please set ranges and shapes firstly if want to genearate random data";
        return;
    }
    auto r_size = m_ranges.size();
    auto s_shape = m_shapes.size();
    if (r_size != 2 * s_shape)
    {
        ERROR() << "ranges.size() == 2 * shapes.size()";
        return;
    }
    if (m_datas.size() != 0)
    {
        m_datas.clear();
    }
    for (uint32_t ss = 0; ss < s_shape; ss++)
    {
        auto shape = m_shapes[ss];
        auto r_max = m_ranges[2 * ss];
        auto r_min = m_ranges[2 * ss + 1];
        float_t *inp_data = new float_t[shape.size()];
        for (int i = 0; i < shape.size(); ++i)
        {
            *(inp_data + i) = r_min + (r_max - r_min) * rand() / (float)RAND_MAX;
        }
        m_datas.push_back(inp_data);
    }
    DEBUG() << "forward use random data generate from max and min value of tensor.";
}

std::vector<float *> Dataset::get_item(void)
{
    return m_datas;
}

} // namespace qtlibn

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
 * @Date: 2020-01-29 14:28:32
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-07-20 10:55:23
 */
#include "core/quantization.h"
#include "core/tensor_slice.h"
#include <map>
namespace aipubt
{

static uint32_t get_tensor_counter()
{
    static uint32_t i = 0;
    return i++;
}

TensorSlice::~TensorSlice()
{
}

TensorSlice::TensorSlice(const TensorPtr t) : Tensor(*t)
{
    m_tensor = t;
    m_name += "_slice" + std::to_string(get_tensor_counter());
    m_slice_offset = std::vector<int>(m_shape.dim(), 0);
}

TensorSlice::TensorSlice(const TensorPtr t, const std::vector<int> &slice_offset, const std::vector<int> &slice_shape) : TensorSlice(t)
{
    if ((int)slice_offset.size() > m_shape.dim())
    {
        ERROR() << " Slice offset size > shape dim";
    }
    else
    {
        m_slice_offset = slice_offset;
        if ((int)slice_offset.size() < m_shape.dim())
        {
            // pad 0
            for (int i = slice_offset.size(); i < m_shape.dim(); i++)
            {
                m_slice_offset.emplace_back(0);
            }
        }
    }
    if ((int)slice_shape.size() > m_shape.dim())
    {
        ERROR() << " Slice shape dim > shape dim";
    }
    else
    {
        // for shape, pad with value of original shape
        for (auto i = 0u; i < slice_shape.size(); ++i)
        {
            m_shape[i] = slice_shape[i];
        }
    }
}
TensorSlice::TensorSlice(const TensorPtr t, const std::vector<int> &slice_offset, const TensorShape &slice_shape) : TensorSlice(t)
{
    if ((int)slice_offset.size() > m_shape.dim())
    {
        ERROR() << " Slice offset size > shape dim";
    }
    else
    {
        m_slice_offset = slice_offset;
        if ((int)slice_offset.size() < m_shape.dim())
        {
            // pad 0
            for (int i = slice_offset.size(); i < m_shape.dim(); i++)
            {
                m_slice_offset.emplace_back(0);
            }
        }
    }
    if ((int)slice_shape.size() > m_shape.dim())
    {
        ERROR() << " Slice shape dim > shape dim";
    }
    else
    {
        // for shape, pad with value of original shape
        for (auto i = 0; i < slice_shape.dim(); ++i)
        {
            m_shape[i] = slice_shape[i];
        }
    }
}

} // namespace aipubt

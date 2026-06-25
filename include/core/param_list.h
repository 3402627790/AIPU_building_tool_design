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
 * @Author: Lang Hu
 * @Date: 2026-06-10 14:25:33
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-06-10 14:25:33
 */
#pragma once
#include "tensor.h"

namespace aipubt
{

struct ReadOnlyParam;
class ParamList;
using ParamListPtr = std::shared_ptr<ParamList>;
using ParamListRef = std::weak_ptr<ParamList>;

class ParamList : public Tensor
{
public:
    ParamList(const std::string &name, Dtype dtype = UINT32)
        : Tensor(TensorShape({0}, DataLayout_Flat), dtype)
    {
        if (name.size())
        {
            m_name = name;
        }
        if (dtype != INT32 && dtype != UINT32)
        {
            ERROR() << "unsupported dtype: " << dtype;
        }
    }
    ParamList(Dtype dtype = UINT32) : ParamList(std::string(), dtype) {}

    void clear()
    {
        m_params.clear();
        update();
    }

    void push_back(const ReadOnlyParam &param)
    {
        m_params.push_back(param);
        update();
    }

    const TensorShape &shape() const
    {
        return m_shape;
    }

    const std::vector<ReadOnlyParam> &data() const
    {
        return m_params;
    }

    void set_data(const std::vector<ReadOnlyParam> &params)
    {
        m_params = params;
        update();
    }

private:
    void update()
    {
        m_shape = TensorShape({(int)m_params.size()}, DataLayout_Flat);
        m_offset->size = mem_size();
    }

    using Tensor::compact;
    using Tensor::compress;
    using Tensor::data;
    using Tensor::decompact;
    using Tensor::decompress;
    using Tensor::device_data;
    using Tensor::fill_tensors_data;
    using Tensor::get_compress_info;
    using Tensor::get_ref_count;
    using Tensor::is_owns_device_data;
    using Tensor::is_owns_host_data;
    using Tensor::make_memory_continuous;
    using Tensor::max;
    using Tensor::min;
    using Tensor::monopolize_data;
    using Tensor::quantization;
    using Tensor::set_data;
    using Tensor::set_device_data;
    using Tensor::set_type;
    using Tensor::shape;
    using Tensor::share_data_from;

    std::vector<ReadOnlyParam> m_params;
};

} // namespace aipubt
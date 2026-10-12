//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2020-2026 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Neo Wang
 * @Date: 2020-02-09 12:16:08
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2026-07-20 10:36:38
 */
#pragma once

#include "tensor.h"

namespace aipubt
{

class TensorSlice;
using TensorSlicePtr = std::shared_ptr<TensorSlice>;
/**
 * @brief a class for Tensor slice, it will owns a base tensor, and  it will represent as a tensor, but shared data from base and other info.
 *
 */
class TensorSlice : public Tensor
{
public:
    TensorSlice() = delete;
    /**
     * @brief Construct a new Tensor object
     *        construct a tensor by copy other, please note that the node owner(m_node) and node consumer(m_consumer) will not be copy
     * 
     * @param t other Tensor
     */
    TensorSlice(const TensorPtr t);
    TensorSlice(const TensorPtr t, const std::vector<int> &slice_offset, const std::vector<int> &slice_shape);
    TensorSlice(const TensorPtr t, const std::vector<int> &slice_offset, const TensorShape &slice_shape);

    ~TensorSlice();

    TensorPtr root_tensor()
    {

        return m_tensor.lock();
    }
    void set_mem_id(int i)
    {
        m_mem_id = i;
    }
    int mem_id()
    {
        return m_mem_id;
    }
    const std::vector<int> &slice_offset()
    {
        return m_slice_offset;
    }
    void set_slice_offset(const std::vector<int> &slice_offset);

protected:
    void setup_slice_offset();
    TensorRef m_tensor;
    std::vector<int> m_slice_offset;
    int m_mem_id = -1;
    friend Node;

private:
    //hide some  functions
    using Tensor::operator=;
};

} // namespace aipubt

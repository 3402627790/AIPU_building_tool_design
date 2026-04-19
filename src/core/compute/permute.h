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
 * @Date: 2020-02-09 12:16:08
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-24 16:40:41
 */
#ifndef CORE_COMPUTE_PERMUTE_H
#define CORE_COMPUTE_PERMUTE_H

#include "core/compute/multitypes.h"
#include "core/tensor.h"

namespace aipubt
{
template <typename T>
void permute_cpu(const T *input, T *output, const std::vector<int> &inp_shape, const std::vector<int> &out_shape, const std::vector<int> &order)
{
    std::vector<size_t> in_stride, out_stride;
    in_stride.resize(inp_shape.size());
    out_stride.resize(inp_shape.size());
    size_t dsize = 1;
    size_t dsize2 = 1;
    for (uint32_t i = 0; i < inp_shape.size(); ++i)
    {
        in_stride[inp_shape.size() - 1 - i] = dsize;
        out_stride[inp_shape.size() - 1 - i] = dsize2;
        dsize *= inp_shape[inp_shape.size() - 1 - i];
        dsize2 *= out_shape[inp_shape.size() - 1 - i];
    }
#pragma omp parallel for
    for (uint32_t i = 0; i < dsize; ++i)
    {
        size_t dsize3 = i;
        int offset = 0;
        for (uint32_t j = 0; j < order.size(); ++j)
        {
            int index = dsize3 / out_stride[j];
            offset += in_stride[order[j]] * index;
            dsize3 -= index * out_stride[j];
        }
        output[i] = input[offset];
    }
}
MULTITYPES_COMPUTE_OP(permute_cpu)

};     // namespace aipubt
#endif //!CORE_COMPUTE_PERMUTE_H

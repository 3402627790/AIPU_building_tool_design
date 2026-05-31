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
 * @Last Modified time: 2024-11-01 14:40:17
 */

#ifndef __QTLIB_INSERTOP_H
#define __QTLIB_INSERTOP_H

#include "quantization_transform/common.h"
#include "quantization_transform/utils/data_type.h"
#include "quantization_transform/utils/math_utils.h"

using dtypes_list = std::vector<aipubt::Tensor::Dtype>;
namespace qtlibn
{
class BaseInsertOp;
class BaseInsertOp
{
public:
    virtual ~BaseInsertOp() {}
    virtual void insert(aipubt::GraphPtr g) = 0;
};

class InsertCastOp : public BaseInsertOp
{
public:
    enum Status
    {
        SKIP = 0,
        UNMATCHED = 1,
        OUTPUT_UNMATCHED = 2,
        MATCHED = 3,
    };

    void insert(aipubt::GraphPtr g);

private:
    Status criteria(aipubt::NodePtr n);
    dtypes_list _match_algo(dtypes_list &cur_dtypes, std::vector<dtypes_list> &dtypes_pool);
    std::vector<dtypes_list> _match_algo_for_output_dtype(dtypes_list &cur_in_dtypes, dtypes_list &cur_output_dtypes, std::vector<aipubt::Op_Dtype> &dtypes_pool, const aipubt::Tensor::Dtype target_dtype);
    void _insert_for_one_node(aipubt::NodePtr n);
    void _insert_subgraph_node(aipubt::NodePtr n);
    void _insert(aipubt::NodePtr n);
    bool is_matched(dtypes_list &in_dtype, dtypes_list &out_dtype, std::vector<aipubt::Op_Dtype> &lib_spec_dtypes);
    std::map<aipubt::OpType, std::vector<aipubt::Op_Dtype>> m_cache_dtypes;
};

} // namespace qtlibn

#endif

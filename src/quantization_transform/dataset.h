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
 * @Date: 2022-08-18 10:50:44
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-11-23 09:30:19
 */

#ifndef __QTLIB_DATASET_H
#define __QTLIB_DATASET_H

#include "quantization_transform/common.h"

namespace qtlibn
{
class Dataset;
using DatasetPtr = std::shared_ptr<Dataset>;

class Dataset
{
public:
    // Dataset(std::vector<float*> datas);
    Dataset(std::vector<float> &ranges, std::vector<aipubt::TensorShape> &shapes);
    Dataset(aipubt::TensorPtr t) : m_tensor(t) {}
    ~Dataset();

    // int32_t len(void) const{};
    std::vector<float *> get_item();

    void generate_data(void);

private:
    bool prepare(void);

private:
    std::vector<float> m_ranges;
    std::vector<aipubt::TensorShape> m_shapes;
    std::vector<float *> m_datas;
    aipubt::TensorPtr m_tensor;
    // int32_t m_len;
};

} // namespace qtlibn

#endif

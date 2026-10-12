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
 * @Date: 2020-01-29 14:28:32 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-23 10:03:41
 */
#include "core/const_tensor.h"
namespace aipubt
{

ConstTensor::ConstTensor(const TensorShape &shape, Dtype type, void *data, size_t data_size) : Tensor(shape, type)
{
    set_const_tensor(true);
    set_data(data, data_size);
}
ConstTensor::ConstTensor(TensorPtr t) : Tensor(*t)
{
    set_const_tensor(true);
    // Preserve device-resident constants: only zero-fill host data when the
    // source carries neither host nor device data.  (Host/device data are
    // mutually exclusive on a Tensor; set_data would clear the device pointer
    // shared by the copy constructor.)
    if (t->data() == nullptr && t->device_data() == nullptr)
    {
        auto d = new char[t->mem_size()];
        memset(d, 0, t->mem_size());
        set_data(d, t->mem_size());
    }
}
ConstTensor::ConstTensor(ConstTensorPtr t) : Tensor(*t)
{
    set_const_tensor(true);
}

} // namespace aipubt

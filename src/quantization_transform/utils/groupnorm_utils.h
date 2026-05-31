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
 * @Date: 2022-03-18 13:22:35
 * @Last Modified by: Ruodi Lu
 * @Last Modified time: 2022-05-24 18:10:59
 */

#ifndef __GROUPNORM_UTILS_H__
#define __GROUPNORM_UTILS_H__

#include "quantization_transform/common.h"
#include "quantization_transform/utils/quantize_utils.h"
#include "quantization_transform/utils/data_type.h"

namespace groupnorm_utils
{

void groupnorm_forward(aipubt::NodePtr node);
void groupnorm_quantize(aipubt::NodePtr node);
}; // namespace groupnorm_utils

#endif

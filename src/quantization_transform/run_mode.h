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
 * @Date: 2022-03-11 18:43:07
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-03 22:25:25
 */

#ifndef __RUNMODE_H__
#define __RUNMODE_H__

#include "quantization_transform/common.h"

namespace qtlibn
{

enum RunMode
{
    Sym = 0,
    Asym = 1,
    FSym = 2,
    FAsym = 3,
    FFAsym = 4,
    PFAsym = 5,

    RunModeSize,
};
const std::string runmode_to_string(const RunMode &runmode);
const RunMode string_to_runmode(const std::string &name);
aipubt::Quantization::Scheme runmode_to_scheme(const RunMode runmode);
aipubt::Quantization::Scheme string_to_scheme(const std::string scheme);

}; // namespace qtlibn

#endif

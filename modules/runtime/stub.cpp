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
 * @Date: 2024-08-21 14:10:22 
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-08-21 14:15:27
 */
#include "wrapper.h"

namespace aipubt
{
using namespace AIPUDriverWrapper;

static bool run_on_driver(const std::string &aipubin, const std::vector<TensorPtr> &inputs, std::vector<TensorPtr> &outputs, const std::string &output_prefix, bool is_dump)
{
    AIPUJob job(aipubin);
    job.set_dump(is_dump);
    job.set_output_prefix(output_prefix);
    job.run(inputs);
    job.get_outputs(outputs);
    return true;
}
ADD_STUB(run_on_driver);
} // namespace aipubt
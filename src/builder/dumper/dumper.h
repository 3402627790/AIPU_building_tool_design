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
// This entire notice must be reproduced on all copies of this file and copies
// of this file may only be made by a person if such person is permitted to do
// so under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Kyle Huang
 * @Date: 2023-03-20 17:00:00
 * @Last Modified by: Kyle Huang
 * @Last Modified time: 2023-03-20 17:00:00
 */
#ifndef BUILDER_DUMPER_H
#define BUILDER_DUMPER_H

#include "builder/builder2.h"
#include <vector>

namespace aipubt
{
class Dumper
{
public:
    virtual bool gen_dumper_files(const std::vector<TensorPtr> &input_tensors, const std::string &dump_prefix, const std::string &output_file) = 0;
    virtual ~Dumper(){};
};
};     // namespace aipubt
#endif //!BUILDER_DUMPER_H
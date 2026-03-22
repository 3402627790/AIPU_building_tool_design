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
 * @Author: Neo Wang
 * @Date: 2020-05-27 10:44:17
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2022-03-07 14:16:53
 */
#ifndef BUILDER_LINKER_LINKER_H
#define BUILDER_LINKER_LINKER_H
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <stdint.h>
#include <string>
#include <vector>

namespace aipubt
{
class LinkerImpl
{
public:
    LinkerImpl() = default;
    virtual ~LinkerImpl() = default;
    virtual void set_code(const std::string &) {}
    virtual void set_codes(std::vector<std::vector<std::string>> &codes) = 0;
    virtual void set_objects(std::vector<std::string> &) {}
    //return the main offset, now we return the first lib's offset
    virtual uint32_t link(std::ostream &oss) = 0;
};
} // namespace aipubt

#endif //! BUILDER_LINKER_LINKER_H
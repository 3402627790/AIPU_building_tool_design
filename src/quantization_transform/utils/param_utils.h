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
 * @Date: 2022-03-13 22:16:06
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-06-02 17:27:08
 */

#ifndef __PARAM_UTILS_H__
#define __PARAM_UTILS_H__

#include "quantization_transform/common.h"

namespace param_utils
{

aipubt::NodeParamValue get_param(aipubt::NodePtr node, const std::string key);

aipubt::Any get_attr(aipubt::NodePtr node, const std::string key);
aipubt::NodeParamValue get_param(std::map<std::string, aipubt::NodeParamValue> &params,
                                 const std::string &key,
                                 bool default_option,
                                 aipubt::NodeParamValue default_value);
int32_t get_lut_items_in_bits(aipubt::NodePtr n);
template <typename T>
T get_attr(std::map<std::string, aipubt::Any> &attrs,
           const std::string &key,
           const bool default_option,
           T default_value)
{
    T ret;
    if (attrs.count(key))
    {
        ret = attrs[key];
    }
    else if (default_option)
    {
        ret = default_value;
    }
    return ret;
}

}; // namespace param_utils

#endif
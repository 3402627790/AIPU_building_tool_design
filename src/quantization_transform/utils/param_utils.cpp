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
 * @Date: 2022-03-14 10:24:39
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-06-02 17:23:40
 */

#include "quantization_transform/utils/param_utils.h"

namespace param_utils
{

aipubt::NodeParamValue get_param(aipubt::NodePtr node, const std::string key)
{
    if (node->params().count(key))
    {
        return (node->params()[key]);
    }
    else
    {
        ERROR() << node->name() << " has no '" << key << "' in params, please check IR.";
        return aipubt::NodeParamValue();
    }
}

aipubt::Any get_attr(aipubt::NodePtr node, const std::string key)
{
    if (node->attrs().count(key))
    {
        return (node->get_attr(key));
    }
    else
    {
        ERROR() << node->name() << " has no '" << key << "' in attrs, please check config.";
        return aipubt::Any();
    }
}

aipubt::NodeParamValue get_param(std::map<std::string, aipubt::NodeParamValue> &params,
                                 const std::string &key,
                                 bool default_option,
                                 aipubt::NodeParamValue default_value)
{
    aipubt::NodeParamValue ret;
    if (params.count(key))
    {
        ret = params[key];
    }
    else if (default_option)
    {
        ret = default_value;
    }
    return ret;
}

int32_t get_lut_items_in_bits(aipubt::NodePtr node)
{
    int32_t l_bits = -1;
    std::vector<int32_t> in_lut_bits;
    for (auto inp : node->inputs())
    {
        in_lut_bits.push_back(inp->quantization().bits());
    }
    auto max_lut_bits = *std::max_element(in_lut_bits.begin(), in_lut_bits.end());

    if (node->attrs().count(Constants::lut_items_in_bits))
    {
        l_bits = node->get_attr<aipubt::NodeParamValue>(Constants::lut_items_in_bits);
    }
    else
    {
        if (max_lut_bits != -1)
        {
            l_bits = max_lut_bits <= 8 ? max_lut_bits : std::min(max_lut_bits, 12);
        }
    }

    if (l_bits <= 8 && max_lut_bits > 8)
    {
        l_bits = std::min(max_lut_bits, 12);
    }

    return l_bits;
}

}; // namespace param_utils
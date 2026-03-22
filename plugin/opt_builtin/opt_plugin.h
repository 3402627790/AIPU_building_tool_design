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
 * @Author: Lang Hu
 * @Date: 2026-03-05 10:22:49
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2026-03-05 10:22:49
 */
#pragma once
#include <builder/plugin.h>
#include <gcompiler_api.h>
#include <utils/utils.h>

#define FUSED_PLUGIN_SCORE UINT32_MAX
#define SINGLE_PLUGIN_SCORE (FUSED_PLUGIN_SCORE - 1)

namespace aipubt
{
inline bool check_plugin_option(const std::string &opt)
{
    auto options = BuilderConfig::get<std::map<std::string, std::string>>("op_plugin_options");
    return options.count(opt) > 0 && std::stoi(options[opt]) != 0;
}

inline TensorPtr set_zerocopy_params(BaseNodePtr sgn,
                                     const std::vector<uint32_t> &params,
                                     const std::string &cst_name)
{
    auto size = params.size();
    auto data = new int[size];
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = params[i];
    }
    auto ret = std::make_shared<Tensor>(cst_name, TensorShape(size), Tensor::INT32);
    ret->set_data(data, size * sizeof(int));
    ret->set_attr(Constants::zerocopy_cst, true);
    sgn->constants()[cst_name] = ret;
    return ret;
}

BuilderParams gen_conv_desc(BaseNodePtr sgnode,
                            BaseNodePtr node,
                            const std::string &target,
                            size_t &max_aiff_hw,
                            size_t &split_num,
                            std::vector<TensorPtr> &constants,
                            std::vector<uint32_t> &splits);

BuilderParams gen_eltwise_desc(BaseNodePtr sgnode,
                               BaseNodePtr node,
                               const std::string &target,
                               std::vector<uint32_t> &params);

BuilderParams gen_lut_desc(BaseNodePtr sgnode,
                           BaseNodePtr node,
                           const std::string &target,
                           std::vector<TensorPtr> &constants);
} // namespace aipubt

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
 * @Date: 2022-03-13 21:01:25
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2022-08-03 22:27:30
 */

#include "quantization_transform/run_mode.h"
#include "quantization_transform/transforms/register.h"

namespace qtlibn
{

const std::string runmode_to_string(const RunMode &run_mode)
{
    auto register_transform_names = TransformRegistry::get_transform_name_map_entry();
    return register_transform_names[run_mode];
}

const RunMode string_to_runmode(const std::string &name)
{
    auto register_transform_names = TransformRegistry::get_transform_name_map_entry();
    for (auto kv : register_transform_names)
    {
        if (kv.second == name)
        {
            return kv.first;
        }
    }
    ERROR() << "Qtlib cannot change the " << name << " to RunMode.";
    return RunModeSize;
}

aipubt::Quantization::Scheme runmode_to_scheme(const RunMode runmode)
{
    std::map<RunMode, aipubt::Quantization::Scheme> _RUNMODE_TO_SCHEME_MAP = {
        {RunMode::Asym, aipubt::Quantization::Scheme::Asymmetric},
        {RunMode::FSym, aipubt::Quantization::Scheme::Symmetric},
        {RunMode::Sym, aipubt::Quantization::Scheme::Symmetric},
        {RunMode::FAsym, aipubt::Quantization::Scheme::Asymmetric},
        {RunMode::FFAsym, aipubt::Quantization::Scheme::Asymmetric},
        {RunMode::PFAsym, aipubt::Quantization::Scheme::Asymmetric},
    };

    return _RUNMODE_TO_SCHEME_MAP[runmode];
}

aipubt::Quantization::Scheme string_to_scheme(const std::string scheme)
{
    std::map<std::string, aipubt::Quantization::Scheme> _STR_TO_SCHEME_MAP = {
        {"asymmetric", aipubt::Quantization::Scheme::Asymmetric},
        {"symmetric", aipubt::Quantization::Scheme::Symmetric},
    };
    std::string scm = scheme;
    std::transform(scm.begin(), scm.end(), scm.begin(), ::tolower);
    if (!_STR_TO_SCHEME_MAP.count(scm))
    {
        ERROR() << "string scheme = " << scm
                << " should be in [Asymmetric, Symmetric], and qtlib we change scheme to Symmetric.";
        scm = "symmetric";
    }
    return _STR_TO_SCHEME_MAP[scm];
}

}; // namespace qtlibn
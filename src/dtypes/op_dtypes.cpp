//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Tom.Gao
 * @Date: 2022-03-03 14:06:21
 * @Last Modified by: Samuel Han
 * @Last Modified time: 2023-12-21 17:17:30
 */
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

#include "builder/plugin.h"
#include "builder/target.h"
#include "checker/plugin.h"
#include "core/any.h"
#include "core/config.h"
#include "core/core.h"
#include "core/graph.h"
#include "core/node.h"
#include "core/plugin_loader.h"
#include "core/quantization.h"
#include "core/subgraph.h"
#include "dtypes/plugin.h"
#include "op_dtypes.h"
#include "utils/string_utils.h"
#include "utils/utils.h"
#include <iterator>
#include <sstream>

namespace aipubt
{

std::string Op_Dtype::to_string(const Op_Dtype &dtype)
{
    std::stringstream ss;
    auto print_dtype_vec = [&ss](const std::vector<Tensor::Dtype> &dtype_vec) {
        static const char *const delim = ", ";
        ss << "[";
        std::copy(dtype_vec.begin(), std::prev(dtype_vec.end(), 1), std::ostream_iterator<Tensor::Dtype>(ss, delim));
        if (dtype_vec.size() > 0)
        {
            ss << dtype_vec.back();
        }
        ss << "]";
    };
    ss << "input: ";
    print_dtype_vec(dtype.in_dtypes);
    ss << ", output: ";
    print_dtype_vec(dtype.out_dtypes);
    return ss.str();
}

bool Op_Dtype::operator<(const Op_Dtype &op_dtype) const
{
    const std::string lhs = this->to_string(*this);
    const std::string rhs = this->to_string(op_dtype);
    return (lhs < rhs) ? true : false;
}

std::ostream &operator<<(std::ostream &os, const Op_Dtype &dtype)
{
    return os << Op_Dtype::to_string(dtype);
}

std::vector<Op_Dtype> Op_Dtypes::get_dtypes(OpType optype)
{
    static std::vector<CPluginInfo *> plugins;
    if (plugins.size() == 0)
    {
        PluginLoader *plugin_loader = PluginLoader::get_instance();
        plugins = ((*plugin_loader)[PluginType_DtypeOp]);
        if (plugins.size() == 0)
        {
            ERROR() << "No plugins for Op_Dtype = " << optype << ", please check your plugin path";
            return {};
        }
    }

    std::vector<Op_Dtype> valid_dtypes;
    std::set<Op_Dtype> valid_dtypes_set;
    for (size_t idx = 0; idx < plugins.size(); ++idx)
    {
        auto m_plugin = reinterpret_cast<DtypeOpPluginInfo *>(plugins[idx]);
        auto opt = m_plugin->init_handle();
        auto cur_plugin_valid_dtypes = opt->get_valid_dtypes(optype);
        valid_dtypes_set.insert(cur_plugin_valid_dtypes.begin(), cur_plugin_valid_dtypes.end());
    }
    valid_dtypes.assign(valid_dtypes_set.begin(), valid_dtypes_set.end());
    return valid_dtypes;
}

}; // namespace aipubt

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
 * @Date: 2023-07-05 15:36:34
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2023-09-13 14:15:07
 */
#ifndef STATISTIC_PROFILE_DATA_PARSER_H
#define STATISTIC_PROFILE_DATA_PARSER_H

#include <core/core.h>
#include <json/json.h>

namespace aipubt
{

using value_t = nlohmann::detail::value_t;
using json = nlohmann::json;

class ProfileDataParser
{
public:
    ProfileDataParser(const std::string &graph_json_path, const std::string &binary_path);
    ProfileDataParser() = delete;

    void parse(json &res);

protected:
    bool fetch_from_binary_x2(json &subgraphs);
    void build_graph_info_x2(json &subgraphs, json &res);

    json m_graph_json;
    std::string m_binary;
};

} // namespace aipubt

#endif // STATISTIC_PROFILE_DATA_PARSER_H

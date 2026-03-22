//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2021 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
/*
 * @Author: Tom.Gao
 * @Date: 2021-01-25 17:06:21
 * @Last Modified by: Lang Hu
 * @Last Modified time: 2024-08-21 10:51:09
 */
#ifndef CHECKER_CHECKER_H
#define CHECKER_CHECKER_H

#include "core/graph.h"
#include "core/plugin_loader.h"

namespace aipubt
{
class IRChecker
{
public:
    IRChecker(const std::string &filename, GraphPtr g);
    IRChecker(std::istream &infile, GraphPtr g);
    bool check_all(bool enable_plugin = true);
    static bool check_graph(GraphPtr g);

protected:
    using Section = std::map<std::string, std::pair<std::string, long>>;
    using SectionList = std::vector<Section>;
    using SectionTable = std::map<std::string, Section>;

    static bool check_graph(GraphPtr g, const SectionTable &sec_table, const std::string &file_name);
    bool check_params();
    bool preprocess(SectionList &sec_list, const std::string &graph_name, const std::string &file_name);
    bool link_subgraphs(GraphPtr g, const std::string &graph_name, std::set<std::string> &linked);
    bool parse_and_syntax();

    SectionList m_section_list;
    GraphPtr m_graph;
    std::string m_file_name;
    std::map<std::string, SectionList> m_section_lists;
    std::map<std::string, GraphPtr> m_graphs;
    std::map<std::string, std::string> m_file_names;
    std::map<std::string, SectionTable> m_tables; // debug messages
    std::vector<CPluginInfo *> m_plugins;
};
bool is_DAG(GraphPtr g);
} // namespace aipubt

#endif //!CHECKER_CHECKER_H
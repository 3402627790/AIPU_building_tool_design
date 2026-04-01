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
 * @Last Modified by: Neo Wang
 * @Last Modified time: 2024-11-14 15:14:02
 */
#include <fstream>
#include <iostream>
#include <sstream>

#include "builder/plugin.h"
#include "builder/target.h"
#include "builder/tools.h"
#include "checker/checker.h"
#include "checker/plugin.h"
#include "core/any.h"
#include "core/config.h"
#include "core/core.h"
#include "core/ifnode.h"
#include "core/loopnode.h"
#include "core/node.h"
#include "core/quantization.h"
#include "core/subgraph.h"
#include "utils/utils.h"

#undef LOG_MODULE
#define LOG_MODULE "[IRChecker] "
#undef LOG_FILEINFO
#define LOG_FILEINFO ""

namespace aipubt
{
using Section = std::map<std::string, std::pair<std::string, long>>;
using SectionList = std::vector<Section>;
using SectionTable = std::map<std::string, Section>;

static constexpr const char *_default_graph = "_default_graph";
static constexpr const char *_global_section = "_global_section";
static const std::vector<std::string> required_head_params = {
    // "model_name",
    "layer_number",
    // "data_format",
    // "batch_size",
};
static const std::vector<std::string> required_layer_params = {
    "layer_id",
    "layer_name",
    "layer_type",
    "layer_bottom",
    "layer_bottom_shape",
    "layer_bottom_type",
    "layer_top",
    "layer_top_shape",
    "layer_top_type",
};

static void set_relu_type(NodePtr node)
{
    auto &params = node->params();
    if (params.count("with_activation") && (std::string)params["with_activation"] != Constants::act_none)
    {
        if (params.count("negative_slope_size") && (long)params["negative_slope_size"] != 0)
        {
            node->set_attr("relu_type", "p_relu");
        }
        else if (params.count("negative_slope_value") && (long)params["negative_slope_value"] != 0)
        {
            node->set_attr("relu_type", "leaky_relu");
        }
        else
        {
            node->set_attr("relu_type", "relu");
        }
    }
    else
    {
        node->set_attr("relu_type", "no_relu");
    }
}

inline void CHK_ERROR(long line_num, const std::string &key, const std::string &msg, const std::string &file_name = "IR_line")
{
    ERROR() << file_name << ":" << line_num << ", key:'" << key << "', MSG: " << msg;
}

inline void CHK_FATAL(long line_num, const std::string &key, const std::string &msg, const std::string &file_name = "IR_line")
{
    FATAL() << file_name << ":" << line_num << ", key:'" << key << "', MSG: " << msg;
}

// map<key, pair<value, line_num>>
static TensorList parse_tensors(Section &section, const std::string &prefix, const std::string &file_name)
{
    std::vector<std::string> outputs = Node::parse_param_value(section[prefix].first);

    std::vector<Tensor::Dtype> out_dtypes = Node::parse_param_value(section[prefix + "_type"].first);
    auto out_shapes = utils::split_string(section[prefix + "_shape"].first, "],", 1);
    if (outputs.size() != out_dtypes.size())
    {
        auto key = prefix + "_type";
        CHK_ERROR(section[key].second, key, "Invalid size: " + section[key].first, file_name);
    }
    if (outputs.size() != out_shapes.size())
    {
        auto key = prefix + "_shape";
        CHK_ERROR(section[key].second, key, "Invalid size: " + section[key].first, file_name);
    }
    std::vector<DataLayout> out_layouts;
    if (section.count(prefix + "_datalayout"))
    {
        std::vector<std::string> layouts = Node::parse_param_value(section[prefix + "_datalayout"].first);
        if (outputs.size() != layouts.size())
        {
            auto key = prefix + "_datalayout";
            CHK_ERROR(section[key].second, key, "Invalid size: " + section[key].first, file_name);
        }
        for (size_t i = 0; i < layouts.size(); ++i)
        {
            auto layout = string_to_datalayout(layouts[i]);
            if (layout == DataLayout_UnSupport)
            {
                layout = DataLayout_Flat;
                auto key = prefix + "_datalayout";
                CHK_ERROR(section[key].second, key, "Unknow dataLayout: " + section[key].first, file_name);
            }
            out_layouts.push_back(layout);
        }
    }
    else
    {
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            TensorShape shape(out_shapes[i]);
            if (shape.dim() == 4)
            {
                out_layouts.push_back(DataLayout_NHWC);
            }
            else if (shape.dim() == 5)
            {
                out_layouts.push_back(DataLayout_NDHWC);
            }
            else
            {
                out_layouts.push_back(DataLayout_Flat);
            }
        }
    }
    std::vector<float> scales;
    if (section.count(prefix + "_scale"))
    {
        BuilderConfig::set(Constants::serialize_scale_zp, true);
        scales = Node::parse_param_value(section[prefix + "_scale"].first);
        if (outputs.size() != scales.size())
        {
            auto key = prefix + "_scale";
            CHK_ERROR(section[key].second, key, "Invalid size: " + section[key].first, file_name);
        }
    }
    else
    {
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            scales.push_back(1.0);
        }
    }
    std::vector<int32_t> zps;
    if (section.count(prefix + "_zp"))
    {
        BuilderConfig::set(Constants::serialize_scale_zp, true);
        zps = Node::parse_param_value(section[prefix + "_zp"].first);
        if (outputs.size() != zps.size())
        {
            auto key = prefix + "_zp";
            CHK_ERROR(section[key].second, key, "Invalid size: " + section[key].first, file_name);
        }
    }
    else
    {
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            zps.push_back(0);
        }
    }

    std::vector<std::vector<float>> ranges;
    if (section.count(prefix + "_range"))
    {
        ranges = Node::parse_param_value(section[prefix + "_range"].first);
        if (outputs.size() != ranges.size())
        {
            auto key = prefix + "_range";
            CHK_ERROR(section[key].second, key, "Invalid size: " + section[key].first, file_name);
        }
    }
    TensorList ret;
    if (outputs.size() == 1 && outputs[0] == "")
    {
        return TensorList();
    }

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        //TODO: check syntax of shape
        TensorShape shape(out_shapes[i]);
        for (int j = 0; j < shape.dim(); ++j)
        {
            if (shape[j] <= 0)
            {
                auto key = prefix + "_shape";
                CHK_ERROR(section[key].second, key, "Invalid tensor shape: " + section[key].first, file_name);
                break;
            }
        }

        // default layout of all tensors is the graph data layout
        shape.set_layout(out_layouts[i]);
        TensorPtr tensor = std::make_shared<Tensor>(outputs[i], shape);
        if (out_dtypes.size() <= i || out_dtypes[i] == Tensor::Dtype::VOID)
        {
            auto key = prefix + "_type";
            CHK_ERROR(section[key].second, key, "Invalid tensor type: " + section[key].first, file_name);
        }
        else
        {
            tensor->set_type((out_dtypes[i]));
        }
        tensor->quantization() = Quantization(scales[i], zps[i]);
        if (ranges.size())
        {
            tensor->set_attr("range", ranges[i]);
            tensor->quantization().set_min(ranges[i][0]);
            tensor->quantization().set_max(ranges[i][1]);
        }
        ret.push_back(tensor);
    }
    return ret;
}

static SectionList sectionize_file(std::istream &infile, const std::string &file_name, bool &ret)
{
    auto is_valid_key = [](const std::string &line) -> bool {
        for (size_t i = 0; i < line.size(); ++i)
        {
            char c = line[i];
            bool ret = (c >= '0' && c <= '9') ||
                       (c >= 'A' && c <= 'Z') ||
                       (c >= 'a' && c <= 'z') ||
                       (c == '_');
            if (!ret)
            {
                return false;
            }
        }
        return true;
    };
    auto is_valid_path = [](const std::string &line) -> bool {
        for (size_t i = 0; i < line.size(); ++i)
        {
            if (line[i] == ' ')
            {
                return false;
            }
        }
        return true;
    };
    auto check_brackets = [](const std::string &line) -> bool {
        auto num_l = std::count(line.begin(), line.end(), '[');
        auto num_r = std::count(line.begin(), line.end(), ']');
        if (num_l == 0 && num_r == 0)
            return true;
        if (num_l != num_r || line[0] != '[' || line[line.size() - 1] != ']')
            return false;
        // check symmetry
        uint32_t num = 0;
        for (size_t i = 0; i < line.size(); ++i)
        {
            if (line[i] == '[')
                num++;
            else if (line[i] == ']')
            {
                if (num == 0)
                    return false;
                num--;
            }
        }
        return (num == 0);
    };
    auto is_valid_value = [](std::string line) -> bool {
        // line = "[[1,10,10,11]]";
        line = utils::replace(line, "[", "");
        line = utils::replace(line, "]", ",");
        for (auto &&str : utils::split_string(line, ','))
        {
            if (utils::is_number(str))
            {
                continue;
            }
            for (size_t i = 0; i < str.size(); ++i)
            {
                char c = str[i];
                bool ret = (c >= '0' && c <= '9') ||
                           (c >= 'A' && c <= 'Z') ||
                           (c >= 'a' && c <= 'z') ||
                           (c == '_') ||
                           (c == '-') || //TODO: remove me when parser fix kws_gru
                           (c == '.') ||
                           (c == '/') ||
                           (c == ':') ||
                           (c == ';') ||
                           (c == '+') || // "+" - ")" are added for dynamic shape expression
                           (c == '*') ||
                           (c == '%') ||
                           (c == '(') ||
                           (c == ')');
                if (!ret)
                    return false;
            }
        }
        return true;
    };

    // INFO() << "parsing graph: " << file_name;
    SectionList sections;
    Section section;
    std::string line;
    long line_num = 1;
    while (std::getline(infile, line))
    {
        line = utils::trim(line);
        if (line.empty())
        {
            if (section.size())
            {
                sections.push_back(std::move(section));
                section = Section();
            }
        }
        else
        {
            if (line.find("=") == std::string::npos)
            {
                CHK_ERROR(line_num, line, "\"=\" is required", file_name);
                ret = false;
            }
            else
            {
                std::istringstream iss(line);
                std::string key, value;
                std::getline(iss, key, '=');
                std::getline(iss, value);
                key = utils::trim(key);
                value = utils::trim(value);
                if (!is_valid_key(key))
                {
                    std::string char_set = "'a-z', 'A-Z', '0-9', '_'";
                    CHK_ERROR(line_num, key, "Invalid symbol in key \"" + key + "\", expected " + char_set, file_name);
                    ret = false;
                }
                if (key == "model_bin")
                {
                    if (!is_valid_path(value))
                    {
                        CHK_ERROR(line_num, line, "Invalid model_bin path. ' ' is NOT allowed in the middle of path", file_name);
                        ret = false;
                    }
                }
                else
                {
                    if (value.empty())
                    {
                        CHK_ERROR(line_num, key, "Invalid value. Try to use: \"" + key + "=[]\" if it's a empty list", file_name);
                        ret = false;
                    }
                    if (!check_brackets(value))
                    {
                        CHK_ERROR(line_num, line, "Invalid value about '[' or ']'", file_name);
                        ret = false;
                    }
                    if (!is_valid_value(value))
                    {
                        std::string char_set = "'a-z', 'A-Z', '0-9', '_', '-', '.', '/', ':', ';', '+', '*', '%', '(', ')'";
                        CHK_ERROR(line_num, line, "Invalid symbol in value \"" + value + "\", expected " + char_set, file_name);
                        ret = false;
                    }
                }
                section[key] = {value, line_num};
            }
        }
        line_num++;
    }
    if (section.size())
    {
        sections.push_back(std::move(section));
        section = Section();
    }

    if (sections.empty())
    {
        ERROR() << "Empty IR!";
        ret = false;
    }
    return sections;
}

static SectionList sectionize_file(const std::string &file_name, bool &ret)
{
    std::ifstream infile;
    infile.open(file_name);
    auto list = sectionize_file(infile, file_name, ret);
    infile.close();
    return list;
}

bool is_DAG(GraphPtr g)
{
    std::set<std::string> visited_tensor_names;
    for (auto n : g->nodes())
    {
        for (auto t : n->outputs())
        {
            auto &name = t->name();
            if (visited_tensor_names.find(name) == visited_tensor_names.end())
            {
                visited_tensor_names.insert(name);
            }
            else
            {
                return false;
            }
        }
    }
    return true;
}

static bool is_connected_graph(GraphPtr g)
{
    std::function<void(NodePtr, std::set<NodePtr> &)> dfs;
    dfs = [&dfs](NodePtr n, std::set<NodePtr> &visited) {
        // INFO() << n->name();
        visited.insert(n);
        for (auto p : n->parents())
        {
            if (visited.find(p) == visited.end())
            {
                dfs(p, visited);
            }
        }
        for (auto c : n->children())
        {
            if (visited.find(c) == visited.end())
            {
                dfs(c, visited);
            }
        }
    };

    std::set<NodePtr> visited;
    if (g->get_input_tensors().size())
    {
        dfs(g->get_input_tensors()[0]->node(), visited);
    }
    else if (g->get_output_tensors().size())
    {
        dfs(g->get_output_tensors()[0]->node(), visited);
    }

    return visited.size() == g->nodes().size();
}

static bool is_empty_graph(GraphPtr g)
{
    for (auto n : g->nodes())
    {
        if (n->type() != OpType_Input)
        {
            return false;
        }
    }
    return true;
}

static SectionTable parse_sections(SectionList &sections, GraphPtr g, bool with_header, const std::string &file_name, bool &ret)
{
    SectionTable sec_table;
    if (sections.empty())
    {
        return sec_table;
    }
    if (!g)
    {
        CHK_ERROR(1, "graph", "Invalid graph pointer!", file_name);
        ret = false;
        return sec_table;
    }
    size_t layer_num = sections.size();
    size_t start_section = with_header ? 1 : 0;
    if (with_header)
    {
        layer_num = layer_num - 1;
        // check required params
        for (auto &key : required_head_params)
        {
            if (!sections[0].count(key))
            {
                CHK_ERROR(1, key, "Parameter required", file_name);
                ret = false;
            }
        }
        if (sections[0].count("model_name"))
        {
            INFO() << "model_name: " << sections[0]["model_name"].first;
        }
        if (sections[0].count("subgraph_name"))
        {
            INFO() << "subgraph_name: " << sections[0]["subgraph_name"].first;
        }
        // parse global keys
        for (auto &kv : sections[0])
        {
            sec_table[_global_section][kv.first] = kv.second;
            if (kv.first == "model_name")
            {
                g->set_name(kv.second.first);
            }
            else if (kv.first == "subgraph_name")
            {
                g->set_name(kv.second.first);
            }
            // else if (kv.first == "workspace")
            // {
            //     g->global_tensors()["workspace"]->shape()[0] = std::stoi(kv.second.first);
            // }
            else if (kv.first == "layer_number")
            {
                auto num = std::stoi(kv.second.first);
                if (num == 0)
                {
                    CHK_ERROR(kv.second.second, kv.first, "layer_number=0, and IR is regarded as empty!", file_name);
                    ret = false;
                }
                else if (num > 0 && num < (int)layer_num)
                {
                    layer_num = num;
                }
            }
            else if (kv.first == "model_bin")
            {
                g->set_attr("model_bin", kv.second.first);
            }
            else if (kv.first == "compat_quantized_model")
            {
                bool quantized = Node::parse_param_value(kv.second.first);
                g->set_attr("compat_quantized_model", quantized);
            }
        }
    }

    for (size_t i = start_section; i < layer_num + start_section; ++i)
    {
        auto &section = sections[i];
        for (auto &key : required_layer_params)
        {
            if (!section.count(key))
            {
                CHK_FATAL(section.begin()->second.second, key, "Parameter required", file_name);
                ret = false;
            }
            else if (section[key].first == "")
            {
                // CHK_ERROR(section[key].second, key, "Invalid param, try to use: \"" + key + "=[]\" if it's a empty list", file_name);
                ret = false;
            }
        }
    }

    // check duplicate layer_name and build lines
    for (size_t i = start_section; i < layer_num + start_section; ++i)
    {
        auto &section = sections[i];
        auto &layer_name = section["layer_name"].first;
        auto line_num = section["layer_name"].second;
        if (sec_table.count(layer_name))
        {
            CHK_FATAL(line_num, "layer_name", "Duplicate layer_name", file_name);
            ret = false;
        }
        else
        {
            for (auto &kv : section)
            {
                sec_table[layer_name][kv.first] = kv.second;
            }
        }
    }

    std::set<int> layer_id_set;
    std::map<std::string, std::pair<std::vector<NodePtr>, TensorPtr>> tensor_map;
    // parse node and tensors
    for (size_t i = start_section; i < layer_num + start_section; ++i)
    {
        auto &section = sections[i];
        auto &layer_type = section["layer_type"].first;
        auto op_type = string_to_optype(layer_type);
        if (op_type == OpType_NoOp)
        {
            auto key = "layer_type";
            auto recommended_type = string_upper_to_optype(layer_type);
            if (recommended_type != OpType_NoOp)
            {
                CHK_ERROR(section[key].second,
                          key,
                          "Unsupported layer_type: " + section[key].first + ", recommended: " + optype_to_string(recommended_type),
                          file_name);
            }
            else
            {
                CHK_ERROR(section[key].second, key, "Unsupported layer_type: " + section[key].first, file_name);
            }
            ret = false;
        }

        // check layout
        const std::vector<OpType> OP_3d_list = {
            OpType_Convolution3D,
            OpType_ConvTranspose3D,
            OpType_Pooling3D,
        };
        bool is_3d_node = false;
        if (std::count(OP_3d_list.begin(), OP_3d_list.end(), op_type) > 0)
        {
            is_3d_node = true;
        }

        bool is_3d_tensor = is_3d_node;
        if (section.count("layout_top_datalayout") > 0)
        {
            auto layout = string_to_datalayout(section["layout_top_datalayout"].first);
            is_3d_tensor = (layout == DataLayout_NDHWC ||
                            layout == DataLayout_NDCHWC16 ||
                            layout == DataLayout_NDCHWC32 ||
                            layout == DataLayout_NCDHWC16 ||
                            layout == DataLayout_NCDHWC32);
        }
        if (is_3d_node && !is_3d_tensor)
        {
            CHK_ERROR(section["layout_top_datalayout"].second,
                      "layout_top_datalayout",
                      "layout_top_datalayout should be 3D (NDHWC/NDCHWC16/NDCHWC32) for 3D layers!",
                      file_name);
            ret = false;
        }

        std::shared_ptr<Node> node = Node::create(section["layer_name"].first, op_type);
        node->set_loc(std::make_shared<Location>(file_name + ":" + std::to_string(section["layer_name"].second), op_type, true));
        for (auto tensor : parse_tensors(section, "layer_top", file_name))
        {
            auto &name = tensor->name();
            if (!tensor_map.count(name))
            {
                tensor_map[name] = {{node}, tensor};
            }
            else
            {
                tensor_map[name].first.push_back(node);
                tensor = tensor_map[name].second;
            }
            node->add_output(tensor);
        }
        for (auto &kv : sections[i])
        {
            if (kv.first.find("layer_") != std::string::npos)
            {
                //TODO: just remove the global keys, not all starts with "layer_"
                continue;
            }
            //TODO: check syntax of optional params
            bool has_expr = utils::startswith(kv.first, "ds_");
            (*node)[kv.first] = Node::parse_param_value(kv.second.first, has_expr);
        }
        int layer_id = Node::parse_param_value(section["layer_id"].first);
        if (layer_id_set.count(layer_id))
        {
            CHK_ERROR(section["layer_id"].second, "layer_id", "Duplicate layer_id", file_name);
            ret = false;
        }
        else if (layer_id < 0)
        {
            auto key = "layer_id";
            CHK_ERROR(section[key].second, key, "Unsupported layer_id: " + section[key].first, file_name);
            ret = false;
        }
        else
        {
            node->set_attr("layer_id", layer_id);
            layer_id_set.insert(layer_id);
        }
        g->add_node(node);
    }
    for (size_t i = start_section; i < layer_num + start_section; ++i)
    {
        auto &section = sections[i];
        for (auto inp : parse_tensors(section, "layer_bottom", file_name))
        {
            if (tensor_map.count(inp->name()) == 0)
            {
                auto key = "layer_bottom";
                CHK_ERROR(section[key].second,
                          key,
                          "No corresponding layer_top to this layer_bottom: " + section[key].first,
                          file_name);
                ret = false;
            }
            else
            {
                auto out = tensor_map[inp->name()].second;
                inp->shape().set_layout(out->shape().layout());
                if (inp->shape() != out->shape())
                {
                    auto key = "layer_bottom_shape";
                    auto msg = "Bottom and top should keep the same shape. Top shape: " + out->shape().to_string() +
                               ", bottom shape: " + inp->shape().to_string();
                    CHK_ERROR(section[key].second, key, msg, file_name);
                    ret = false;
                }
                if (inp->type() != out->type())
                {
                    auto key = "layer_bottom_type";
                    auto msg = "Bottom and top should keep the same type. Top type: " + Tensor::dtype_to_string(out->type()) +
                               ", bottom type: " + Tensor::dtype_to_string(inp->type());
                    CHK_ERROR(section[key].second, key, msg, file_name);
                    ret = false;
                }
            }
        }
    }

    // link nodes
    for (size_t i = start_section; i < layer_num + start_section; ++i)
    {
        auto &section = sections[i];
        auto node = (*g)[section["layer_name"].first];
        std::vector<std::string> inputs = Node::parse_param_value(sections[i]["layer_bottom"].first);
        for (auto inp : inputs)
        {
            if (inp.empty())
            {
                continue;
            }
            if (tensor_map.count(inp))
            {
                auto &kv = tensor_map[inp];
                // kv.first->add_child(node);
                node->add_input(kv.second);
            }
            else
            {
                auto key = "layer_bottom";
                CHK_ERROR(section[key].second, key, "It is not an output of any node: " + section[key].first, file_name);
                ret = false;
            }
        }
    }

    if (is_empty_graph(g))
    {
        WARN() << "Empty IR, and no real layers in this graph";
    }

    if (!is_DAG(g))
    {
        std::string png_name = "Non-DAG.png";
        ERROR() << "Directed acyclic graph (DAG) is required. Please refer to the png file: " << utils::get_abspath(png_name);
        g->to_png(png_name);
        ret = false;
    }

    if (!is_connected_graph(g))
    {
        std::string png_name = "Non-Connected-graph.png";
        ERROR() << "Connected graph is required. Please refer to the png file: " << utils::get_abspath(png_name);
        g->to_png(png_name);
        ret = false;
    }

    for (auto node : *g)
    {
        set_relu_type(node);
        auto &params = node->params();
        std::vector<std::string> constants;
        std::vector<std::string> scales;
        for (auto &kv : params)
        {
            if (utils::endswith(kv.first, "_offset"))
            {
                auto key = kv.first;
                utils::replace(key, "_offset", "");
                bool is_constant = true;
                for (auto post_fix : {"_size", "_type", "_offset", "_shape"})
                {
                    if (!params.count(key + post_fix))
                    {
                        is_constant = false;
                        break;
                    }
                }
                if (is_constant)
                {
                    constants.push_back(key);
                }
            }
            else if (utils::endswith(kv.first, "_value"))
            {
                auto key = kv.first;
                utils::replace(key, "_value", "");
                bool is_scale = true;
                for (auto post_fix : {"_value", "_type"})
                {
                    if (!params.count(key + post_fix))
                    {
                        is_scale = false;
                        break;
                    }
                }
                if (is_scale)
                {
                    scales.push_back(key);
                }
            }
        }

        for (auto &key : constants)
        {
            if (utils::startswith(key, "layer_top") &&
                params.count(key + "_size") &&
                params[key + "_size"].arr.size() == node->outputs().size())
            {
                std::vector<uint32_t> sizes = params[key + "_size"];
                std::vector<TensorShape> shape = params[key + "_shape"];
                std::vector<uint32_t> offset = params[key + "_offset"];
                std::vector<Tensor::Dtype> types = params[key + "_type"];
                if (shape.size() == sizes.size() && offset.size() == sizes.size() && types.size() == sizes.size())
                {
                    for (size_t i = 0; i < sizes.size(); ++i)
                    {
                        auto name = key;
                        utils::replace(name, "layer_top", node->outputs()[i]->name());
                        TensorPtr weight = std::make_shared<Tensor>(name, shape[i]);
                        weight->memory_offset()->relative_offset = offset[i];
                        weight->set_type(types[i]);
                        weight->set_data(nullptr, sizes[i]);
                        node->constants()[name] = weight;
                    }
                    for (auto post_fix : {"_size", "_type", "_offset", "_shape"})
                    {
                        params.erase(key + post_fix);
                    }
                }
            }
            else if (params.count(key + "_size") && static_cast<int>(params[key + "_size"]))
            {
                TensorShape shape = params[key + "_shape"];
                //trick for one dimension shape
                if (shape.size() == 1 && (int)params[key + "_shape"] != 0)
                {
                    // shape[0] = params[key + "_shape"];
                    // TODO: work around for lut_shape=256, and we should remove it later
                    shape = TensorShape((int)params[key + "_shape"]);
                }
                TensorPtr weight = std::make_shared<Tensor>(node->name() + key, shape);
                // this is file offset!!! not the memory offset TODO!!!
                weight->memory_offset()->relative_offset = params[key + "_offset"];
                weight->set_type(params[key + "_type"]);
                int size = params[key + "_size"];
                weight->set_data(nullptr, size);
                for (auto post_fix : {"_size", "_type", "_offset", "_shape"})
                {
                    params.erase(key + post_fix);
                }
                node->constants()[key] = weight;
            }
        }

        //deal with loop out
        if (node->type() == OpType_ForRange)
        {
            std::vector<std::string> outputs = Node::parse_param_value(node->params()["proxy_output_tensors"]);
            TensorList p_outputs;
            for (auto &t : outputs)
            {
                if (tensor_map.count(t))
                {
                    p_outputs.push_back(tensor_map[t].second);
                }
                else
                {
                    ERROR() << "Missing proxy output tensors!";
                }
            }
            node->set_attr("output_tensors", p_outputs);
        }
    }

    //check_input_output_tensors
    auto check_tensors = [&tensor_map, &sec_table, &file_name](std::vector<std::string> &tensor_names, const std::string &key) {
        bool has_tensors = true;
        for (auto &t : tensor_names)
        {
            if (!tensor_map.count(t))
            {
                CHK_ERROR(sec_table[_global_section][key].second, key, "Tensor \"" + t + "\" does not exits in the graph!", file_name);
                has_tensors = false;
            }
        }
        return has_tensors;
    };

    g->record_input_tensors();
    g->record_output_tensors();

    std::vector<std::string> input_tensors = Node::parse_param_value(sections[0]["input_tensors"].first);
    bool is_empty_input = input_tensors.size() == 1 && input_tensors[0].empty();
    if (input_tensors.size() > 0 && !is_empty_input)
    {
        if (check_tensors(input_tensors, "input_tensors"))
        {
            std::vector<TensorPtr> inps;
            for (auto t : input_tensors)
            {
                inps.push_back(tensor_map[t].second);
            }
            g->record_input_tensors(inps);
        }
        else
        {
            ret = false;
        }
    }

    std::vector<std::string> output_tensors = Node::parse_param_value(sections[0]["output_tensors"].first);
    if (output_tensors.size() > 0)
    {
        if (check_tensors(output_tensors, "output_tensors"))
        {
            std::vector<TensorPtr> outs;
            for (auto &t : output_tensors)
            {
                outs.push_back(tensor_map[t].second);
            }
            g->record_output_tensors(outs);
        }
        else
        {
            ret = false;
        }
    }
    else
    {
        WARN() << "\"output_tensors\" is not specified in IR. Use the out tensors of the end node instead:";
        WARN() << "[" << utils::join_string(",", g->get_output_tensors()) << "]";
    }
    for (auto inp : g->get_input_tensors())
    {
        for (auto out : g->get_output_tensors())
        {
            if (inp == out && inp->node() != out->node())
            {
                auto key = "output_tensors";
                long line = 1;
                if (sec_table[_global_section].count(key))
                {
                    line = sec_table[_global_section][key].second;
                }
                CHK_ERROR(line, key, "Output and Input should NOT share the same tensor: " + inp->name(), file_name);
                ret = false;
            }
        }
    }
    return sec_table;
}

IRChecker::IRChecker(const std::string &file_name, GraphPtr g)
{
    auto file_path = utils::get_abspath(file_name);
    m_file_name = file_path;
    if (!g)
    {
        g = Graph::create();
    }
    m_graph = g;
    bool ret = true;
    m_section_list = sectionize_file(file_path, ret);
    if (!ret)
    {
        m_section_list = SectionList();
    }
}

IRChecker::IRChecker(std::istream &infile, GraphPtr g)
{
    m_file_name = "";
    if (!g)
    {
        g = Graph::create();
    }
    m_graph = g;
    bool ret = true;
    m_section_list = sectionize_file(infile, "", ret);
    if (!ret)
    {
        m_section_list = SectionList();
    }
}

// subgraph partition and including
bool IRChecker::preprocess(SectionList &sec_list, const std::string &graph_name, const std::string &file_name)
{
    if (m_section_lists.count(graph_name))
    {
        return true;
    }
    bool ret = true;
    m_section_lists[graph_name] = SectionList();
    m_file_names[graph_name] = file_name;
    for (size_t i = 0; i < sec_list.size(); ++i)
    {
        auto &sec = sec_list[i];
        if (sec.count("subgraph_name") && sec.count("layer_number")) // subgraph declaration
        {
            auto &name = sec["subgraph_name"].first;
            size_t sec_num = std::stoi(sec["layer_number"].first) + 1;
            m_file_names[name] = file_name;
            for (size_t j = i; j < i + sec_num; ++j)
            {
                m_section_lists[name].push_back(sec_list[j]);
            }
            i += sec_num - 1;
        }
        else if (sec.count("subgraph_name") && sec.count("subgraph_input_type"))
        {
            auto &path = sec["txt_path"].first;
            auto &bin_path = sec["bin_path"].first;
            if (m_section_lists.count(path) || !utils::path_exists(path))
            {
                continue;
            }
            auto list = sectionize_file(path, ret);
            if (!ret || list.empty())
            {
                continue;
            }
            auto &name = sec["subgraph_name"].first;
            // modify model bin if bin is present
            if (bin_path.size() && utils::path_exists(bin_path))
            {
                list[0]["model_bin"].first = bin_path;
            }
            ret &= preprocess(list, name, path);
        }
        else if (sec.count("subgraph")) // subgraph scope
        {
            m_section_lists[sec["subgraph"].first].push_back(sec);
        }
        else
        {
            m_section_lists[graph_name].push_back(sec);
        }
    }
    return ret;
}

bool IRChecker::link_subgraphs(GraphPtr g, const std::string &graph_name, std::set<std::string> &linked)
{
    auto check_params = [](NodePtr node, const std::vector<std::string> &params) -> bool {
        for (auto &&param : params)
        {
            if (!node->params().count(param))
            {
                ERROR() << "Missing param \"" << param << "\" for node: " << node->name();
                return false;
            }
        }
        return true;
    };
    auto check_call = [&](NodePtr node, const std::vector<GraphPtr> &sg_list) -> bool {
        if (sg_list.size() != 1)
        {
            return false;
        }
        auto sg = sg_list[0];
        if (node->inputs().size() != sg->get_input_tensors().size() ||
            node->outputs().size() != sg->get_output_tensors().size())
        {
            return false;
        }
        for (size_t i = 0; i < node->inputs().size(); ++i)
        {
            auto t0 = node->inputs()[i];
            auto t1 = sg->get_input_tensors()[i];
            if (t0->type() != t1->type() || t0->shape() != t1->shape())
            {
                return false;
            }
        }
        for (size_t i = 0; i < node->outputs().size(); ++i)
        {
            auto t0 = node->outputs()[i];
            auto t1 = sg->get_output_tensors()[i];
            if (t0->type() != t1->type() || t0->shape() != t1->shape())
            {
                return false;
            }
        }
        return true;
    };
    auto check_loop = [&](NodePtr node, const std::vector<GraphPtr> &sg_list) -> bool {
        if (sg_list.size() != 1)
        {
            return false;
        }
        auto sg = sg_list[0];
        if (!check_params(node, {"dependency_inputs_num", "scan_outputs_num"}))
        {
            return false;
        }
        size_t depend_input_num = (int)node->params()["dependency_inputs_num"];
        size_t scan_output_num = (int)node->params()["scan_outputs_num"];
        if (node->inputs().size() < depend_input_num + 2 ||
            node->inputs().size() != sg->get_input_tensors().size())
        {
            return false;
        }
        if (node->inputs()[0]->type() != Tensor::INT32 ||
            node->inputs()[0]->shape().size() != 1 ||
            node->inputs()[1]->type() != Tensor::UINT8 ||
            node->inputs()[1]->shape().size() != 1)
        {
            return false;
        }
        for (size_t i = 0; i < node->inputs().size(); ++i)
        {
            auto t0 = node->inputs()[i];
            auto t1 = sg->get_input_tensors()[i];
            if (t0->type() != t1->type() || t0->shape() != t1->shape())
            {
                return false;
            }
        }
        if (node->outputs().size() < 1 ||
            node->outputs().size() != depend_input_num + scan_output_num ||
            node->outputs().size() + 1 != sg->get_output_tensors().size())
        {
            return false;
        }
        if (sg->get_output_tensors()[0]->type() != Tensor::UINT8 ||
            sg->get_output_tensors()[0]->shape().size() != 1)
        {
            return false;
        }
        for (size_t i = 0; i < depend_input_num; ++i)
        {
            auto t0 = node->outputs()[i];
            auto t1 = sg->get_output_tensors()[i + 1];
            if (t0->type() != t1->type() || t0->shape() != t1->shape())
            {
                return false;
            }
        }
        for (size_t i = depend_input_num; i < node->outputs().size(); ++i)
        {
            auto t0 = node->outputs()[i];
            auto t1 = sg->get_output_tensors()[i + 1];
            if (t0->type() != t1->type())
            {
                return false;
            }
            if (t0->shape().dim() == 0 ||
                t0->shape().dim() < t1->shape().dim() ||
                t0->shape().size() < t1->shape().size())
            {
                return false;
            }
            if (t1->shape().dim() > 0)
            {
                if (t0->shape().dim() != t1->shape().dim() &&
                    t0->shape().dim() != t1->shape().dim() + 1)
                {
                    return false;
                }
                int t1_start_dim = t0->shape().dim() == t1->shape().dim() ? 1 : 0;
                int t0_dim_offset = t0->shape().dim() == t1->shape().dim() ? 0 : 1;
                for (int d = t1_start_dim; d < t1->shape().dim(); ++d)
                {
                    if (t0->shape()[d + t0_dim_offset] != t1->shape()[d])
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    };
    auto check_if = [&](NodePtr node, const std::vector<GraphPtr> &sg_list) -> bool {
        if (sg_list.size() != 2)
        {
            return false;
        }
        auto then_sg = sg_list[0];
        auto else_sg = sg_list[1];
        if (!check_params(node, {"then_branch_inputs_num", "else_branch_inputs_num"}))
        {
            return false;
        }
        size_t then_input_num = (int)node->params()["then_branch_inputs_num"];
        size_t else_input_num = (int)node->params()["else_branch_inputs_num"];
        if (node->inputs().size() != 1 + then_input_num + else_input_num ||
            then_sg->get_input_tensors().size() != then_input_num ||
            else_sg->get_input_tensors().size() != else_input_num)
        {
            return false;
        }
        if (node->inputs()[0]->type() != Tensor::UINT8 ||
            node->inputs()[0]->shape().size() != 1)
        {
            return false;
        }
        for (size_t i = 1; i < node->inputs().size() - else_input_num; ++i)
        {
            auto t0 = node->inputs()[i];
            auto t1 = then_sg->get_input_tensors()[i - 1];
            if (t0->type() != t1->type() || t0->shape() != t1->shape())
            {
                return false;
            }
        }
        for (size_t i = 1 + then_input_num; i < node->inputs().size(); ++i)
        {
            auto t0 = node->inputs()[i];
            auto t1 = else_sg->get_input_tensors()[i - (1 + then_input_num)];
            if (t0->type() != t1->type() || t0->shape() != t1->shape())
            {
                return false;
            }
        }
        if (node->outputs().size() < 1 ||
            node->outputs().size() != then_sg->get_output_tensors().size() ||
            node->outputs().size() != else_sg->get_output_tensors().size())
        {
            return false;
        }
        for (size_t i = 0; i < node->outputs().size(); ++i)
        {
            auto out = node->outputs()[i];
            auto then_out = then_sg->get_output_tensors()[i];
            auto else_out = else_sg->get_output_tensors()[i];
            if (out->type() != then_out->type() ||
                out->type() != else_out->type())
            {
                return false;
            }
            if (out->shape().dim() != then_out->shape().dim() ||
                out->shape().dim() != else_out->shape().dim())
            {
                return false;
            }
            if (then_out->shape().size() >= else_out->shape().size())
            {
                if (out->shape() != then_out->shape())
                {
                    return false;
                }
            }
            else
            {
                if (out->shape() != else_out->shape())
                {
                    return false;
                }
            }
            for (int d = 0; d < out->shape().dim(); ++d)
            {
                if (out->shape()[d] < then_out->shape()[d] ||
                    out->shape()[d] < else_out->shape()[d])
                {
                    return false;
                }
            }
        }
        return true;
    };
    auto get_subgraphs = [&](NodePtr node, const std::vector<std::string> &sg_names) {
        std::vector<GraphPtr> ret;
        for (auto &&sg_name : sg_names)
        {
            if (!m_graphs.count(sg_name))
            {
                ERROR() << "Missing subgraph: " << sg_name;
                return std::vector<GraphPtr>();
            }
            if (linked.count(m_file_names[sg_name] + "/" + sg_name))
            {
                ERROR() << "Cannot link subgraph \"" << sg_name << "\", as it would create a cycle!";
                return std::vector<GraphPtr>();
            }
            ret.push_back(m_graphs[sg_name]);
        }
        bool check_ret = false;
        if (node->type() == OpType_Call)
        {
            check_ret = check_call(node, ret);
        }
        else if (node->type() == OpType_Loop)
        {
            check_ret = check_loop(node, ret);
        }
        else if (node->type() == OpType_If)
        {
            check_ret = check_if(node, ret);
        }
        if (!check_ret)
        {
            ERROR() << "Check subgraph inputs/outputs failed for node: " << node->name();
            return std::vector<GraphPtr>();
        }
        return ret;
    };
    bool ret = true;
    linked.insert(m_file_names[graph_name] + "/" + graph_name);
    auto nodes = g->nodes();
    for (auto &&node : nodes)
    {
        if (node->type() == OpType_Call)
        {
            // call node will embed as a subgraph into main graph
            if (!check_params(node, {"call_subgraph"}))
            {
                ret = false;
                continue;
            }
            std::string sg_name = node->params()["call_subgraph"];
            auto sg_list = get_subgraphs(node, {sg_name});
            if (sg_list.empty())
            {
                ret = false;
                continue;
            }
            auto sg = sg_list[0];
            auto emb = sg->clone();
            // map io tensor
            std::vector<NodePtr> m;
            auto inputs = node->inputs();
            auto outputs = node->outputs();
            node->clear_inputs();
            node->clear_outputs();
            auto g_inputs = emb->get_input_tensors();
            auto g_outputs = emb->get_output_tensors();
            auto g_nodes = emb->nodes();
            for (auto n : g_nodes)
            {
                if (n->type() == OpType_Input)
                {
                    emb->remove_node(n);
                    continue;
                }
                for (auto inp : n->inputs())
                {
                    for (auto i = 0u; i < g_inputs.size(); ++i)
                    {
                        if (inp == g_inputs[i])
                        {
                            n->replace_input_by(inp, inputs[i]);
                            break;
                        }
                    }
                }
                for (auto out : n->outputs())
                {
                    out->set_name(g->get_valid_tensor_name(node->name() + "/" + out->name())); //renaming
                    for (auto i = 0u; i < g_outputs.size(); ++i)
                    {
                        if (out == g_outputs[i])
                        {
                            n->add_output(outputs[i], n->remove_output(out));
                            for (auto &consumer : out->consumers())
                            {
                                consumer->add_input(outputs[i], consumer->remove_input(out));
                            }
                            break;
                        }
                    }
                }
                n->set_name(g->get_valid_node_name(node->name() + "/" + n->name())); //renaming
                emb->remove_node(n, false);
                g->add_node(n);
                m.push_back(n);
            }
            // unfold call
            auto nsg = SubGraph::create(g, m);
            for (auto &kv : sg->attrs())
            {
                nsg->set_attr(kv.first, kv.second);
            }
            auto sgn = nsg->as_node(true, true);
            sgn->set_attr("own_weight", true);
            sgn->set_attr("is_call", true);
            sgn->set_attr(Constants::skip_unfold, true);
            sgn->params()["call_subgraph"] = sg_name;
            sgn->set_name(node->name());
            g->remove_node(node);
            ret &= link_subgraphs(sgn->subgraph(), sg_name, linked);
            sgn->unfold();
        }
        else if (node->type() == OpType_Loop)
        {
            if (!check_params(node, {"body"}))
            {
                ret = false;
                continue;
            }
            std::string sg_name = node->params()["body"];
            auto sg_list = get_subgraphs(node, {sg_name});
            if (sg_list.empty())
            {
                ret = false;
                continue;
            }
            auto sg = sg_list[0];
            auto sgn = LoopNode::create(node->graph(), node, sg->clone());
            auto nsg = sgn->subgraph();
            for (auto &kv : sg->attrs())
            {
                nsg->set_attr(kv.first, kv.second);
            }
            ret &= link_subgraphs(nsg, sg_name, linked);
        }
        else if (node->type() == OpType_If)
        {
            if (!check_params(node, {"then_branch", "else_branch"}))
            {
                ret = false;
                continue;
            }
            std::string then_sg_name = node->params()["then_branch"];
            std::string else_sg_name = node->params()["else_branch"];
            auto sg_list = get_subgraphs(node, {then_sg_name, else_sg_name});
            if (sg_list.empty())
            {
                ret = false;
                continue;
            }
            auto then_sg = sg_list[0];
            auto else_sg = sg_list[1];
            auto sgn = IfNode::create(node->graph(), node, then_sg->clone(), else_sg->clone());
            auto nsg = sgn->subgraph();
            auto then_branch_attrs = then_sg->attrs();
            auto else_branch_attrs = else_sg->attrs();
            nsg->set_attr("then_branch_attrs", then_branch_attrs);
            nsg->set_attr("else_branch_attrs", else_branch_attrs);
            linked.insert(m_file_names[then_sg_name] + "/" + then_sg_name);
            ret &= link_subgraphs(nsg, else_sg_name, linked);
            linked.erase(m_file_names[then_sg_name] + "/" + then_sg_name);
        }
    }
    linked.erase(m_file_names[graph_name] + "/" + graph_name);
    return ret;
}

bool IRChecker::parse_and_syntax()
{
    if (m_section_list.empty())
    {
        return false;
    }
    bool ret = preprocess(m_section_list, _default_graph, m_file_name);
    for (auto &kv : m_section_lists)
    {
        auto &name = kv.first;
        GraphPtr graph = m_graph;
        if (name != _default_graph)
        {
            graph = Graph::create();
        }
        m_graphs[name] = graph;
        bool with_header = true;
        auto file_name = m_file_names[name];
        m_tables[name] = parse_sections(kv.second, graph, with_header, file_name, ret);
    }
    std::set<std::string> linked;
    ret &= link_subgraphs(m_graph, _default_graph, linked);
    if (ret)
    {
        for (auto n : m_graph->nodes())
        {
            auto sgn = SubGraphNode::from_node(n);
            if (sgn)
            {
                auto sg = std::static_pointer_cast<SubGraph>(sgn->subgraph());
                sg->update_real_nodes();
            }
        }
        // infer symbol shape if graph inputs have ds_output_shape
        // set default ds_output_shape if some input nodes don't have
        bool has_ds_output_shape = false;
        for (auto t : m_graph->get_input_tensors())
        {
            auto n = t->node();
            if (n->params().count("ds_output_shape"))
            {
                has_ds_output_shape = true;
                break;
            }
        }
        if (has_ds_output_shape)
        {
            for (auto t : m_graph->get_input_tensors())
            {
                auto n = t->node();
                if (!n->params().count("ds_output_shape"))
                {
                    std::vector<Expr> s = t->shape().symbol();
                    n->params()["ds_output_shape"] = std::vector<std::vector<Expr>>{s};
                }
            }
            infershape(m_graph);
        }
    }
    if (!m_graph->check_sg())
    {
        ERROR() << "Check subgraph failed!";
        ret = false;
    }
    return ret;
}

static const std::string shape_to_string(const ShapeList &shapes)
{
    std::ostringstream oss;
    size_t size = shapes.size();
    if (size)
    {
        oss << shapes[0].to_string();
        for (size_t i = 1; i < size; ++i)
        {
            oss << ", " << shapes[i].to_string();
        }
        oss << "(" << datalayout_to_string(shapes[0].layout()) << ")";
    }
    return oss.str();
}

static bool check_infer_shape(BaseNodePtr node, const ShapeList &valid_shapes, std::vector<IRError> &errors)
{
    bool ret = true;
    ShapeList shapes;
    for (auto out : node->outputs())
    {
        shapes.push_back(out->shape());
    }
    if (shapes.size() == valid_shapes.size())
    {
        for (size_t i = 0; i < shapes.size(); i++)
        {
            if ((std::vector<int>)shapes[i] != (std::vector<int>)valid_shapes[i])
            {
                ret = false;
                break;
            }
        }
    }
    else
    {
        ret = false;
    }
    if (!ret)
    {
        auto key = "layer_top_shape";
        auto msg = "Invalid shape: " + shape_to_string(shapes) + ", expected: " + shape_to_string(valid_shapes);
        errors.push_back(IRError(IRErr_InvalidParam, key) << msg);
    }
    return ret;
}

static const std::map<OpType, CheckerOpPluginInfo *> &get_checker_plugins()
{
    static std::map<OpType, CheckerOpPluginInfo *> plugins;
    if (plugins.size())
    {
        return plugins;
    }
    auto disable_plugins = BuilderConfig::get<std::vector<std::string>>("disable-plugins");
    PluginLoader *m_plugin_loader = PluginLoader::get_instance();
    auto all_plugins = (*m_plugin_loader)[PluginType_IRCheckerOp];
    for (auto _plugin : all_plugins)
    {
        CheckerOpPluginInfo *plugin = reinterpret_cast<CheckerOpPluginInfo *>(_plugin);
        auto opt = plugin->init_handle();
        auto &name = plugin->name;
        if (contains(disable_plugins, name))
        {
            DEBUG() << "Disbaled Plugin: " << name;
            continue;
        }
        auto op_type = opt->get_op_type();
        if (plugins.count(op_type))
        {
            WARN("Two plugins (%s, %s) point to the same op_type(%s). Apply the first one by default.",
                 plugins[op_type]->name.c_str(),
                 plugin->name.c_str(),
                 optype_to_string(op_type).c_str());
        }
        else
        {
            plugins[opt->get_op_type()] = plugin;
        }
    }
    return plugins;
}

bool IRChecker::check_graph(GraphPtr g)
{
    return check_graph(g, SectionTable(), std::string());
}

bool IRChecker::check_graph(GraphPtr g, const SectionTable &sec_table, const std::string &file_name)
{
    if (!g)
    {
        return false;
    }
    bool ret = true;
    auto plugins = get_checker_plugins();
    for (auto n : g->nodes())
    {
        if (plugins.count(n->type()))
        {
            auto plugin = plugins[n->type()];
            auto opt = plugin->init_handle();
            auto errs = opt->check(n);
            bool ret_op = errs.empty();
            if (ret_op)
            {
                ret_op &= check_infer_shape(n, opt->infer_shape(n), errs);
            }
            if (!ret_op)
            {
                int layer_id = n->get_attr("layer_id");
                ERROR() << "[" << plugin->name << "] "
                        << "layer_id: " << layer_id
                        << ", layer_name: " << n->name()
                        << ", layer_type: " << optype_to_string(n->type());
                Section sec;
                if (sec_table.count(n->name()))
                {
                    sec = sec_table.at(n->name());
                }
                for (auto &err : errs)
                {
                    if (sec.count(err.key()))
                    {
                        CHK_ERROR(sec.at(err.key()).second, err.key(), err.msg(), file_name);
                    }
                    else
                    {
                        CHK_ERROR(-1, err.key(), err.msg(), file_name);
                    }
                }
            }
            ret &= ret_op;
        }
    }
    return ret;
}

bool IRChecker::check_params()
{
    return check_graph(m_graph, m_tables[_default_graph], m_file_name);
}

bool IRChecker::check_all(bool enable_plugin)
{
    PluginLoader *m_plugin_loader = PluginLoader::get_instance();
    m_plugins = (*m_plugin_loader)[PluginType_IRCheckerOp];
    // DEBUG() << "Load checker plugins: " << m_plugins.size();
    INFO() << "Start to check IR: " << m_file_name;
    bool ret = parse_and_syntax();
    if (ret && enable_plugin)
    {
        ret &= check_params();
    }
    if (ret)
    {
        if (enable_plugin)
        {
            INFO() << "IRChecker: All IR pass";
        }
        else
        {
            INFO() << "IRChecker: All IR pass (Checker Plugin disabled)";
        }
    }
    else
    {
        ERROR() << "IRChecker: Invalid IR. Please double check the IR";
    }
    // m_graph->to_svg("graph.svg");
    return ret;
}
} // namespace aipubt

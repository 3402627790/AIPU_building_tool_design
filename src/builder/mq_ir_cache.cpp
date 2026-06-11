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

#include "builder/mq_ir_cache.h"

#include "core/any.h"
#include "core/constants.h"
#include "core/node.h"
#include "core/tensor.h"
#include "utils/utils.h"

#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

namespace aipubt
{
namespace
{

template <typename T>
void set_typed_json_attr(const std::string& key, const std::string& type, const T& value,
                         utils::JsonObject& json_obj)
{
    utils::JsonObject attr_obj(utils::JsonObject::OBJ);
    attr_obj["type"] = type;
    attr_obj["value"] = value;
    json_obj[key] = attr_obj;
}

bool get_json_object_field(const utils::JsonObject& object, const std::string& key, utils::JsonObject& value)
{
    if (object.t != utils::JsonObject::OBJ)
        return false;

    auto fields = object.obj();
    for (const auto& field : fields)
    {
        if (field.first == key)
        {
            value = field.second;
            return true;
        }
    }

    return false;
}

// Helper function: serialize an Any value to JsonObject
bool serialize_any_to_json(const std::string& key, const Any& value, utils::JsonObject& json_obj)
{
    if (!value.value())
        return false;

    auto type_id = value.get_type_id();

    if (type_id == GET_ANY_TYPEID<int>())
    {
        set_typed_json_attr(key, "int", value.cast<int>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<int64_t>())
    {
        set_typed_json_attr(key, "int64_t", value.cast<int64_t>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<float>())
    {
        set_typed_json_attr(key, "float", value.cast<float>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<double>())
    {
        set_typed_json_attr(key, "double", value.cast<double>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<bool>())
    {
        set_typed_json_attr(key, "bool", value.cast<bool>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::string>())
    {
        set_typed_json_attr(key, "string", value.cast<std::string>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::vector<int>>())
    {
        auto vec = value.cast<std::vector<int>>();
        utils::JsonObject arr(utils::JsonObject::ARR);
        for (int v : vec)
            arr.push_back(v);
        set_typed_json_attr(key, "vector<int>", arr, json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::vector<float>>())
    {
        auto vec = value.cast<std::vector<float>>();
        utils::JsonObject arr(utils::JsonObject::ARR);
        for (float v : vec)
            arr.push_back(v);
        set_typed_json_attr(key, "vector<float>", arr, json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::vector<std::string>>())
    {
        auto vec = value.cast<std::vector<std::string>>();
        utils::JsonObject arr(utils::JsonObject::ARR);
        for (const auto& v : vec)
            arr.push_back(v);
        set_typed_json_attr(key, "vector<string>", arr, json_obj);
        return true;
    }

    return false;
}

// Helper function: deserialize typed JsonObject to Any and set as node/tensor attribute
bool deserialize_json_to_any(const std::string& key, const utils::JsonObject& json_value,
                              std::function<void(const std::string&, const Any&)> set_attr_func)
{
    utils::JsonObject type_obj;
    utils::JsonObject value_obj;
    if (!get_json_object_field(json_value, "type", type_obj) ||
        !get_json_object_field(json_value, "value", value_obj) ||
        type_obj.t != utils::JsonObject::STR)
    {
        return false;
    }

    std::string type = type_obj;
    if (type == "int")
    {
        set_attr_func(key, Any((int)value_obj));
        return true;
    }
    else if (type == "int64_t")
    {
        set_attr_func(key, Any((int64_t)value_obj));
        return true;
    }
    else if (type == "float")
    {
        set_attr_func(key, Any((float)value_obj));
        return true;
    }
    else if (type == "double")
    {
        set_attr_func(key, Any((double)value_obj));
        return true;
    }
    else if (type == "bool")
    {
        if (value_obj.t != utils::JsonObject::BOOL)
            return false;
        set_attr_func(key, Any((bool)value_obj));
        return true;
    }
    else if (type == "string")
    {
        if (value_obj.t != utils::JsonObject::STR)
            return false;
        set_attr_func(key, Any((std::string)value_obj));
        return true;
    }
    else if (type == "vector<int>")
    {
        if (value_obj.t != utils::JsonObject::ARR)
            return false;
        std::vector<int> vec;
        for (const auto& item : value_obj.array())
            vec.push_back((int)item);
        set_attr_func(key, Any(vec));
        return true;
    }
    else if (type == "vector<float>")
    {
        if (value_obj.t != utils::JsonObject::ARR)
            return false;
        std::vector<float> vec;
        for (const auto& item : value_obj.array())
            vec.push_back((float)item);
        set_attr_func(key, Any(vec));
        return true;
    }
    else if (type == "vector<string>")
    {
        if (value_obj.t != utils::JsonObject::ARR)
            return false;
        std::vector<std::string> vec;
        for (const auto& item : value_obj.array())
        {
            if (item.t != utils::JsonObject::STR)
                return false;
            vec.push_back((std::string)item);
        }
        set_attr_func(key, Any(vec));
        return true;
    }

    return false;
}

// Save all node and tensor attributes to a JSON file
bool save_graph_attributes(GraphPtr graph, const std::string& json_path)
{
    utils::JsonObject root(utils::JsonObject::OBJ);
    utils::JsonObject nodes_arr(utils::JsonObject::ARR);

    int total_node_attrs = 0;
    int total_tensor_attrs = 0;

    // Iterate through all nodes
    for (auto node : graph->nodes())
    {
        utils::JsonObject node_obj(utils::JsonObject::OBJ);
        node_obj["node_name"] = node->name();
        node_obj["layer_id"] = node->get_attr<int>("layer_id");

        // Serialize node attributes
        utils::JsonObject node_attrs(utils::JsonObject::OBJ);
        int node_attr_count = 0;

        for (const auto& attr : node->attrs())
        {
            if (serialize_any_to_json(attr.first, attr.second, node_attrs))
            {
                node_attr_count++;
            }
        }

        if (node_attr_count > 0)
        {
            node_obj["node_attrs"] = node_attrs;
            total_node_attrs += node_attr_count;
        }

        // Serialize tensor attributes for each output tensor
        utils::JsonObject tensors_arr(utils::JsonObject::ARR);
        int tensor_attr_count = 0;

        for (size_t i = 0; i < node->outputs().size(); ++i)
        {
            auto tensor = node->outputs()[i];
            utils::JsonObject tensor_obj(utils::JsonObject::OBJ);
            tensor_obj["tensor_name"] = tensor->name();
            tensor_obj["tensor_index"] = (int)i;

            utils::JsonObject tensor_attrs(utils::JsonObject::OBJ);
            int count = 0;

            for (const auto& attr : tensor->attrs())
            {
                if (serialize_any_to_json(attr.first, attr.second, tensor_attrs))
                {
                    count++;
                }
            }

            if (count > 0)
            {
                tensor_obj["tensor_attrs"] = tensor_attrs;
                tensors_arr.push_back(tensor_obj);
                tensor_attr_count += count;
            }
        }

        if (tensor_attr_count > 0)
        {
            node_obj["output_tensors"] = tensors_arr;
            total_tensor_attrs += tensor_attr_count;
        }

        // Only add node to JSON if it has attributes
        if (node_attr_count > 0 || tensor_attr_count > 0)
        {
            nodes_arr.push_back(node_obj);
        }
    }

    root["nodes"] = nodes_arr;
    root["total_nodes"] = (int)nodes_arr.array().size();
    root["total_node_attrs"] = total_node_attrs;
    root["total_tensor_attrs"] = total_tensor_attrs;

    // Write to file
    std::ofstream ofs(json_path);
    if (!ofs.is_open())
    {
        ERROR() << "Failed to open attributes file for writing: " << json_path;
        return false;
    }

    ofs << root.to_string();
    ofs.close();

    INFO() << "Saved " << total_node_attrs << " node attributes and "
           << total_tensor_attrs << " tensor attributes to " << json_path;

    return true;
}

// Load node and tensor attributes from a JSON file
bool load_graph_attributes(GraphPtr graph, const std::string& json_path)
{
    std::ifstream ifs(json_path);
    if (!ifs.is_open())
    {
        ERROR() << "Failed to open attributes file for reading: " << json_path;
        return false;
    }

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    ifs.close();

    utils::JsonObject root;
    root = utils::JsonObject::parse(buffer.str());

    if (root.t != utils::JsonObject::OBJ || !root.exist("nodes"))
    {
        ERROR() << "Invalid attributes JSON format";
        return false;
    }

    auto nodes_arr = root["nodes"].array();
    int loaded_node_attrs = 0;
    int loaded_tensor_attrs = 0;

    // Create a map from node name to node pointer for fast lookup
    std::map<std::string, NodePtr> node_map;
    for (auto node : graph->nodes())
    {
        node_map[node->name()] = node;
    }

    // Iterate through saved nodes
    for (auto& node_json : nodes_arr)
    {
        if (node_json.t != utils::JsonObject::OBJ)
            continue;

        std::string node_name = node_json["node_name"];
        auto it = node_map.find(node_name);
        if (it == node_map.end())
        {
            WARN() << "Node not found in graph: " << node_name;
            continue;
        }

        NodePtr node = it->second;

        // Load node attributes
        if (node_json.exist("node_attrs"))
        {
            auto node_attrs = node_json["node_attrs"].obj();
            for (const auto& kv : node_attrs)
            {
                if (deserialize_json_to_any(kv.first, kv.second,
                    [&node](const std::string& key, const Any& value) {
                        node->set_attr(key, value);
                    }))
                {
                    loaded_node_attrs++;
                }
            }
        }

        // Load tensor attributes
        if (node_json.exist("output_tensors"))
        {
            auto tensors_arr = node_json["output_tensors"].array();
            for (auto& tensor_json : tensors_arr)
            {
                if (tensor_json.t != utils::JsonObject::OBJ)
                    continue;

                int tensor_index = tensor_json["tensor_index"];

                if (tensor_index >= 0 && tensor_index < (int)node->outputs().size())
                {
                    auto tensor = node->outputs()[tensor_index];

                    if (tensor_json.exist("tensor_attrs"))
                    {
                        auto tensor_attrs = tensor_json["tensor_attrs"].obj();
                        for (const auto& kv : tensor_attrs)
                        {
                            if (deserialize_json_to_any(kv.first, kv.second,
                                [&tensor](const std::string& key, const Any& value) {
                                    tensor->set_attr(key, value);
                                }))
                            {
                                loaded_tensor_attrs++;
                            }
                        }
                    }
                }
            }
        }
    }

    INFO() << "Loaded " << loaded_node_attrs << " node attributes and "
           << loaded_tensor_attrs << " tensor attributes from " << json_path;

    return true;
}


// File names used inside the cache directory pointed at by --mq_save_ir / --mq_load_ir.
constexpr const char *kMqIrCacheDef = "lowered.def";
constexpr const char *kMqIrCacheBin = "lowered.bin";
constexpr const char *kMqIrCacheAttrs = "lowered.attrs.json";

inline std::string mq_ir_cache_def_path(const std::string &dir)
{
    return dir + "/" + kMqIrCacheDef;
}

inline std::string mq_ir_cache_bin_path(const std::string &dir)
{
    return dir + "/" + kMqIrCacheBin;
}

inline std::string mq_ir_cache_attrs_path(const std::string &dir)
{
    return dir + "/" + kMqIrCacheAttrs;
}

} // namespace

// Try to load a previously saved post-lowering graph from `dir`
GraphPtr load_mq_ir_cache(const std::string &dir)
{
    auto def_path = mq_ir_cache_def_path(dir);
    auto bin_path = mq_ir_cache_bin_path(dir);
    auto attrs_path = mq_ir_cache_attrs_path(dir);

    if (!utils::file_exists(def_path) || !utils::file_exists(bin_path))
    {
        WARN() << "mq_load_ir: cache files not found in " << dir << " (expected "
               << kMqIrCacheDef << " and " << kMqIrCacheBin << ").";
        return nullptr;
    }

    INFO() << "mq_load_ir: loading cached MQ IR from " << dir;
    auto g = Graph::parse(def_path);
    if (!g)
    {
        ERROR() << "mq_load_ir: failed to parse cached IR " << def_path;
        return nullptr;
    }

    // Load node and tensor attributes from JSON file if it exists
    if (utils::file_exists(attrs_path))
    {
        INFO() << "mq_load_ir: loading cached attributes from " << attrs_path;
        if (!load_graph_attributes(g, attrs_path))
        {
            WARN() << "mq_load_ir: failed to load attributes, continuing without them";
        }
    }
    else
    {
        DEBUG() << "mq_load_ir: no attributes file found, skipping attribute loading";
    }

    return g;
}

// Serialize the current graph into `dir`
void save_mq_ir_cache(GraphPtr graph, const std::string &dir)
{
    if (!utils::path_exists(dir) && !utils::mkdir(dir))
    {
        ERROR() << "mq_save_ir: cannot create cache directory " << dir;
        return;
    }

    graph->set_attr(Constants::serialize_scale_zp, true);
    graph->set_attr(Constants::dump_debug_IR, true);

    auto abs_dir = utils::get_abspath(dir);
    auto def_path = mq_ir_cache_def_path(abs_dir);
    auto bin_path = mq_ir_cache_bin_path(abs_dir);
    auto attrs_path = mq_ir_cache_attrs_path(abs_dir);

    INFO() << "mq_save_ir: dumping post-lowering MQ IR to " << def_path;
    graph->serialize(def_path, bin_path);

    // Save node and tensor attributes to separate JSON file
    INFO() << "mq_save_ir: dumping attributes to " << attrs_path;
    if (!save_graph_attributes(graph, attrs_path))
    {
        WARN() << "mq_save_ir: failed to save attributes file";
    }
}

} // namespace aipubt

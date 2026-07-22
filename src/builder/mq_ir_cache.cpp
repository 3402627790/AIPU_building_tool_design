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

#include "builder/cps_elf.h"
#include "builder/gop_plugin.h"
#include "builder/lowering/lowering.h"
#include "builder/readonly_param.h"
#include "core/any.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/param_list.h"
#include "core/placeholder.h"
#include "core/node.h"
#include "core/surface.h"
#include "core/subgraph.h"
#include "core/tensor.h"
#include "utils/utils.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <vector>

namespace aipubt
{
namespace
{

class BoolConfigGuard
{
public:
    BoolConfigGuard(const std::string &key, bool value) : m_key(key), m_had_value(BuilderConfig::exist(key))
    {
        m_old_value = BuilderConfig::get<bool>(key);
        BuilderConfig::set(key, value);
    }

    ~BoolConfigGuard()
    {
        if (m_had_value)
        {
            BuilderConfig::set(m_key, m_old_value);
        }
        else
        {
            BuilderConfig::erase(m_key);
        }
    }

private:
    std::string m_key;
    bool m_had_value = false;
    bool m_old_value = false;
};

template <typename T>
void set_typed_json_attr(const std::string& key, const std::string& type, const T& value,
                         utils::JsonObject& json_obj)
{
    utils::JsonObject attr_obj(utils::JsonObject::OBJ);
    attr_obj["type"] = type;
    attr_obj["value"] = value;
    json_obj[key] = attr_obj;
}

utils::JsonObject serialize_layout(DataLayout layout)
{
    return datalayout_to_string(layout);
}

utils::JsonObject serialize_layout_list(const LayoutList &layouts)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (auto layout : layouts)
    {
        arr.push_back(serialize_layout(layout));
    }
    return arr;
}

utils::JsonObject serialize_layout_combinations(const LayoutCombinations &layout_combinations)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (const auto &layouts : layout_combinations)
    {
        arr.push_back(serialize_layout_list(layouts));
    }
    return arr;
}

utils::JsonObject serialize_tensor_shape(const TensorShape &shape)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    std::vector<int> dims = shape;
    for (auto dim : dims)
    {
        arr.push_back(dim);
    }
    return arr;
}

utils::JsonObject serialize_tensor_dtype(Tensor::Dtype dtype)
{
    return Tensor::dtype_to_string(dtype);
}

bool deserialize_layout(const utils::JsonObject &obj, DataLayout &layout)
{
    if (obj.t == utils::JsonObject::STR)
    {
        layout = string_to_datalayout((std::string)obj);
        return layout != DataLayout_UnSupport;
    }
    if (obj.t == utils::JsonObject::INT)
    {
        layout = static_cast<DataLayout>((int)obj);
        return true;
    }
    return false;
}

bool deserialize_tensor_shape(const utils::JsonObject &obj, TensorShape &shape, DataLayout layout)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;

    std::vector<int> dims;
    for (const auto &item : obj.array())
    {
        dims.push_back((int)item);
    }
    shape = TensorShape(dims, layout);
    return true;
}

bool deserialize_tensor_dtype(const utils::JsonObject &obj, Tensor::Dtype &dtype)
{
    if (obj.t == utils::JsonObject::STR)
    {
        dtype = Tensor::string_to_dtype((std::string)obj);
        return true;
    }
    if (obj.t == utils::JsonObject::INT)
    {
        dtype = static_cast<Tensor::Dtype>((int)obj);
        return true;
    }
    return false;
}

bool deserialize_layout_list(const utils::JsonObject &obj, LayoutList &layouts)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;

    layouts.clear();
    for (const auto &item : obj.array())
    {
        DataLayout layout = DataLayout_UnSupport;
        if (!deserialize_layout(item, layout))
            return false;
        layouts.push_back(layout);
    }
    return true;
}

bool deserialize_layout_combinations(const utils::JsonObject &obj, LayoutCombinations &layout_combinations)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;

    layout_combinations.clear();
    for (const auto &item : obj.array())
    {
        LayoutList layouts;
        if (!deserialize_layout_list(item, layouts))
            return false;
        layout_combinations.push_back(layouts);
    }
    return true;
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

bool has_json_object_field(const utils::JsonObject& object, const std::string& key)
{
    utils::JsonObject value;
    return get_json_object_field(object, key, value);
}

void dump_graph_io_tensor_ptrs(GraphPtr graph, const std::string &tag)
{
    if (!graph)
        return;

    auto input_tensors = graph->get_input_tensors();
    INFO() << tag << ": input_tensors size = " << input_tensors.size();
    for (size_t i = 0; i < input_tensors.size(); ++i)
    {
        auto tensor = input_tensors[i];
        INFO() << tag << ": input_tensors[" << i << "] name = "
               << (tensor ? tensor->name() : std::string("<null>"))
               << ", ptr = " << tensor.get();
    }

    auto output_tensors = graph->get_output_tensors();
    INFO() << tag << ": output_tensors size = " << output_tensors.size();
    for (size_t i = 0; i < output_tensors.size(); ++i)
    {
        auto tensor = output_tensors[i];
        INFO() << tag << ": output_tensors[" << i << "] name = "
               << (tensor ? tensor->name() : std::string("<null>"))
               << ", ptr = " << tensor.get();
    }
}

std::string pointer_id(const std::string &kind, const void *ptr)
{
    std::ostringstream oss;
    oss << kind << ":0x" << std::hex << reinterpret_cast<uintptr_t>(ptr);
    return oss.str();
}

utils::JsonObject make_pointer_ref(const std::string &id)
{
    utils::JsonObject ref_obj(utils::JsonObject::OBJ);
    ref_obj["kind"] = "pointer";
    ref_obj["id"] = id;
    return ref_obj;
}

bool is_pointer_ref(utils::JsonObject obj)
{
    if (obj.t != utils::JsonObject::OBJ || !obj.exist("kind") || !obj.exist("id"))
        return false;
    return obj["kind"].t == utils::JsonObject::STR &&
           obj["id"].t == utils::JsonObject::STR &&
           (std::string)obj["kind"] == "pointer";
}

NodePtr find_node_by_name(GraphPtr graph, const std::string &node_name)
{
    if (!graph)
        return nullptr;
    for (auto node : graph->nodes())
    {
        if (node->name() == node_name)
            return node;
    }
    return nullptr;
}

TensorPtr find_output_tensor(NodePtr node, utils::JsonObject tensor_json);
TensorPtr find_constant_tensor(NodePtr node, utils::JsonObject tensor_json);
TensorPtr find_placeholder_tensor(NodePtr node, utils::JsonObject tensor_json);

class PointerRefTable
{
public:
    explicit PointerRefTable(GraphPtr graph = nullptr) : m_graph(graph)
    {
    }

    std::string add_tensor(TensorPtr tensor)
    {
        if (!tensor)
            return "";

        std::string id = pointer_id("tensor", tensor.get());
        if (m_objects.count(id))
            return id;
        m_tensors[id] = tensor;

        utils::JsonObject obj(utils::JsonObject::OBJ);
        obj["object_kind"] = "tensor";
        obj["tensor_name"] = tensor->name();

        if (m_graph)
        {
            for (const auto &kv : m_graph->global_tensors())
            {
                if (kv.second == tensor)
                {
                    obj["role"] = "global";
                    obj["global_key"] = kv.first;
                    m_objects[id] = obj;
                    return id;
                }
            }
        }

        auto owner = tensor->node();
        if (owner)
        {
            obj["owner_node"] = owner->name();

            for (size_t i = 0; i < owner->outputs().size(); ++i)
            {
                if (owner->outputs()[i] == tensor)
                {
                    obj["role"] = "output";
                    obj["tensor_index"] = (int)i;
                    m_objects[id] = obj;
                    return id;
                }
            }

            for (const auto &kv : owner->constants())
            {
                if (kv.second == tensor)
                {
                    obj["role"] = "constant";
                    obj["constant_key"] = kv.first;
                    m_objects[id] = obj;
                    return id;
                }
            }

            for (size_t i = 0; i < owner->placeholders().size(); ++i)
            {
                if (owner->placeholders()[i] == tensor)
                {
                    obj["role"] = "placeholder";
                    obj["tensor_index"] = (int)i;
                    m_objects[id] = obj;
                    return id;
                }
            }
        }

        obj["role"] = "named";
        m_objects[id] = obj;
        return id;
    }

    utils::JsonObject tensor_ref(TensorPtr tensor)
    {
        return make_pointer_ref(add_tensor(tensor));
    }

    std::string add_node(NodePtr node)
    {
        if (!node)
            return "";

        std::string id = pointer_id("node", node.get());
        if (m_objects.count(id))
            return id;

        utils::JsonObject obj(utils::JsonObject::OBJ);
        obj["object_kind"] = "node";
        obj["node_name"] = node->name();
        if (node->graph())
        {
            obj["graph_name"] = node->graph()->name();
        }
        m_objects[id] = obj;
        return id;
    }

    utils::JsonObject node_ref(NodePtr node)
    {
        return make_pointer_ref(add_node(node));
    }

    std::string add_memory(MemoryObjectPtr memory)
    {
        if (!memory)
            return "";

        std::string id = pointer_id("memory", memory.get());
        if (m_objects.count(id))
            return id;

        utils::JsonObject obj(utils::JsonObject::OBJ);
        obj["object_kind"] = "memory";
        obj["relative_offset"] = (uint64_t)memory->relative_offset;
        obj["m_base"] = (uint64_t)memory->m_base;
        obj["align"] = (uint64_t)memory->align;
        obj["mask"] = (uint64_t)memory->mask;
        obj["size"] = (uint64_t)memory->size;
        obj["foot_print"] = (uint64_t)memory->foot_print;
        obj["mem_flag"] = (uint64_t)memory->mem_flag;
        obj["buffer_id"] = (int64_t)memory->buffer_id;
        obj["bss_id"] = (int64_t)memory->bss_id;

        m_objects[id] = obj;
        if (memory->base_offset)
        {
            m_objects[id]["base_offset"] = memory_ref(memory->base_offset);
        }
        if (memory->next_block)
        {
            m_objects[id]["next_block"] = memory_ref(memory->next_block);
        }
        return id;
    }

    utils::JsonObject memory_ref(MemoryObjectPtr memory)
    {
        return make_pointer_ref(add_memory(memory));
    }

    utils::JsonObject to_json() const
    {
        utils::JsonObject obj(utils::JsonObject::OBJ);
        for (const auto &kv : m_objects)
        {
            obj[kv.first] = kv.second;
        }
        return obj;
    }

    utils::JsonObject serialize_payloads(const std::string &bin_path, const std::string &payload_file_name) const
    {
        utils::JsonObject payloads(utils::JsonObject::OBJ);
        std::ofstream ofs;
        std::map<char *, utils::JsonObject> data_map;
        int payload_count = 0;

        for (const auto &kv : m_tensors)
        {
            auto tensor = kv.second;
            if (!tensor)
                continue;

            auto obj_it = m_objects.find(kv.first);
            if (obj_it != m_objects.end())
            {
                const auto &obj = obj_it->second;
                utils::JsonObject role;
                if (get_json_object_field(obj, "role", role) && role.t == utils::JsonObject::STR &&
                    (std::string)role == "constant")
                {
                    continue;
                }
            }

            char *data = tensor->data<char>();
            size_t size = tensor->mem_size();
            if (!data || size == 0)
            {
                DEBUG() << "Skip tensor payload, no data: " << tensor->name();
                continue;
            }

            utils::JsonObject data_obj(utils::JsonObject::OBJ);
            auto data_it = data_map.find(data);
            if (data_it != data_map.end())
            {
                data_obj = data_it->second;
            }
            else
            {
                if (!ofs.is_open())
                {
                    ofs.open(bin_path, std::ios::binary);
                    if (!ofs.is_open())
                    {
                        ERROR() << "Failed to open tensor payload file for writing: " << bin_path;
                        return payloads;
                    }
                }

                int64_t offset = (int64_t)ofs.tellp();
                ofs.write(data, size);
                if (ofs.fail())
                {
                    WARN() << "Failed to write tensor payload for " << tensor->name();
                    continue;
                }

                data_obj["file"] = payload_file_name;
                data_obj["offset"] = offset;
                data_obj["size"] = (int64_t)size;
                data_map[data] = data_obj;
                payload_count++;
            }

            utils::JsonObject payload(utils::JsonObject::OBJ);
            payload["object_kind"] = "tensor";
            payload["tensor_name"] = tensor->name();
            payload["data"] = data_obj;
            payloads[kv.first] = payload;
        }

        if (payload_count > 0)
        {
            INFO() << "Saved " << payload_count << " tensor payloads to " << bin_path;
        }
        return payloads;
    }

private:
    GraphPtr m_graph;
    std::map<std::string, utils::JsonObject> m_objects;
    std::map<std::string, TensorPtr> m_tensors;
};

TensorPtr lookup_tensor_by_desc(GraphPtr graph, utils::JsonObject desc)
{
    if (!graph || desc.t != utils::JsonObject::OBJ)
        return nullptr;

    if (desc.exist("role") && desc["role"].t == utils::JsonObject::STR &&
        (std::string)desc["role"] == "global" &&
        desc.exist("global_key") && desc["global_key"].t == utils::JsonObject::STR)
    {
        std::string key = (std::string)desc["global_key"];
        if (graph->global_tensors().count(key))
            return graph->global_tensors()[key];
    }

    if (desc.exist("owner_node") && desc["owner_node"].t == utils::JsonObject::STR)
    {
        auto node = find_node_by_name(graph, (std::string)desc["owner_node"]);
        if (node && desc.exist("role") && desc["role"].t == utils::JsonObject::STR)
        {
            std::string role = (std::string)desc["role"];
            if (role == "output")
                return find_output_tensor(node, desc);
            if (role == "constant")
                return find_constant_tensor(node, desc);
            if (role == "placeholder")
                return find_placeholder_tensor(node, desc);
        }
    }

    if (desc.exist("tensor_name") && desc["tensor_name"].t == utils::JsonObject::STR)
    {
        std::string tensor_name = (std::string)desc["tensor_name"];
        for (const auto &kv : graph->global_tensors())
        {
            if (kv.second && kv.second->name() == tensor_name)
                return kv.second;
        }
        for (auto node : graph->nodes())
        {
            for (auto tensor : node->outputs())
            {
                if (tensor->name() == tensor_name)
                    return tensor;
            }
            for (auto &kv : node->constants())
            {
                if (kv.second->name() == tensor_name)
                    return kv.second;
            }
            for (auto tensor : node->placeholders())
            {
                if (tensor->name() == tensor_name)
                    return tensor;
            }
        }
    }

    return nullptr;
}

TensorPtr lookup_tensor_ref(GraphPtr graph, utils::JsonObject object_table, utils::JsonObject ref)
{
    if (!is_pointer_ref(ref) || object_table.t != utils::JsonObject::OBJ)
        return nullptr;

    std::string id = (std::string)ref["id"];
    if (!object_table.exist(id))
        return nullptr;

    return lookup_tensor_by_desc(graph, object_table[id]);
}

TensorPtr lookup_tensor_by_desc_in_graphs(GraphPtr graph, utils::JsonObject desc)
{
    auto tensor = lookup_tensor_by_desc(graph, desc);
    if (tensor)
        return tensor;

    if (!graph)
        return nullptr;

    for (auto subgraph : graph->get_all_graphs())
    {
        if (subgraph == graph)
            continue;
        tensor = lookup_tensor_by_desc(subgraph, desc);
        if (tensor)
            return tensor;
    }

    return nullptr;
}

NodePtr lookup_node_by_desc(GraphPtr graph, utils::JsonObject desc)
{
    if (!graph || desc.t != utils::JsonObject::OBJ ||
        !desc.exist("node_name") || desc["node_name"].t != utils::JsonObject::STR)
    {
        return nullptr;
    }

    std::string node_name = (std::string)desc["node_name"];
    std::string graph_name;
    if (desc.exist("graph_name") && desc["graph_name"].t == utils::JsonObject::STR)
    {
        graph_name = (std::string)desc["graph_name"];
    }

    for (auto candidate_graph : graph->get_all_graphs())
    {
        if (!graph_name.empty() && candidate_graph->name() != graph_name)
            continue;
        auto node = find_node_by_name(candidate_graph, node_name);
        if (node)
            return node;
    }

    return find_node_by_name(graph, node_name);
}

NodePtr lookup_node_ref(GraphPtr graph, utils::JsonObject object_table, utils::JsonObject ref)
{
    if (!is_pointer_ref(ref) || object_table.t != utils::JsonObject::OBJ)
        return nullptr;

    std::string id = (std::string)ref["id"];
    if (!object_table.exist(id))
        return nullptr;

    return lookup_node_by_desc(graph, object_table[id]);
}

utils::JsonObject serialize_node_ref_list(const std::vector<NodePtr> &nodes, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (auto node : nodes)
    {
        if (node)
            arr.push_back(ref_table.node_ref(node));
    }
    return arr;
}

utils::JsonObject serialize_node_ref_levels(const std::vector<std::vector<NodePtr>> &levels, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (const auto &level : levels)
    {
        arr.push_back(serialize_node_ref_list(level, ref_table));
    }
    return arr;
}

bool deserialize_node_ref_list(GraphPtr graph,
                               const utils::JsonObject &object_table,
                               const utils::JsonObject &obj,
                               std::vector<NodePtr> &nodes)
{
    if (!graph || object_table.t != utils::JsonObject::OBJ || obj.t != utils::JsonObject::ARR)
        return false;

    nodes.clear();
    for (const auto &item : obj.array())
    {
        auto node = lookup_node_ref(graph, object_table, item);
        if (!node)
            return false;
        nodes.push_back(node);
    }
    return true;
}

bool deserialize_node_ref_levels(GraphPtr graph,
                                 const utils::JsonObject &object_table,
                                 const utils::JsonObject &obj,
                                 std::vector<std::vector<NodePtr>> &levels)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;

    levels.clear();
    for (const auto &item : obj.array())
    {
        std::vector<NodePtr> level;
        if (!deserialize_node_ref_list(graph, object_table, item, level))
            return false;
        levels.push_back(level);
    }
    return true;
}

struct MqIrCacheContext
{
    std::map<std::string, ConstTensorPtr> const_tensor_cache;
    std::map<std::string, SurfacePtr> surface_cache;
    std::map<std::string, MemoryObjectPtr> memory_cache;
};

std::string pointer_ref_id(utils::JsonObject ref)
{
    if (!is_pointer_ref(ref))
        return "";
    return (std::string)ref["id"];
}

MemoryObjectPtr deserialize_memory_ref(utils::JsonObject object_table,
                                       utils::JsonObject ref,
                                       MqIrCacheContext *context);

void restore_memory_fields(MemoryObjectPtr memory,
                           utils::JsonObject object_table,
                           utils::JsonObject desc,
                           MqIrCacheContext *context)
{
    if (!memory || desc.t != utils::JsonObject::OBJ)
        return;

    if (desc.exist("relative_offset"))
        memory->relative_offset = (uint64_t)desc["relative_offset"];
    if (desc.exist("m_base"))
        memory->m_base = (uint64_t)desc["m_base"];
    if (desc.exist("align"))
        memory->align = (uint32_t)(uint64_t)desc["align"];
    if (desc.exist("mask"))
        memory->mask = (uint64_t)desc["mask"];
    if (desc.exist("size"))
        memory->size = (uint64_t)desc["size"];
    if (desc.exist("foot_print"))
        memory->foot_print = (uint64_t)desc["foot_print"];
    if (desc.exist("mem_flag"))
        memory->mem_flag = (uint32_t)(uint64_t)desc["mem_flag"];
    if (desc.exist("buffer_id"))
        memory->buffer_id = (int32_t)(int64_t)desc["buffer_id"];
    if (desc.exist("bss_id"))
        memory->bss_id = (int32_t)(int64_t)desc["bss_id"];
    if (desc.exist("base_offset"))
        memory->base_offset = deserialize_memory_ref(object_table, desc["base_offset"], context);
    if (desc.exist("next_block"))
        memory->next_block = deserialize_memory_ref(object_table, desc["next_block"], context);
}

MemoryObjectPtr deserialize_memory_ref(utils::JsonObject object_table,
                                       utils::JsonObject ref,
                                       MqIrCacheContext *context)
{
    if (!is_pointer_ref(ref) || object_table.t != utils::JsonObject::OBJ)
        return nullptr;

    auto id = pointer_ref_id(ref);
    if (context && context->memory_cache.count(id))
        return context->memory_cache[id];
    if (id.empty() || !object_table.exist(id))
        return nullptr;

    auto desc = object_table[id];
    if (desc.t != utils::JsonObject::OBJ ||
        !desc.exist("object_kind") ||
        desc["object_kind"].t != utils::JsonObject::STR ||
        (std::string)desc["object_kind"] != "memory")
    {
        return nullptr;
    }

    auto memory = std::make_shared<MemoryObject>(0);
    if (context)
        context->memory_cache[id] = memory;
    restore_memory_fields(memory, object_table, desc, context);
    return memory;
}

utils::JsonObject serialize_memory_list(const std::vector<MemoryObjectPtr> &memories, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (auto memory : memories)
    {
        arr.push_back(ref_table.memory_ref(memory));
    }
    return arr;
}

bool deserialize_memory_list(utils::JsonObject object_table,
                             utils::JsonObject obj,
                             std::vector<MemoryObjectPtr> &memories,
                             MqIrCacheContext *context)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;

    memories.clear();
    for (auto item : obj.array())
    {
        auto memory = deserialize_memory_ref(object_table, item, context);
        if (!memory)
            return false;
        memories.push_back(memory);
    }
    return true;
}

utils::JsonObject serialize_expr(Expr expr)
{
    if (!expr)
        return utils::JsonObject();
    return expr->to_string();
}

bool deserialize_expr(const utils::JsonObject &obj, Expr &expr)
{
    if (obj.t == utils::JsonObject::STR)
    {
        expr = Expr((std::string)obj);
        return true;
    }
    if (obj.t == utils::JsonObject::INT || obj.t == utils::JsonObject::FLOAT)
    {
        expr = Expr((double)obj);
        return true;
    }
    return false;
}

utils::JsonObject serialize_dim3(const Dim3 &dim)
{
    utils::JsonObject obj(utils::JsonObject::OBJ);
    obj["x"] = serialize_expr(dim.x);
    obj["y"] = serialize_expr(dim.y);
    obj["z"] = serialize_expr(dim.z);
    return obj;
}

bool deserialize_dim3(utils::JsonObject obj, Dim3 &dim)
{
    if (obj.t != utils::JsonObject::OBJ)
        return false;
    return deserialize_expr(obj["x"], dim.x) &&
           deserialize_expr(obj["y"], dim.y) &&
           deserialize_expr(obj["z"], dim.z);
}

utils::JsonObject serialize_placeholder(PlaceholderPtr placeholder)
{
    utils::JsonObject obj(utils::JsonObject::OBJ);
    int64_t offset = 0;
    if (placeholder)
    {
        offset = (int64_t)placeholder->memory_offset()->relative_offset;
        if (placeholder->attrs().count(Constants::expr_offset))
        {
            Expr expr = placeholder->get_attr(Constants::expr_offset);
            obj["expr_offset"] = serialize_expr(expr);
        }
    }
    obj["offset"] = offset;
    return obj;
}

PlaceholderPtr deserialize_placeholder(utils::JsonObject obj)
{
    auto placeholder = std::make_shared<Placeholder>();
    if (obj.t != utils::JsonObject::OBJ)
        return placeholder;
    if (obj.exist("offset") && obj["offset"].t == utils::JsonObject::INT)
    {
        *placeholder = (uint32_t)(int64_t)obj["offset"];
    }
    if (obj.exist("expr_offset"))
    {
        Expr expr;
        if (deserialize_expr(obj["expr_offset"], expr))
        {
            placeholder->set_attr(Constants::expr_offset, expr);
        }
    }
    return placeholder;
}

utils::JsonObject serialize_placeholder_list(const std::vector<PlaceholderPtr> &placeholders)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (auto placeholder : placeholders)
    {
        arr.push_back(serialize_placeholder(placeholder));
    }
    return arr;
}

bool deserialize_placeholder_list(const utils::JsonObject &obj, std::vector<PlaceholderPtr> &placeholders)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;
    placeholders.clear();
    for (const auto &item : obj.array())
    {
        placeholders.push_back(deserialize_placeholder(item));
    }
    return true;
}

utils::JsonObject serialize_const_tensor_ref(ConstTensorPtr tensor, PointerRefTable &ref_table)
{
    std::shared_ptr<ConstTensor> const_tensor = tensor;
    TensorPtr tensor_ptr = std::static_pointer_cast<Tensor>(const_tensor);
    return ref_table.tensor_ref(tensor_ptr);
}

ConstTensorPtr deserialize_const_tensor_ref(GraphPtr graph,
                                            utils::JsonObject object_table,
                                            utils::JsonObject ref,
                                            MqIrCacheContext *context)
{
    auto id = pointer_ref_id(ref);
    if (context && context->const_tensor_cache.count(id))
        return context->const_tensor_cache[id];

    auto tensor = lookup_tensor_ref(graph, object_table, ref);
    if (!tensor)
        return nullptr;

    auto existing = std::dynamic_pointer_cast<ConstTensor>(tensor);
    ConstTensorPtr const_tensor = existing ? ConstTensorPtr(existing) : ConstTensorPtr(tensor);
    if (context && !id.empty())
        context->const_tensor_cache[id] = const_tensor;
    return const_tensor;
}

utils::JsonObject serialize_const_tensor_list(const std::vector<ConstTensorPtr> &tensors, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (auto tensor : tensors)
    {
        arr.push_back(serialize_const_tensor_ref(tensor, ref_table));
    }
    return arr;
}

bool deserialize_const_tensor_list(GraphPtr graph,
                                   utils::JsonObject object_table,
                                   utils::JsonObject obj,
                                   std::vector<ConstTensorPtr> &tensors,
                                   MqIrCacheContext *context)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;
    tensors.clear();
    for (const auto &item : obj.array())
    {
        auto tensor = deserialize_const_tensor_ref(graph, object_table, item, context);
        if (!tensor)
            return false;
        tensors.push_back(tensor);
    }
    return true;
}

utils::JsonObject serialize_surface_ref(SurfacePtr surface, PointerRefTable &ref_table)
{
    utils::JsonObject obj(utils::JsonObject::OBJ);
    TensorPtr tensor = surface;
    obj["tensor"] = ref_table.tensor_ref(tensor);
    auto desc = surface->get_surf_desc();
    const uint32_t *words = reinterpret_cast<const uint32_t *>(&desc);
    obj["desc0"] = (uint64_t)words[0];
    obj["desc1"] = (uint64_t)words[1];
    return obj;
}

SurfacePtr deserialize_surface_ref(GraphPtr graph,
                                   utils::JsonObject object_table,
                                   utils::JsonObject obj,
                                   MqIrCacheContext *context)
{
    if (obj.t != utils::JsonObject::OBJ || !obj.exist("tensor"))
        return nullptr;

    auto id = pointer_ref_id(obj["tensor"]);
    if (context && context->surface_cache.count(id))
        return context->surface_cache[id];

    auto tensor = lookup_tensor_ref(graph, object_table, obj["tensor"]);
    if (!tensor)
        return nullptr;

    auto surface = std::dynamic_pointer_cast<Surface>(tensor);
    if (!surface)
    {
        surface = std::make_shared<Surface>(tensor);
    }

    if (obj.exist("desc0") && obj.exist("desc1"))
    {
        cpsSurfDesc desc{};
        uint32_t words[2] = {
            (uint32_t)(uint64_t)obj["desc0"],
            (uint32_t)(uint64_t)obj["desc1"],
        };
        std::memcpy(&desc, words, sizeof(desc));
        surface->set_surf_desc(desc);
    }

    if (context && !id.empty())
        context->surface_cache[id] = surface;
    return surface;
}

utils::JsonObject serialize_readonly_param(const ReadOnlyParam &param, PointerRefTable &ref_table);
bool deserialize_readonly_param(GraphPtr graph,
                                utils::JsonObject object_table,
                                utils::JsonObject obj,
                                ReadOnlyParam &param,
                                MqIrCacheContext *context);

utils::JsonObject serialize_readonly_params(const std::vector<ReadOnlyParam> &params, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (const auto &param : params)
    {
        arr.push_back(serialize_readonly_param(param, ref_table));
    }
    return arr;
}

bool deserialize_readonly_params(GraphPtr graph,
                                 const utils::JsonObject &object_table,
                                 const utils::JsonObject &obj,
                                 std::vector<ReadOnlyParam> &params,
                                 MqIrCacheContext *context)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;
    params.clear();
    for (const auto &item : obj.array())
    {
        ReadOnlyParam param((uint64_t)0);
        if (!deserialize_readonly_param(graph, object_table, item, param, context))
            return false;
        params.push_back(param);
    }
    return true;
}

utils::JsonObject serialize_readonly_param(const ReadOnlyParam &param, PointerRefTable &ref_table)
{
    utils::JsonObject obj(utils::JsonObject::OBJ);
    if (param.const_tensor)
    {
        obj["kind"] = "const_tensor";
        obj["value"] = serialize_const_tensor_ref(param.const_tensor, ref_table);
    }
    else if (param.surface)
    {
        obj["kind"] = "surface";
        obj["value"] = serialize_surface_ref(param.surface, ref_table);
    }
    else if (param.param_list)
    {
        obj["kind"] = "param_list";
        obj["name"] = param.param_list->name();
        obj["dtype"] = serialize_tensor_dtype(param.param_list->type());
        obj["value"] = serialize_readonly_params(param.param_list->params(), ref_table);
    }
    else if (param.tensor)
    {
        auto placeholder = std::dynamic_pointer_cast<Placeholder>(param.tensor);
        if (placeholder)
        {
            obj["kind"] = "placeholder";
            obj["value"] = serialize_placeholder(placeholder);
        }
        else
        {
            obj["kind"] = "tensor";
            obj["value"] = ref_table.tensor_ref(param.tensor);
        }
    }
    else if (param.expr)
    {
        obj["kind"] = "expr";
        obj["value"] = serialize_expr(param.expr);
    }
    else
    {
        obj["kind"] = "value";
        obj["value"] = (int64_t)param.value;
    }
    return obj;
}

bool deserialize_readonly_param(GraphPtr graph,
                                utils::JsonObject object_table,
                                utils::JsonObject obj,
                                ReadOnlyParam &param,
                                MqIrCacheContext *context)
{
    if (obj.t != utils::JsonObject::OBJ || !obj.exist("kind") || !obj.exist("value") ||
        obj["kind"].t != utils::JsonObject::STR)
        return false;

    std::string kind = (std::string)obj["kind"];
    if (kind == "value")
    {
        param = ReadOnlyParam((uint64_t)(int64_t)obj["value"]);
        return true;
    }
    if (kind == "expr")
    {
        Expr expr;
        if (!deserialize_expr(obj["value"], expr))
            return false;
        param = ReadOnlyParam(expr);
        return true;
    }
    if (kind == "tensor")
    {
        auto tensor = lookup_tensor_ref(graph, object_table, obj["value"]);
        if (!tensor)
            return false;
        param = ReadOnlyParam(tensor);
        return true;
    }
    if (kind == "placeholder")
    {
        param = ReadOnlyParam(deserialize_placeholder(obj["value"]));
        return true;
    }
    if (kind == "const_tensor")
    {
        auto tensor = deserialize_const_tensor_ref(graph, object_table, obj["value"], context);
        if (!tensor)
            return false;
        param = ReadOnlyParam(tensor);
        return true;
    }
    if (kind == "surface")
    {
        auto surface = deserialize_surface_ref(graph, object_table, obj["value"], context);
        if (!surface)
            return false;
        param = ReadOnlyParam(surface);
        return true;
    }
    if (kind == "param_list")
    {
        std::vector<ReadOnlyParam> params;
        if (!deserialize_readonly_params(graph, object_table, obj["value"], params, context))
            return false;
        Tensor::Dtype dtype = Tensor::UINT32;
        if (obj.exist("dtype"))
            deserialize_tensor_dtype(obj["dtype"], dtype);
        auto param_list = std::make_shared<ParamList>(obj.exist("name") ? (std::string)obj["name"] : std::string(), dtype);
        param_list->set_params(params);
        param = ReadOnlyParam(param_list);
        return true;
    }
    return false;
}

utils::JsonObject serialize_kernel_infos(const KernelInfos &kernel_infos, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (const auto &info : kernel_infos)
    {
        utils::JsonObject obj(utils::JsonObject::OBJ);
        obj["kernel_name"] = info.kernel_name;
        obj["grid"] = serialize_dim3(info.grid);
        obj["block"] = serialize_dim3(info.block);
        obj["sharemem_size"] = (uint64_t)info.sharemem_size;
        obj["args"] = serialize_readonly_params(info.args, ref_table);
        arr.push_back(obj);
    }
    return arr;
}

bool deserialize_kernel_infos(GraphPtr graph,
                              utils::JsonObject object_table,
                              const utils::JsonObject &obj,
                              KernelInfos &kernel_infos,
                              MqIrCacheContext *context)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;
    kernel_infos.clear();
    for (auto item : obj.array())
    {
        if (item.t != utils::JsonObject::OBJ)
            return false;
        KernelInfo info;
        info.kernel_name = item.exist("kernel_name") ? (std::string)item["kernel_name"] : std::string();
        if (!deserialize_dim3(item["grid"], info.grid) ||
            !deserialize_dim3(item["block"], info.block))
            return false;
        info.sharemem_size = item.exist("sharemem_size") ? (size_t)(uint64_t)item["sharemem_size"] : 0;
        if (!deserialize_readonly_params(graph, object_table, item["args"], info.args, context))
            return false;
        kernel_infos.push_back(info);
    }
    return true;
}

utils::JsonObject serialize_ws_map(const std::map<NodePtr, std::set<TensorPtr>> &ws_map, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (const auto &kv : ws_map)
    {
        utils::JsonObject item(utils::JsonObject::OBJ);
        item["node"] = ref_table.node_ref(kv.first);
        utils::JsonObject tensors(utils::JsonObject::ARR);
        for (auto tensor : kv.second)
        {
            tensors.push_back(ref_table.tensor_ref(tensor));
        }
        item["tensors"] = tensors;
        arr.push_back(item);
    }
    return arr;
}

bool deserialize_ws_map(GraphPtr graph,
                        utils::JsonObject object_table,
                        const utils::JsonObject &obj,
                        std::map<NodePtr, std::set<TensorPtr>> &ws_map)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;
    ws_map.clear();
    for (auto item : obj.array())
    {
        if (item.t != utils::JsonObject::OBJ || !item.exist("node") || !item.exist("tensors"))
            return false;
        auto node = lookup_node_ref(graph, object_table, item["node"]);
        if (!node || item["tensors"].t != utils::JsonObject::ARR)
            return false;
        for (const auto &tensor_ref : item["tensors"].array())
        {
            auto tensor = lookup_tensor_ref(graph, object_table, tensor_ref);
            if (!tensor)
                return false;
            ws_map[node].insert(tensor);
        }
    }
    return true;
}

utils::JsonObject serialize_surface_tensor_map(const std::map<TensorPtr, NodePtr> &surface_tensors, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (const auto &kv : surface_tensors)
    {
        utils::JsonObject item(utils::JsonObject::OBJ);
        item["tensor"] = ref_table.tensor_ref(kv.first);
        item["node"] = ref_table.node_ref(kv.second);
        arr.push_back(item);
    }
    return arr;
}

bool deserialize_surface_tensor_map(GraphPtr graph,
                                    utils::JsonObject object_table,
                                    const utils::JsonObject &obj,
                                    std::map<TensorPtr, NodePtr> &surface_tensors)
{
    if (obj.t != utils::JsonObject::ARR)
        return false;
    surface_tensors.clear();
    for (auto item : obj.array())
    {
        if (item.t != utils::JsonObject::OBJ || !item.exist("tensor") || !item.exist("node"))
            return false;
        auto tensor = lookup_tensor_ref(graph, object_table, item["tensor"]);
        auto node = lookup_node_ref(graph, object_table, item["node"]);
        if (!tensor || !node)
            return false;
        surface_tensors[tensor] = node;
    }
    return true;
}

utils::JsonObject serialize_string_map(const std::map<std::string, std::string> &map_value)
{
    utils::JsonObject obj(utils::JsonObject::OBJ);
    for (const auto &kv : map_value)
    {
        obj[kv.first] = kv.second;
    }
    return obj;
}

bool deserialize_string_map(const utils::JsonObject &obj, std::map<std::string, std::string> &map_value)
{
    if (obj.t != utils::JsonObject::OBJ)
        return false;
    map_value.clear();
    for (const auto &kv : obj.obj())
    {
        if (kv.second.t != utils::JsonObject::STR)
            return false;
        map_value[kv.first] = (std::string)kv.second;
    }
    return true;
}

// Helper function: serialize an Any value to JsonObject
bool serialize_any_to_json(const std::string& key, const Any& value, utils::JsonObject& json_obj, PointerRefTable *ref_table = nullptr)
{
    if (!value.value())
        return false;

    auto type_id = value.get_type_id();

    if (type_id == GET_ANY_TYPEID<int>())
    {
        set_typed_json_attr(key, "int", value.cast<int>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<int32_t>())
    {
        set_typed_json_attr(key, "int32_t", value.cast<int32_t>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<uint32_t>())
    {
        set_typed_json_attr(key, "uint32_t", value.cast<uint32_t>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<int64_t>())
    {
        set_typed_json_attr(key, "int64_t", value.cast<int64_t>(), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<uint64_t>())
    {
        set_typed_json_attr(key, "uint64_t", (int64_t)value.cast<uint64_t>(), json_obj);
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
    else if (type_id == GET_ANY_TYPEID<DataLayout>())
    {
        set_typed_json_attr(key, "DataLayout", serialize_layout(value.cast<DataLayout>()), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<LayoutList>())
    {
        set_typed_json_attr(key, "LayoutList", serialize_layout_list(value.cast<LayoutList>()), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<LayoutCombinations>())
    {
        set_typed_json_attr(key, "LayoutCombinations", serialize_layout_combinations(value.cast<LayoutCombinations>()), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<TensorPtr>() && ref_table)
    {
        set_typed_json_attr(key, "TensorPtr", ref_table->tensor_ref(value.cast<TensorPtr>()), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<NodePtr>() && ref_table)
    {
        auto node = value.cast<NodePtr>();
        if (!node)
            return false;
        set_typed_json_attr(key, "NodePtr", ref_table->node_ref(node), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::vector<NodePtr>>() && ref_table)
    {
        set_typed_json_attr(key, "vector<NodePtr>", serialize_node_ref_list(value.cast<std::vector<NodePtr>>(), *ref_table), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::vector<std::vector<NodePtr>>>() && ref_table)
    {
        set_typed_json_attr(key, "vector<vector<NodePtr>>",
                            serialize_node_ref_levels(value.cast<std::vector<std::vector<NodePtr>>>(), *ref_table),
                            json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<Expr>())
    {
        set_typed_json_attr(key, "Expr", serialize_expr(value.cast<Expr>()), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<KernelInfos>() && ref_table)
    {
        set_typed_json_attr(key, "KernelInfos", serialize_kernel_infos(value.cast<KernelInfos>(), *ref_table), json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::vector<PlaceholderPtr>>())
    {
        set_typed_json_attr(key, "vector<PlaceholderPtr>",
                            serialize_placeholder_list(value.cast<std::vector<PlaceholderPtr>>()),
                            json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::vector<ConstTensorPtr>>() && ref_table)
    {
        set_typed_json_attr(key, "vector<ConstTensorPtr>",
                            serialize_const_tensor_list(value.cast<std::vector<ConstTensorPtr>>(), *ref_table),
                            json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::vector<MemoryObjectPtr>>() && ref_table)
    {
        set_typed_json_attr(key, "vector<MemoryObjectPtr>",
                            serialize_memory_list(value.cast<std::vector<MemoryObjectPtr>>(), *ref_table),
                            json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::map<NodePtr, std::set<TensorPtr>>>() && ref_table)
    {
        set_typed_json_attr(key, "map<NodePtr,set<TensorPtr>>",
                            serialize_ws_map(value.cast<std::map<NodePtr, std::set<TensorPtr>>>(), *ref_table),
                            json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::map<TensorPtr, NodePtr>>() && ref_table)
    {
        set_typed_json_attr(key, "map<TensorPtr,NodePtr>",
                            serialize_surface_tensor_map(value.cast<std::map<TensorPtr, NodePtr>>(), *ref_table),
                            json_obj);
        return true;
    }
    else if (type_id == GET_ANY_TYPEID<std::map<std::string, std::string>>())
    {
        set_typed_json_attr(key, "map<string,string>",
                            serialize_string_map(value.cast<std::map<std::string, std::string>>()),
                            json_obj);
        return true;
    }

    return false;
}

// Helper function: deserialize typed JsonObject to Any and set as node/tensor attribute
bool deserialize_json_to_any(const std::string& key, const utils::JsonObject& json_value,
                              std::function<void(const std::string&, const Any&)> set_attr_func,
                              GraphPtr graph = nullptr,
                              const utils::JsonObject *object_table = nullptr,
                              MqIrCacheContext *context = nullptr)
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
    else if (type == "int32_t")
    {
        set_attr_func(key, Any((int32_t)(int)value_obj));
        return true;
    }
    else if (type == "uint32_t")
    {
        set_attr_func(key, Any((uint32_t)(int)value_obj));
        return true;
    }
    else if (type == "int64_t")
    {
        set_attr_func(key, Any((int64_t)value_obj));
        return true;
    }
    else if (type == "uint64_t")
    {
        set_attr_func(key, Any((uint64_t)(int64_t)value_obj));
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
    else if (type == "DataLayout")
    {
        DataLayout layout = DataLayout_UnSupport;
        if (!deserialize_layout(value_obj, layout))
            return false;
        set_attr_func(key, Any(layout));
        return true;
    }
    else if (type == "LayoutList")
    {
        LayoutList layouts;
        if (!deserialize_layout_list(value_obj, layouts))
            return false;
        set_attr_func(key, Any(layouts));
        return true;
    }
    else if (type == "LayoutCombinations")
    {
        LayoutCombinations layout_combinations;
        if (!deserialize_layout_combinations(value_obj, layout_combinations))
            return false;
        set_attr_func(key, Any(layout_combinations));
        return true;
    }
    else if (type == "TensorPtr")
    {
        if (!graph || !object_table)
            return false;
        auto tensor = lookup_tensor_ref(graph, *object_table, value_obj);
        if (!tensor)
            return false;
        set_attr_func(key, Any(tensor));
        return true;
    }
    else if (type == "NodePtr")
    {
        if (!graph || !object_table)
            return false;
        auto node = lookup_node_ref(graph, *object_table, value_obj);
        if (!node)
            return false;
        set_attr_func(key, Any(node));
        return true;
    }
    else if (type == "vector<NodePtr>")
    {
        if (!graph || !object_table)
            return false;
        std::vector<NodePtr> nodes;
        if (!deserialize_node_ref_list(graph, *object_table, value_obj, nodes))
            return false;
        set_attr_func(key, Any(nodes));
        return true;
    }
    else if (type == "vector<vector<NodePtr>>")
    {
        if (!graph || !object_table)
            return false;
        std::vector<std::vector<NodePtr>> levels;
        if (!deserialize_node_ref_levels(graph, *object_table, value_obj, levels))
            return false;
        set_attr_func(key, Any(levels));
        return true;
    }
    else if (type == "Expr")
    {
        Expr expr;
        if (!deserialize_expr(value_obj, expr))
            return false;
        set_attr_func(key, Any(expr));
        return true;
    }
    else if (type == "KernelInfos")
    {
        if (!graph || !object_table)
            return false;
        KernelInfos kernel_infos;
        if (!deserialize_kernel_infos(graph, *object_table, value_obj, kernel_infos, context))
            return false;
        set_attr_func(key, Any(kernel_infos));
        return true;
    }
    else if (type == "vector<PlaceholderPtr>")
    {
        std::vector<PlaceholderPtr> placeholders;
        if (!deserialize_placeholder_list(value_obj, placeholders))
            return false;
        set_attr_func(key, Any(placeholders));
        return true;
    }
    else if (type == "vector<ConstTensorPtr>")
    {
        if (!graph || !object_table)
            return false;
        std::vector<ConstTensorPtr> tensors;
        if (!deserialize_const_tensor_list(graph, *object_table, value_obj, tensors, context))
            return false;
        set_attr_func(key, Any(tensors));
        return true;
    }
    else if (type == "vector<MemoryObjectPtr>")
    {
        if (!object_table)
            return false;
        std::vector<MemoryObjectPtr> memories;
        if (!deserialize_memory_list(*object_table, value_obj, memories, context))
            return false;
        set_attr_func(key, Any(memories));
        return true;
    }
    else if (type == "map<NodePtr,set<TensorPtr>>")
    {
        if (!graph || !object_table)
            return false;
        std::map<NodePtr, std::set<TensorPtr>> ws_map;
        if (!deserialize_ws_map(graph, *object_table, value_obj, ws_map))
            return false;
        set_attr_func(key, Any(ws_map));
        return true;
    }
    else if (type == "map<TensorPtr,NodePtr>")
    {
        if (!graph || !object_table)
            return false;
        std::map<TensorPtr, NodePtr> surface_tensors;
        if (!deserialize_surface_tensor_map(graph, *object_table, value_obj, surface_tensors))
            return false;
        set_attr_func(key, Any(surface_tensors));
        return true;
    }
    else if (type == "map<string,string>")
    {
        std::map<std::string, std::string> map_value;
        if (!deserialize_string_map(value_obj, map_value))
            return false;
        set_attr_func(key, Any(map_value));
        return true;
    }

    return false;
}

int serialize_attrs_to_json(const std::map<std::string, Any> &attrs, utils::JsonObject &json_obj, PointerRefTable *ref_table = nullptr)
{
    int attr_count = 0;
    for (const auto &attr : attrs)
    {
        if (serialize_any_to_json(attr.first, attr.second, json_obj, ref_table))
        {
            attr_count++;
        }
    }
    return attr_count;
}

int deserialize_attrs_from_json(const utils::JsonObject &json_obj,
                                std::function<void(const std::string&, const Any&)> set_attr_func,
                                GraphPtr graph = nullptr,
                                const utils::JsonObject *object_table = nullptr,
                                MqIrCacheContext *context = nullptr)
{
    if (json_obj.t != utils::JsonObject::OBJ)
        return 0;

    int attr_count = 0;
    for (const auto &kv : json_obj.obj())
    {
        if (deserialize_json_to_any(kv.first, kv.second, set_attr_func, graph, object_table, context))
        {
            attr_count++;
        }
    }
    return attr_count;
}

utils::JsonObject serialize_tensor_metadata(TensorPtr tensor, size_t tensor_index, int &tensor_attr_count, PointerRefTable *ref_table = nullptr)
{
    utils::JsonObject tensor_obj(utils::JsonObject::OBJ);
    tensor_obj["tensor_name"] = tensor->name();
    tensor_obj["tensor_index"] = (int)tensor_index;
    tensor_obj["shape"] = serialize_tensor_shape(tensor->shape());
    tensor_obj["dtype"] = serialize_tensor_dtype(tensor->type());
    tensor_obj["layout"] = serialize_layout(tensor->shape().layout());
    if (ref_table)
    {
        tensor_obj["pointer_ref"] = ref_table->tensor_ref(tensor);
        tensor_obj["memory_offset"] = ref_table->memory_ref(tensor->memory_offset());
    }

    utils::JsonObject tensor_attrs(utils::JsonObject::OBJ);
    int count = serialize_attrs_to_json(tensor->attrs(), tensor_attrs, ref_table);
    if (count > 0)
    {
        tensor_obj["tensor_attrs"] = tensor_attrs;
        tensor_attr_count += count;
    }

    return tensor_obj;
}

utils::JsonObject serialize_global_tensors(GraphPtr graph, int &tensor_attr_count, PointerRefTable &ref_table)
{
    utils::JsonObject globals(utils::JsonObject::OBJ);
    size_t tensor_index = 0;
    for (const auto &kv : graph->global_tensors())
    {
        if (!kv.second)
            continue;

        auto tensor_obj = serialize_tensor_metadata(kv.second, tensor_index++, tensor_attr_count, &ref_table);
        tensor_obj["global_key"] = kv.first;
        globals[kv.first] = tensor_obj;
    }
    return globals;
}

utils::JsonObject serialize_node_metadata(NodePtr node, int &node_attr_count, int &tensor_attr_count, PointerRefTable *ref_table = nullptr)
{
    utils::JsonObject node_obj(utils::JsonObject::OBJ);
    node_obj["node_name"] = node->name();
    if (node->graph())
    {
        node_obj["graph_name"] = node->graph()->name();
    }
    node_obj["op_type"] = optype_to_string(node->type());
    if (node->attrs().count("layer_id"))
    {
        node_obj["layer_id"] = node->get_attr<int>("layer_id");
    }

    utils::JsonObject node_attrs(utils::JsonObject::OBJ);
    int count = serialize_attrs_to_json(node->attrs(), node_attrs, ref_table);
    if (count > 0)
    {
        node_obj["node_attrs"] = node_attrs;
        node_attr_count += count;
    }

    utils::JsonObject tensors_arr(utils::JsonObject::ARR);
    for (size_t i = 0; i < node->outputs().size(); ++i)
    {
        auto tensor_obj = serialize_tensor_metadata(node->outputs()[i], i, tensor_attr_count, ref_table);
        if (has_json_object_field(tensor_obj, "tensor_attrs") || has_json_object_field(tensor_obj, "layout"))
        {
            tensors_arr.push_back(tensor_obj);
        }
    }
    if (!tensors_arr.array().empty())
    {
        node_obj["output_tensors"] = tensors_arr;
    }

    utils::JsonObject constants_arr(utils::JsonObject::ARR);
    int constant_index = 0;
    for (const auto &kv : node->constants())
    {
        auto tensor_obj = serialize_tensor_metadata(kv.second, constant_index++, tensor_attr_count, ref_table);
        tensor_obj["constant_key"] = kv.first;
        constants_arr.push_back(tensor_obj);
    }
    if (!constants_arr.array().empty())
    {
        node_obj["constant_tensors"] = constants_arr;
    }

    utils::JsonObject placeholders_arr(utils::JsonObject::ARR);
    for (size_t i = 0; i < node->placeholders().size(); ++i)
    {
        placeholders_arr.push_back(serialize_tensor_metadata(node->placeholders()[i], i, tensor_attr_count, ref_table));
    }
    if (!placeholders_arr.array().empty())
    {
        node_obj["placeholder_tensors"] = placeholders_arr;
    }

    return node_obj;
}

TensorPtr find_output_tensor(NodePtr node, utils::JsonObject tensor_json)
{
    if (tensor_json.t != utils::JsonObject::OBJ)
        return nullptr;

    if (tensor_json.exist("tensor_name"))
    {
        std::string tensor_name = (std::string)tensor_json["tensor_name"];
        for (auto tensor : node->outputs())
        {
            if (tensor->name() == tensor_name)
            {
                return tensor;
            }
        }
    }

    if (tensor_json.exist("tensor_index"))
    {
        int tensor_index = tensor_json["tensor_index"];
        if (tensor_index >= 0 && tensor_index < (int)node->outputs().size())
        {
            return node->outputs()[tensor_index];
        }
    }

    return nullptr;
}

TensorPtr find_constant_tensor(NodePtr node, utils::JsonObject tensor_json)
{
    if (tensor_json.t != utils::JsonObject::OBJ)
        return nullptr;

    if (tensor_json.exist("constant_key"))
    {
        std::string key = (std::string)tensor_json["constant_key"];
        if (node->constants().count(key))
        {
            return node->constants()[key];
        }
    }

    if (tensor_json.exist("tensor_name"))
    {
        std::string tensor_name = (std::string)tensor_json["tensor_name"];
        for (auto &kv : node->constants())
        {
            if (kv.second->name() == tensor_name)
            {
                return kv.second;
            }
        }
    }

    return nullptr;
}

TensorPtr find_placeholder_tensor(NodePtr node, utils::JsonObject tensor_json)
{
    if (tensor_json.t != utils::JsonObject::OBJ)
        return nullptr;

    if (tensor_json.exist("tensor_name"))
    {
        std::string tensor_name = (std::string)tensor_json["tensor_name"];
        for (auto tensor : node->placeholders())
        {
            if (tensor->name() == tensor_name)
            {
                return tensor;
            }
        }
    }

    if (tensor_json.exist("tensor_index"))
    {
        int tensor_index = tensor_json["tensor_index"];
        if (tensor_index >= 0 && tensor_index < (int)node->placeholders().size())
        {
            return node->placeholders()[tensor_index];
        }
    }

    return nullptr;
}

utils::JsonObject serialize_tensor_ref_list(const std::vector<TensorPtr> &tensors, PointerRefTable &ref_table)
{
    utils::JsonObject arr(utils::JsonObject::ARR);
    for (auto tensor : tensors)
    {
        arr.push_back(ref_table.tensor_ref(tensor));
    }
    return arr;
}

utils::JsonObject serialize_graph_references(GraphPtr graph, PointerRefTable &ref_table)
{
    utils::JsonObject refs(utils::JsonObject::OBJ);
    refs["input_tensors"] = serialize_tensor_ref_list(graph->get_input_tensors(), ref_table);
    refs["output_tensors"] = serialize_tensor_ref_list(graph->get_output_tensors(), ref_table);

    utils::JsonObject globals(utils::JsonObject::OBJ);
    for (const auto &kv : graph->global_tensors())
    {
        if (kv.second)
            globals[kv.first] = ref_table.tensor_ref(kv.second);
    }
    refs["global_tensors"] = globals;
    return refs;
}

void restore_tensor_ref_list(GraphPtr graph,
                             utils::JsonObject object_table,
                             utils::JsonObject refs,
                             std::vector<TensorPtr> &tensors)
{
    if (refs.t != utils::JsonObject::ARR)
        return;

    for (const auto &ref : refs.array())
    {
        auto tensor = lookup_tensor_ref(graph, object_table, ref);
        if (tensor)
            tensors.push_back(tensor);
    }
}

void restore_graph_references(GraphPtr graph, utils::JsonObject graph_json, const utils::JsonObject *object_table)
{
    if (!graph || !object_table || graph_json.t != utils::JsonObject::OBJ || !graph_json.exist("graph_references"))
        return;

    auto refs = graph_json["graph_references"];
    if (refs.t != utils::JsonObject::OBJ)
        return;

    if (refs.exist("input_tensors"))
    {
        std::vector<TensorPtr> tensors;
        restore_tensor_ref_list(graph, *object_table, refs["input_tensors"], tensors);
        if (!tensors.empty())
            graph->record_input_tensors(tensors);
    }
    if (refs.exist("output_tensors"))
    {
        std::vector<TensorPtr> tensors;
        restore_tensor_ref_list(graph, *object_table, refs["output_tensors"], tensors);
        if (!tensors.empty())
            graph->record_output_tensors(tensors);
    }
    if (refs.exist("global_tensors") && refs["global_tensors"].t == utils::JsonObject::OBJ)
    {
        for (const auto &kv : refs["global_tensors"].obj())
        {
            auto tensor = lookup_tensor_ref(graph, *object_table, kv.second);
            if (tensor)
                graph->global_tensors()[kv.first] = tensor;
        }
    }
}

int restore_object_payloads(GraphPtr graph,
                            utils::JsonObject object_table,
                            utils::JsonObject object_payloads,
                            const std::string &bin_path)
{
    if (!graph || object_table.t != utils::JsonObject::OBJ || object_payloads.t != utils::JsonObject::OBJ)
        return 0;

    std::ifstream ifs(bin_path, std::ios::binary);
    if (!ifs.is_open())
    {
        WARN() << "mq_load_ir: tensor payload file not found, skipping: " << bin_path;
        return 0;
    }

    int restored = 0;
    for (const auto &kv : object_payloads.obj())
    {
        const std::string &id = kv.first;
        if (!object_table.exist(id))
            continue;

        utils::JsonObject data_obj;
        if (!get_json_object_field(kv.second, "data", data_obj) || data_obj.t != utils::JsonObject::OBJ)
            continue;

        utils::JsonObject offset_obj;
        utils::JsonObject size_obj;
        if (!get_json_object_field(data_obj, "offset", offset_obj) ||
            !get_json_object_field(data_obj, "size", size_obj) ||
            offset_obj.t != utils::JsonObject::INT ||
            size_obj.t != utils::JsonObject::INT)
        {
            continue;
        }

        int64_t offset = offset_obj;
        int64_t size = size_obj;
        if (offset < 0 || size <= 0)
            continue;

        auto tensor = lookup_tensor_by_desc_in_graphs(graph, object_table[id]);
        if (!tensor)
        {
            WARN() << "mq_load_ir: cannot resolve tensor payload ref " << id;
            continue;
        }

        size_t data_size = (size_t)size;
        char *data = new char[data_size];
        ifs.seekg(offset);
        ifs.read(data, data_size);
        if (ifs.fail())
        {
            WARN() << "mq_load_ir: failed to read tensor payload for " << tensor->name();
            delete[] data;
            ifs.clear();
            continue;
        }

        tensor->set_data(data, data_size);
        restored++;
    }

    INFO() << "mq_load_ir: restored " << restored << " tensor payloads from " << bin_path;
    return restored;
}

int restore_tensor_metadata(TensorPtr tensor,
                            utils::JsonObject tensor_json,
                            GraphPtr graph,
                            const utils::JsonObject *object_table,
                            MqIrCacheContext *context = nullptr)
{
    if (!tensor || tensor_json.t != utils::JsonObject::OBJ)
        return 0;

    DataLayout layout = tensor->shape().layout();
    if (tensor_json.exist("layout"))
    {
        if (deserialize_layout(tensor_json["layout"], layout))
        {
            tensor->shape().set_layout(layout);
        }
    }

    if (tensor_json.exist("shape"))
    {
        TensorShape shape;
        if (deserialize_tensor_shape(tensor_json["shape"], shape, layout))
        {
            tensor->shape() = shape;
        }
    }

    if (tensor_json.exist("dtype"))
    {
        Tensor::Dtype dtype = tensor->type();
        if (deserialize_tensor_dtype(tensor_json["dtype"], dtype))
        {
            tensor->set_type(dtype);
        }
    }

    if (object_table && context && tensor_json.exist("memory_offset"))
    {
        auto memory = deserialize_memory_ref(*object_table, tensor_json["memory_offset"], context);
        if (memory)
        {
            *tensor->memory_offset() = *memory;
            context->memory_cache[pointer_ref_id(tensor_json["memory_offset"])] = tensor->memory_offset();
        }
    }

    if (!tensor_json.exist("tensor_attrs"))
        return 0;

    return deserialize_attrs_from_json(tensor_json["tensor_attrs"],
        [&tensor](const std::string& key, const Any& value) {
            tensor->set_attr(key, value);
        }, graph, object_table, context);
}

TensorPtr create_tensor_from_metadata(const std::string &fallback_name, utils::JsonObject tensor_json)
{
    std::string name = fallback_name;
    if (tensor_json.t == utils::JsonObject::OBJ &&
        tensor_json.exist("tensor_name") &&
        tensor_json["tensor_name"].t == utils::JsonObject::STR)
    {
        name = (std::string)tensor_json["tensor_name"];
    }

    DataLayout layout = DataLayout_Unkown;
    if (tensor_json.t == utils::JsonObject::OBJ && tensor_json.exist("layout"))
    {
        deserialize_layout(tensor_json["layout"], layout);
    }

    TensorShape shape(0);
    if (tensor_json.t == utils::JsonObject::OBJ && tensor_json.exist("shape"))
    {
        deserialize_tensor_shape(tensor_json["shape"], shape, layout);
    }

    Tensor::Dtype dtype = Tensor::UINT8;
    if (tensor_json.t == utils::JsonObject::OBJ && tensor_json.exist("dtype"))
    {
        deserialize_tensor_dtype(tensor_json["dtype"], dtype);
    }

    return std::make_shared<Tensor>(name, shape, dtype);
}

int restore_global_tensors_from_metadata(GraphPtr graph,
                                         utils::JsonObject graph_json,
                                         const utils::JsonObject *object_table,
                                         MqIrCacheContext *context = nullptr)
{
    if (!graph || graph_json.t != utils::JsonObject::OBJ ||
        !graph_json.exist("global_tensors") ||
        graph_json["global_tensors"].t != utils::JsonObject::OBJ)
    {
        return 0;
    }

    std::map<std::string, TensorPtr> restored_globals;
    for (const auto &kv : graph_json["global_tensors"].obj())
    {
        TensorPtr tensor = nullptr;
        auto existing = graph->global_tensors().find(kv.first);
        if (existing != graph->global_tensors().end())
        {
            tensor = existing->second;
        }
        if (!tensor)
        {
            tensor = create_tensor_from_metadata(kv.first, kv.second);
        }
        restored_globals[kv.first] = tensor;
    }

    graph->global_tensors() = restored_globals;

    int loaded_tensor_attrs = 0;
    for (const auto &kv : graph_json["global_tensors"].obj())
    {
        auto tensor_it = graph->global_tensors().find(kv.first);
        if (tensor_it != graph->global_tensors().end())
        {
            loaded_tensor_attrs += restore_tensor_metadata(tensor_it->second, kv.second, graph, object_table, context);
        }
    }

    INFO() << "mq_load_ir: restored " << graph->global_tensors().size()
           << " global tensors from attributes";
    return loaded_tensor_attrs;
}

utils::JsonObject serialize_graph_metadata(GraphPtr graph, int &graph_attr_count, int &node_attr_count, int &tensor_attr_count, PointerRefTable &ref_table);

utils::JsonObject serialize_subgraph_metadata(GraphPtr graph, int &graph_attr_count, int &node_attr_count, int &tensor_attr_count, PointerRefTable &ref_table)
{
    return serialize_graph_metadata(graph, graph_attr_count, node_attr_count, tensor_attr_count, ref_table);
}

void add_subgraph_metadata(utils::JsonObject &node_obj, NodePtr node, int &graph_attr_count, int &node_attr_count, int &tensor_attr_count, PointerRefTable &ref_table)
{
    auto sgn = SubGraphNode::from_node(node);
    if (!sgn)
    {
        return;
    }
    node_obj["subgraph"] = serialize_subgraph_metadata(sgn->subgraph(), graph_attr_count, node_attr_count, tensor_attr_count, ref_table);
}

utils::JsonObject serialize_graph_metadata(GraphPtr graph, int &graph_attr_count, int &node_attr_count, int &tensor_attr_count, PointerRefTable &ref_table)
{
    utils::JsonObject graph_obj(utils::JsonObject::OBJ);
    utils::JsonObject nodes_arr(utils::JsonObject::ARR);

    graph_obj["graph_name"] = graph->name();
    graph_obj["layout"] = serialize_layout(graph->layout());
    utils::JsonObject graph_attrs(utils::JsonObject::OBJ);
    int count = serialize_attrs_to_json(graph->attrs(), graph_attrs, &ref_table);
    if (count > 0)
    {
        graph_obj["graph_attrs"] = graph_attrs;
        graph_attr_count += count;
    }

    for (auto node : graph->nodes())
    {
        auto node_obj = serialize_node_metadata(node, node_attr_count, tensor_attr_count, &ref_table);
        add_subgraph_metadata(node_obj, node, graph_attr_count, node_attr_count, tensor_attr_count, ref_table);
        if (has_json_object_field(node_obj, "node_attrs") ||
            has_json_object_field(node_obj, "output_tensors") ||
            has_json_object_field(node_obj, "constant_tensors") ||
            has_json_object_field(node_obj, "placeholder_tensors") ||
            has_json_object_field(node_obj, "subgraph"))
        {
            nodes_arr.push_back(node_obj);
        }
    }

    graph_obj["nodes"] = nodes_arr;
    graph_obj["graph_references"] = serialize_graph_references(graph, ref_table);
    return graph_obj;
}

void restore_graph_metadata(GraphPtr graph, utils::JsonObject graph_json, const utils::JsonObject *object_table = nullptr, MqIrCacheContext *context = nullptr);
void restore_node_metadata(GraphPtr graph, NodePtr node, utils::JsonObject node_json, int &loaded_node_attrs, int &loaded_tensor_attrs, const utils::JsonObject *object_table = nullptr, MqIrCacheContext *context = nullptr);

NodePtr fold_subgraph_from_metadata(GraphPtr graph,
                                    const std::string &node_name,
                                    utils::JsonObject node_json,
                                    std::map<std::string, NodePtr> &node_map)
{
    if (!graph || node_json.t != utils::JsonObject::OBJ || !node_json.exist("subgraph"))
    {
        return nullptr;
    }

    auto subgraph_json = node_json["subgraph"];
    if (subgraph_json.t != utils::JsonObject::OBJ || !subgraph_json.exist("nodes"))
    {
        return nullptr;
    }

    std::vector<NodePtr> subgraph_nodes;
    for (auto &sub_node_json : subgraph_json["nodes"].array())
    {
        if (sub_node_json.t != utils::JsonObject::OBJ || !sub_node_json.exist("node_name"))
        {
            continue;
        }

        std::string sub_node_name = (std::string)sub_node_json["node_name"];
        NodePtr sub_node = nullptr;
        auto it = node_map.find(sub_node_name);
        if (it != node_map.end())
        {
            sub_node = it->second;
        }
        else if (sub_node_json.exist("subgraph"))
        {
            sub_node = fold_subgraph_from_metadata(graph, sub_node_name, sub_node_json, node_map);
        }

        if (!sub_node)
        {
            WARN() << "Subgraph internal node not found in graph: " << sub_node_name;
            return nullptr;
        }
        subgraph_nodes.push_back(sub_node);
    }

    if (subgraph_nodes.empty())
    {
        return nullptr;
    }

    auto sg = SubGraph::create(graph, subgraph_nodes);
    if (subgraph_json.exist("graph_name"))
    {
        sg->set_name((std::string)subgraph_json["graph_name"]);
    }
    if (subgraph_json.exist("layout"))
    {
        DataLayout layout = DataLayout_UnSupport;
        if (deserialize_layout(subgraph_json["layout"], layout))
        {
            sg->set_layout(layout);
        }
    }

    auto sgn = sg->as_node(true, true);
    for (auto node : subgraph_nodes)
    {
        node_map.erase(node->name());
    }
    sgn->set_name(node_name);
    node_map[node_name] = sgn;
    return sgn;
}

NodePtr replace_node_with_subgraph(GraphPtr graph,
                                   NodePtr node,
                                   utils::JsonObject subgraph_json,
                                   const utils::JsonObject *object_table = nullptr,
                                   MqIrCacheContext *context = nullptr)
{
    if (!graph || !node || subgraph_json.t != utils::JsonObject::OBJ)
    {
        return node;
    }

    auto sg = SubGraph::create(graph);
    if (subgraph_json.exist("graph_name"))
    {
        sg->set_name((std::string)subgraph_json["graph_name"]);
    }
    if (subgraph_json.exist("layout"))
    {
        DataLayout layout = DataLayout_UnSupport;
        if (deserialize_layout(subgraph_json["layout"], layout))
        {
            sg->set_layout(layout);
        }
    }

    auto sub_node = SubGraphNode::create(sg, node->name());
    sub_node->params() = node->params();
    for (auto &kv : node->attrs())
    {
        sub_node->set_attr(kv.first, kv.second);
    }
    for (auto &kv : node->constants())
    {
        sub_node->constants()[kv.first] = kv.second;
    }
    for (auto t : node->placeholders())
    {
        sub_node->placeholders().push_back(t);
    }

    graph->replace_node_by(node, sub_node);
    restore_graph_metadata(sg, subgraph_json, object_table, context);
    return sub_node;
}

void restore_graph_metadata(GraphPtr graph, utils::JsonObject graph_json, const utils::JsonObject *object_table, MqIrCacheContext *context)
{
    if (!graph || graph_json.t != utils::JsonObject::OBJ)
    {
        return;
    }

    if (graph_json.exist("layout"))
    {
        DataLayout layout = DataLayout_UnSupport;
        if (deserialize_layout(graph_json["layout"], layout))
        {
            graph->set_layout(layout);
        }
    }
    restore_global_tensors_from_metadata(graph, graph_json, object_table, context);
    if (!graph_json.exist("nodes"))
    {
        if (graph_json.exist("graph_attrs"))
        {
            deserialize_attrs_from_json(graph_json["graph_attrs"],
                                        [&graph](const std::string &key, const Any &value) {
                                            graph->set_attr(key, value);
                                        },
                                        graph,
                                        object_table,
                                        context);
        }
        restore_graph_references(graph, graph_json, object_table);
        return;
    }

    std::map<std::string, NodePtr> node_map;
    for (auto node : graph->nodes())
    {
        node_map[node->name()] = node;
    }

    auto nodes_arr = graph_json["nodes"].array();
    int loaded_node_attrs = 0;
    int loaded_tensor_attrs = 0;
    for (auto &node_json : nodes_arr)
    {
        if (node_json.t != utils::JsonObject::OBJ || !node_json.exist("node_name"))
        {
            continue;
        }
        std::string node_name = (std::string)node_json["node_name"];
        auto it = node_map.find(node_name);
        NodePtr node = nullptr;
        if (it == node_map.end())
        {
            node = fold_subgraph_from_metadata(graph, node_name, node_json, node_map);
            if (!node)
            {
                WARN() << "Node not found in graph: " << node_name;
                continue;
            }
        }
        else
        {
            node = it->second;
        }
        restore_node_metadata(graph, node, node_json, loaded_node_attrs, loaded_tensor_attrs, object_table, context);
    }

    if (graph_json.exist("graph_attrs"))
    {
        deserialize_attrs_from_json(graph_json["graph_attrs"],
                                    [&graph](const std::string &key, const Any &value) {
                                        graph->set_attr(key, value);
                                    },
                                    graph,
                                    object_table,
                                    context);
    }
    restore_graph_references(graph, graph_json, object_table);
}

void restore_node_metadata(GraphPtr graph, NodePtr node, utils::JsonObject node_json, int &loaded_node_attrs, int &loaded_tensor_attrs, const utils::JsonObject *object_table, MqIrCacheContext *context)
{
    if (!node)
    {
        return;
    }

    if (node_json.exist("node_attrs"))
    {
        loaded_node_attrs += deserialize_attrs_from_json(node_json["node_attrs"],
            [&node](const std::string& key, const Any& value) {
                node->set_attr(key, value);
            }, graph, object_table, context);
    }

    if (node_json.exist("output_tensors"))
    {
        auto tensors_arr = node_json["output_tensors"].array();
        for (auto& tensor_json : tensors_arr)
        {
            loaded_tensor_attrs += restore_tensor_metadata(find_output_tensor(node, tensor_json), tensor_json, graph, object_table, context);
        }
    }

    if (node_json.exist("constant_tensors"))
    {
        auto tensors_arr = node_json["constant_tensors"].array();
        for (auto& tensor_json : tensors_arr)
        {
            loaded_tensor_attrs += restore_tensor_metadata(find_constant_tensor(node, tensor_json), tensor_json, graph, object_table, context);
        }
    }

    if (node_json.exist("placeholder_tensors"))
    {
        auto tensors_arr = node_json["placeholder_tensors"].array();
        for (auto& tensor_json : tensors_arr)
        {
            loaded_tensor_attrs += restore_tensor_metadata(find_placeholder_tensor(node, tensor_json), tensor_json, graph, object_table, context);
        }
    }

    if (node_json.exist("subgraph"))
    {
        auto sgn = SubGraphNode::from_node(node);
        if (sgn)
        {
            restore_graph_metadata(sgn->subgraph(), node_json["subgraph"], object_table, context);
        }
        else
        {
            replace_node_with_subgraph(graph, node, node_json["subgraph"], object_table, context);
        }
    }
}

std::string path_basename(const std::string &path)
{
    auto pos = path.find_last_of("/\\");
    if (pos == std::string::npos)
        return path;
    return path.substr(pos + 1);
}

// Save graph, node and tensor metadata that the textual IR does not fully represent.
bool save_graph_attributes(GraphPtr graph, const std::string& json_path, const std::string &payload_bin_path)
{
    utils::JsonObject root(utils::JsonObject::OBJ);
    utils::JsonObject graph_obj(utils::JsonObject::OBJ);
    utils::JsonObject nodes_arr(utils::JsonObject::ARR);

    int total_graph_attrs = 0;
    int total_node_attrs = 0;
    int total_tensor_attrs = 0;
    PointerRefTable ref_table(graph);

    graph_obj["graph_name"] = graph->name();
    graph_obj["layout"] = serialize_layout(graph->layout());
    utils::JsonObject graph_attrs(utils::JsonObject::OBJ);
    int root_graph_attrs = serialize_attrs_to_json(graph->attrs(), graph_attrs, &ref_table);
    if (root_graph_attrs > 0)
    {
        graph_obj["graph_attrs"] = graph_attrs;
        total_graph_attrs += root_graph_attrs;
    }
    graph_obj["graph_references"] = serialize_graph_references(graph, ref_table);
    graph_obj["global_tensors"] = serialize_global_tensors(graph, total_tensor_attrs, ref_table);
    root["graph"] = graph_obj;

    for (auto node : graph->nodes())
    {
        auto node_obj = serialize_node_metadata(node, total_node_attrs, total_tensor_attrs, &ref_table);
        add_subgraph_metadata(node_obj, node, total_graph_attrs, total_node_attrs, total_tensor_attrs, ref_table);
        if (has_json_object_field(node_obj, "node_attrs") ||
            has_json_object_field(node_obj, "output_tensors") ||
            has_json_object_field(node_obj, "constant_tensors") ||
            has_json_object_field(node_obj, "placeholder_tensors") ||
            has_json_object_field(node_obj, "subgraph"))
        {
            nodes_arr.push_back(node_obj);
        }
    }

    root["nodes"] = nodes_arr;
    root["object_table"] = ref_table.to_json();
    auto object_payloads = ref_table.serialize_payloads(payload_bin_path, path_basename(payload_bin_path));
    if (object_payloads.t == utils::JsonObject::OBJ && !object_payloads.obj().empty())
    {
        root["object_payloads"] = object_payloads;
    }
    root["total_nodes"] = (int)nodes_arr.array().size();
    root["total_graph_attrs"] = total_graph_attrs;
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

    INFO() << "Saved " << total_graph_attrs << " graph attributes, "
           << total_node_attrs << " node attributes and "
           << total_tensor_attrs << " tensor attributes to " << json_path;

    return true;
}

// Load node and tensor attributes from a JSON file
bool load_graph_attributes(GraphPtr graph, const std::string& json_path, const std::string &payload_bin_path)
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

    MqIrCacheContext context;
    const utils::JsonObject *object_table = nullptr;
    utils::JsonObject object_table_obj;
    if (root.exist("object_table") && root["object_table"].t == utils::JsonObject::OBJ)
    {
        object_table_obj = root["object_table"];
        object_table = &object_table_obj;
    }

    if (root.exist("graph"))
    {
        auto graph_json = root["graph"];
        if (graph_json.t == utils::JsonObject::OBJ)
        {
            restore_global_tensors_from_metadata(graph, graph_json, object_table, &context);
            if (graph_json.exist("layout"))
            {
                DataLayout layout = DataLayout_UnSupport;
                if (deserialize_layout(graph_json["layout"], layout))
                {
                    graph->set_layout(layout);
                }
            }
        }
    }

    auto nodes_arr = root["nodes"].array();
    int loaded_node_attrs = 0;
    int loaded_tensor_attrs = 0;

    // Create a map from node name to node pointer for fast lookup.
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

        std::string node_name = (std::string)node_json["node_name"];
        auto it = node_map.find(node_name);
        NodePtr node = nullptr;
        if (it == node_map.end())
        {
            node = fold_subgraph_from_metadata(graph, node_name, node_json, node_map);
            if (!node)
            {
                WARN() << "Node not found in graph: " << node_name;
                continue;
            }
        }
        else
        {
            node = it->second;
        }

        restore_node_metadata(graph, node, node_json, loaded_node_attrs, loaded_tensor_attrs, object_table, &context);
    }

    if (root.exist("graph") && root["graph"].t == utils::JsonObject::OBJ)
    {
        if (root["graph"].exist("graph_attrs"))
        {
            deserialize_attrs_from_json(root["graph"]["graph_attrs"],
                                        [&graph](const std::string &key, const Any &value) {
                                            graph->set_attr(key, value);
                                        },
                                        graph,
                                        object_table,
                                        &context);
        }
        restore_graph_references(graph, root["graph"], object_table);
    }

    if (object_table && root.exist("object_payloads") && root["object_payloads"].t == utils::JsonObject::OBJ)
    {
        restore_object_payloads(graph, *object_table, root["object_payloads"], payload_bin_path);
    }

    INFO() << "Loaded " << loaded_node_attrs << " node attributes and "
           << loaded_tensor_attrs << " tensor attributes from " << json_path;

    return true;
}

bool load_graph_global_tensors(GraphPtr graph, const std::string& json_path)
{
    if (!graph)
        return false;

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

    if (root.t != utils::JsonObject::OBJ || !root.exist("graph") || root["graph"].t != utils::JsonObject::OBJ)
    {
        ERROR() << "Invalid attributes JSON format";
        return false;
    }

    const utils::JsonObject *object_table = nullptr;
    utils::JsonObject object_table_obj;
    if (root.exist("object_table") && root["object_table"].t == utils::JsonObject::OBJ)
    {
        object_table_obj = root["object_table"];
        object_table = &object_table_obj;
    }

    restore_global_tensors_from_metadata(graph, root["graph"], object_table);
    return true;
}


// File names used inside the cache directory pointed at by --mq_save_ir / --mq_load_ir.
std::string mq_ir_cache_stage_label(const std::string &stage)
{
    if (stage.empty())
        return "optimize";
    return stage;
}

inline std::string mq_ir_cache_def_name(const std::string &stage)
{
    return mq_ir_cache_stage_label(stage) + ".def";
}

inline std::string mq_ir_cache_bin_name(const std::string &stage)
{
    return mq_ir_cache_stage_label(stage) + ".bin";
}

inline std::string mq_ir_cache_attrs_name(const std::string &stage)
{
    return mq_ir_cache_stage_label(stage) + ".attrs.json";
}

inline std::string mq_ir_cache_payload_bin_name(const std::string &stage)
{
    return mq_ir_cache_stage_label(stage) + ".payload.bin";
}

inline std::string mq_ir_cache_elf_name(const std::string &stage)
{
    return mq_ir_cache_stage_label(stage) + ".elf";
}

inline std::string mq_ir_cache_def_path(const std::string &dir, const std::string &stage)
{
    return dir + "/" + mq_ir_cache_def_name(stage);
}

inline std::string mq_ir_cache_bin_path(const std::string &dir, const std::string &stage)
{
    return dir + "/" + mq_ir_cache_bin_name(stage);
}

inline std::string mq_ir_cache_attrs_path(const std::string &dir, const std::string &stage)
{
    return dir + "/" + mq_ir_cache_attrs_name(stage);
}

inline std::string mq_ir_cache_payload_bin_path(const std::string &dir, const std::string &stage)
{
    return dir + "/" + mq_ir_cache_payload_bin_name(stage);
}

inline std::string mq_ir_cache_elf_path(const std::string &dir, const std::string &stage)
{
    return dir + "/" + mq_ir_cache_elf_name(stage);
}

} // namespace

// Try to load a previously saved post-lowering graph from `dir`
GraphPtr load_mq_ir_cache(const std::string &dir, const std::string &stage)
{
    auto def_path = mq_ir_cache_def_path(dir, stage);
    auto bin_path = mq_ir_cache_bin_path(dir, stage);
    auto attrs_path = mq_ir_cache_attrs_path(dir, stage);

    if (!utils::file_exists(def_path) || !utils::file_exists(bin_path))
    {
        WARN() << "mq_load_ir: cache files not found in " << dir << " (expected "
               << mq_ir_cache_def_name(stage) << " and " << mq_ir_cache_bin_name(stage) << ").";
        return nullptr;
    }

    INFO() << "mq_load_ir: loading cached MQ IR from " << dir << " at stage " << mq_ir_cache_stage_label(stage);
    auto g = Graph::parse(def_path);
    if (!g)
    {
        ERROR() << "mq_load_ir: failed to parse cached IR " << def_path;
        return nullptr;
    }

    // Rebuild runtime-only GOP plugin metadata before folding subgraphs back from JSON.
    // plugin_impl is a C++ object pointer and is intentionally not serialized.
    if (utils::file_exists(attrs_path))
    {
        load_graph_global_tensors(g, attrs_path);
    }
    apply_gopplugins(g);

    return g;
}

bool load_mq_ir_cache_attributes(GraphPtr graph, const std::string &dir, const std::string &stage)
{
    auto attrs_path = mq_ir_cache_attrs_path(dir, stage);
    auto payload_bin_path = mq_ir_cache_payload_bin_path(dir, stage);
    if (!utils::file_exists(attrs_path))
    {
        DEBUG() << "mq_load_ir: no attributes file found, skipping attribute loading";
        return true;
    }

    INFO() << "mq_load_ir: loading cached attributes from " << attrs_path;
    bool loaded = load_graph_attributes(graph, attrs_path, payload_bin_path);
    dump_graph_io_tensor_ptrs(graph, "mq_load_ir: after reloading attributes");
    return loaded;
}

std::shared_ptr<HybridELF> load_mq_ir_cache_elf(const std::string &dir, const std::string &stage)
{
    auto elf_path = mq_ir_cache_elf_path(dir, stage);
    if (!utils::file_exists(elf_path))
    {
        ERROR() << "mq_load_ir: cached HybridELF not found: " << elf_path;
        return nullptr;
    }

    std::ifstream ifs(elf_path, std::ios::binary);
    if (!ifs.is_open())
    {
        ERROR() << "mq_load_ir: failed to open cached HybridELF: " << elf_path;
        return nullptr;
    }

    auto elf = std::make_shared<HybridELF>();
    if (!elf->load(ifs))
    {
        ERROR() << "mq_load_ir: failed to load cached HybridELF: " << elf_path;
        return nullptr;
    }

    if (!elf->get_cpself())
    {
        ERROR() << "mq_load_ir: cached HybridELF has no cpself note: " << elf_path;
        return nullptr;
    }

    INFO() << "mq_load_ir: loaded cached HybridELF from " << elf_path;
    return elf;
}

// Serialize the current graph into `dir`
void save_mq_ir_cache(GraphPtr graph, const std::string &dir, const std::string &stage)
{
    if (!utils::path_exists(dir) && !utils::mkdir(dir))
    {
        ERROR() << "mq_save_ir: cannot create cache directory " << dir;
        return;
    }

    auto abs_dir = utils::get_abspath(dir);
    auto def_path = mq_ir_cache_def_path(abs_dir, stage);
    auto bin_path = mq_ir_cache_bin_path(abs_dir, stage);
    auto attrs_path = mq_ir_cache_attrs_path(abs_dir, stage);
    auto payload_bin_path = mq_ir_cache_payload_bin_path(abs_dir, stage);

    INFO() << "mq_save_ir: dumping MQ IR at stage " << mq_ir_cache_stage_label(stage) << " to " << def_path;
    {
        BoolConfigGuard serialize_scale_zp(Constants::serialize_scale_zp, true);
        BoolConfigGuard dump_debug_ir(Constants::dump_debug_IR, true);
        graph->serialize(def_path, bin_path);
    }

    // Save node and tensor attributes to separate JSON file
    INFO() << "mq_save_ir: dumping attributes to " << attrs_path;
    if (!save_graph_attributes(graph, attrs_path, payload_bin_path))
    {
        WARN() << "mq_save_ir: failed to save attributes file";
    }
}

bool save_mq_ir_cache_elf(const std::shared_ptr<HybridELF> &elf,
                          const std::string &dir,
                          const std::string &stage)
{
    if (!elf)
    {
        ERROR() << "mq_save_ir: cannot save a null HybridELF";
        return false;
    }

    if (!utils::path_exists(dir) && !utils::mkdir(dir))
    {
        ERROR() << "mq_save_ir: cannot create cache directory " << dir;
        return false;
    }

    auto abs_dir = utils::get_abspath(dir);
    auto elf_path = mq_ir_cache_elf_path(abs_dir, stage);
    std::ofstream ofs(elf_path, std::ios::binary);
    if (!ofs.is_open())
    {
        ERROR() << "mq_save_ir: failed to open HybridELF cache file: " << elf_path;
        return false;
    }

    if (!elf->save(ofs))
    {
        ERROR() << "mq_save_ir: failed to save HybridELF to " << elf_path;
        return false;
    }

    INFO() << "mq_save_ir: saved HybridELF to " << elf_path;
    return true;
}

} // namespace aipubt

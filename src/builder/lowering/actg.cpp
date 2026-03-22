#include "builder/lowering/actg.h"
#include "builder/lowering/lowering.h"
#include "builder/plugin.h"
#include "core/config.h"
#include "core/constants.h"
#include "core/graph.h"
#include "core/graph_matcher.h"
#include "core/logger.h"
#include "core/node.h"
#include "core/plugin_loader.h"
#include "core/proxy_tensor.h"
#include "core/subgraph.h"
#include <algorithm>
#include <builder/target.h>
#include <set>
#include <vector>
namespace aipubt
{
static bool eq_node_type(NodePtr node, OpType optype)
{
    if (!SubGraphNode::is_subgraph_node(node))
        return node->type() == optype;
    SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(node);
    auto nodes = sgn->get_real_node();
    return nodes.size() == 1 && nodes[0]->type() == optype;
}

static bool neq_node_type(NodePtr node, OpType optype)
{
    return !eq_node_type(node, optype);
}

static std::map<std::string, NodeParamValue> &to_params(NodePtr node)
{
    if (!SubGraphNode::is_subgraph_node(node))
        return node->params();
    SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(node);
    auto nodes = sgn->get_real_node();
    aipu_assert(nodes.size() == 1);
    return nodes[0]->params();
}

static void unset_plugin(NodePtr node)
{
    if (SubGraphNode::is_subgraph_node(node))
    {
        SubGraphNodePtr sgn = std::static_pointer_cast<SubGraphNode>(node);
        if (sgn->attrs().count(Constants::plugin_attrs_name))
        {
            sgn->erase_attr(Constants::plugin_attrs_name);
            sgn->erase_attr("_match_list");
        }
    }
}

static bool check_valid_layout(LayoutList &layout, std::vector<int32_t> &channel, std::vector<int32_t> &c_base)
{
    LayoutList valid_layout = {DataLayout_NCHWC32, DataLayout_NCHWC16,
                               DataLayout_NCHWC32T8, DataLayout_NCHWC16T8,
                               DataLayout_NCHWC32T1, DataLayout_NCHWC16T1};

    bool result = true;
    for (auto i = 0u; i < layout.size(); i++)
    {
        auto &l = layout[i];
        if (std::find(valid_layout.begin(), valid_layout.end(), l) == valid_layout.end())
        {
            if (l == DataLayout_NHWC && channel <= c_base)
            {
                continue;
            }
            result = false;
            break;
        }
    }
    return result;
}

static LayoutList expand_layout_for_actg(LayoutList &layout_list, uint32_t total_num, std::vector<std::vector<int32_t>> regions)
{
    aipu_assert(layout_list.size() == regions.size());

    LayoutList new_layout;
    new_layout.resize(total_num);
    for (uint32_t i = 0; i < regions.size(); i++)
    {
        auto layout = layout_list[i];
        for (auto id : regions[i])
        {
            if (id != -1)
                new_layout[id] = layout;
        }
    }
    return new_layout;
}

static std::vector<int32_t> get_origin_channel(TensorPtrList tensors, std::vector<std::vector<int32_t>> regions, std::vector<int32_t> &c_base)
{
    std::map<uint32_t, int32_t> byte_to_c_base = {{1, 32}, {2, 16}, {4, 8}};
    std::vector<int32_t> channels;
    for (uint32_t i = 0; i < regions.size(); i++)
    {
        auto id = regions[i][0];
        channels.push_back(tensors[id]->shape().c());
        uint32_t byte = tensors[id]->dtype_size(tensors[id]->type());
        c_base.push_back(byte_to_c_base[byte]);
    }
    return channels;
}

static void change_to_actg_layout(SubGraphNodePtr sgnode)
{
    LayoutCombinations input_layouts = sgnode->get_attr(Constants::valid_input_layouts);
    LayoutCombinations output_layouts = sgnode->get_attr(Constants::valid_output_layouts);
    std::vector<int32_t> actg_enables = sgnode->get_attr("actg_enables");
    std::vector<std::vector<int32_t>> actg_regions = sgnode->get_attr("actg_regions");
    std::vector<int32_t> wrb_crop_enables = sgnode->get_attr("wrb_crop_enables");
    std::vector<std::vector<int32_t>> wrb_crop_regions = sgnode->get_attr("wrb_crop_regions");

    if (std::find(actg_enables.begin(), actg_enables.end(), 1) != actg_enables.end())
    {
        std::vector<int32_t> c_base;
        auto in_channel = get_origin_channel(sgnode->inputs(), actg_regions, c_base);

        LayoutCombinations new_input_layouts;
        for (auto &layout_list : input_layouts)
        {
            if (check_valid_layout(layout_list, in_channel, c_base))
            {
                auto tmp = expand_layout_for_actg(layout_list, sgnode->inputs().size(), actg_regions);
                new_input_layouts.push_back(tmp);
            }
        }

        if (new_input_layouts.size() == 0)
        {
            BuilderOpPluginInfo *plugin = sgnode->get_attr(Constants::plugin_attrs_name);
            ERROR() << "Plugin(" << plugin->class_name << ") not support actg input layout";
        }

        sgnode->set_attr(Constants::valid_input_layouts, new_input_layouts);
    }

    if (std::find(wrb_crop_enables.begin(), wrb_crop_enables.end(), 1) != wrb_crop_enables.end())
    {
        LayoutCombinations new_output_layouts;
        for (auto &layout_list : output_layouts)
        {
            std::vector<int32_t> c_base;
            auto out_channel = get_origin_channel(sgnode->outputs(), actg_regions, c_base);
            if (check_valid_layout(layout_list, out_channel, c_base))
            {
                auto tmp = expand_layout_for_actg(layout_list, sgnode->outputs().size(), wrb_crop_regions);
                new_output_layouts.push_back(tmp);
            }
        }

        if (new_output_layouts.size() == 0)
        {
            BuilderOpPluginInfo *plugin = sgnode->get_attr(Constants::plugin_attrs_name);
            ERROR() << "Plugin(" << plugin->class_name << ") not support wrb_crop output Layout";
        }

        sgnode->set_attr(Constants::valid_output_layouts, new_output_layouts);
    }
}

static bool get_actg_concat_nodes(NodePtr node, std::vector<NodePtr> &actg_concat_list, std::vector<int32_t> &actg_enables)
{
    bool change = false;

    auto input_size = node->inputs().size();
    actg_enables = std::vector<int32_t>(input_size, 0);
    actg_concat_list = std::vector<NodePtr>(input_size, nullptr);

    std::vector<int32_t> actg_supports = node->get_attr("actg_supports");
    if (actg_supports.size() == 0)
        actg_supports = std::vector<int32_t>(input_size, 0);

    for (auto idx = 0u; idx < input_size; idx++)
    {
        auto input = node->inputs()[idx];
        auto actg_node = input->node();
        if (actg_node == nullptr || (neq_node_type(actg_node, OpType_Concat) && neq_node_type(actg_node, OpType_InternalConcatenated)))
            continue;
        int32_t actg_input_size = actg_node->inputs().size();
        int32_t axis = to_params(actg_node)["axis"];
        int32_t output_size = actg_node->outputs().size();
        int32_t child_size = actg_node->children().size();
        actg_enables[idx] = int32_t(actg_supports[idx] && axis == 1 && actg_input_size == 2 && output_size == 1 && child_size == 1);
        if (actg_enables[idx])
        {
            actg_concat_list[idx] = actg_node;
            change = true;
        }
    }

    return change;
}

static bool get_wrb_crop_nodes(NodePtr node, std::vector<NodePtrList> &wrb_crop_nodes_list, std::vector<int32_t> &wrb_crop_enables)
{
    bool change = false;
    auto output_size = node->outputs().size();
    wrb_crop_enables = std::vector<int32_t>(output_size, 0);
    wrb_crop_nodes_list = std::vector<NodePtrList>(output_size, NodePtrList());

    std::vector<int32_t> wrb_crop_supports = node->get_attr("wrb_crop_supports");
    if (wrb_crop_supports.size() == 0)
        wrb_crop_supports = std::vector<int32_t>(output_size, 0);

    for (auto idx = 0u; idx < output_size; idx++)
    {
        auto output = node->outputs()[idx];
        auto tmp_nodes = output->consumers();
        std::vector<NodePtr> wrb_crop_nodes;
        wrb_crop_enables[idx] = wrb_crop_supports[idx];
        bool require = false;
        for (auto &wrb_crop_node : tmp_nodes)
        {
            if (wrb_crop_node != nullptr && (eq_node_type(wrb_crop_node, OpType_Crop) || eq_node_type(wrb_crop_node, OpType_InternalCrop)))
            {
                wrb_crop_nodes.push_back(wrb_crop_node);
                require = true;
            }
        }
        wrb_crop_enables[idx] = wrb_crop_enables[idx] && require && wrb_crop_nodes.size() <= 2 && wrb_crop_nodes.size() > 0;
        if (wrb_crop_enables[idx])
        {
            wrb_crop_nodes_list[idx] = wrb_crop_nodes;
            change = true;
        }
    }

    return change;
}

static std::vector<std::vector<std::string>> get_actg_region_names(NodePtr node, std::vector<NodePtr> &actg_concat_list)
{
    auto mid_dim = 0; // for actg_region
    auto top_dim = 1;
    auto left_dim = 2;
    auto bottom_dim = 3;
    auto right_dim = 4;

    auto top_idx = 0; // for agtg_node input
    auto mid_idx = 1;
    auto bottom_idx = 2;
    auto left_idx = 3;
    auto right_idx = 4;
    auto input_idx_to_dim = std::vector<int32_t>(5, -1);
    input_idx_to_dim[mid_idx] = mid_dim;
    input_idx_to_dim[top_idx] = top_dim;
    input_idx_to_dim[bottom_idx] = bottom_dim;
    input_idx_to_dim[left_idx] = left_dim;
    input_idx_to_dim[right_idx] = right_dim;

    std::vector<std::vector<std::string>> actg_region_names;
    for (auto idx = 0u; idx < actg_concat_list.size(); idx++)
    {

        auto tmp = std::vector<std::string>(5, "");
        if (actg_concat_list[idx] == nullptr)
        {
            tmp[mid_dim] = node->inputs()[idx]->name();
        }
        else
        {
            auto actg_node = actg_concat_list[idx];
            for (auto j = 0; j < (int32_t)actg_node->inputs().size(); j++)
            {
                tmp[input_idx_to_dim[j]] = actg_node->inputs()[j]->name();
            }
        }
        actg_region_names.push_back(tmp);
    }

    return actg_region_names;
}

static std::vector<std::vector<std::string>> get_wrb_crop_region_names(NodePtr node, std::vector<NodePtrList> &wrb_crop_nodes_list,
                                                                       std::map<std::string, CropInfo> &name_to_crop_info)
{
    std::vector<std::vector<std::string>> wrb_crop_region_names;
    for (auto idx = 0u; idx < wrb_crop_nodes_list.size(); idx++)
    {

        auto tmp = std::vector<std::string>(3, "");
        tmp[0] = node->outputs()[idx]->name();
        CropInfo crop_info;
        auto out_shape = std::vector<int32_t>(node->outputs()[idx]->shape());
        crop_info.start_point = std::vector<int32_t>(out_shape.size(), 0);
        crop_info.length = out_shape;
        crop_info.origin_length = out_shape;
        name_to_crop_info[node->outputs()[idx]->name()] = crop_info;
        if (wrb_crop_nodes_list[idx].size() != 0)
        {
            auto wrb_crop_nodes = wrb_crop_nodes_list[idx];
            for (auto j = 0; j < (int32_t)wrb_crop_nodes.size(); j++)
            {
                tmp[j + 1] = wrb_crop_nodes[j]->outputs()[0]->name();
                auto in_shape = std::vector<int32_t>(wrb_crop_nodes[j]->inputs()[0]->shape());
                auto out_shape = std::vector<int32_t>(wrb_crop_nodes[j]->outputs()[0]->shape());
                std::vector<int32_t> start_point;
                std::vector<std::vector<int32_t>> crops = to_params(wrb_crop_nodes[j])["crops"];
                for (auto tmp : crops)
                {
                    start_point.push_back(tmp[0]);
                }
                CropInfo crop_info;
                crop_info.start_point = start_point;
                crop_info.length = out_shape;
                crop_info.origin_length = in_shape;

                name_to_crop_info[wrb_crop_nodes[j]->outputs()[0]->name()] = crop_info;
            }
        }
        wrb_crop_region_names.push_back(tmp);
    }

    return wrb_crop_region_names;
}

static std::vector<std::vector<int32_t>> get_actg_regions(SubGraphNodePtr new_node, std::vector<std::vector<std::string>> &actg_region_names)
{
    std::map<std::string, int32_t> name_to_idx;
    for (auto idx = 0u; idx < new_node->inputs().size(); idx++)
    {
        name_to_idx[new_node->inputs()[idx]->name()] = idx;
    }

    std::vector<std::vector<int32_t>> actg_regions;
    for (auto &regions_name : actg_region_names)
    {
        auto region_idx = std::vector<int32_t>(regions_name.size(), -1);
        for (auto i = 0u; i < regions_name.size(); i++)
        {
            auto name = regions_name[i];
            if (name.size() != 0)
                region_idx[i] = name_to_idx[name];
        }
        actg_regions.push_back(region_idx);
    }

    return actg_regions;
}

static std::vector<std::vector<int32_t>> get_wrb_crop_regions(SubGraphNodePtr new_node, std::vector<std::vector<std::string>> &wrb_crop_region_names,
                                                              std::vector<CropInfo> &output_crops, std::map<std::string, CropInfo> &name_to_crop_info)
{
    std::map<std::string, int32_t> name_to_idx;
    for (auto idx = 0u; idx < new_node->outputs().size(); idx++)
    {
        name_to_idx[new_node->outputs()[idx]->name()] = idx;
        output_crops.push_back(name_to_crop_info[new_node->outputs()[idx]->name()]);
    }

    std::vector<std::vector<int32_t>> wrb_crop_regions;
    for (auto &regions_name : wrb_crop_region_names)
    {
        auto region_idx = std::vector<int32_t>(regions_name.size(), -1);
        for (auto i = 0u; i < regions_name.size(); i++)
        {
            auto name = regions_name[i];
            if (name.size() != 0)
                region_idx[i] = name_to_idx[name];
        }
        wrb_crop_regions.push_back(region_idx);
    }

    return wrb_crop_regions;
}

static void rename_constants(SubGraphNodePtr node, NodePtrList m_nodes)
{
    for (auto sn : m_nodes)
    {
        auto &constants = node->constants();
        if (SubGraphNode::is_subgraph_node(sn))
        {
            for (auto cts : sn->constants())
            {
                auto target_name = cts.first;
                auto replace_name = sn->name() + "/" + target_name;
                aipu_assert(constants.count(replace_name));
                constants[target_name] = constants[replace_name];
                constants.erase(replace_name);
            }
        }
    }
}

bool actg(GraphPtr graph)
{
    // graph->to_svg("graph_begin.svg");
    auto count = 0;
    std::set<NodePtr> fused_nodes;
    NodePtrList all_nodes = graph->nodes();
    for (auto node : all_nodes)
    {
        if (fused_nodes.count(node))
            continue;

        std::vector<NodePtr> actg_concat_list;
        std::vector<int32_t> actg_enables;
        bool is_actg_concat = get_actg_concat_nodes(node, actg_concat_list, actg_enables);

        std::vector<NodePtrList> wrb_crop_nodes_list;
        std::vector<int32_t> wrb_crop_enables;
        bool is_wrb_crop = get_wrb_crop_nodes(node, wrb_crop_nodes_list, wrb_crop_enables);

        bool change = is_actg_concat || is_wrb_crop;
        if (!change)
            continue;

        auto actg_region_names = get_actg_region_names(node, actg_concat_list);
        std::map<std::string, CropInfo> name_to_crop_info;
        auto wrb_crop_region_names = get_wrb_crop_region_names(node, wrb_crop_nodes_list, name_to_crop_info);

        // asnode again
        NodePtrList new_mm;
        for (auto idx = 0u; idx < actg_concat_list.size(); idx++)
        {
            auto actg_node = actg_concat_list[idx];
            if (actg_node != nullptr)
            {
                new_mm.push_back(actg_node);
            }
        }
        new_mm.push_back(node);
        for (auto idx = 0u; idx < wrb_crop_nodes_list.size(); idx++)
        {
            auto wrb_crop_nodes = wrb_crop_nodes_list[idx];
            for (auto &wrb_crop_node : wrb_crop_nodes)
            {
                new_mm.push_back(wrb_crop_node);
            }
        }
        fused_nodes.insert(new_mm.begin(), new_mm.end());

        auto new_sg = SubGraph::create(graph, new_mm);
        auto new_node = new_sg->as_node(true, false);
        bool unified_view = true;
        const char *_unified_view = getenv(Constants::SUBGRAPH_VIEW);
        if (_unified_view)
        {
            unified_view = (atoi(_unified_view) != 0);
        }
        new_node->set_unified_view(unified_view);
        for (auto kv : node->attrs())
        {
            new_node->set_attr(kv.first, kv.second);
        }
        for (auto n : new_mm)
        {
            unset_plugin(n);
        }
        rename_constants(new_node, new_mm);

        auto actg_regions = get_actg_regions(new_node, actg_region_names);
        std::vector<CropInfo> output_crops;
        auto wrb_crop_regions = get_wrb_crop_regions(new_node, wrb_crop_region_names, output_crops, name_to_crop_info);

        new_node->set_attr("actg_enables", actg_enables);
        new_node->set_attr("actg_regions", actg_regions);
        new_node->set_attr("wrb_crop_enables", wrb_crop_enables);
        new_node->set_attr("wrb_crop_regions", wrb_crop_regions);
        new_node->set_attr("output_crops", output_crops);

        change_to_actg_layout(new_node);

        count++;
    }
    INFO() << "new sgnode with actg: " << count;
    // graph->to_svg("graph_actg.svg");
    return count != 0;
}
}; // namespace aipubt
